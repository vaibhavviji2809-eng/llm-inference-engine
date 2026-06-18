from __future__ import annotations

import math
from dataclasses import asdict

import torch
from torch import nn

from kv_cache import KVCache, LayerKVCache
from .config import TransformerConfig
from .modules import LayerNorm, Linear, PositionalEmbedding, TokenEmbedding, gelu, softmax
from .flash_attention import FlashAttention, NaiveAttention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.q_proj = Linear(config.d_model, config.d_model)
        self.k_proj = Linear(config.d_model, config.d_model)
        self.v_proj = Linear(config.d_model, config.d_model)
        self.out_proj = Linear(config.d_model, config.d_model)

    def project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.shape

        query = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        return query, key, value

    def attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool,
    ) -> torch.Tensor:
        _, _, query_len, _ = query.shape
        _, _, key_len, _ = key.shape

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(query_len, key_len, device=query.device, dtype=torch.bool),
                diagonal=1 + key_len - query_len,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))
        attention = softmax(scores, dim=-1)
        context = torch.matmul(attention, value)
        return context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        query, key, value = self.project(x)
        context = self.attend(query, key, value, is_causal=True)

        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(context)

    def forward_incremental(
        self,
        x: torch.Tensor,
        layer_cache: LayerKVCache | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.shape
        query, key, value = self.project(x)

        if layer_cache is not None:
            full_key = torch.cat([layer_cache.keys, key], dim=2)
            full_value = torch.cat([layer_cache.values, value], dim=2)
        else:
            full_key = key
            full_value = value

        context = self.attend(query, full_key, full_value, is_causal=False)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(context), key, value


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.fc1 = Linear(config.d_model, config.d_ff)
        self.fc2 = Linear(config.d_ff, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = LayerNorm(config.d_model, eps=config.eps)
        if config.attention_backend == "flash":
            self.attn = FlashAttention(
                d_model=config.d_model,
                num_heads=config.num_heads,
                block_size=config.flash_block_size,
            )
        else:
            self.attn = NaiveAttention(
                d_model=config.d_model,
                num_heads=config.num_heads,
            )
        self.ln2 = LayerNorm(config.d_model, eps=config.eps)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

    def forward_incremental(
        self,
        x: torch.Tensor,
        layer_cache: LayerKVCache | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_input = self.ln1(x)
        attn_output, key, value = self.attn.forward_incremental(attn_input, layer_cache)
        x = x + attn_output
        x = x + self.ffn(self.ln2(x))
        return x, key, value


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = TokenEmbedding(config.vocab_size, config.d_model)
        self.position_embedding = PositionalEmbedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.final_norm = LayerNorm(config.d_model, eps=config.eps)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError("Sequence length exceeds max_seq_len.")

        token_vectors = self.token_embedding(token_ids)
        position_vectors = self.position_embedding(seq_len).unsqueeze(0).expand(
            batch_size, seq_len, self.config.d_model
        )
        hidden = token_vectors + position_vectors

        for block in self.blocks:
            hidden = block(hidden)

        hidden = self.final_norm(hidden)
        logits = torch.matmul(hidden, self.token_embedding.weight.transpose(0, 1))
        return logits

    def forward_incremental(
        self,
        token_ids: torch.Tensor,
        cache: KVCache,
        position_offset: int,
    ) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        total_seq_len = position_offset + seq_len
        if total_seq_len > self.config.max_seq_len:
            raise ValueError("Sequence length exceeds max_seq_len while using KV cache.")

        token_vectors = self.token_embedding(token_ids)
        position_vectors = self.position_embedding.weight[
            position_offset : position_offset + seq_len
        ]
        hidden = token_vectors + position_vectors.unsqueeze(0).expand(
            batch_size, seq_len, self.config.d_model
        )

        for layer_index, block in enumerate(self.blocks):
            layer_cache = cache.get(layer_index)
            hidden, key, value = block.forward_incremental(hidden, layer_cache)
            cache.update(layer_index, key, value)

        hidden = self.final_norm(hidden)
        logits = torch.matmul(hidden, self.token_embedding.weight.transpose(0, 1))
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        generated = prompt_ids

        for _ in range(max_new_tokens):
            window = generated[:, -self.config.max_seq_len :]
            logits = self.forward(window)
            next_token_logits = logits[:, -1, :]
            if temperature <= 0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                scaled = next_token_logits / temperature
                probs = softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    @torch.no_grad()
    def generate_with_kv_cache(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        generated = prompt_ids.clone()
        cache = KVCache(num_layers=len(self.blocks))

        prompt_window = generated[:, -self.config.max_seq_len :]
        prompt_prefix_len = prompt_window.size(1)
        logits = self.forward_incremental(prompt_window, cache, position_offset=0)

        for _ in range(max_new_tokens):
            next_token_logits = logits[:, -1, :]
            if temperature <= 0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                scaled = next_token_logits / temperature
                probs = softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if cache.sequence_length() >= self.config.max_seq_len:
                window = generated[:, -self.config.max_seq_len :]
                cache = KVCache(num_layers=len(self.blocks))
                logits = self.forward_incremental(window, cache, position_offset=0)
            else:
                logits = self.forward_incremental(
                    next_token,
                    cache,
                    position_offset=cache.sequence_length(),
                )

        if generated.size(1) > prompt_prefix_len + max_new_tokens:
            return generated[:, -(prompt_prefix_len + max_new_tokens) :]
        return generated

    def summary(self) -> dict[str, int]:
        trainable_params = sum(parameter.numel() for parameter in self.parameters())
        details = asdict(self.config)
        details["trainable_params"] = trainable_params
        return details
