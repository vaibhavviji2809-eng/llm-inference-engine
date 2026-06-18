from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from kv_cache import LayerKVCache
from .modules import Linear, softmax


@dataclass
class AttentionBenchmark:
    name: str
    milliseconds: float
    peak_memory_bytes: int
    tokens_per_second: float


class _AttentionBase(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)

    def project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        query = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        return query.permute(0, 2, 1, 3), key.permute(0, 2, 1, 3), value.permute(0, 2, 1, 3)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, heads, seq_len, head_dim = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, heads * head_dim)

    def estimate_memory_bytes(self, batch_size: int, query_len: int, key_len: int) -> int:
        return batch_size * self.num_heads * query_len * key_len * 4

    def attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward_incremental(
        self,
        x: torch.Tensor,
        layer_cache: LayerKVCache | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value = self.project(x)
        if layer_cache is not None:
            key = torch.cat([layer_cache.keys, key], dim=2)
            value = torch.cat([layer_cache.values, value], dim=2)
        context = self.attend(query, key, value, is_causal=layer_cache is None)
        return self.out_proj(self.merge_heads(context)), key[:, :, -x.size(1) :, :], value[:, :, -x.size(1) :, :]


class NaiveAttention(_AttentionBase):
    def attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool,
    ) -> torch.Tensor:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if is_causal:
            q_len, k_len = scores.size(-2), scores.size(-1)
            causal_mask = torch.triu(
                torch.ones(q_len, k_len, device=query.device, dtype=torch.bool),
                diagonal=1 + k_len - q_len,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))
        attention = softmax(scores, dim=-1)
        return torch.matmul(attention, value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query, key, value = self.project(x)
        context = self.attend(query, key, value, is_causal=True)
        return self.out_proj(self.merge_heads(context))


class FlashAttention(_AttentionBase):
    def __init__(self, d_model: int, num_heads: int, block_size: int = 64) -> None:
        super().__init__(d_model=d_model, num_heads=num_heads)
        self.block_size = block_size

    def _block_mask(
        self,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
    ) -> torch.Tensor:
        return key_positions > query_positions

    def attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool,
    ) -> torch.Tensor:
        if not is_causal:
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attention = softmax(scores, dim=-1)
            return torch.matmul(attention, value)

        batch_size, heads, query_len, _ = query.shape
        key_len = key.size(2)
        scale = 1.0 / math.sqrt(self.head_dim)

        output = torch.zeros_like(query)
        running_max = torch.full(
            (batch_size, heads, query_len, 1),
            float("-inf"),
            device=query.device,
            dtype=query.dtype,
        )
        running_sum = torch.zeros((batch_size, heads, query_len, 1), device=query.device, dtype=query.dtype)
        query_positions = torch.arange(query_len, device=query.device).view(1, 1, query_len, 1)

        for start in range(0, key_len, self.block_size):
            stop = min(start + self.block_size, key_len)
            key_positions = torch.arange(start, stop, device=query.device).view(1, 1, 1, -1)
            key_block = key[:, :, start:stop, :]
            value_block = value[:, :, start:stop, :]

            scores = torch.matmul(query, key_block.transpose(-2, -1)) * scale
            scores = scores.masked_fill(self._block_mask(query_positions, key_positions), float("-inf"))

            block_max = torch.amax(scores, dim=-1, keepdim=True)
            new_max = torch.maximum(running_max, block_max)
            max_shift = torch.exp(running_max - new_max)
            block_shift = torch.exp(scores - new_max)

            output = output * max_shift + torch.matmul(block_shift, value_block)
            running_sum = running_sum * max_shift + torch.sum(block_shift, dim=-1, keepdim=True)
            running_max = new_max

        return output / torch.clamp(running_sum, min=1e-9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query, key, value = self.project(x)
        context = self.attend(query, key, value, is_causal=True)
        return self.out_proj(self.merge_heads(context))

    def estimate_memory_bytes(self, batch_size: int, query_len: int, key_len: int) -> int:
        block_memory = batch_size * self.num_heads * query_len * self.block_size * 4
        return min(super().estimate_memory_bytes(batch_size, query_len, key_len), block_memory)
