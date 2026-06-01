from __future__ import annotations

import torch
from torch import nn

from kv_cache import LayerKVCache
from transformer.modules import softmax, gelu


def quantize_tensor_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    max_abs = torch.max(torch.abs(weight)).clamp(min=1e-8)
    scale = max_abs / 127.0
    quantized = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
    return quantized.contiguous(), scale.to(torch.float32)


class QuantizedLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        quantized_weight, scale = quantize_tensor_int8(weight)
        self.register_buffer("qweight", quantized_weight)
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias.detach().to(torch.float32).contiguous())

    def weight(self) -> torch.Tensor:
        return self.qweight.to(torch.float32) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x.to(torch.float32), self.weight()) + self.bias


class QuantizedTokenEmbedding(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        quantized_weight, scale = quantize_tensor_int8(weight)
        self.register_buffer("qweight", quantized_weight)
        self.register_buffer("scale", scale)

    @property
    def weight(self) -> torch.Tensor:
        return self.qweight.to(torch.float32) * self.scale

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class QuantizedPositionalEmbedding(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        quantized_weight, scale = quantize_tensor_int8(weight)
        self.register_buffer("qweight", quantized_weight)
        self.register_buffer("scale", scale)

    @property
    def weight(self) -> torch.Tensor:
        return self.qweight.to(torch.float32) * self.scale

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.weight[:seq_len]


class QuantizedLayerNorm(nn.Module):
    def __init__(self, gamma: torch.Tensor, beta: torch.Tensor, eps: float) -> None:
        super().__init__()
        self.register_buffer("gamma", gamma.detach().to(torch.float32).contiguous())
        self.register_buffer("beta", beta.detach().to(torch.float32).contiguous())
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        mean = torch.mean(x, dim=-1, keepdim=True)
        variance = torch.mean((x - mean) * (x - mean), dim=-1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return normalized * self.gamma + self.beta


class QuantizedMultiHeadSelfAttention(nn.Module):
    def __init__(self, attn_module) -> None:
        super().__init__()
        self.num_heads = attn_module.num_heads
        self.head_dim = attn_module.head_dim
        self.q_proj = QuantizedLinear(attn_module.q_proj.weight.data, attn_module.q_proj.bias.data)
        self.k_proj = QuantizedLinear(attn_module.k_proj.weight.data, attn_module.k_proj.bias.data)
        self.v_proj = QuantizedLinear(attn_module.v_proj.weight.data, attn_module.v_proj.bias.data)
        self.out_proj = QuantizedLinear(
            attn_module.out_proj.weight.data,
            attn_module.out_proj.bias.data,
        )

    def project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.shape
        query = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        return query.permute(0, 2, 1, 3), key.permute(0, 2, 1, 3), value.permute(0, 2, 1, 3)

    def attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool,
    ) -> torch.Tensor:
        _, _, query_len, _ = query.shape
        _, _, key_len, _ = key.shape
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(query_len, key_len, device=query.device, dtype=torch.bool),
                diagonal=1 + key_len - query_len,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))
        attention = softmax(scores, dim=-1)
        return torch.matmul(attention, value)

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


class QuantizedFeedForward(nn.Module):
    def __init__(self, ffn_module) -> None:
        super().__init__()
        self.fc1 = QuantizedLinear(ffn_module.fc1.weight.data, ffn_module.fc1.bias.data)
        self.fc2 = QuantizedLinear(ffn_module.fc2.weight.data, ffn_module.fc2.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(gelu(self.fc1(x)))
