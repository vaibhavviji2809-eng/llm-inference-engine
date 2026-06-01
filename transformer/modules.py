from __future__ import annotations

import math

import torch
from torch import nn


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    shifted = x - torch.amax(x, dim=dim, keepdim=True)
    numerator = torch.exp(shifted)
    denominator = torch.sum(numerator, dim=dim, keepdim=True)
    return numerator / denominator


def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        scale = 1.0 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * scale)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight) + self.bias


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(max_seq_len, d_model) * 0.02)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.weight[:seq_len]


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x, dim=-1, keepdim=True)
        variance = torch.mean((x - mean) * (x - mean), dim=-1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return normalized * self.gamma + self.beta
