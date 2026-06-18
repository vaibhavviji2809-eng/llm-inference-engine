from __future__ import annotations

from dataclasses import dataclass
import time

import torch

from .attention import AttentionBenchmark, FlashAttention, NaiveAttention


@dataclass
class FlashAttentionComparison:
    batch_size: int
    seq_len: int
    d_model: int
    num_heads: int
    block_size: int
    repeats: int
    naive: AttentionBenchmark
    flash: AttentionBenchmark

    @property
    def speedup(self) -> float:
        return self.naive.milliseconds / self.flash.milliseconds if self.flash.milliseconds > 0 else 0.0

    @property
    def memory_savings(self) -> float:
        if self.naive.peak_memory_bytes <= 0:
            return 0.0
        return 1.0 - (self.flash.peak_memory_bytes / self.naive.peak_memory_bytes)


def _benchmark_model(model: torch.nn.Module, inputs: torch.Tensor, repeats: int, name: str) -> AttentionBenchmark:
    model.eval()
    with torch.no_grad():
        start = time.perf_counter()
        output = None
        for _ in range(repeats):
            output = model(inputs)
        elapsed = (time.perf_counter() - start) / repeats
    assert output is not None
    peak_memory_bytes = getattr(model, "estimate_memory_bytes", lambda *_: 0)(
        inputs.size(0),
        inputs.size(1),
        inputs.size(1),
    )
    tokens_per_second = (inputs.size(0) * inputs.size(1)) / elapsed if elapsed > 0 else 0.0
    return AttentionBenchmark(
        name=name,
        milliseconds=elapsed * 1000.0,
        peak_memory_bytes=peak_memory_bytes,
        tokens_per_second=tokens_per_second,
    )


def compare_naive_and_flash_attention(
    inputs: torch.Tensor,
    num_heads: int,
    block_size: int = 64,
    repeats: int = 10,
) -> FlashAttentionComparison:
    batch_size, seq_len, d_model = inputs.shape
    naive = NaiveAttention(d_model=d_model, num_heads=num_heads)
    flash = FlashAttention(d_model=d_model, num_heads=num_heads, block_size=block_size)

    naive_result = _benchmark_model(naive, inputs, repeats, "naive_attention")
    flash_result = _benchmark_model(flash, inputs, repeats, "flash_attention")

    return FlashAttentionComparison(
        batch_size=batch_size,
        seq_len=seq_len,
        d_model=d_model,
        num_heads=num_heads,
        block_size=block_size,
        repeats=repeats,
        naive=naive_result,
        flash=flash_result,
    )


__all__ = [
    "AttentionBenchmark",
    "NaiveAttention",
    "FlashAttention",
    "FlashAttentionComparison",
    "compare_naive_and_flash_attention",
]
