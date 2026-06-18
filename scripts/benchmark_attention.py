from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformer.flash_attention import compare_naive_and_flash_attention


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("benchmarks/results/attention_report.json"),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("benchmarks/results/attention_report.md"),
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    inputs = torch.randn(args.batch_size, args.seq_len, args.d_model)
    comparison = compare_naive_and_flash_attention(
        inputs=inputs,
        num_heads=args.heads,
        block_size=args.block_size,
        repeats=args.repeats,
    )

    report = {
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "d_model": args.d_model,
        "heads": args.heads,
        "block_size": args.block_size,
        "results": [
            {
                "method": comparison.naive.name,
                "milliseconds": comparison.naive.milliseconds,
                "tokens_per_second": comparison.naive.tokens_per_second,
                "estimated_memory_bytes": comparison.naive.peak_memory_bytes,
            },
            {
                "method": comparison.flash.name,
                "milliseconds": comparison.flash.milliseconds,
                "tokens_per_second": comparison.flash.tokens_per_second,
                "estimated_memory_bytes": comparison.flash.peak_memory_bytes,
            },
        ],
        "speedup": comparison.speedup,
        "memory_savings": comparison.memory_savings,
    }

    markdown = f"""# Attention Benchmark Report

| Method | Milliseconds | Tokens/sec | Estimated Memory (bytes) |
| --- | ---: | ---: | ---: |
| {report["results"][0]["method"]} | {report["results"][0]["milliseconds"]:.4f} | {report["results"][0]["tokens_per_second"]:.2f} | {report["results"][0]["estimated_memory_bytes"]} |
| {report["results"][1]["method"]} | {report["results"][1]["milliseconds"]:.4f} | {report["results"][1]["tokens_per_second"]:.2f} | {report["results"][1]["estimated_memory_bytes"]} |

Speedup: `{report["speedup"]:.2f}x`
Estimated memory savings: `{report["memory_savings"]:.2%}`
"""

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.markdown_out.write_text(markdown, encoding="utf-8")

    print(f"wrote {args.json_out}")
    print(f"wrote {args.markdown_out}")


if __name__ == "__main__":
    main()
