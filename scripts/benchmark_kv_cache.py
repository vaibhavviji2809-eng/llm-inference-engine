from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.app.runtime import TransformerRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("benchmarks/results/tiny_transformer.pt"),
    )
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--sample-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    runtime = TransformerRuntime(checkpoint_path=args.checkpoint)
    report = runtime.benchmark(
        prompt=args.prompt,
        max_new_tokens=args.sample_tokens,
        repeats=args.repeats,
    )
    uncached = next(result for result in report["results"] if result["method"] == "full_recompute")
    cached = next(result for result in report["results"] if result["method"] == "kv_cache")

    print("KV cache benchmark")
    print(f"prompt={args.prompt!r} sample_tokens={args.sample_tokens} repeats={args.repeats}")
    print("")
    print(f"{'Method':<24}{'Seconds':<14}{'Tokens/sec':<14}")
    print(f"{'Full recompute':<24}{uncached['seconds']:<14.6f}{uncached['tokens_per_second']:<14.2f}")
    print(f"{'KV cache':<24}{cached['seconds']:<14.6f}{cached['tokens_per_second']:<14.2f}")
    print("")
    print("uncached sample:")
    print(uncached["sample"])
    print("")
    print("cached sample:")
    print(cached["sample"])


if __name__ == "__main__":
    main()
