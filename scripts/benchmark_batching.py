from __future__ import annotations

import argparse
from pathlib import Path
import sys

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
    parser.add_argument("--sample-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["Hello", "Hello how", "Hello there", "Hello how are"],
    )
    args = parser.parse_args()

    runtime = TransformerRuntime(checkpoint_path=args.checkpoint)
    report = runtime.benchmark_batching(
        prompts=args.prompts,
        max_new_tokens=args.sample_tokens,
        repeats=args.repeats,
    )
    serial = next(result for result in report["results"] if result["method"] == "serial_kv_cache")
    batched = next(result for result in report["results"] if result["method"] != "serial_kv_cache")

    print("Continuous batching benchmark")
    print(f"prompts={len(args.prompts)} sample_tokens={args.sample_tokens} repeats={args.repeats}")
    print("")
    print(f"{'Method':<24}{'Seconds':<14}{'Tokens/sec':<14}{'Batch Size':<14}")
    print(
        f"{'Serial KV cache':<24}{serial['seconds']:<14.6f}{serial['tokens_per_second']:<14.2f}"
        f"{serial['avg_batch_size']:<14.2f}"
    )
    print(
        f"{'Batched KV cache':<24}{batched['seconds']:<14.6f}{batched['tokens_per_second']:<14.2f}"
        f"{batched['avg_batch_size']:<14.2f}"
    )
    print("")
    print(f"speedup_vs_serial={batched['speedup_vs_serial']:.2f}x")
    print(f"cache_hit_rate={batched['cache_hit_rate']:.2%}")
    print(f"cache_rebuilds={batched['cache_rebuilds']}")
    print(f"prefill_groups={batched['prefill_groups']}")
    print(f"decode_groups={batched['decode_groups']}")
    print("")
    print("serial samples:")
    for output in serial["samples"]:
        print(output)
        print("")
    print("batched samples:")
    for output in batched["samples"]:
        print(output)
        print("")


if __name__ == "__main__":
    main()
