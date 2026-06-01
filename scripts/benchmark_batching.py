from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from batching import BatchGenerateRequest, ContinuousBatcher
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
    batcher = ContinuousBatcher(runtime.model)
    prompt_ids = [runtime.encode_prompt(prompt) for prompt in args.prompts]

    serial_start = time.perf_counter()
    for _ in range(args.repeats):
        serial_outputs = [
            runtime.model.generate_with_kv_cache(
                ids,
                max_new_tokens=args.sample_tokens,
                temperature=0.0,
            )
            for ids in prompt_ids
        ]
    serial_elapsed = (time.perf_counter() - serial_start) / args.repeats

    batch_start = time.perf_counter()
    batched_report = None
    for _ in range(args.repeats):
        requests = [
            BatchGenerateRequest(
                request_id=f"req-{index}",
                prompt_ids=ids,
                max_new_tokens=args.sample_tokens,
                temperature=0.0,
            )
            for index, ids in enumerate(prompt_ids)
        ]
        batched_report = batcher.generate(requests)
    batched_elapsed = (time.perf_counter() - batch_start) / args.repeats

    request_count = len(args.prompts)
    total_tokens = request_count * args.sample_tokens
    print("Continuous batching benchmark")
    print(f"prompts={request_count} sample_tokens={args.sample_tokens} repeats={args.repeats}")
    print("")
    print(f"{'Method':<24}{'Seconds':<14}{'Tokens/sec':<14}")
    print(f"{'Serial requests':<24}{serial_elapsed:<14.6f}{(total_tokens / serial_elapsed):<14.2f}")
    print(f"{'Continuous batching':<24}{batched_elapsed:<14.6f}{(total_tokens / batched_elapsed):<14.2f}")
    print("")
    print("serial samples:")
    for output in serial_outputs:
        print(runtime.decode_tokens(output))
        print("")
    print("batched samples:")
    assert batched_report is not None
    for result in batched_report["results"]:
        print(runtime.decode_tokens(result["generated"]))
        print("")


if __name__ == "__main__":
    main()
