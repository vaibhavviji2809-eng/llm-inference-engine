from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantization import (
    create_fp16_model,
    create_int8_weight_only_model,
    estimate_model_bytes,
)
from scripts.generate import normalize_config
from server.app.runtime import TransformerRuntime
from tokenizer import CharTokenizer
from transformer import DecoderOnlyTransformer, TransformerConfig
from transformer.modules import softmax


def bytes_to_mib(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def load_model(checkpoint: Path) -> tuple[DecoderOnlyTransformer, CharTokenizer, torch.Tensor]:
    payload = torch.load(checkpoint, map_location="cpu")
    corpus_text = (PROJECT_ROOT / "data" / "tiny_corpus.txt").read_text(encoding="utf-8")
    tokenizer = CharTokenizer(corpus_text)
    config = TransformerConfig(**normalize_config(payload["config"]))
    model = DecoderOnlyTransformer(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    token_stream = torch.tensor(tokenizer.encode(corpus_text), dtype=torch.long)
    return model, tokenizer, token_stream


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> float:
    probabilities = softmax(logits.to(torch.float32), dim=-1)
    gathered = probabilities.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return float((-torch.log(gathered + 1e-9).mean()).item())


def validation_loss(model: DecoderOnlyTransformer, token_stream: torch.Tensor, seq_len: int) -> float:
    usable = (token_stream.size(0) - 1) // seq_len
    inputs = token_stream[: usable * seq_len].view(usable, seq_len)
    targets = token_stream[1 : usable * seq_len + 1].view(usable, seq_len)
    with torch.no_grad():
        logits = model(inputs)
    return cross_entropy_loss(logits, targets)


def benchmark_generation(
    model: DecoderOnlyTransformer,
    prompt_ids: torch.Tensor,
    sample_tokens: int,
    repeats: int,
) -> tuple[float, torch.Tensor]:
    import time

    start = time.perf_counter()
    output = prompt_ids
    for _ in range(repeats):
        output = model.generate_with_kv_cache(
            prompt_ids,
            max_new_tokens=sample_tokens,
            temperature=0.0,
        )
    elapsed = time.perf_counter() - start
    return elapsed / repeats, output


def collect_quantization_report(
    checkpoint: Path,
    prompt: str,
    sample_tokens: int,
    repeats: int,
    seq_len: int,
) -> dict[str, Any]:
    fp32_model, tokenizer, token_stream = load_model(checkpoint)
    fp16_model = create_fp16_model(fp32_model)
    int8_model = create_int8_weight_only_model(fp32_model)
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

    variants = [
        ("FP32", fp32_model),
        ("FP16", fp16_model),
        ("INT8", int8_model),
    ]

    results: list[dict[str, Any]] = []
    for label, model in variants:
        loss = validation_loss(model, token_stream, seq_len)
        seconds, output = benchmark_generation(model, prompt_ids, sample_tokens, repeats)
        results.append(
            {
                "type": label,
                "memory_mib": bytes_to_mib(estimate_model_bytes(model)),
                "seconds": seconds,
                "tokens_per_second": sample_tokens / seconds,
                "loss": loss,
                "sample": tokenizer.decode(output[0].tolist()),
            }
        )

    baseline_loss = results[0]["loss"]
    for result in results:
        result["loss_delta"] = result["loss"] - baseline_loss

    return {
        "prompt": prompt,
        "sample_tokens": sample_tokens,
        "repeats": repeats,
        "results": results,
    }


def render_markdown(report: dict[str, Any]) -> str:
    kv_rows = report["kv_cache"]["results"]
    batch_rows = report["batching"]["results"]
    quant_rows = report["quantization"]["results"]
    return f"""# Benchmark Report

Generated automatically for the current local environment.

## Environment

- Python: `{report["environment"]["python"]}`
- PyTorch: `{report["environment"]["torch"]}`
- CUDA available: `{report["environment"]["cuda_available"]}`
- Platform: `{report["environment"]["platform"]}`

## KV Cache

| Method | Seconds | Tokens/sec |
| --- | ---: | ---: |
| {kv_rows[0]["method"]} | {kv_rows[0]["seconds"]:.6f} | {kv_rows[0]["tokens_per_second"]:.2f} |
| {kv_rows[1]["method"]} | {kv_rows[1]["seconds"]:.6f} | {kv_rows[1]["tokens_per_second"]:.2f} |

## Batching

| Method | Seconds | Tokens/sec |
| --- | ---: | ---: |
| {batch_rows[0]["method"]} | {batch_rows[0]["seconds"]:.6f} | {batch_rows[0]["tokens_per_second"]:.2f} |
| {batch_rows[1]["method"]} | {batch_rows[1]["seconds"]:.6f} | {batch_rows[1]["tokens_per_second"]:.2f} |

## Quantization

| Type | Memory (MiB) | Seconds | Tokens/sec | Loss | Loss Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| {quant_rows[0]["type"]} | {quant_rows[0]["memory_mib"]:.4f} | {quant_rows[0]["seconds"]:.6f} | {quant_rows[0]["tokens_per_second"]:.2f} | {quant_rows[0]["loss"]:.4f} | {quant_rows[0]["loss_delta"]:.4f} |
| {quant_rows[1]["type"]} | {quant_rows[1]["memory_mib"]:.4f} | {quant_rows[1]["seconds"]:.6f} | {quant_rows[1]["tokens_per_second"]:.2f} | {quant_rows[1]["loss"]:.4f} | {quant_rows[1]["loss_delta"]:.4f} |
| {quant_rows[2]["type"]} | {quant_rows[2]["memory_mib"]:.4f} | {quant_rows[2]["seconds"]:.6f} | {quant_rows[2]["tokens_per_second"]:.2f} | {quant_rows[2]["loss"]:.4f} | {quant_rows[2]["loss_delta"]:.4f} |

## Notes

- KV cache and batching are measured on the current tiny checkpoint.
- CUDA kernel execution is not included in this report because this machine does not have a CUDA runtime.
- Batched decoding now uses per-request KV cache state grouped by matching cache lengths.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("benchmarks/results/tiny_transformer.pt"),
    )
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--sample-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["Hello", "Hello how", "Hello there", "Hello how are"],
    )
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("benchmarks/results/latest_report.json"),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("benchmarks/results/latest_report.md"),
    )
    args = parser.parse_args()

    runtime = TransformerRuntime(checkpoint_path=args.checkpoint)
    report = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "platform": platform.platform(),
        },
        "kv_cache": runtime.benchmark(
            prompt=args.prompt,
            max_new_tokens=args.sample_tokens,
            repeats=args.repeats,
        ),
        "batching": runtime.benchmark_batching(
            prompts=args.prompts,
            max_new_tokens=args.sample_tokens,
            repeats=args.repeats,
        ),
        "quantization": collect_quantization_report(
            checkpoint=args.checkpoint,
            prompt=args.prompt,
            sample_tokens=args.sample_tokens,
            repeats=args.repeats,
            seq_len=args.seq_len,
        ),
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.markdown_out.write_text(render_markdown(report), encoding="utf-8")

    print(f"wrote {args.json_out}")
    print(f"wrote {args.markdown_out}")


if __name__ == "__main__":
    main()
