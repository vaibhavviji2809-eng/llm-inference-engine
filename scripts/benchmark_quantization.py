from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

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
from tokenizer import CharTokenizer
from transformer import DecoderOnlyTransformer, TransformerConfig
from transformer.modules import softmax


def load_model(checkpoint: Path) -> tuple[DecoderOnlyTransformer, CharTokenizer, torch.Tensor]:
    payload = torch.load(checkpoint, map_location="cpu")
    corpus_text = Path("data/tiny_corpus.txt").read_text(encoding="utf-8")
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
) -> tuple[float, str]:
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


def bytes_to_mib(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("benchmarks/results/tiny_transformer.pt"),
    )
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--sample-tokens", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=16)
    args = parser.parse_args()

    fp32_model, tokenizer, token_stream = load_model(args.checkpoint)
    fp16_model = create_fp16_model(fp32_model)
    int8_model = create_int8_weight_only_model(fp32_model)

    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)

    variants = [
        ("FP32", fp32_model),
        ("FP16", fp16_model),
        ("INT8", int8_model),
    ]

    results: list[dict[str, object]] = []
    for label, model in variants:
        loss = validation_loss(model, token_stream, args.seq_len)
        seconds, output = benchmark_generation(model, prompt_ids, args.sample_tokens, args.repeats)
        results.append(
            {
                "type": label,
                "memory_mib": bytes_to_mib(estimate_model_bytes(model)),
                "seconds": seconds,
                "tokens_per_second": args.sample_tokens / seconds,
                "loss": loss,
                "sample": tokenizer.decode(output[0].tolist()),
            }
        )

    baseline_loss = float(results[0]["loss"])

    print("Quantization benchmark")
    print(f"prompt={args.prompt!r} sample_tokens={args.sample_tokens} repeats={args.repeats}")
    print("")
    print(
        f"{'Type':<10}{'Memory (MiB)':<16}{'Seconds':<14}{'Tokens/sec':<14}{'Loss':<12}{'Loss Delta':<12}"
    )
    for result in results:
        loss_delta = float(result["loss"]) - baseline_loss
        print(
            f"{result['type']:<10}"
            f"{float(result['memory_mib']):<16.4f}"
            f"{float(result['seconds']):<14.6f}"
            f"{float(result['tokens_per_second']):<14.2f}"
            f"{float(result['loss']):<12.4f}"
            f"{loss_delta:<12.4f}"
        )

    print("")
    for result in results:
        print(f"{result['type']} sample:")
        print(result["sample"])
        print("")


if __name__ == "__main__":
    main()
