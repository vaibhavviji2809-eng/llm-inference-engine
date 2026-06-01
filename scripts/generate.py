from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tokenizer import CharTokenizer
from transformer import DecoderOnlyTransformer, TransformerConfig


def normalize_config(config_payload: dict) -> dict:
    allowed_keys = {
        "vocab_size",
        "max_seq_len",
        "d_model",
        "num_heads",
        "num_layers",
        "d_ff",
        "dropout",
        "eps",
    }
    return {key: value for key, value in config_payload.items() if key in allowed_keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("benchmarks/results/tiny_transformer.pt"),
    )
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--sample-tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu")
    corpus_text = Path("data/tiny_corpus.txt").read_text(encoding="utf-8")
    tokenizer = CharTokenizer(corpus_text)
    config = TransformerConfig(**normalize_config(payload["config"]))
    model = DecoderOnlyTransformer(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
    generated = model.generate(
        prompt_ids,
        max_new_tokens=args.sample_tokens,
        temperature=args.temperature,
    )
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
