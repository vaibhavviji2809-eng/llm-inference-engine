from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tokenizer import CharTokenizer
from transformer import DecoderOnlyTransformer, TransformerConfig
from transformer.modules import softmax


def build_batch(
    token_stream: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = token_stream.size(0) - seq_len - 1
    starts = torch.randint(low=0, high=max_start, size=(batch_size,))
    inputs = torch.stack([token_stream[start : start + seq_len] for start in starts])
    targets = torch.stack(
        [token_stream[start + 1 : start + seq_len + 1] for start in starts]
    )
    return inputs, targets


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = softmax(logits, dim=-1)
    gathered = probabilities.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return -torch.log(gathered + 1e-9).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--sample-tokens", type=int, default=20)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("benchmarks/results/tiny_transformer.pt"),
    )
    args = parser.parse_args()

    corpus_path = Path("data/tiny_corpus.txt")
    text = corpus_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    token_stream = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.seq_len,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
    )
    model = DecoderOnlyTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for step in range(1, args.steps + 1):
        inputs, targets = build_batch(token_stream, args.batch_size, args.seq_len)
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % 50 == 0 or step == args.steps:
            print(f"step={step:04d} loss={loss.item():.4f}")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(config),
            "model_summary": model.summary(),
            "state_dict": model.state_dict(),
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
        },
        args.checkpoint,
    )
    print(f"saved checkpoint to {args.checkpoint}")

    model.eval()
    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
    generated = model.generate(prompt_ids, max_new_tokens=args.sample_tokens, temperature=0.0)
    print("sample:")
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
