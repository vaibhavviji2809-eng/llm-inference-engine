from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformer import DecoderOnlyTransformer, TransformerConfig


def main() -> None:
    config = TransformerConfig(vocab_size=32, max_seq_len=8, d_model=32, num_heads=4, num_layers=2, d_ff=64)
    model = DecoderOnlyTransformer(config)
    inputs = torch.randint(0, 32, (2, 8))
    logits = model(inputs)
    print("shape", tuple(logits.shape))


if __name__ == "__main__":
    main()
