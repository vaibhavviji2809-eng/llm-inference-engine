from dataclasses import dataclass


@dataclass
class TransformerConfig:
    vocab_size: int
    max_seq_len: int = 32
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.0
    eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        return self.d_model // self.num_heads
