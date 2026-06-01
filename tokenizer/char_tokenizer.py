from __future__ import annotations


class CharTokenizer:
    def __init__(self, text: str) -> None:
        symbols = sorted(set(text))
        self.stoi = {symbol: index for index, symbol in enumerate(symbols)}
        self.itos = {index: symbol for symbol, index in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[character] for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.itos[token_id] for token_id in token_ids)
