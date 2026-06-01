from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LayerKVCache:
    keys: torch.Tensor
    values: torch.Tensor


class KVCache:
    def __init__(self, num_layers: int) -> None:
        self.layers: list[LayerKVCache | None] = [None for _ in range(num_layers)]

    def get(self, layer_index: int) -> LayerKVCache | None:
        return self.layers[layer_index]

    def update(
        self,
        layer_index: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> LayerKVCache:
        current = self.layers[layer_index]
        if current is None:
            updated = LayerKVCache(keys=keys, values=values)
        else:
            updated = LayerKVCache(
                keys=torch.cat([current.keys, keys], dim=2),
                values=torch.cat([current.values, values], dim=2),
            )
        self.layers[layer_index] = updated
        return updated

    def sequence_length(self) -> int:
        first_layer = next((layer for layer in self.layers if layer is not None), None)
        if first_layer is None:
            return 0
        return int(first_layer.keys.size(2))
