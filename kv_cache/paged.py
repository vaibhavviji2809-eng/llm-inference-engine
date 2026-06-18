from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .cache import KVCache, LayerKVCache


@dataclass
class KVPage:
    page_id: int
    start_index: int
    keys: torch.Tensor
    values: torch.Tensor

    @property
    def length(self) -> int:
        return int(self.keys.size(2))

    @property
    def capacity(self) -> int:
        return int(self.keys.size(2))


@dataclass
class KVBlock:
    block_id: int
    page_size: int
    pages: list[KVPage] = field(default_factory=list)
    hits: int = 0
    misses: int = 0

    def append(self, keys: torch.Tensor, values: torch.Tensor, page_id_start: int = 0) -> int:
        seq_len = int(keys.size(2))
        created = 0
        for offset in range(0, seq_len, self.page_size):
            page_keys = keys[:, :, offset : offset + self.page_size, :].clone()
            page_values = values[:, :, offset : offset + self.page_size, :].clone()
            self.pages.append(
                KVPage(
                    page_id=page_id_start + created,
                    start_index=offset,
                    keys=page_keys,
                    values=page_values,
                )
            )
            self.hits += int(page_keys.size(2))
            created += 1
        return created

    def materialize(self) -> LayerKVCache | None:
        if not self.pages:
            return None
        keys = torch.cat([page.keys for page in self.pages], dim=2)
        values = torch.cat([page.values for page in self.pages], dim=2)
        return LayerKVCache(keys=keys, values=values)

    def sequence_length(self) -> int:
        return sum(page.length for page in self.pages)


class KVAllocator:
    def __init__(self, page_size: int) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        self.page_size = page_size
        self.next_page_id = 0
        self.free_pages: list[int] = []

    def allocate_page(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        start_index: int,
    ) -> KVPage:
        page_id = self.free_pages.pop() if self.free_pages else self.next_page_id
        if not self.free_pages:
            self.next_page_id += 1
        return KVPage(page_id=page_id, start_index=start_index, keys=keys.clone(), values=values.clone())

    def release_page(self, page: KVPage) -> None:
        self.free_pages.append(page.page_id)


class PagedKVCache:
    def __init__(self, num_layers: int, page_size: int = 16) -> None:
        self.page_size = page_size
        self.allocator = KVAllocator(page_size=page_size)
        self.blocks: list[KVBlock | None] = [None for _ in range(num_layers)]

    def get(self, layer_index: int) -> KVBlock | None:
        return self.blocks[layer_index]

    def update(self, layer_index: int, keys: torch.Tensor, values: torch.Tensor) -> KVBlock:
        block = self.blocks[layer_index]
        if block is None:
            block = KVBlock(block_id=layer_index, page_size=self.page_size)
        created = block.append(keys, values, page_id_start=self.allocator.next_page_id)
        self.allocator.next_page_id += created
        self.blocks[layer_index] = block
        return block

    def materialize(self, layer_index: int) -> LayerKVCache | None:
        block = self.blocks[layer_index]
        if block is None:
            return None
        return block.materialize()

    def sequence_length(self) -> int:
        for block in self.blocks:
            if block is not None:
                return block.sequence_length()
        return 0

    def to_dense(self) -> KVCache:
        cache = KVCache(num_layers=len(self.blocks))
        for index, block in enumerate(self.blocks):
            if block is None:
                continue
            cache.layers[index] = block.materialize()
        return cache
