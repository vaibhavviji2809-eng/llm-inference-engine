from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .cache import KVCache, LayerKVCache


@dataclass
class KVPage:
    page_id: int
    start_index: int
    page_size: int
    keys: torch.Tensor
    values: torch.Tensor

    @property
    def length(self) -> int:
        return int(self.keys.size(2))

    @property
    def capacity(self) -> int:
        return self.page_size

    @property
    def utilization(self) -> float:
        capacity = self.capacity
        return float(self.length / capacity) if capacity > 0 else 0.0


@dataclass
class KVBlock:
    block_id: int
    page_size: int
    pages: list[KVPage] = field(default_factory=list)
    stored_tokens: int = 0
    evicted_tokens: int = 0

    def append_page(self, page: KVPage) -> None:
        self.pages.append(page)
        self.stored_tokens += page.length

    def append(self, keys: torch.Tensor, values: torch.Tensor, allocator: "KVAllocator") -> int:
        seq_len = int(keys.size(2))
        created = 0
        for offset in range(0, seq_len, self.page_size):
            page_keys = keys[:, :, offset : offset + self.page_size, :].clone()
            page_values = values[:, :, offset : offset + self.page_size, :].clone()
            page = KVPage(
                page_id=allocator.acquire_page_id(),
                start_index=offset,
                page_size=self.page_size,
                keys=page_keys,
                values=page_values,
            )
            self.append_page(page)
            created += 1
        return created

    def evict_oldest_page(self, allocator: "KVAllocator") -> KVPage | None:
        if not self.pages:
            return None
        page = self.pages.pop(0)
        self.evicted_tokens += page.length
        allocator.release_page_id(page.page_id)
        return page

    def evict_to_fit(self, allocator: "KVAllocator", max_pages: int | None) -> int:
        if max_pages is None or max_pages < 0:
            return 0
        evicted = 0
        while len(self.pages) > max_pages:
            if self.evict_oldest_page(allocator) is None:
                break
            evicted += 1
        return evicted

    def materialize(self) -> LayerKVCache | None:
        if not self.pages:
            return None
        keys = torch.cat([page.keys for page in self.pages], dim=2)
        values = torch.cat([page.values for page in self.pages], dim=2)
        return LayerKVCache(keys=keys, values=values)

    def sequence_length(self) -> int:
        return sum(page.length for page in self.pages)

    def capacity_tokens(self) -> int:
        return len(self.pages) * self.page_size

    def fragmentation_ratio(self) -> float:
        capacity = self.capacity_tokens()
        if capacity == 0:
            return 0.0
        return max(0.0, 1.0 - (self.sequence_length() / capacity))


@dataclass
class AllocatorStats:
    allocated_pages: int = 0
    reused_pages: int = 0
    evicted_pages: int = 0
    allocated_blocks: int = 0
    reused_blocks: int = 0


class KVAllocator:
    def __init__(self, page_size: int) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        self.page_size = page_size
        self.next_page_id = 0
        self.free_page_ids: list[int] = []
        self.next_block_id = 0
        self.free_block_ids: list[int] = []
        self.stats = AllocatorStats()

    def acquire_page_id(self) -> int:
        if self.free_page_ids:
            self.stats.reused_pages += 1
            return self.free_page_ids.pop()
        page_id = self.next_page_id
        self.next_page_id += 1
        self.stats.allocated_pages += 1
        return page_id

    def release_page_id(self, page_id: int) -> None:
        self.free_page_ids.append(page_id)
        self.stats.evicted_pages += 1

    def acquire_block_id(self) -> int:
        if self.free_block_ids:
            self.stats.reused_blocks += 1
            return self.free_block_ids.pop()
        block_id = self.next_block_id
        self.next_block_id += 1
        self.stats.allocated_blocks += 1
        return block_id

    def release_block_id(self, block_id: int) -> None:
        self.free_block_ids.append(block_id)

    def snapshot(self) -> dict[str, int]:
        return {
            "allocated_pages": self.stats.allocated_pages,
            "reused_pages": self.stats.reused_pages,
            "evicted_pages": self.stats.evicted_pages,
            "allocated_blocks": self.stats.allocated_blocks,
            "reused_blocks": self.stats.reused_blocks,
            "free_pages": len(self.free_page_ids),
            "free_blocks": len(self.free_block_ids),
        }


class PagedKVCache:
    def __init__(self, num_layers: int, page_size: int = 16, max_pages_per_block: int | None = None) -> None:
        if max_pages_per_block is not None and max_pages_per_block <= 0:
            raise ValueError("max_pages_per_block must be positive or None")
        self.page_size = page_size
        self.max_pages_per_block = max_pages_per_block
        self.allocator = KVAllocator(page_size=page_size)
        self.blocks: list[KVBlock | None] = [None for _ in range(num_layers)]

    def get(self, layer_index: int) -> KVBlock | None:
        return self.blocks[layer_index]

    def _ensure_block(self, layer_index: int) -> KVBlock:
        block = self.blocks[layer_index]
        if block is None:
            block = KVBlock(
                block_id=self.allocator.acquire_block_id(),
                page_size=self.page_size,
            )
            self.blocks[layer_index] = block
        return block

    def update(self, layer_index: int, keys: torch.Tensor, values: torch.Tensor) -> KVBlock:
        block = self._ensure_block(layer_index)
        block.append(keys, values, self.allocator)
        block.evict_to_fit(self.allocator, self.max_pages_per_block)
        if not block.pages:
            self.release_layer(layer_index)
            return self._ensure_block(layer_index)
        return block

    def release_layer(self, layer_index: int) -> None:
        block = self.blocks[layer_index]
        if block is None:
            return
        while block.pages:
            page = block.pages.pop(0)
            block.evicted_tokens += page.length
            self.allocator.release_page_id(page.page_id)
        self.allocator.release_block_id(block.block_id)
        self.blocks[layer_index] = None

    def materialize(self, layer_index: int) -> LayerKVCache | None:
        block = self.blocks[layer_index]
        if block is None:
            return None
        return block.materialize()

    def sequence_length(self) -> int:
        return max((block.sequence_length() for block in self.blocks if block is not None), default=0)

    def fragmentation_metrics(self) -> dict[str, float | int]:
        active_blocks = [block for block in self.blocks if block is not None]
        active_pages = sum(len(block.pages) for block in active_blocks)
        active_tokens = sum(block.sequence_length() for block in active_blocks)
        total_capacity = sum(block.capacity_tokens() for block in active_blocks)
        page_utilization = (active_tokens / total_capacity) if total_capacity > 0 else 0.0
        fragmentation_ratio = 1.0 - page_utilization if total_capacity > 0 else 0.0
        return {
            **self.allocator.snapshot(),
            "active_blocks": len(active_blocks),
            "active_pages": active_pages,
            "active_tokens": active_tokens,
            "total_capacity_tokens": total_capacity,
            "page_utilization": page_utilization,
            "fragmentation_ratio": fragmentation_ratio,
            "avg_pages_per_block": (active_pages / len(active_blocks)) if active_blocks else 0.0,
        }

    def to_dense(self) -> KVCache:
        cache = KVCache(num_layers=len(self.blocks))
        for index, block in enumerate(self.blocks):
            if block is None:
                continue
            cache.layers[index] = block.materialize()
        return cache
