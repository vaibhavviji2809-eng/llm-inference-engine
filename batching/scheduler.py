from __future__ import annotations

from dataclasses import dataclass
import time

import torch

from kv_cache import KVCache, LayerKVCache
from transformer import DecoderOnlyTransformer
from transformer.modules import softmax


@dataclass
class BatchGenerateRequest:
    request_id: str
    prompt_ids: torch.Tensor
    max_new_tokens: int
    temperature: float = 0.0


class ContinuousBatcher:
    def __init__(self, model: DecoderOnlyTransformer) -> None:
        self.model = model

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperatures: list[float],
    ) -> list[torch.Tensor]:
        sampled: list[torch.Tensor] = []
        for row_index, temperature in enumerate(temperatures):
            next_token_logits = logits[row_index, -1, :]
            if temperature <= 0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True).view(1, 1)
            else:
                scaled = next_token_logits / temperature
                probs = softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).view(1, 1)
            sampled.append(next_token)
        return sampled

    def _stack_caches(self, caches: list[KVCache]) -> KVCache:
        batch_cache = KVCache(num_layers=len(caches[0].layers))
        for layer_index in range(len(caches[0].layers)):
            layer_entries = [cache.get(layer_index) for cache in caches]
            if any(entry is None for entry in layer_entries):
                continue
            keys = torch.cat([entry.keys for entry in layer_entries if entry is not None], dim=0)
            values = torch.cat([entry.values for entry in layer_entries if entry is not None], dim=0)
            batch_cache.layers[layer_index] = LayerKVCache(keys=keys, values=values)
        return batch_cache

    def _split_cache(self, batch_cache: KVCache, batch_size: int) -> list[KVCache]:
        caches = [KVCache(num_layers=len(batch_cache.layers)) for _ in range(batch_size)]
        for layer_index, layer_cache in enumerate(batch_cache.layers):
            if layer_cache is None:
                continue
            for batch_index in range(batch_size):
                caches[batch_index].layers[layer_index] = LayerKVCache(
                    keys=layer_cache.keys[batch_index : batch_index + 1].clone(),
                    values=layer_cache.values[batch_index : batch_index + 1].clone(),
                )
        return caches

    def _prefill_group(self, items: list[dict]) -> None:
        prompt_batch = torch.cat([item["generated"] for item in items], dim=0)
        batch_cache = KVCache(num_layers=len(self.model.blocks))
        logits = self.model.forward_incremental(prompt_batch, batch_cache, position_offset=0)
        split_caches = self._split_cache(batch_cache, len(items))
        next_tokens = self._sample_next_token(logits, [item["temperature"] for item in items])

        for index, item in enumerate(items):
            item["cache"] = split_caches[index]
            item["generated"] = torch.cat([item["generated"], next_tokens[index]], dim=1)
            item["remaining_tokens"] -= 1
            item["last_token"] = next_tokens[index]

    def _decode_group(self, items: list[dict]) -> None:
        token_batch = torch.cat([item["last_token"] for item in items], dim=0)
        batch_cache = self._stack_caches([item["cache"] for item in items])
        position_offset = batch_cache.sequence_length()
        logits = self.model.forward_incremental(token_batch, batch_cache, position_offset=position_offset)
        split_caches = self._split_cache(batch_cache, len(items))
        next_tokens = self._sample_next_token(logits, [item["temperature"] for item in items])

        for index, item in enumerate(items):
            item["cache"] = split_caches[index]
            item["generated"] = torch.cat([item["generated"], next_tokens[index]], dim=1)
            item["remaining_tokens"] -= 1
            item["last_token"] = next_tokens[index]

    def generate(self, requests: list[BatchGenerateRequest]) -> dict:
        if not requests:
            return {
                "request_count": 0,
                "steps": 0,
                "latency_seconds": 0.0,
                "tokens_per_second": 0.0,
                "results": [],
            }

        active = [
            {
                "request_id": request.request_id,
                "generated": request.prompt_ids.clone(),
                "remaining_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "prompt_length": int(request.prompt_ids.size(1)),
                "cache": KVCache(num_layers=len(self.model.blocks)),
                "last_token": None,
            }
            for request in requests
        ]

        steps = 0
        emitted_tokens = 0
        start = time.perf_counter()

        prefill_groups: dict[int, list[dict]] = {}
        for item in active:
            if item["remaining_tokens"] <= 0:
                continue
            prefill_groups.setdefault(item["prompt_length"], []).append(item)

        for items in prefill_groups.values():
            self._prefill_group(items)
            emitted_tokens += len(items)
            steps += 1

        while any(item["remaining_tokens"] > 0 for item in active):
            decode_groups: dict[int, list[dict]] = {}
            for item in active:
                if item["remaining_tokens"] <= 0:
                    continue
                cache_len = item["cache"].sequence_length()
                if cache_len >= self.model.config.max_seq_len:
                    window = item["generated"][:, -self.model.config.max_seq_len :]
                    item["cache"] = KVCache(num_layers=len(self.model.blocks))
                    logits = self.model.forward_incremental(window, item["cache"], position_offset=0)
                    next_token = self._sample_next_token(logits, [item["temperature"]])[0]
                    item["generated"] = torch.cat([item["generated"], next_token], dim=1)
                    item["remaining_tokens"] -= 1
                    item["last_token"] = next_token
                    emitted_tokens += 1
                    steps += 1
                else:
                    decode_groups.setdefault(cache_len, []).append(item)

            for items in decode_groups.values():
                self._decode_group(items)
                emitted_tokens += len(items)
                steps += 1

        elapsed = time.perf_counter() - start
        return {
            "request_count": len(active),
            "steps": steps,
            "latency_seconds": elapsed,
            "tokens_per_second": emitted_tokens / elapsed if elapsed > 0 else 0.0,
            "mode": "batched_kv_cache",
            "results": [
                {
                    "request_id": item["request_id"],
                    "generated": item["generated"],
                    "prompt_length": item["prompt_length"],
                    "cache_length": item["cache"].sequence_length(),
                }
                for item in active
            ],
        }
