from __future__ import annotations

from dataclasses import dataclass
import time

import torch

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
            }
            for request in requests
        ]

        steps = 0
        emitted_tokens = 0
        start = time.perf_counter()

        while any(item["remaining_tokens"] > 0 for item in active):
            step_inputs: list[torch.Tensor] = []
            request_indexes: list[int] = []

            for index, item in enumerate(active):
                if item["remaining_tokens"] <= 0:
                    continue
                window = item["generated"][:, -self.model.config.max_seq_len :]
                step_inputs.append(window)
                request_indexes.append(index)

            if not step_inputs:
                break

            max_len = max(step_input.size(1) for step_input in step_inputs)
            padded_batch = []
            for step_input in step_inputs:
                pad_len = max_len - step_input.size(1)
                if pad_len > 0:
                    pad = torch.full(
                        (1, pad_len),
                        fill_value=int(step_input[0, 0].item()),
                        dtype=step_input.dtype,
                    )
                    step_input = torch.cat([pad, step_input], dim=1)
                padded_batch.append(step_input)

            batch_input = torch.cat(padded_batch, dim=0)
            logits = self.model(batch_input)

            for batch_row, request_index in enumerate(request_indexes):
                item = active[request_index]
                current_len = step_inputs[batch_row].size(1)
                offset = max_len - current_len
                next_token_logits = logits[batch_row, offset + current_len - 1, :]
                if item["temperature"] <= 0:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True).view(1, 1)
                else:
                    scaled = next_token_logits / item["temperature"]
                    probs = softmax(scaled, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).view(1, 1)
                item["generated"] = torch.cat([item["generated"], next_token], dim=1)
                item["remaining_tokens"] -= 1
                emitted_tokens += 1

            steps += 1

        elapsed = time.perf_counter() - start
        return {
            "request_count": len(active),
            "steps": steps,
            "latency_seconds": elapsed,
            "tokens_per_second": emitted_tokens / elapsed if elapsed > 0 else 0.0,
            "results": [
                {
                    "request_id": item["request_id"],
                    "generated": item["generated"],
                    "prompt_length": item["prompt_length"],
                }
                for item in active
            ],
        }
