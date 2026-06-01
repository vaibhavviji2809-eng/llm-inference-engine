from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
import sys
import time

import torch

from batching import BatchGenerateRequest, ContinuousBatcher
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tokenizer import CharTokenizer
from transformer import DecoderOnlyTransformer, TransformerConfig


DEFAULT_CHECKPOINT = PROJECT_ROOT / "benchmarks" / "results" / "tiny_transformer.pt"
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "tiny_corpus.txt"


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


class TransformerRuntime:
    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT) -> None:
        self.checkpoint_path = checkpoint_path
        payload = torch.load(checkpoint_path, map_location="cpu")
        corpus_text = DEFAULT_CORPUS.read_text(encoding="utf-8")
        self.tokenizer = CharTokenizer(corpus_text)
        self.config = TransformerConfig(**normalize_config(payload["config"]))
        self.model = DecoderOnlyTransformer(self.config)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.device = "cpu"
        self.batcher = ContinuousBatcher(self.model)

    def model_card(self) -> dict:
        return {
            "checkpoint": str(self.checkpoint_path),
            "device": self.device,
            "config": asdict(self.config),
            "summary": self.model.summary(),
        }

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        missing = sorted({character for character in prompt if character not in self.tokenizer.stoi})
        if missing:
            raise ValueError(
                "Prompt contains characters outside the toy tokenizer vocabulary: "
                + ", ".join(repr(character) for character in missing)
            )
        return torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)

    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids[0].tolist())

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        use_cache: bool,
    ) -> dict:
        prompt_ids = self.encode_prompt(prompt)
        start = time.perf_counter()
        if use_cache:
            generated = self.model.generate_with_kv_cache(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        else:
            generated = self.model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        elapsed = time.perf_counter() - start

        return {
            "prompt": prompt,
            "completion": self.decode_tokens(generated),
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "use_kv_cache": use_cache,
            "latency_seconds": elapsed,
            "tokens_per_second": max_new_tokens / elapsed if elapsed > 0 else None,
        }

    def stream_text(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        use_cache: bool,
    ) -> list[str]:
        prompt_ids = self.encode_prompt(prompt)
        if use_cache:
            generated = self.model.generate_with_kv_cache(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        else:
            generated = self.model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        full_text = self.decode_tokens(generated)
        prompt_text = self.decode_tokens(prompt_ids)
        completion = full_text[len(prompt_text) :]
        return [character for character in completion]

    def benchmark(
        self,
        prompt: str,
        max_new_tokens: int,
        repeats: int,
    ) -> dict:
        prompt_ids = self.encode_prompt(prompt)

        def measure(use_cache: bool) -> tuple[float, torch.Tensor]:
            generated = prompt_ids
            start = time.perf_counter()
            for _ in range(repeats):
                if use_cache:
                    generated = self.model.generate_with_kv_cache(
                        prompt_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=0.0,
                    )
                else:
                    generated = self.model.generate(
                        prompt_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=0.0,
                    )
            elapsed = time.perf_counter() - start
            return elapsed / repeats, generated

        uncached_seconds, uncached_output = measure(use_cache=False)
        cached_seconds, cached_output = measure(use_cache=True)

        return {
            "prompt": prompt,
            "sample_tokens": max_new_tokens,
            "repeats": repeats,
            "results": [
                {
                    "method": "full_recompute",
                    "seconds": uncached_seconds,
                    "tokens_per_second": max_new_tokens / uncached_seconds,
                    "sample": self.decode_tokens(uncached_output),
                },
                {
                    "method": "kv_cache",
                    "seconds": cached_seconds,
                    "tokens_per_second": max_new_tokens / cached_seconds,
                    "sample": self.decode_tokens(cached_output),
                },
            ],
        }

    def batch_generate_text(
        self,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
    ) -> dict:
        requests = [
            BatchGenerateRequest(
                request_id=f"req-{index}",
                prompt_ids=self.encode_prompt(prompt),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            for index, prompt in enumerate(prompts)
        ]
        report = self.batcher.generate(requests)
        return {
            "request_count": report["request_count"],
            "steps": report["steps"],
            "latency_seconds": report["latency_seconds"],
            "tokens_per_second": report["tokens_per_second"],
            "results": [
                {
                    "request_id": result["request_id"],
                    "completion": self.decode_tokens(result["generated"]),
                }
                for result in report["results"]
            ],
            "mode": report["mode"],
        }

    def benchmark_batching(
        self,
        prompts: list[str],
        max_new_tokens: int,
        repeats: int,
    ) -> dict:
        prompt_ids = [self.encode_prompt(prompt) for prompt in prompts]

        serial_start = time.perf_counter()
        serial_outputs = []
        for _ in range(repeats):
            serial_outputs = [
                self.model.generate_with_kv_cache(
                    prompt_ids_item,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                )
                for prompt_ids_item in prompt_ids
            ]
        serial_elapsed = (time.perf_counter() - serial_start) / repeats

        batched_report = None
        batch_start = time.perf_counter()
        for _ in range(repeats):
            requests = [
                BatchGenerateRequest(
                    request_id=f"req-{index}",
                    prompt_ids=prompt_ids_item,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                )
                for index, prompt_ids_item in enumerate(prompt_ids)
            ]
            batched_report = self.batcher.generate(requests)
        batched_elapsed = (time.perf_counter() - batch_start) / repeats

        total_tokens = len(prompts) * max_new_tokens
        assert batched_report is not None
        return {
            "prompts": prompts,
            "sample_tokens": max_new_tokens,
            "repeats": repeats,
            "results": [
                {
                    "method": "serial_kv_cache",
                    "seconds": serial_elapsed,
                    "tokens_per_second": total_tokens / serial_elapsed,
                    "samples": [self.decode_tokens(output) for output in serial_outputs],
                },
                {
                    "method": batched_report["mode"],
                    "seconds": batched_elapsed,
                    "tokens_per_second": total_tokens / batched_elapsed,
                    "steps": batched_report["steps"],
                    "samples": [
                        self.decode_tokens(result["generated"])
                        for result in batched_report["results"]
                    ],
                },
            ],
        }


@lru_cache(maxsize=1)
def get_runtime() -> TransformerRuntime:
    return TransformerRuntime()
