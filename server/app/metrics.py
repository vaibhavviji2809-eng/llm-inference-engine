from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import mean
from typing import Any


@dataclass
class MetricsStore:
    requests: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=200))
    benchmarks: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=50))
    batch_runs: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=50))

    def record_request(self, payload: dict[str, Any]) -> None:
        self.requests.append(payload)

    def record_benchmark(self, payload: dict[str, Any]) -> None:
        self.benchmarks.append(payload)

    def record_batch_run(self, payload: dict[str, Any]) -> None:
        self.batch_runs.append(payload)

    def summary(self) -> dict[str, Any]:
        request_latencies = [
            entry["latency_seconds"]
            for entry in self.requests
            if entry.get("latency_seconds") is not None
        ]
        request_tps = [
            entry["tokens_per_second"]
            for entry in self.requests
            if entry.get("tokens_per_second") is not None
        ]
        batch_efficiency = [
            entry["results"][1].get("speedup_vs_serial")
            for entry in self.benchmarks
            if entry.get("results") and len(entry["results"]) > 1 and entry["results"][1].get("speedup_vs_serial") is not None
        ]
        batch_sizes = [
            entry.get("avg_batch_size")
            for entry in self.batch_runs
            if entry.get("avg_batch_size") is not None
        ]
        cache_hit_rates = [
            entry.get("cache_hit_rate")
            for entry in list(self.requests) + list(self.batch_runs)
            if entry.get("cache_hit_rate") is not None
        ]
        for benchmark in self.benchmarks:
            for result in benchmark.get("results", []):
                if result.get("cache_hit_rate") is not None:
                    cache_hit_rates.append(result["cache_hit_rate"])
        vram_allocated = [
            entry.get("vram_allocated_mb")
            for entry in list(self.requests) + list(self.batch_runs) + list(self.benchmarks)
            if entry.get("vram_allocated_mb") is not None
        ]
        for benchmark in self.benchmarks:
            for result in benchmark.get("results", []):
                if result.get("vram_allocated_mb") is not None:
                    vram_allocated.append(result["vram_allocated_mb"])
        return {
            "request_count": len(self.requests),
            "benchmark_count": len(self.benchmarks),
            "batch_run_count": len(self.batch_runs),
            "avg_latency_seconds": mean(request_latencies) if request_latencies else None,
            "avg_tokens_per_second": mean(request_tps) if request_tps else None,
            "avg_batch_speedup": mean(batch_efficiency) if batch_efficiency else None,
            "avg_batch_size": mean(batch_sizes) if batch_sizes else None,
            "avg_cache_hit_rate": mean(cache_hit_rates) if cache_hit_rates else None,
            "avg_vram_allocated_mb": mean(vram_allocated) if vram_allocated else None,
            "recent_requests": list(self.requests)[-10:],
            "recent_benchmarks": list(self.benchmarks)[-10:],
            "recent_batch_runs": list(self.batch_runs)[-10:],
        }


metrics_store = MetricsStore()
