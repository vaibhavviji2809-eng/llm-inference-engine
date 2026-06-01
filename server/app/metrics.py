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
        return {
            "request_count": len(self.requests),
            "benchmark_count": len(self.benchmarks),
            "batch_run_count": len(self.batch_runs),
            "avg_latency_seconds": mean(request_latencies) if request_latencies else None,
            "avg_tokens_per_second": mean(request_tps) if request_tps else None,
            "recent_requests": list(self.requests)[-10:],
            "recent_benchmarks": list(self.benchmarks)[-10:],
            "recent_batch_runs": list(self.batch_runs)[-10:],
        }


metrics_store = MetricsStore()
