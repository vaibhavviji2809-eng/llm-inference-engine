from __future__ import annotations

import unittest

from server.app.metrics import MetricsStore


class MetricsAndDashboardTests(unittest.TestCase):
    def test_metrics_summary_includes_batch_signals(self) -> None:
        store = MetricsStore()
        store.record_request(
            {
                "prompt": "Hello",
                "latency_seconds": 0.25,
                "tokens_per_second": 40.0,
                "use_kv_cache": True,
            }
        )
        store.record_benchmark(
            {
                "results": [
                    {"method": "serial_kv_cache", "seconds": 1.0, "tokens_per_second": 10.0},
                    {"method": "batched_kv_cache", "seconds": 0.5, "tokens_per_second": 20.0, "speedup_vs_serial": 2.0},
                ]
            }
        )
        store.record_batch_run({"avg_batch_size": 3.5})

        summary = store.summary()

        self.assertEqual(summary["request_count"], 1)
        self.assertEqual(summary["benchmark_count"], 1)
        self.assertEqual(summary["batch_run_count"], 1)
        self.assertAlmostEqual(summary["avg_latency_seconds"], 0.25)
        self.assertAlmostEqual(summary["avg_tokens_per_second"], 40.0)
        self.assertAlmostEqual(summary["avg_batch_speedup"], 2.0)
        self.assertAlmostEqual(summary["avg_batch_size"], 3.5)


if __name__ == "__main__":
    unittest.main()
