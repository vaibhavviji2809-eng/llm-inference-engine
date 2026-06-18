# Benchmarks

Track these comparisons over time:

| Benchmark | Baseline | Target |
| --- | --- | --- |
| CPU matmul | Python or NumPy | naive CUDA |
| naive CUDA | global-memory only | tiled CUDA |
| attention | naive NxN attention | FlashAttention |
| decode throughput | no KV cache | KV cache |
| precision | FP32 | FP16 / INT8 |
| serving | serial requests | continuous batching |

## CUDA Matmul Benchmark

Files:

- `benchmarks/matmul_benchmark.cu`
- `cuda_kernels/matmul.cu`
- `scripts/build_matmul_benchmark.ps1`
- `scripts/run_matmul_benchmark.ps1`

Expected output table:

| Method | Time (ms) | GFLOP/s | Max Error |
| --- | ---: | ---: | ---: |
| CPU |  |  |  |
| Naive CUDA |  |  |  |
| Tiled CUDA |  |  |  |

## Attention Benchmark

Files:

- `benchmarks/attention_benchmark.cu`
- `cuda_kernels/softmax.cu`
- `cuda_kernels/attention.cu`
- `scripts/build_attention_benchmark.ps1`
- `scripts/run_attention_benchmark.ps1`
- `scripts/benchmark_attention.py`

Expected output table:

| Method | Milliseconds | Tokens/sec | Estimated Memory (bytes) |
| --- | ---: | ---: | ---: |
| naive_attention |  |  |  |
| flash_attention |  |  |  |

The CUDA path now includes the output projection stage after `Attention Weights x V`.

## Consolidated Reporting

Generate a single JSON and Markdown benchmark report with:

```powershell
py scripts\generate_benchmark_report.py
```

Artifacts:

- `benchmarks/results/latest_report.json`
- `benchmarks/results/latest_report.md`

The consolidated batching section includes cache-aware metadata such as average batch size, cache rebuilds, and serial-vs-batched speedup.
