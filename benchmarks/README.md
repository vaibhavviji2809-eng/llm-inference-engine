# Benchmarks

Track these comparisons over time:

| Benchmark | Baseline | Target |
| --- | --- | --- |
| CPU matmul | Python or NumPy | naive CUDA |
| naive CUDA | global-memory only | tiled CUDA |
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
