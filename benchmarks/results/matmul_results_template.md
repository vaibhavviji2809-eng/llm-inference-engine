# Matrix Multiplication Benchmark Results

Fill this in on a CUDA-enabled machine after running:

```powershell
.\scripts\build_matmul_benchmark.ps1
.\scripts\run_matmul_benchmark.ps1 -M 512 -N 512 -K 512 -WarmupRuns 3 -TimedRuns 10
```

## Environment

- GPU:
- CUDA Toolkit:
- Driver:
- OS:

## Results

| Method | Time (ms) | GFLOP/s | Max Error |
| --- | ---: | ---: | ---: |
| CPU |  |  |  |
| Naive CUDA |  |  |  |
| Tiled CUDA |  |  |  |

## Notes

- Shared-memory tiling should reduce global-memory traffic versus the naive kernel.
- `Max Error` should remain close to floating-point noise relative to the CPU reference.
