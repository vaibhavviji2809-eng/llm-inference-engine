# Quantization Benchmark Results

Run:

```powershell
py scripts\benchmark_quantization.py --prompt Hello --sample-tokens 24 --repeats 10
```

## Results

| Type | Memory (MiB) | Seconds | Tokens/sec | Loss | Loss Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 |  |  |  |  | 0.0000 |
| FP16 |  |  |  |  |  |
| INT8 |  |  |  |  |  |

## Notes

- `Loss Delta` is measured relative to the FP32 checkpoint on the toy corpus.
- On CPU, FP16 and INT8 may not always be faster than FP32; the benchmark still captures the memory/accuracy tradeoff cleanly.
