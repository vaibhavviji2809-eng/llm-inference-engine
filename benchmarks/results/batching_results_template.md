# Continuous Batching Benchmark Results

Run:

```powershell
py scripts\benchmark_batching.py --sample-tokens 16 --repeats 5
```

## Results

| Method | Seconds | Tokens/sec |
| --- | ---: | ---: |
| Serial requests |  |  |
| Continuous batching |  |  |

## Notes

- This first-pass benchmark is CPU-focused and groups decode steps into shared model passes.
- A stronger future version should combine batching with per-request KV caches.
