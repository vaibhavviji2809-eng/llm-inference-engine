# Benchmark Report

Generated automatically for the current local environment.

## Environment

- Python: `3.11.0`
- PyTorch: `2.9.0+cpu`
- CUDA available: `False`
- Platform: `Windows-10-10.0.26200-SP0`

## KV Cache

| Method | Seconds | Tokens/sec |
| --- | ---: | ---: |
| full_recompute | 0.025582 | 312.72 |
| kv_cache | 0.024281 | 329.47 |

## Batching

| Method | Seconds | Tokens/sec |
| --- | ---: | ---: |
| serial_kv_cache | 0.119177 | 268.51 |
| batched_kv_cache | 0.210851 | 151.77 |

## Quantization

| Type | Memory (MiB) | Seconds | Tokens/sec | Loss | Loss Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 0.2656 | 0.032230 | 248.21 | 0.2030 | 0.0000 |
| FP16 | 0.1328 | 0.047318 | 169.07 | 0.2030 | -0.0000 |
| INT8 | 0.0709 | 0.041745 | 191.64 | 0.2023 | -0.0007 |

## Notes

- KV cache and batching are measured on the current tiny checkpoint.
- CUDA kernel execution is not included in this report because this machine does not have a CUDA runtime.
- Batched decoding now uses per-request KV cache state grouped by matching cache lengths.
