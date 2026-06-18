# Attention Results

| Method | Milliseconds | Tokens/sec | Estimated Memory (bytes) |
| --- | ---: | ---: | ---: |
| naive_attention |  |  |  |
| flash_attention |  |  |  |

Notes:

- `flash_attention` should avoid materializing the full `N x N` attention matrix.
- compare the estimated memory footprint and measured latency on a GPU-enabled machine.
