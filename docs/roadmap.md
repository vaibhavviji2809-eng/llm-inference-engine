# Roadmap

This project is being developed in phases so the repo shows clear technical progress instead of one large unfinished jump.

## Completed

### Phase 1: Transformer From Scratch

- token embeddings
- positional embeddings
- multi-head self-attention
- LayerNorm
- feed-forward network
- manual softmax
- decoder-only generation

### Phase 2: CUDA Matmul Foundations

- naive CUDA matmul kernel
- tiled shared-memory CUDA matmul kernel
- benchmark harness for `CPU / Naive CUDA / Tiled CUDA`

Note:

- the code is implemented, but local execution still requires a CUDA-enabled machine

### Phase 3: KV Cache

- per-layer cache representation
- incremental decoding path
- throughput comparison against full recompute

### Phase 4: Quantization

- FP32 baseline
- FP16 comparison path
- INT8 weight-only quantization path
- memory, throughput, and loss comparison script

### Phase 5: Continuous Batching

- grouped decode requests
- shared-pass scheduling
- throughput comparison against serial handling

### Phase 6: Inference Server Expansion

- request queueing
- streaming responses
- cache-aware request handling
- server-side benchmark reporting

### Phase 7A: CUDA Softmax

- numerically stable row-wise softmax kernel
- row max / row sum helpers
- shared-memory and warp-reduction structure

### Phase 7B: CUDA Attention

- explicit QK^T kernel
- softmax normalization kernel
- AV kernel
- attention pipeline launcher

### Phase 8: FlashAttention

- `NaiveAttention`
- `FlashAttention`
- blockwise softmax accumulation without materializing an NxN matrix

### Phase 9: PagedAttention

- `KVPage`
- `KVBlock`
- `KVAllocator`
- `PagedKVCache`

### Phase 10: Real GPU Benchmarks

- attention benchmark script surface
- CUDA kernel benchmark surface
- consolidated JSON and Markdown reporting

### Phase 11: Production Dashboard

- tokens/sec
- batch size
- latency
- cache hit rate
- VRAM usage when available

## Next Up

The implementation phases are complete. The only remaining work is to capture real CUDA benchmark numbers on a GPU machine and fold those measurements into the report.
