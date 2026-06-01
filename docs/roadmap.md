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

## Next Up

### Phase 5: Continuous Batching

Build a scheduler that:

- groups active decode requests
- advances them in a shared pass
- measures throughput gains versus serial handling

### Phase 6: Inference Server Expansion

Improve serving with:

- request queueing
- streaming responses
- cache-aware request handling
- server-side benchmark reporting

### Phase 7: Memory Optimizations

Implement simplified versions of:

- paged attention
- flash-style attention

### Phase 8: Profiling Dashboard

Track:

- latency
- throughput
- memory usage
- tokens per second
- benchmark comparisons

### Phase 9: External Comparisons

Benchmark against:

- PyTorch
- ONNX Runtime
- TensorRT
- vLLM

### Phase 10: Technical Writing

Turn the build process into a series of short, concrete blog posts:

1. Building a Transformer From Scratch
2. Writing CUDA Kernels
3. KV Cache Optimization
4. Quantization Tradeoffs
5. Inference Engine Architecture
