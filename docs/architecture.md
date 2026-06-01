# CUDA-Accelerated Transformer Inference Engine From Scratch

This repository is structured as a long-horizon systems project:

- `transformer/`: decoder-only transformer built from low-level matrix multiplications
- `tokenizer/`: minimal tokenizer implementations
- `cuda_kernels/`: handwritten CUDA kernels for matmul, softmax, and attention
- `kv_cache/`: incremental decoding and cache reuse
- `quantization/`: FP16 and INT8 experiments
- `batching/`: continuous batching scheduler
- `server/`: FastAPI inference service
- `dashboard/`: profiling and benchmarking UI
- `benchmarks/`: scripts and results
- `docs/`: architecture and blog drafts
- `research/`: implementation notes and paper reading summaries

Current status:

- Phase 1 is implemented in Python with a decoder-only transformer built from scratch.
- Phase 3 includes KV-cached decoding plus API-level benchmarking endpoints.
- Phase 2 now includes handwritten naive and tiled CUDA matmul kernels plus a benchmark harness.
- Phase 4 now includes FP32, FP16, and INT8 comparison utilities for memory, speed, and loss.
- Phase 5 now includes a first-pass continuous batching scheduler, benchmark script, and batch API endpoint.
- The environment on this machine is CPU-only, so CUDA phases are scaffolded but not executable here yet.
- The FastAPI server can now load the checkpoint, generate text, stream output, serve chat-style prompts, batch prompts together, expose benchmark results, and render a lightweight dashboard.
