# CUDA-Accelerated Transformer Inference Engine From Scratch

An end-to-end systems project for learning and demonstrating how modern LLM inference stacks are built under the hood.

This repository is focused on implementing the core pieces ourselves:

- a decoder-only transformer built from low-level matrix operations
- KV-cached incremental decoding
- quantization experiments across FP32, FP16, and INT8
- handwritten CUDA kernels for matrix multiplication
- a FastAPI inference server and benchmark surface
- cache-aware batching metrics with speedup and batch-size reporting

The goal is not just to get a model to generate text. The goal is to understand and expose the engineering layers that matter to real inference systems: kernels, memory movement, caching, batching, precision tradeoffs, and serving architecture.

## Why This Project

This is the kind of project that maps well to:

- AI infrastructure roles
- inference and systems engineering internships
- research engineering applications
- systems-focused graduate programs

If someone opens the repo, they should be able to see both implementation depth and a clear roadmap toward production-style inference features.

## Current Status

Implemented now:

- Phase 1: decoder-only transformer from scratch
- Phase 2: naive and tiled CUDA matmul kernels plus benchmark harness
- Phase 3: KV cache with incremental decoding and throughput benchmark
- Phase 4: FP32, FP16, and INT8 comparison utilities
- Phase 5: first-pass continuous batching benchmark and API surface
- Phase 7A: numerically stable CUDA softmax kernel
- Phase 7B: CUDA attention pipeline kernel
- Phase 8: NaiveAttention and FlashAttention implementations
- Phase 9: paged KV cache abstractions
- Phase 10: attention and CUDA benchmark surfaces
- Phase 11: dashboard metrics for tokens/sec, VRAM, batch size, latency, and cache hits
- Inference server: FastAPI endpoints for generation, streaming, chat, batching, model info, benchmarking, and dashboard metrics
- Reporting layer: auto-generated JSON and Markdown benchmark summaries with batching metadata

Latest additions:

- numerically stable CUDA softmax and attention kernels
- `NaiveAttention` and `FlashAttention`
- paged KV cache abstractions
- attention benchmark entry point
- dashboard metrics for cache hits and VRAM visibility

Current machine limitation:

- the local environment is CPU-only, so CUDA kernels are implemented but the real GPU benchmark numbers still need to be collected on a CUDA-enabled machine with `nvcc`

## What's Inside

```text
llm-inference-engine/
|- transformer/
|- tokenizer/
|- cuda_kernels/
|- quantization/
|- kv_cache/
|- batching/
|- server/
|- dashboard/
|- benchmarks/
|- docs/
`- research/
```

High-signal folders:

- [transformer](C:/Users/vaibh/Documents/Codex/2026-05-30/lets-do-a-huge-project/transformer): from-scratch decoder model, attention, LayerNorm, feed-forward blocks
- [kv_cache](C:/Users/vaibh/Documents/Codex/2026-05-30/lets-do-a-huge-project/kv_cache): cache objects and incremental decode support
- [quantization](C:/Users/vaibh/Documents/Codex/2026-05-30/lets-do-a-huge-project/quantization): FP16 and INT8 conversion helpers
- [cuda_kernels](C:/Users/vaibh/Documents/Codex/2026-05-30/lets-do-a-huge-project/cuda_kernels): handwritten CUDA kernels
- [server/app](C:/Users/vaibh/Documents/Codex/2026-05-30/lets-do-a-huge-project/server/app): inference runtime and API surface
- [benchmarks](C:/Users/vaibh/Documents/Codex/2026-05-30/lets-do-a-huge-project/benchmarks): benchmark harnesses and result templates

## Implemented Features

### Transformer

- token embeddings
- positional embeddings
- multi-head self-attention
- manual softmax
- LayerNorm
- feed-forward network
- transformer blocks
- full decoder-only architecture

Implementation constraints:

- no PyTorch transformer APIs
- weights stored as `torch.nn.Parameter`
- linear layers built with `torch.matmul`
- attention projections are explicit in code

### KV Cache

- per-layer key/value cache storage
- incremental decode path
- cache vs full-recompute benchmark

Example benchmark result from this machine:

- full recompute: about `95.80 tokens/sec`
- KV cache: about `300.32 tokens/sec`

### Quantization

- FP32 baseline
- FP16 model copy
- INT8 weight-only quantization

Example benchmark result from this machine:

- FP32: `0.2656 MiB`, `341.06 tokens/sec`
- FP16: `0.1328 MiB`, `173.05 tokens/sec`
- INT8: `0.0709 MiB`, `251.84 tokens/sec`

### Server

- `GET /health`
- `GET /model`
- `GET /metrics/summary`
- `GET /dashboard`
- `POST /generate`
- `POST /stream`
- `POST /chat`
- `POST /benchmark`
- `POST /batch_generate`
- `POST /benchmark_batching`

### Reporting

- consolidated JSON benchmark export
- consolidated Markdown benchmark report
- benchmark artifacts written under `benchmarks/results/`
- latest local benchmark snapshot: [latest_report.md](C:/Users/vaibh/Documents/Codex/2026-05-30/lets-do-a-huge-project/benchmarks/results/latest_report.md)

## Quick Start

Install Python dependencies:

```bash
py -m pip install -r requirements.txt
```

Train the tiny demo model:

```bash
py scripts/train_tiny_transformer.py --steps 250 --prompt Hello
```

Generate text:

```bash
py scripts/generate.py --prompt Hello --temperature 0.0
```

Benchmark KV cache:

```bash
py scripts/benchmark_kv_cache.py --prompt Hello --sample-tokens 32 --repeats 20
```

Benchmark quantization:

```bash
py scripts/benchmark_quantization.py --prompt Hello --sample-tokens 24 --repeats 10
```

Benchmark attention:

```bash
py scripts/benchmark_attention.py
```

Benchmark batching:

```bash
py scripts/benchmark_batching.py --sample-tokens 16 --repeats 5
```

Generate a consolidated benchmark report:

```bash
py scripts/generate_benchmark_report.py
```

Run the FastAPI server:

```bash
py -m uvicorn server.app.main:app --reload
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"Hello\",\"max_new_tokens\":16,\"temperature\":0.0,\"use_kv_cache\":true}"
```

## Limitations Right Now

- the toy tokenizer only supports characters seen in [data/tiny_corpus.txt](C:/Users/vaibh/Documents/Codex/2026-05-30/lets-do-a-huge-project/data/tiny_corpus.txt)
- the current checkpoint is intentionally tiny and is meant for architecture experimentation, not model quality
- CUDA benchmark numbers still need to be collected on a machine with `nvcc` and an NVIDIA runtime
- the repository is otherwise phase-complete for the current scope

## Roadmap

The near-term roadmap is in [docs/roadmap.md](C:/Users/vaibh/Documents/Codex/2026-05-30/lets-do-a-huge-project/docs/roadmap.md).

Current focus:

1. Capture real CUDA benchmark numbers on a GPU machine.
2. Compare the full stack against PyTorch, TensorRT, and vLLM on real hardware.
3. Extend the reporting layer with those measured numbers once available.

## Documentation

- [Architecture Notes](C:/Users/vaibh/Documents/Codex/2026-05-30/lets-do-a-huge-project/docs/architecture.md)
- [Roadmap](C:/Users/vaibh/Documents/Codex/2026-05-30/lets-do-a-huge-project/docs/roadmap.md)
- [Blog Draft: Building a Transformer From Scratch](C:/Users/vaibh/Documents/Codex/2026-05-30/lets-do-a-huge-project/docs/blog/part1_transformer_from_scratch.md)
