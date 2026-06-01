# CUDA-Accelerated Transformer Inference Engine From Scratch

This repository is being shaped into the kind of end-to-end systems project that stands out for AI infrastructure, research engineering, and systems-focused graduate applications.

## Project Layout

```text
llm-inference-engine/
├── transformer/
├── tokenizer/
├── cuda_kernels/
├── quantization/
├── kv_cache/
├── batching/
├── server/
├── dashboard/
├── benchmarks/
├── docs/
└── research/
```

## Current Milestone

Phase 1 is implemented:

- token embeddings
- positional embeddings
- multi-head self-attention
- layer normalization
- feed-forward network
- manual softmax
- transformer block
- full decoder-only architecture

The model is intentionally low-level:

- no PyTorch transformer APIs
- learned weights are stored as `torch.nn.Parameter`
- linear layers are expressed with `torch.matmul`
- every attention projection is visible in code

Phase 3 now has a working first pass too:

- per-layer KV cache objects
- incremental decode path that reuses cached keys and values
- a benchmark script for `tokens/sec` before and after cache reuse

## Quick Start

Install Python dependencies:

```bash
py -m pip install -r requirements.txt
```

Run the tiny training demo:

```bash
py scripts/train_tiny_transformer.py --steps 250 --prompt Hello
```

Generate from a saved checkpoint:

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

Run the FastAPI scaffold:

```bash
py -m uvicorn server.app.main:app --reload
```

Server endpoints:

- `GET /health`
- `GET /model`
- `POST /generate`
- `POST /chat`
- `POST /benchmark`

Example generate request:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"Hello\",\"max_new_tokens\":16,\"temperature\":0.0,\"use_kv_cache\":true}"
```

## Environment Status

This machine currently has:

- `Python 3.11`
- `torch 2.9.0+cpu`
- no CUDA-capable device available

That means the transformer and server phases are runnable here, while the CUDA kernels are currently scaffolded for later execution on a GPU-enabled machine.

One current limitation is that the toy character tokenizer only supports characters seen in `data/tiny_corpus.txt`. The server now returns a clear `400` if a prompt contains unsupported characters.

## Next Build Targets

1. Run the CUDA matmul benchmark on a GPU machine and record `CPU / Naive CUDA / Tiled CUDA`.
2. Extend handwritten CUDA coverage from matmul to softmax and attention.
3. Build continuous batching over cached decode streams.
4. Add benchmark dashboards comparing this engine against production runtimes.
5. Expand quantization from toy PTQ to larger-model evaluation and calibration.
