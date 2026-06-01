# Continuous Batching

Implemented in a first-pass CPU-friendly form:

- `batching/scheduler.py` groups active decode requests into shared model passes
- `scripts/benchmark_batching.py` compares serial handling against batched generation
- `POST /batch_generate` exposes batched generation through the FastAPI server

Current behavior:

- active prompts are advanced together in batched decode steps
- shorter prompts continue participating until their requested output length is reached
- throughput is reported against serial request handling

Next improvements:

- combine batching with per-request KV caches
- add request queueing and fairness policies
- move the same design onto GPU once CUDA execution is available
