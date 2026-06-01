# KV Cache

Implemented in the first-pass form:

- `kv_cache/cache.py` defines per-layer key/value storage.
- `DecoderOnlyTransformer.generate_with_kv_cache()` reuses cached keys and values across decode steps.
- `scripts/benchmark_kv_cache.py` measures full recompute vs cached generation.

Current behavior:

1. Run the full prompt once and store per-layer `K` and `V`.
2. For each new token, project only the new token to `Q`, `K`, `V`.
3. Reuse cached history for attention instead of recomputing prior tokens.
4. If the active sequence exceeds `max_seq_len`, rebuild the cache from the newest window.

Next improvements:

- expose cache stats in the server layer
- batch multiple cached decode streams together
- combine with quantized weights and CUDA kernels
