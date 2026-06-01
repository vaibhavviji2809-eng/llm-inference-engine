# Quantization

Implemented in a first-pass form:

- `FP32` baseline from the trained checkpoint
- `FP16` model copy via half-precision weights and activations
- `INT8` weight-only quantization for embeddings and linear layers

Files:

- `quantization/quantized_modules.py`
- `quantization/convert.py`
- `scripts/benchmark_quantization.py`

Metrics tracked:

- model memory footprint
- generation speed in tokens/sec
- validation loss on the toy corpus
- loss delta relative to the FP32 baseline
