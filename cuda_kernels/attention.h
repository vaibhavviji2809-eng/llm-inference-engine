#pragma once

#include <cuda_runtime.h>

void launch_attention_pipeline(
    const float* q,
    const float* k,
    const float* v,
    float* scores,
    float* attention,
    float* output,
    int batch,
    int heads,
    int query_len,
    int key_len,
    int head_dim,
    cudaStream_t stream
);
