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

void launch_attention_with_output_projection(
    const float* q,
    const float* k,
    const float* v,
    const float* out_weight,
    const float* out_bias,
    float* scores,
    float* attention,
    float* attention_output,
    float* projected_output,
    int batch,
    int heads,
    int query_len,
    int key_len,
    int head_dim,
    int out_dim,
    cudaStream_t stream
);
