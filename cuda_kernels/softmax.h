#pragma once

#include <cuda_runtime.h>

void launch_softmax_rowwise(
    const float* input,
    float* output,
    int rows,
    int cols,
    cudaStream_t stream
);

void launch_rowwise_max(
    const float* input,
    float* output,
    int rows,
    int cols,
    cudaStream_t stream
);

void launch_rowwise_sum(
    const float* input,
    float* output,
    int rows,
    int cols,
    cudaStream_t stream
);
