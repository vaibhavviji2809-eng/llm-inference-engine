#include <cuda_runtime.h>

#include <float.h>

namespace {

constexpr int kWarpSize = 32;

__inline__ __device__ float warpReduceMax(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__inline__ __device__ float warpReduceSum(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__inline__ __device__ float blockReduceMax(float value) {
    __shared__ float shared[kWarpSize];
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    value = warpReduceMax(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();

    value = (threadIdx.x < blockDim.x / kWarpSize) ? shared[lane] : -FLT_MAX;
    if (warp == 0) {
        value = warpReduceMax(value);
    }
    if (threadIdx.x == 0) {
        shared[0] = value;
    }
    __syncthreads();
    return shared[0];
}

__inline__ __device__ float blockReduceSum(float value) {
    __shared__ float shared[kWarpSize];
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    value = warpReduceSum(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();

    value = (threadIdx.x < blockDim.x / kWarpSize) ? shared[lane] : 0.0f;
    if (warp == 0) {
        value = warpReduceSum(value);
    }
    if (threadIdx.x == 0) {
        shared[0] = value;
    }
    __syncthreads();
    return shared[0];
}

}  // namespace

__global__ void softmax_rowwise_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const float* rowInput = input + row * cols;
    float* rowOutput = output + row * cols;

    float localMax = -FLT_MAX;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        localMax = fmaxf(localMax, rowInput[col]);
    }

    const float rowMax = blockReduceMax(localMax);
    __syncthreads();

    float localSum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const float shifted = rowInput[col] - rowMax;
        const float value = expf(shifted);
        rowOutput[col] = value;
        localSum += value;
    }

    const float sum = blockReduceSum(localSum);
    __syncthreads();

    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        rowOutput[col] /= sum;
    }
}

__global__ void rowwise_max_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const float* rowInput = input + row * cols;
    float localMax = -FLT_MAX;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        localMax = fmaxf(localMax, rowInput[col]);
    }

    const float rowMax = blockReduceMax(localMax);
    if (threadIdx.x == 0) {
        output[row] = rowMax;
    }
}

__global__ void rowwise_sum_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const float* rowInput = input + row * cols;
    float localSum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        localSum += rowInput[col];
    }

    const float rowSum = blockReduceSum(localSum);
    if (threadIdx.x == 0) {
        output[row] = rowSum;
    }
}

void launch_softmax_rowwise(
    const float* input,
    float* output,
    int rows,
    int cols,
    cudaStream_t stream
) {
    const int threads = 256;
    softmax_rowwise_kernel<<<rows, threads, 0, stream>>>(input, output, rows, cols);
}

void launch_rowwise_max(
    const float* input,
    float* output,
    int rows,
    int cols,
    cudaStream_t stream
) {
    const int threads = 256;
    rowwise_max_kernel<<<rows, threads, 0, stream>>>(input, output, rows, cols);
}

void launch_rowwise_sum(
    const float* input,
    float* output,
    int rows,
    int cols,
    cudaStream_t stream
) {
    const int threads = 256;
    rowwise_sum_kernel<<<rows, threads, 0, stream>>>(input, output, rows, cols);
}
