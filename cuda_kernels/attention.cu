#include <cuda_runtime.h>

#include <float.h>

namespace {

constexpr int kTile = 16;

__global__ void qk_matmul_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    float* __restrict__ scores,
    int batch,
    int heads,
    int query_len,
    int key_len,
    int head_dim,
    float scale
) {
    const int batchHead = blockIdx.z;
    const int b = batchHead / heads;
    const int h = batchHead % heads;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch || h >= heads || row >= query_len || col >= key_len) {
        return;
    }

    const float* qBase = q + (((b * heads + h) * query_len + row) * head_dim);
    const float* kBase = k + (((b * heads + h) * key_len + col) * head_dim);

    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        dot += qBase[d] * kBase[d];
    }
    scores[(((b * heads + h) * query_len + row) * key_len + col)] = dot * scale;
}

__global__ void causal_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int heads,
    int query_len,
    int key_len
) {
    const int batchHead = blockIdx.x;
    const int row = threadIdx.x;
    if (batchHead >= batch * heads || row >= query_len) {
        return;
    }

    const int b = batchHead / heads;
    const int h = batchHead % heads;
    const float* rowInput = input + (((b * heads + h) * query_len + row) * key_len);
    float* rowOutput = output + (((b * heads + h) * query_len + row) * key_len);

    float rowMax = -FLT_MAX;
    for (int col = 0; col < key_len; ++col) {
        if (col > row) {
            continue;
        }
        rowMax = fmaxf(rowMax, rowInput[col]);
    }

    float rowSum = 0.0f;
    for (int col = 0; col < key_len; ++col) {
        if (col > row) {
            rowOutput[col] = 0.0f;
            continue;
        }
        const float value = expf(rowInput[col] - rowMax);
        rowOutput[col] = value;
        rowSum += value;
    }

    const float denom = fmaxf(rowSum, 1e-9f);
    for (int col = 0; col < key_len; ++col) {
        rowOutput[col] /= denom;
    }
}

__global__ void av_matmul_kernel(
    const float* __restrict__ attn,
    const float* __restrict__ v,
    float* __restrict__ output,
    int batch,
    int heads,
    int query_len,
    int value_len,
    int head_dim
) {
    const int batchHead = blockIdx.z;
    const int b = batchHead / heads;
    const int h = batchHead % heads;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch || h >= heads || row >= query_len || col >= head_dim) {
        return;
    }

    const float* attnBase = attn + (((b * heads + h) * query_len + row) * value_len);
    float* outBase = output + (((b * heads + h) * query_len + row) * head_dim);

    float sum = 0.0f;
    for (int k = 0; k < value_len; ++k) {
        const float* vBase = v + (((b * heads + h) * value_len + k) * head_dim);
        sum += attnBase[k] * vBase[col];
    }
    outBase[col] = sum;
}

}  // namespace

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
) {
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    dim3 qkBlock(kTile, kTile);
    dim3 qkGrid((key_len + kTile - 1) / kTile, (query_len + kTile - 1) / kTile, batch * heads);
    qk_matmul_kernel<<<qkGrid, qkBlock, 0, stream>>>(q, k, scores, batch, heads, query_len, key_len, head_dim, scale);

    dim3 softmaxBlock(256);
    causal_softmax_kernel<<<batch * heads, softmaxBlock, 0, stream>>>(
        scores, attention, batch, heads, query_len, key_len
    );

    dim3 avBlock(kTile, kTile);
    dim3 avGrid((head_dim + kTile - 1) / kTile, (query_len + kTile - 1) / kTile, batch * heads);
    av_matmul_kernel<<<avGrid, avBlock, 0, stream>>>(
        attention, v, output, batch, heads, query_len, key_len, head_dim
    );
}
