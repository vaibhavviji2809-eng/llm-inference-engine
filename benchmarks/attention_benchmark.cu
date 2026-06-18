#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../cuda_kernels/attention.h"

namespace {

struct BenchmarkResult {
    std::string method;
    double milliseconds;
    double tokensPerSecond;
    double maxError;
};

void checkCuda(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        std::cerr << message << ": " << cudaGetErrorString(status) << '\n';
        std::exit(1);
    }
}

std::vector<float> randomVector(size_t size, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(size);
    for (float& value : data) {
        value = dist(rng);
    }
    return data;
}

float maxAbsError(const std::vector<float>& reference, const std::vector<float>& actual) {
    float maxError = 0.0f;
    for (size_t i = 0; i < reference.size(); ++i) {
        maxError = std::max(maxError, std::abs(reference[i] - actual[i]));
    }
    return maxError;
}

inline size_t index4d(int b, int h, int q, int d, int heads, int queryLen, int headDim) {
    return (((static_cast<size_t>(b) * heads + h) * queryLen + q) * headDim) + d;
}

inline size_t index3d(int b, int q, int d, int queryLen, int outDim) {
    return (static_cast<size_t>(b) * queryLen + q) * outDim + d;
}

std::vector<float> cpuAttentionReference(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    const std::vector<float>& outWeight,
    const std::vector<float>& outBias,
    int batch,
    int heads,
    int queryLen,
    int keyLen,
    int headDim,
    int outDim
) {
    std::vector<float> output(batch * queryLen * outDim, 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(headDim));

    for (int b = 0; b < batch; ++b) {
        for (int qRow = 0; qRow < queryLen; ++qRow) {
            std::vector<float> merged(heads * headDim, 0.0f);
            for (int h = 0; h < heads; ++h) {
                std::vector<float> logits(keyLen, -INFINITY);
                float maxLogit = -INFINITY;
                for (int kCol = 0; kCol < keyLen; ++kCol) {
                    if (kCol > qRow) {
                        continue;
                    }
                    float score = 0.0f;
                    for (int d = 0; d < headDim; ++d) {
                        score += q[index4d(b, h, qRow, d, heads, queryLen, headDim)]
                            * k[index4d(b, h, kCol, d, heads, keyLen, headDim)];
                    }
                    score *= scale;
                    logits[kCol] = score;
                    maxLogit = std::max(maxLogit, score);
                }

                float denom = 0.0f;
                for (int kCol = 0; kCol < keyLen; ++kCol) {
                    if (logits[kCol] == -INFINITY) {
                        continue;
                    }
                    logits[kCol] = std::exp(logits[kCol] - maxLogit);
                    denom += logits[kCol];
                }
                denom = std::max(denom, 1e-9f);

                for (int d = 0; d < headDim; ++d) {
                    float acc = 0.0f;
                    for (int kCol = 0; kCol < keyLen; ++kCol) {
                        if (kCol > qRow) {
                            continue;
                        }
                        acc += (logits[kCol] / denom)
                            * v[index4d(b, h, kCol, d, heads, keyLen, headDim)];
                    }
                    merged[h * headDim + d] = acc;
                }
            }

            for (int outCol = 0; outCol < outDim; ++outCol) {
                float sum = outBias[outCol];
                for (int inCol = 0; inCol < heads * headDim; ++inCol) {
                    sum += merged[inCol] * outWeight[inCol * outDim + outCol];
                }
                output[index3d(b, qRow, outCol, queryLen, outDim)] = sum;
            }
        }
    }

    return output;
}

BenchmarkResult benchmarkCuda(
    const std::string& method,
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    const std::vector<float>& outWeight,
    const std::vector<float>& outBias,
    const std::vector<float>& reference,
    int batch,
    int heads,
    int queryLen,
    int keyLen,
    int headDim,
    int outDim,
    int warmupRuns,
    int timedRuns
) {
    const size_t qBytes = q.size() * sizeof(float);
    const size_t kBytes = k.size() * sizeof(float);
    const size_t vBytes = v.size() * sizeof(float);
    const size_t weightBytes = outWeight.size() * sizeof(float);
    const size_t biasBytes = outBias.size() * sizeof(float);
    const size_t scoreBytes = static_cast<size_t>(batch) * heads * queryLen * keyLen * sizeof(float);
    const size_t attentionBytes = scoreBytes;
    const size_t attnOutBytes = static_cast<size_t>(batch) * heads * queryLen * headDim * sizeof(float);
    const size_t projBytes = static_cast<size_t>(batch) * queryLen * outDim * sizeof(float);

    float* devQ = nullptr;
    float* devK = nullptr;
    float* devV = nullptr;
    float* devWeight = nullptr;
    float* devBias = nullptr;
    float* devScores = nullptr;
    float* devAttention = nullptr;
    float* devAttnOut = nullptr;
    float* devProjOut = nullptr;

    checkCuda(cudaMalloc(&devQ, qBytes), "cudaMalloc Q failed");
    checkCuda(cudaMalloc(&devK, kBytes), "cudaMalloc K failed");
    checkCuda(cudaMalloc(&devV, vBytes), "cudaMalloc V failed");
    checkCuda(cudaMalloc(&devWeight, weightBytes), "cudaMalloc weight failed");
    checkCuda(cudaMalloc(&devBias, biasBytes), "cudaMalloc bias failed");
    checkCuda(cudaMalloc(&devScores, scoreBytes), "cudaMalloc scores failed");
    checkCuda(cudaMalloc(&devAttention, attentionBytes), "cudaMalloc attention failed");
    checkCuda(cudaMalloc(&devAttnOut, attnOutBytes), "cudaMalloc attention output failed");
    checkCuda(cudaMalloc(&devProjOut, projBytes), "cudaMalloc projected output failed");

    checkCuda(cudaMemcpy(devQ, q.data(), qBytes, cudaMemcpyHostToDevice), "copy Q failed");
    checkCuda(cudaMemcpy(devK, k.data(), kBytes, cudaMemcpyHostToDevice), "copy K failed");
    checkCuda(cudaMemcpy(devV, v.data(), vBytes, cudaMemcpyHostToDevice), "copy V failed");
    checkCuda(cudaMemcpy(devWeight, outWeight.data(), weightBytes, cudaMemcpyHostToDevice), "copy weight failed");
    checkCuda(cudaMemcpy(devBias, outBias.data(), biasBytes, cudaMemcpyHostToDevice), "copy bias failed");

    cudaEvent_t start;
    cudaEvent_t stop;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start failed");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop failed");

    for (int i = 0; i < warmupRuns; ++i) {
        launch_attention_with_output_projection(
            devQ,
            devK,
            devV,
            devWeight,
            devBias,
            devScores,
            devAttention,
            devAttnOut,
            devProjOut,
            batch,
            heads,
            queryLen,
            keyLen,
            headDim,
            outDim,
            nullptr
        );
    }
    checkCuda(cudaGetLastError(), "warmup launch failed");
    checkCuda(cudaDeviceSynchronize(), "warmup sync failed");

    checkCuda(cudaEventRecord(start), "record start failed");
    for (int i = 0; i < timedRuns; ++i) {
        launch_attention_with_output_projection(
            devQ,
            devK,
            devV,
            devWeight,
            devBias,
            devScores,
            devAttention,
            devAttnOut,
            devProjOut,
            batch,
            heads,
            queryLen,
            keyLen,
            headDim,
            outDim,
            nullptr
        );
    }
    checkCuda(cudaGetLastError(), "timed launch failed");
    checkCuda(cudaEventRecord(stop), "record stop failed");
    checkCuda(cudaEventSynchronize(stop), "sync stop failed");

    float totalMilliseconds = 0.0f;
    checkCuda(cudaEventElapsedTime(&totalMilliseconds, start, stop), "elapsed time failed");

    std::vector<float> actual(reference.size());
    checkCuda(cudaMemcpy(actual.data(), devProjOut, projBytes, cudaMemcpyDeviceToHost), "copy output failed");

    checkCuda(cudaEventDestroy(start), "destroy start failed");
    checkCuda(cudaEventDestroy(stop), "destroy stop failed");
    checkCuda(cudaFree(devQ), "free Q failed");
    checkCuda(cudaFree(devK), "free K failed");
    checkCuda(cudaFree(devV), "free V failed");
    checkCuda(cudaFree(devWeight), "free weight failed");
    checkCuda(cudaFree(devBias), "free bias failed");
    checkCuda(cudaFree(devScores), "free scores failed");
    checkCuda(cudaFree(devAttention), "free attention failed");
    checkCuda(cudaFree(devAttnOut), "free attnOut failed");
    checkCuda(cudaFree(devProjOut), "free projOut failed");

    const double milliseconds = totalMilliseconds / timedRuns;
    const double tokensPerSecond = (static_cast<double>(batch) * queryLen) / (milliseconds / 1000.0);
    return {method, milliseconds, tokensPerSecond, maxAbsError(reference, actual)};
}

}  // namespace

int main(int argc, char** argv) {
    const int batch = argc > 1 ? std::atoi(argv[1]) : 2;
    const int queryLen = argc > 2 ? std::atoi(argv[2]) : 64;
    const int keyLen = argc > 3 ? std::atoi(argv[3]) : queryLen;
    const int heads = argc > 4 ? std::atoi(argv[4]) : 4;
    const int headDim = argc > 5 ? std::atoi(argv[5]) : 16;
    const int outDim = argc > 6 ? std::atoi(argv[6]) : heads * headDim;
    const int warmupRuns = argc > 7 ? std::atoi(argv[7]) : 3;
    const int timedRuns = argc > 8 ? std::atoi(argv[8]) : 10;

    int deviceCount = 0;
    const cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    if (deviceStatus != cudaSuccess || deviceCount == 0) {
        std::cerr << "No CUDA device available. This benchmark requires a GPU-enabled machine.\n";
        return 2;
    }

    std::mt19937 rng(42);
    const std::vector<float> q = randomVector(static_cast<size_t>(batch) * heads * queryLen * headDim, rng);
    const std::vector<float> k = randomVector(static_cast<size_t>(batch) * heads * keyLen * headDim, rng);
    const std::vector<float> v = randomVector(static_cast<size_t>(batch) * heads * keyLen * headDim, rng);
    const std::vector<float> outWeight = randomVector(static_cast<size_t>(heads) * headDim * outDim, rng);
    const std::vector<float> outBias = randomVector(outDim, rng);
    const std::vector<float> reference = cpuAttentionReference(
        q,
        k,
        v,
        outWeight,
        outBias,
        batch,
        heads,
        queryLen,
        keyLen,
        headDim,
        outDim
    );

    const BenchmarkResult result = benchmarkCuda(
        "CUDA attention + output projection",
        q,
        k,
        v,
        outWeight,
        outBias,
        reference,
        batch,
        heads,
        queryLen,
        keyLen,
        headDim,
        outDim,
        warmupRuns,
        timedRuns
    );

    std::cout << "Attention benchmark\n";
    std::cout << "Shape: batch=" << batch << " query_len=" << queryLen << " key_len=" << keyLen
              << " heads=" << heads << " head_dim=" << headDim << " out_dim=" << outDim << "\n\n";
    std::cout << std::left
              << std::setw(28) << "Method"
              << std::setw(16) << "Time (ms)"
              << std::setw(16) << "Tokens/sec"
              << std::setw(16) << "Max Error"
              << '\n';
    std::cout << std::left
              << std::setw(28) << result.method
              << std::setw(16) << std::fixed << std::setprecision(4) << result.milliseconds
              << std::setw(16) << std::fixed << std::setprecision(2) << result.tokensPerSecond
              << std::setw(16) << std::scientific << std::setprecision(3) << result.maxError
              << '\n';

    return 0;
}
