#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../cuda_kernels/matmul.h"

namespace {

struct BenchmarkResult {
    std::string method;
    double milliseconds;
    double gflops;
    double maxError;
};

void checkCuda(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        std::cerr << message << ": " << cudaGetErrorString(status) << '\n';
        std::exit(1);
    }
}

std::vector<float> randomMatrix(int size, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(size);
    for (float& value : data) {
        value = dist(rng);
    }
    return data;
}

std::vector<float> cpuMatmul(
    const std::vector<float>& A,
    const std::vector<float>& B,
    int M,
    int N,
    int K
) {
    std::vector<float> C(M * N, 0.0f);
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int inner = 0; inner < K; ++inner) {
                sum += A[row * K + inner] * B[inner * N + col];
            }
            C[row * N + col] = sum;
        }
    }
    return C;
}

double maxAbsError(const std::vector<float>& reference, const std::vector<float>& actual) {
    double maxError = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double error = std::abs(static_cast<double>(reference[i]) - actual[i]);
        maxError = std::max(maxError, error);
    }
    return maxError;
}

template <typename Fn>
BenchmarkResult benchmarkCpu(
    const std::string& method,
    Fn&& fn,
    const std::vector<float>& reference,
    int M,
    int N,
    int K
) {
    const auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> output = fn();
    const auto stop = std::chrono::high_resolution_clock::now();
    const double milliseconds =
        std::chrono::duration<double, std::milli>(stop - start).count();
    const double gflops = (2.0 * M * N * K) / (milliseconds * 1e6);
    return {method, milliseconds, gflops, maxAbsError(reference, output)};
}

BenchmarkResult benchmarkCudaKernel(
    const std::string& method,
    void (*launcher)(const float*, const float*, float*, int, int, int, cudaStream_t),
    const std::vector<float>& hostA,
    const std::vector<float>& hostB,
    const std::vector<float>& reference,
    int M,
    int N,
    int K,
    int warmupRuns,
    int timedRuns
) {
    const size_t bytesA = static_cast<size_t>(M) * K * sizeof(float);
    const size_t bytesB = static_cast<size_t>(K) * N * sizeof(float);
    const size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);

    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;
    checkCuda(cudaMalloc(&devA, bytesA), "cudaMalloc A failed");
    checkCuda(cudaMalloc(&devB, bytesB), "cudaMalloc B failed");
    checkCuda(cudaMalloc(&devC, bytesC), "cudaMalloc C failed");

    checkCuda(cudaMemcpy(devA, hostA.data(), bytesA, cudaMemcpyHostToDevice), "copy A failed");
    checkCuda(cudaMemcpy(devB, hostB.data(), bytesB, cudaMemcpyHostToDevice), "copy B failed");

    cudaEvent_t start;
    cudaEvent_t stop;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start failed");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop failed");

    for (int i = 0; i < warmupRuns; ++i) {
        launcher(devA, devB, devC, M, N, K, nullptr);
    }
    checkCuda(cudaGetLastError(), "warmup kernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "warmup synchronize failed");

    checkCuda(cudaEventRecord(start), "event start failed");
    for (int i = 0; i < timedRuns; ++i) {
        launcher(devA, devB, devC, M, N, K, nullptr);
    }
    checkCuda(cudaGetLastError(), "timed kernel launch failed");
    checkCuda(cudaEventRecord(stop), "event stop failed");
    checkCuda(cudaEventSynchronize(stop), "event sync failed");

    float totalMilliseconds = 0.0f;
    checkCuda(cudaEventElapsedTime(&totalMilliseconds, start, stop), "elapsed time failed");

    std::vector<float> output(M * N);
    checkCuda(cudaMemcpy(output.data(), devC, bytesC, cudaMemcpyDeviceToHost), "copy C failed");

    checkCuda(cudaEventDestroy(start), "destroy start event failed");
    checkCuda(cudaEventDestroy(stop), "destroy stop event failed");
    checkCuda(cudaFree(devA), "free A failed");
    checkCuda(cudaFree(devB), "free B failed");
    checkCuda(cudaFree(devC), "free C failed");

    const double milliseconds = totalMilliseconds / timedRuns;
    const double gflops = (2.0 * M * N * K) / (milliseconds * 1e6);
    return {method, milliseconds, gflops, maxAbsError(reference, output)};
}

}  // namespace

int main(int argc, char** argv) {
    const int M = argc > 1 ? std::atoi(argv[1]) : 512;
    const int N = argc > 2 ? std::atoi(argv[2]) : 512;
    const int K = argc > 3 ? std::atoi(argv[3]) : 512;
    const int warmupRuns = argc > 4 ? std::atoi(argv[4]) : 3;
    const int timedRuns = argc > 5 ? std::atoi(argv[5]) : 10;

    int deviceCount = 0;
    const cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    if (deviceStatus != cudaSuccess || deviceCount == 0) {
        std::cerr << "No CUDA device available. This benchmark requires a GPU-enabled machine.\n";
        return 2;
    }

    std::mt19937 rng(42);
    const std::vector<float> A = randomMatrix(M * K, rng);
    const std::vector<float> B = randomMatrix(K * N, rng);
    const std::vector<float> reference = cpuMatmul(A, B, M, N, K);

    const BenchmarkResult cpuResult = benchmarkCpu(
        "CPU",
        [&]() { return cpuMatmul(A, B, M, N, K); },
        reference,
        M,
        N,
        K
    );
    const BenchmarkResult naiveResult = benchmarkCudaKernel(
        "Naive CUDA",
        launch_matmul_naive,
        A,
        B,
        reference,
        M,
        N,
        K,
        warmupRuns,
        timedRuns
    );
    const BenchmarkResult tiledResult = benchmarkCudaKernel(
        "Tiled CUDA",
        launch_matmul_tiled,
        A,
        B,
        reference,
        M,
        N,
        K,
        warmupRuns,
        timedRuns
    );

    std::cout << "Matrix multiplication benchmark\n";
    std::cout << "Shape: M=" << M << " N=" << N << " K=" << K << "\n\n";
    std::cout << std::left
              << std::setw(16) << "Method"
              << std::setw(16) << "Time (ms)"
              << std::setw(16) << "GFLOP/s"
              << std::setw(16) << "Max Error"
              << '\n';
    for (const BenchmarkResult& result : {cpuResult, naiveResult, tiledResult}) {
        std::cout << std::left
                  << std::setw(16) << result.method
                  << std::setw(16) << std::fixed << std::setprecision(4) << result.milliseconds
                  << std::setw(16) << std::fixed << std::setprecision(2) << result.gflops
                  << std::setw(16) << std::scientific << std::setprecision(3) << result.maxError
                  << '\n';
    }

    return 0;
}
