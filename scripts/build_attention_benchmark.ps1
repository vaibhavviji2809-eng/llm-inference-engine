param(
    [string]$Output = "benchmarks\\attention_benchmark.exe"
)

$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if (-not $nvcc) {
    Write-Error "nvcc was not found. Install the CUDA Toolkit and rerun this script."
    exit 1
}

$source = "benchmarks\\attention_benchmark.cu"
$kernels = @(
    "cuda_kernels\\softmax.cu",
    "cuda_kernels\\attention.cu"
)

& $nvcc.Source "-O3" "--std=c++17" $source @kernels "-o" $Output
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Output "Built $Output"
