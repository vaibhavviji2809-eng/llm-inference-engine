param(
    [string]$Output = "benchmarks\\matmul_benchmark.exe"
)

$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if (-not $nvcc) {
    Write-Error "nvcc was not found. Install the CUDA Toolkit and rerun this script."
    exit 1
}

$source = "benchmarks\\matmul_benchmark.cu"
$kernel = "cuda_kernels\\matmul.cu"

& $nvcc.Source "-O3" "--std=c++17" $source $kernel "-o" $Output
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Output "Built $Output"
