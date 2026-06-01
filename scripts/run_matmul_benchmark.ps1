param(
    [int]$M = 512,
    [int]$N = 512,
    [int]$K = 512,
    [int]$WarmupRuns = 3,
    [int]$TimedRuns = 10
)

$binary = "benchmarks\\matmul_benchmark.exe"
if (-not (Test-Path $binary)) {
    Write-Error "Benchmark binary not found. Run scripts\\build_matmul_benchmark.ps1 first."
    exit 1
}

& $binary $M $N $K $WarmupRuns $TimedRuns
