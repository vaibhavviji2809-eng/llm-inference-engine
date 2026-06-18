param(
    [int]$BatchSize = 2,
    [int]$QueryLen = 64,
    [int]$KeyLen = 64,
    [int]$Heads = 4,
    [int]$HeadDim = 16,
    [int]$OutDim = 64,
    [int]$WarmupRuns = 3,
    [int]$TimedRuns = 10
)

$binary = "benchmarks\\attention_benchmark.exe"
if (-not (Test-Path $binary)) {
    Write-Error "Benchmark binary not found. Run scripts\\build_attention_benchmark.ps1 first."
    exit 1
}

& $binary $BatchSize $QueryLen $KeyLen $Heads $HeadDim $OutDim $WarmupRuns $TimedRuns
