#Requires -Version 5.1
param(
  [string]$BaseUrl = "http://localhost:8001",
  [int]$TopK = 3,
  [string]$Dataset = "data/eval/erp_eval.json",
  [string]$ReportDir = "data/eval/reports"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$PythonExe = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
  $PythonExe = "python"
}

Write-Host "Running retrieval evaluation..."
Write-Host "  Base URL  : $BaseUrl"
Write-Host "  Top-K     : $TopK"
Write-Host "  Dataset   : $Dataset"
Write-Host "  Report dir: $ReportDir"
Write-Host ""

& $PythonExe "python_rag/tests/eval_retrieval.py" `
  --dataset $Dataset `
  --base-url $BaseUrl `
  --top-k $TopK `
  --report-dir $ReportDir
