#Requires -Version 5.1
<#
.SYNOPSIS
  Runs the same checks as GitHub Actions CI (Python + backend + frontend).

.DESCRIPTION
  Uses .venv\Scripts\python.exe when present. Pass -SkipPip to skip pip install
  when dependencies are already installed.
#>
param(
  [switch]$SkipPip
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$PythonExe = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
  $PythonExe = "python"
}

Write-Host "=== Python (compileall + unittest) ===" -ForegroundColor Cyan
if (-not $SkipPip) {
  & $PythonExe -m pip install -r python_rag\requirements.txt
}
& $PythonExe -m compileall -q python_rag
& $PythonExe -m unittest discover -s python_rag\tests -p "test_*.py"

Write-Host "`n=== Backend (npm ci, build, test) ===" -ForegroundColor Cyan
Push-Location (Join-Path $Root "backend")
try {
  npm ci
  npm run build
  npm test
} finally {
  Pop-Location
}

Write-Host "`n=== Frontend (npm ci, test, build) ===" -ForegroundColor Cyan
$env:CI = "true"
Push-Location (Join-Path $Root "frontend")
try {
  npm ci
  npm test
  npm run build
} finally {
  Pop-Location
}

Write-Host "`nAll CI checks passed locally." -ForegroundColor Green
