#Requires -Version 5.1
<#
.SYNOPSIS
  Starts Python RAG, Express backend, and React frontend in separate PowerShell windows.

.DESCRIPTION
  Reads PYTHON_RAG_URL from project-root .env to pick the uvicorn port (default 8001).
  Run from anywhere:  .\scripts\start-dev.ps1
#>

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$EnvFile = Join-Path $Root ".env"
$RagPort = 8001

if (Test-Path $EnvFile) {
  Get-Content $EnvFile | ForEach-Object {
    if ($_ -match '^\s*PYTHON_RAG_URL\s*=\s*https?://[^:]+:(\d+)') {
      $RagPort = [int]$Matches[1]
    }
  }
}

$PythonExe = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
  $PythonExe = "python"
}

$PyWindow = @"
Set-Location "$Root"
if (Test-Path '.venv\Scripts\Activate.ps1') { . '.venv\Scripts\Activate.ps1' }
Set-Location 'python_rag'
& "$PythonExe" -m uvicorn main:app --host 0.0.0.0 --port $RagPort
"@

$BackendWindow = @"
Set-Location "$Root\backend"
npm.cmd run dev
"@

$FrontendWindow = @"
Set-Location "$Root\frontend"
npm.cmd start
"@

Write-Host "Starting dev stack from: $Root"
Write-Host "  Python RAG -> http://localhost:$RagPort  (match PYTHON_RAG_URL in .env)"
Write-Host "  Backend    -> http://localhost:5000"
Write-Host "  Frontend   -> http://localhost:3000 (see terminal for exact URL)"
Write-Host ""

Start-Process powershell -WorkingDirectory $Root -ArgumentList @("-NoExit", "-Command", $PyWindow)
Start-Sleep -Milliseconds 400
Start-Process powershell -WorkingDirectory $Root -ArgumentList @("-NoExit", "-Command", $BackendWindow)
Start-Sleep -Milliseconds 400
Start-Process powershell -WorkingDirectory $Root -ArgumentList @("-NoExit", "-Command", $FrontendWindow)

Write-Host "Three windows opened. First run: POST http://localhost:$RagPort/reindex if the index is empty."
