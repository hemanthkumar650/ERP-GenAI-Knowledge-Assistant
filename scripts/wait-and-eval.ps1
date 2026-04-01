#Requires -Version 5.1
<#
.SYNOPSIS
  Waits for the Python RAG service to be healthy, then runs retrieval evaluation.

.DESCRIPTION
  Optionally launches start-dev.ps1 first, then polls GET /health until ready.
  Use -RequireIndexed to wait until indexed_chunks > 0 (recommended before eval).
#>
param(
  [string]$BaseUrl = "",
  [int]$TopK = 3,
  [string]$Dataset = "data/eval/erp_eval.json",
  [string]$ReportDir = "data/eval/reports",
  [int]$MaxWaitSeconds = 180,
  [int]$PollSeconds = 2,
  [switch]$RequireIndexed,
  [switch]$StartDev,
  [switch]$ReindexFirst
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

function Get-ProjectBaseUrl {
  $envFile = Join-Path $Root ".env"
  if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
      if ($_ -match '^\s*PYTHON_RAG_URL\s*=\s*(.+)$') {
        return $Matches[1].Trim().TrimEnd("/")
      }
    }
  }
  return "http://localhost:8001"
}

if ([string]::IsNullOrWhiteSpace($BaseUrl)) {
  $BaseUrl = Get-ProjectBaseUrl
}

$healthUrl = "$($BaseUrl.TrimEnd('/'))/health"
$reindexUrl = "$($BaseUrl.TrimEnd('/'))/reindex"

if ($StartDev) {
  Write-Host "Starting dev stack (new windows)..."
  & "$Root\scripts\start-dev.ps1"
  Start-Sleep -Seconds 2
}

$deadline = (Get-Date).AddSeconds($MaxWaitSeconds)
Write-Host "Waiting for RAG health: $healthUrl (max ${MaxWaitSeconds}s)"
if ($RequireIndexed) {
  Write-Host "  (will wait until indexed_chunks > 0)"
}

while ((Get-Date) -lt $deadline) {
  try {
    $h = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 5
    $ok = ($h.status -eq "ok") -and ($h.vector_db_loaded -eq $true)
    if (-not $ok) {
      throw "not ready"
    }
    if ($RequireIndexed -and ([int]$h.indexed_chunks -le 0)) {
      throw "no chunks yet"
    }
    Write-Host "RAG is ready. indexed_chunks: $($h.indexed_chunks)"
    break
  } catch {
    Write-Host "  ... waiting ($($_.Exception.Message))"
    Start-Sleep -Seconds $PollSeconds
  }
}

try {
  $final = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 10
} catch {
  Write-Error "RAG did not become healthy within ${MaxWaitSeconds}s. Start Python RAG or increase -MaxWaitSeconds."
  exit 1
}

if (($final.status -ne "ok") -or ($final.vector_db_loaded -ne $true)) {
  Write-Error "RAG health check failed: $($final | ConvertTo-Json -Compress)"
  exit 1
}

if ($RequireIndexed -and ([int]$final.indexed_chunks -le 0)) {
  Write-Error "indexed_chunks is 0. Add PDFs, run reindex, or drop -RequireIndexed."
  exit 1
}

if ($ReindexFirst) {
  Write-Host "POST reindex: $reindexUrl"
  try {
    Invoke-RestMethod -Uri $reindexUrl -Method Post -TimeoutSec 600 | Out-Null
  } catch {
    Write-Error "Reindex failed: $($_.Exception.Message)"
    exit 1
  }
}

Write-Host ""
& "$Root\scripts\run-eval.ps1" -BaseUrl $BaseUrl -TopK $TopK -Dataset $Dataset -ReportDir $ReportDir
exit $LASTEXITCODE
