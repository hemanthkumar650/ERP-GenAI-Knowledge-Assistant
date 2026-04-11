#!/usr/bin/env bash
# Same checks as GitHub Actions CI (Python + backend + frontend + .NET worker).
# Usage: ./scripts/ci-local.sh          # runs pip install first
#        ./scripts/ci-local.sh --skip-pip
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
  else
    PYTHON="python"
  fi
fi

SKIP_PIP=false
if [[ "${1:-}" == "--skip-pip" ]]; then
  SKIP_PIP=true
fi

echo "=== Python (compileall + unittest) ==="
if [[ "$SKIP_PIP" == false ]]; then
  "$PYTHON" -m pip install -r python_rag/requirements.txt
fi
"$PYTHON" -m compileall -q python_rag
"$PYTHON" -m unittest discover -s python_rag/tests -p "test_*.py"

echo ""
echo "=== Backend (npm ci, build, test) ==="
(
  cd "$ROOT/backend"
  npm ci
  npm run build
  npm test
)

echo ""
echo "=== Frontend (npm ci, test, build) ==="
export CI=true
(
  cd "$ROOT/frontend"
  npm ci
  npm test
  npm run build
)

echo ""
echo "=== .NET worker (dotnet build) ==="
(
  cd "$ROOT/dotnet_worker"
  dotnet build -c Release
)

echo ""
echo "All CI checks passed locally."
