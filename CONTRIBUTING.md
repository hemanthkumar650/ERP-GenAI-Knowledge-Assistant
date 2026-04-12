# Contributing

Thanks for helping improve this project.

1. **Pull requests** — GitHub will show a short checklist (summary, testing, no secrets).
2. **Local checks** — Before pushing, run the same core checks CI uses:
   - Windows: [`scripts/ci-local.ps1`](scripts/ci-local.ps1) (optional: `-SkipPip`)
   - Linux / macOS / WSL: [`scripts/ci-local.sh`](scripts/ci-local.sh) (optional: `--skip-pip`)
   - Prereqs: **Node 20**, **Python 3.11**, and **.NET 8 SDK**
3. **Security** — Do not open public issues for undisclosed vulnerabilities; see [SECURITY.md](SECURITY.md).
