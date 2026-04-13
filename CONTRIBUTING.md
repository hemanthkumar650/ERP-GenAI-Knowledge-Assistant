# Contributing

Thanks for helping improve this project.

1. **Issues** — Use the **bug** or **feature** templates when reporting problems or ideas (security: use the private advisory link from the template chooser).
2. **Pull requests** — GitHub will show a short checklist (summary, testing, no secrets).
3. **Local checks** — Before pushing, run the same core checks CI uses:
   - Windows: [`scripts/ci-local.ps1`](scripts/ci-local.ps1) (optional: `-SkipPip`)
   - Linux / macOS / WSL: [`scripts/ci-local.sh`](scripts/ci-local.sh) (optional: `--skip-pip`)
   - Prereqs: **Node 20**, **Python 3.11**, and **.NET 8 SDK**
4. **Security** — Do not open public issues for undisclosed vulnerabilities; see [SECURITY.md](SECURITY.md).
