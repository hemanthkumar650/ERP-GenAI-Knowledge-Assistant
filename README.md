# ERP GenAI Knowledge Assistant

[![CI](https://github.com/hemanthkumar650/ERP-GenAI-Knowledge-Assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/hemanthkumar650/ERP-GenAI-Knowledge-Assistant/actions/workflows/ci.yml)

License: **MIT** — see [LICENSE](LICENSE). Contributing: [CONTRIBUTING.md](CONTRIBUTING.md).

**A production-style, full-stack RAG application** that lets employees ask questions about ERP and HR policy PDFs in plain English—and get **grounded answers with source citations**, not generic LLM guesses.

I built this to demonstrate how **Retrieval-Augmented Generation** fits into a realistic enterprise stack: separate UI, API gateway, AI service, vector store, and optional background automation.

---

## What I built

- **Multi-service architecture:** React + TypeScript frontend, Node.js/Express API layer, Python FastAPI RAG service, and a **.NET 8 worker** that can detect policy file changes and trigger re-indexing (mirroring how real companies split web apps, AI, and ops jobs).
- **End-to-end RAG:** PDF ingestion → text cleaning → **token-aware chunking** → **Azure OpenAI embeddings** → **Chroma** vector storage → semantic retrieval → **grounded generation** with strict prompts so answers stay tied to retrieved context.
- **Trust and transparency:** Responses include **source files and retrieved chunks** (similarity scores) so users can verify what the model used.
- **Metadata-enriched indexing:** During ingestion/indexing, each chunk is tagged with **policy_type**, **effective_date**, **department**, and **version** to support governance-friendly retrieval and filtering.
- **Developer experience:** Health checks, reindex API, Docker Compose for the whole stack, environment-based configuration, **GitHub Actions CI** (Python, Node, **.NET 8 worker**, Docker Compose **config + image builds** on `main`), and **Dependabot** for npm/pip/NuGet/**Docker base images**/GitHub Actions updates.

---

## Why it matters

Internal policy search is usually slow and error-prone. This project shows I can ship a **GenAI feature** that is not “chat with GPT,” but **document-grounded Q&A** with a clear path from PDFs to answers—something hiring teams care about for **AI engineering** and **full-stack** roles.

---

## Tech stack

| Layer | Technologies |
|--------|----------------|
| **Frontend** | React 18, TypeScript |
| **API gateway** | Node.js, Express, TypeScript, Axios, Helmet, CORS |
| **RAG & AI** | Python, FastAPI, ChromaDB, Azure OpenAI (chat + embeddings), **BM25 + RRF hybrid retrieval** (`rank-bm25`), pypdf, tiktoken |
| **Automation** | .NET 8 Worker Service, HttpClient |
| **Ops** | Docker, Docker Compose |

---

## Architecture (high level)

```text
Browser (React)
    → Node/Express (:5000) — REST + proxy to Python
        → FastAPI RAG (:8000/8001) — retrieve + generate
            → Azure OpenAI + Chroma (vector DB)

Optional: .NET worker watches policy PDFs → calls /reindex on the Python service
```

**Services**

| Folder | Role |
|--------|------|
| `frontend/` | Chat UI, health, sources, evidence preview |
| `backend/` | `/api/chat`, `/api/search`, `/api/reindex`, `/api/health` |
| `python_rag/` | Indexing, retrieval, embeddings, grounded answers |
| `dotnet_worker/` | Background reindex when policy files change |

---

## How the RAG pipeline works

1. Policy **PDFs** live under `data/policies/`.
2. Text is **cleaned and chunked** (token limits + overlap) for better retrieval.
3. For each document, policy metadata is extracted (`policy_type`, `effective_date`, `department`, `version`) and attached to every chunk.
4. Chunks are **embedded** and stored in **Chroma** for similarity search.
5. On each question, the system **retrieves top-k chunks** using **hybrid retrieval** by default: **dense vectors (Chroma)** plus **BM25 keyword scores** over the same chunks, merged with **reciprocal rank fusion (RRF)** so exact policy numbers and acronyms surface reliably. Set `HYBRID_RETRIEVAL=false` in `.env` to use vector-only search.
6. A **grounded prompt** is built and **Azure OpenAI** returns the final answer—with **sources** returned to the client.

---

## Getting started

**Prerequisites:** Node.js 20+, Python 3.11+, Azure OpenAI (chat + embedding deployments). Optional: .NET 8 SDK for the worker, Docker for Compose.

**1. Clone and configure**

```powershell
cd <project-root>
Copy-Item .env.example .env
# Edit .env: Azure keys, endpoint, deployments; POLICIES_PATH; PYTHON_RAG_URL must match your Python port
```

**2. Install**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r python_rag\requirements.txt

cd backend; npm install; cd ..
cd frontend; npm install; cd ..
```

**Optional — run both Node test suites from the repo root:**

```powershell
cd <project-root>
npm test
```

**3. Run (three terminals — order: Python → backend → frontend)**

```powershell
# Terminal 1 — RAG API (port must match PYTHON_RAG_URL in .env)
cd python_rag
python -m uvicorn main:app --host 0.0.0.0 --port 8001
```

```powershell
# Terminal 2
cd backend
npm run dev
```

```powershell
# Terminal 3
cd frontend
npm start
```

**4. Index policies (first run or after new PDFs)**

```powershell
Invoke-RestMethod -Method Post http://localhost:8001/reindex
```

Open the app URL from the React output (often `http://localhost:3000`). Ask a question; confirm **sources** and **chunks** appear.

### Verify hybrid retrieval and read API output (PowerShell)

Hybrid search (**BM25 + vector + RRF**) is **on by default**. Dependencies include `rank-bm25` — install from the **project root** (not `backend/`):

```powershell
cd <project-root>
pip install -r python_rag\requirements.txt
```

In `.env`, leave hybrid enabled or set explicitly:

```env
HYBRID_RETRIEVAL=true
RRF_K=60
HYBRID_CANDIDATE_MULTIPLIER=2
```

Use `HYBRID_RETRIEVAL=false` for **vector-only** retrieval.

**Inspect responses as full JSON** — `Invoke-RestMethod` otherwise truncates long fields in the console:

```powershell
# Direct Python RAG
Invoke-RestMethod -Method Post http://localhost:8001/search -ContentType "application/json" -Body '{"query":"expense reimbursement policy 2.21","topK":3}' | ConvertTo-Json -Depth 8

# Via backend proxy (frontend uses these paths)
Invoke-RestMethod "http://localhost:5000/api/chunks?limit=3" | ConvertTo-Json -Depth 8
```

Each retrieved chunk includes:

| Field | Meaning |
|--------|--------|
| `source` | PDF filename (e.g. `Employee-Expense-Reimbursement-Policy-2023-APPROVED-External-Use.pdf`) |
| `chunk_id` | Stable id within that file (e.g. `…pdf::chunk-5`) |
| `similarity` | Normalized score (vector distance, or fused RRF score when hybrid is on) |
| `retrieval` | `"hybrid"` when BM25+vector+RRF is used, `"vector"` when hybrid is off |
| `policy_type` | Extracted policy category (e.g. Finance, HR, IT) |
| `effective_date` | Parsed effective date when available |
| `department` | Sponsoring/owner department parsed from policy header |
| `version` | Parsed policy version (or year fallback) |

Restart the Python service after changing `.env` or installing packages.

**Optional — .NET worker:** `cd dotnet_worker` → `dotnet run` (auto reindex on file changes).

**Docker:** `docker compose up --build` from the project root.

---

## Retrieval evaluation (quick quality check)

Run the starter eval set and measure source hit-rate at top-k:

```powershell
cd <project-root>
python python_rag/tests/eval_retrieval.py --dataset data/eval/erp_eval.json --base-url http://localhost:8001 --top-k 3
```

Shortcut command:

```powershell
cd <project-root>
.\scripts\run-eval.ps1
```

Wait until the RAG API is up (and optionally until documents are indexed), then run eval:

```powershell
cd <project-root>
.\scripts\wait-and-eval.ps1 -RequireIndexed
```

First-time setup in one flow (opens dev windows, waits for health, requires a non-empty index, then eval):

```powershell
.\scripts\wait-and-eval.ps1 -StartDev -RequireIndexed -MaxWaitSeconds 300
```

Output includes:
- `Hit@1` (correct source keyword in first result)
- `Hit@k` (correct source keyword appears within top-k results)
- A short miss list so you can tune chunking/retrieval and compare before vs after changes.
- Auto-generated files in `data/eval/reports/` (folder is gitignored so reports stay local):
  - `retrieval-eval-<timestamp>.csv` (row-level results)
  - `retrieval-eval-<timestamp>.md` (shareable summary)

---

## API (backend, used via `/api/...` from the frontend)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Backend status |
| GET | `/api/health` | Python + vector DB status |
| GET | `/api/chunks` | Sample indexed chunks |
| POST | `/api/chat` | Grounded Q&A |
| POST | `/api/search` | Semantic retrieval |
| POST | `/api/reindex` | Rebuild vector index |

With the **Python RAG** service running, OpenAPI (Swagger) UI is at **`http://localhost:<port>/docs`** — use the same port as `PYTHON_RAG_URL` in `.env` (for example `8001`).

Example chat body:

```json
{
  "message": "What is the expense reimbursement policy?",
  "conversationId": "optional-session-id"
}
```

---

## Repository layout

```text
.github/           CI, Dependabot, PR + issue templates
CONTRIBUTING.md    How to contribute and run local checks
frontend/          React app
backend/           Express API
python_rag/        FastAPI RAG service
dotnet_worker/     Policy watcher + reindex trigger
data/policies/     PDFs (local)
docker-compose.yml
.dockerignore      Smaller/faster `docker compose build` context
.env.example       Template only — copy to .env (never commit secrets)
LICENSE            MIT license
SECURITY.md        Vulnerability reporting (private disclosure)
```

---

## Security note

`.env` is gitignored. Use `.env.example` as a template only; **do not commit API keys**. For production, use a secret store or CI secrets.

See **[SECURITY.md](SECURITY.md)** for how to report vulnerabilities privately.

---

## Continuous integration (local)

GitHub Actions runs tests and builds on push/PR (including **`dotnet_worker`** and compose validation). To run the **core checks** locally you need **Node20**, **Python 3.11**, and **.NET 8 SDK** (for the worker). On Windows before you push:

```powershell
cd <project-root>
.\scripts\ci-local.ps1
```

Skip reinstalling Python packages when your venv is already set up:

```powershell
.\scripts\ci-local.ps1 -SkipPip
```

Or:

```powershell
npm run ci:local
```

On **Linux, macOS, or WSL** (after `chmod +x scripts/ci-local.sh` once):

```bash
cd <project-root>
./scripts/ci-local.sh
./scripts/ci-local.sh --skip-pip
```

---

## Troubleshooting (short)

- **Port in use:** Find the process (`netstat -ano | findstr :5000`) and end it, or use one server per port.
- **`indexed_chunks` is 0:** Add PDFs under `data/policies`, fix `POLICIES_PATH`, then call `/reindex`.
- **Chat 500:** Confirm Azure variables in `.env` and that `PYTHON_RAG_URL` matches the running Python port.
- **PowerShell + npm:** Use `npm.cmd` if `npm` is blocked.

---