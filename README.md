# ERP-GenAI-Knowledge-Assistant

Full-stack ERP policy assistant built with React, Node.js, Python, and .NET.

This repository demonstrates a multi-service Retrieval-Augmented Generation (RAG) architecture:

- `frontend/`: React + TypeScript chat UI
- `backend/`: Node.js + Express + TypeScript API (proxies to Python)
- `python_rag/`: FastAPI + Chroma + Azure OpenAI RAG service
- `dotnet_worker/`: .NET 8 background worker for policy change monitoring and reindex triggers

## Architecture

```text
User
  ->
React dev server (:3000 or :3001 if 3000 is busy)
  proxies /api ->
Node/Express API (:5000)
  forwards to ->
Python RAG service (:8000 or :8001 — must match PYTHON_RAG_URL in .env)
  ->
Azure OpenAI + Chroma (persistent vector store under data/chroma_db)

Policy file changes (optional)
  ->
.NET worker
  ->
POST Python /reindex
```

### What each service does

- **frontend**: Chat UI, health, sources, retrieved chunks; `/api/*` is proxied to the Node backend (`frontend/src/setupProxy.js`).
- **backend**: `GET /health`, `GET /api/*`, `POST /api/chat`, `POST /api/search`, `POST /api/reindex` — calls Python using `PYTHON_RAG_URL`.
- **python_rag**: PDF ingestion, chunking, embeddings, Chroma retrieval, grounded answers. Loads `.env` from the **project root** (`python_rag/config.py`).
- **dotnet_worker**: Polls the policies folder and calls Python `/reindex` when PDFs change (optional for local dev).

## Project layout

```text
Project/
|-- frontend/
|-- backend/
|-- python_rag/
|-- dotnet_worker/
|-- data/
|   |-- policies/          <- place ERP policy PDFs here
|   `-- chroma_db/         <- Chroma persistence (gitignored if large)
|-- docker-compose.yml
|-- .env.example
`-- README.md
```

## Tech stack

- **Frontend:** React 18, TypeScript, react-scripts, http-proxy-middleware
- **Backend:** Node.js, Express, TypeScript, Axios, Helmet, CORS
- **RAG:** Python 3.11+, FastAPI, Chroma, Azure OpenAI, pypdf, tiktoken
- **Worker:** .NET 8 Worker Service, HttpClient
- **DevOps:** Docker, Docker Compose

## Environment variables

Create `.env` at the **project root** (copy from `.env.example`):

```powershell
Copy-Item .env.example .env
```

Fill at least:

| Variable | Purpose |
|----------|---------|
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | e.g. `https://<resource>.openai.azure.com/` |
| `AZURE_OPENAI_API_VERSION` | API version for your resource |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Chat model deployment name |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding deployment name |
| `POLICIES_PATH` | Folder containing `*.pdf` policies (see below) |
| `CHROMA_PATH` | Chroma DB directory (default `data/chroma_db` relative to cwd when running Python) |
| `PYTHON_RAG_URL` | URL the **backend** uses to call Python (must match the port you start uvicorn on) |

**Policy folder:** Put PDFs in `data/policies` at the repo root. When you run uvicorn from `python_rag/`, use:

```env
POLICIES_PATH=../data/policies
```

**Backend → Python:** If Python listens on port `8001`, set:

```env
PYTHON_RAG_URL=http://localhost:8001
```

If Python listens on `8000`, use `http://localhost:8000`. Restart the backend after changing this.

Optional: Langfuse keys in `.env` for tracing (see `python_rag/observability.py`).

### Secrets and GitHub

- **Do not commit** `.env` — it is listed in `.gitignore` and should contain real API keys only on your machine or in a secure secret store (CI variables, Azure Key Vault, etc.).
- **Do commit** `.env.example` (placeholders only, no real keys).
- Before pushing, run `git status` and confirm `.env` does **not** appear as a new file to be added.
- If `.env` was ever committed by mistake, remove it from Git history, rotate the exposed keys in Azure, and use `git rm --cached .env` so future commits stop tracking it.

## Prerequisites

- Node.js 20+
- Python 3.11+
- .NET 8 SDK (only if you run `dotnet_worker`)
- Azure OpenAI resource with chat + embedding deployments

Optional: Docker Desktop for `docker compose`.

## Local setup (first time)

From the project root:

```powershell
cd C:\Users\heman\Desktop\Project
```

### 1. Python virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r python_rag\requirements.txt
```

### 2. Backend

```powershell
cd backend
npm.cmd install
cd ..
```

### 3. Frontend

```powershell
cd frontend
npm.cmd install
cd ..
```

### 4. .NET worker (optional)

```powershell
cd dotnet_worker
dotnet restore
cd ..
```

## Run locally

Use **three terminals** (four if you run the .NET worker). Order: **Python → backend → frontend**.

### Terminal 1: Python RAG

```powershell
cd C:\Users\heman\Desktop\Project
.\.venv\Scripts\Activate.ps1
cd python_rag
python -m uvicorn main:app --host 0.0.0.0 --port 8001
```

Use the same port as `PYTHON_RAG_URL` in `.env`. For a single process without the reloader, omit `--reload` (helps avoid duplicate listeners on Windows).

**First-time or after adding PDFs**, rebuild the index (or use **Reindex** in the UI):

```powershell
Invoke-RestMethod -Method Post http://localhost:8001/reindex
```

Check `indexed_chunks` in `http://localhost:8001/health` — it should be greater than zero after PDFs are indexed.

### Terminal 2: Backend API

```powershell
cd C:\Users\heman\Desktop\Project\backend
npm.cmd run dev
```

Listens on `http://localhost:5000` (or `PORT` from `.env`).

### Terminal 3: Frontend

```powershell
cd C:\Users\heman\Desktop\Project\frontend
npm.cmd start
```

Opens CRA on `http://localhost:3000` (or **3001** if 3000 is in use — use the URL printed in the terminal).

### Terminal 4 (optional): .NET worker

```powershell
cd C:\Users\heman\Desktop\Project\dotnet_worker
dotnet run
```

Ensures `PoliciesPath` / `PythonRag:BaseUrl` in `appsettings.json` match your folders and Python URL.

## Access points

| Service | URL |
|---------|-----|
| Frontend | `http://localhost:3000` (or printed port) |
| Backend | `http://localhost:5000/health` |
| Python RAG | `http://localhost:<port>/health` (port = uvicorn + `PYTHON_RAG_URL`) |

**Smoke test**

1. `Invoke-RestMethod http://localhost:5000/health` — JSON with `pythonRagUrl` pointing at your Python URL.
2. Open the frontend URL; **Refresh Health** should show `ok`.
3. Ask a policy question; you should get an answer, **Sources**, and **Retrieved Evidence** chunks.

## API overview

### Backend (via frontend proxy: `/api/...`)

- `GET /health` — backend status
- `GET /api/health` — proxied Python health
- `GET /api/chunks?limit=6` — sample indexed chunks
- `POST /api/chat` — grounded chat
- `POST /api/search` — semantic search
- `POST /api/reindex` — rebuild vector index

### Example chat request

```json
{
  "message": "What is the company policy on expense reimbursement?",
  "conversationId": "optional-session-id"
}
```

### Example search request

```json
{
  "query": "expense reimbursement",
  "topK": 3
}
```

## Docker

From the project root:

```powershell
docker compose build
docker compose up
```

Stop:

```powershell
docker compose down
```

Services and ports are defined in `docker-compose.yml`.

## What the .NET worker does

The worker is **not** the main API. It watches the configured policies directory for changes and `POST`s to Python `/reindex` so the vector store stays in sync. For local development you can skip it and reindex manually or use the UI button.

## Troubleshooting

### Port already in use (`EADDRINUSE` / `WinError 10048`)

Something is still bound to that port (often a previous dev server).

**Find PID (example for 5000):**

```powershell
netstat -ano | findstr :5000
```

**Stop the process** (replace `<PID>`):

```powershell
taskkill /PID <PID> /F
```

Repeat for `8001`, `3000`, etc., as needed. Only run **one** uvicorn instance per Python port.

### Backend: `listen EADDRINUSE :::5000`

Kill the process on port 5000 (see above), then `npm.cmd run dev` again.

### Python: `only one usage of each socket address` on 8001

Another Python (or app) is already listening. Either use that instance, or free the port and start again.

### Frontend: `Unexpected token '<'` or API errors

- Backend must be on **5000** (or update the proxy in `frontend/src/setupProxy.js`).
- Python must match **`PYTHON_RAG_URL`** in `.env`.
- The dev server proxies `/api` to the backend; restart `npm start` after changing `setupProxy.js`.

### `indexed_chunks` is 0

- Confirm PDFs exist under `data/policies` (or the folder in `POLICIES_PATH`).
- Run `POST` reindex on the Python base URL (see Terminal 1).
- Ensure `POLICIES_PATH` resolves correctly when cwd is `python_rag/` (e.g. `../data/policies`).

### Chat returns 500

- Check the **Python terminal** for stack traces (often missing or invalid Azure credentials).
- Verify all four `AZURE_OPENAI_*` variables in `.env` and restart Python after edits.

### `npm` blocked in PowerShell

Use `npm.cmd`:

```powershell
npm.cmd install
npm.cmd start
```

### .NET worker build errors

Install .NET 8 SDK, e.g.:

```powershell
winget install Microsoft.DotNet.SDK.8
```

Ensure `dotnet_worker` references match a standard .NET 8 worker project (hosted services, `Microsoft.Extensions.Hosting`).
