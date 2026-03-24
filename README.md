# ERP GenAI Knowledge Assistant

**A production-style, full-stack RAG application** that lets employees ask questions about ERP and HR policy PDFs in plain English—and get **grounded answers with source citations**, not generic LLM guesses.

I built this to demonstrate how **Retrieval-Augmented Generation** fits into a realistic enterprise stack: separate UI, API gateway, AI service, vector store, and optional background automation.

---

## What I built

- **Multi-service architecture:** React + TypeScript frontend, Node.js/Express API layer, Python FastAPI RAG service, and a **.NET 8 worker** that can detect policy file changes and trigger re-indexing (mirroring how real companies split web apps, AI, and ops jobs).
- **End-to-end RAG:** PDF ingestion → text cleaning → **token-aware chunking** → **Azure OpenAI embeddings** → **Chroma** vector storage → semantic retrieval → **grounded generation** with strict prompts so answers stay tied to retrieved context.
- **Trust and transparency:** Responses include **source files and retrieved chunks** (similarity scores) so users can verify what the model used.
- **Developer experience:** Health checks, reindex API, Docker Compose for the whole stack, and environment-based configuration.

---

## Why it matters

Internal policy search is usually slow and error-prone. This project shows I can ship a **GenAI feature** that is not “chat with GPT,” but **document-grounded Q&A** with a clear path from PDFs to answers—something hiring teams care about for **AI engineering** and **full-stack** roles.

---

## Tech stack

| Layer | Technologies |
|--------|----------------|
| **Frontend** | React 18, TypeScript |
| **API gateway** | Node.js, Express, TypeScript, Axios, Helmet, CORS |
| **RAG & AI** | Python, FastAPI, ChromaDB, Azure OpenAI (chat + embeddings), pypdf, tiktoken |
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
3. Chunks are **embedded** and stored in **Chroma** for similarity search.
4. On each question, the system **retrieves top-k chunks**, builds a **grounded prompt**, and calls **Azure OpenAI** for the final answer—with **sources** returned to the client.

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

**Optional — .NET worker:** `cd dotnet_worker` → `dotnet run` (auto reindex on file changes).

**Docker:** `docker compose up --build` from the project root.

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
frontend/          React app
backend/           Express API
python_rag/        FastAPI RAG service
dotnet_worker/     Policy watcher + reindex trigger
data/policies/     PDFs (local)
docker-compose.yml
.env.example       Template only — copy to .env (never commit secrets)
```

---

## Security note

`.env` is gitignored. Use `.env.example` as a template only; **do not commit API keys**. For production, use a secret store or CI secrets.

---

## Troubleshooting (short)

- **Port in use:** Find the process (`netstat -ano | findstr :5000`) and end it, or use one server per port.
- **`indexed_chunks` is 0:** Add PDFs under `data/policies`, fix `POLICIES_PATH`, then call `/reindex`.
- **Chat 500:** Confirm Azure variables in `.env` and that `PYTHON_RAG_URL` matches the running Python port.
- **PowerShell + npm:** Use `npm.cmd` if `npm` is blocked.

---