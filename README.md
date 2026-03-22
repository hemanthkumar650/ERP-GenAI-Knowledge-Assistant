# ERP-GenAI-Knowledge-Assistant

Full-stack ERP policy assistant built with React, Node.js, Python, and .NET.

This repository demonstrates a multi-service Retrieval-Augmented Generation (RAG) architecture:

- `frontend/`: React + TypeScript chat UI
- `backend/`: Node.js + Express + TypeScript API
- `python_rag/`: FastAPI + Chroma + Azure OpenAI RAG service
- `dotnet_worker/`: .NET 8 background worker for policy change monitoring and reindex triggers

## Architecture

```text
User
  ->
React frontend (:3000)
  ->
Node/Express API (:5000)
  ->
Python RAG service (:8000)
  ->
Azure OpenAI + Chroma

Policy file changes
  ->
.NET worker
  ->
Python /reindex
```

### What each service does

- `frontend`: renders the chat experience, health status, sources, and retrieved chunks
- `backend`: exposes `/api/*` endpoints and forwards chat/search/reindex calls to Python
- `python_rag`: performs retrieval, grounding, embeddings, answer generation, and indexing
- `dotnet_worker`: polls `data/policies` for changes and triggers reindexing automatically

## Project Layout

```text
Project/
|-- frontend/
|   |-- public/
|   |-- src/
|   |-- package.json
|   |-- tsconfig.json
|   `-- Dockerfile
|-- backend/
|   |-- src/
|   |   |-- config/
|   |   |-- routes/
|   |   |-- services/
|   |   `-- types/
|   |-- package.json
|   |-- tsconfig.json
|   `-- Dockerfile
|-- python_rag/
|   |-- utils/
|   |-- main.py
|   |-- rag_pipeline.py
|   |-- embed_index.py
|   |-- requirements.txt
|   `-- Dockerfile
|-- dotnet_worker/
|   |-- Services/
|   |-- Workers/
|   |-- Program.cs
|   |-- Worker.csproj
|   |-- appsettings.json
|   `-- Dockerfile
|-- data/
|   |-- policies/
|   `-- chroma_db/
|-- docker-compose.yml
|-- .env.example
`-- README.md
```

## Tech Stack

- Frontend: React 18, TypeScript, react-scripts
- Backend: Node.js, Express, TypeScript, Axios, Helmet, CORS
- RAG service: Python 3.11+, FastAPI, Chroma, Azure OpenAI, pypdf, tiktoken
- Worker: .NET 8 Worker Service, C# 12, HttpClient, hosted background service
- DevOps: Docker, Docker Compose, environment-based config
- Observability: optional Langfuse integration

## Environment Variables

Create `.env` from `.env.example`:

```powershell
Copy-Item .env.example .env
```

Required values:

```env
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small

POLICIES_PATH=data/policies
CHROMA_PATH=data/chroma_db
CHROMA_COLLECTION=erp_policies
TOP_K=3

LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com

PORT=5000
NODE_ENV=development
PYTHON_RAG_URL=http://localhost:8000
```

## Prerequisites

- Node.js 20+
- Python 3.11+
- .NET 8 SDK
- Azure OpenAI deployment credentials

Optional:

- Docker Desktop

## Local Setup

From the project root:

```powershell
cd C:\Users\heman\Desktop\Project
```

### 1. Python dependencies

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r python_rag\requirements.txt
```

### 2. Backend dependencies

```powershell
cd backend
npm.cmd install
cd ..
```

### 3. Frontend dependencies

```powershell
cd frontend
npm.cmd install
cd ..
```

### 4. .NET worker dependencies

```powershell
cd dotnet_worker
dotnet restore
cd ..
```

## Run Locally

Open four terminals.

### Terminal 1: Python RAG

```powershell
cd C:\Users\heman\Desktop\Project
.\.venv\Scripts\Activate.ps1
cd python_rag
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Terminal 2: Backend API

```powershell
cd C:\Users\heman\Desktop\Project\backend
npm.cmd run dev
```

### Terminal 3: Frontend

```powershell
cd C:\Users\heman\Desktop\Project\frontend
npm.cmd start
```

### Terminal 4: .NET worker

```powershell
cd C:\Users\heman\Desktop\Project\dotnet_worker
dotnet run
```

## Access Points

- Frontend: `http://localhost:3000`
- Backend health: `http://localhost:5000/health`
- Python health: `http://localhost:8000/health`

### Quick smoke test

1. Start Python, backend, and frontend.
2. Open `http://localhost:3000`.
3. Ask a policy question.
4. Confirm you receive an answer, sources, and retrieved chunks.
5. Start the `.NET` worker if you want automatic reindex-on-change behavior.

## API Overview

### Backend endpoints

- `GET /health`: backend service health
- `GET /api/health`: proxied Python health
- `GET /api/chunks?limit=6`: preview indexed chunks
- `POST /api/chat`: ask a grounded question
- `POST /api/search`: semantic retrieval
- `POST /api/reindex`: rebuild the vector index

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

Build and run all services:

```powershell
docker compose build
docker compose up
```

Stop everything:

```powershell
docker compose down
```

## What .NET Does Here

The `.NET` service is a background worker, not the main API.

Its role is to:

- watch `data/policies` for changes
- detect when policy files are added or updated
- call the Python RAG service `/reindex` endpoint
- keep the Chroma-backed knowledge base fresh

This lets the interview project show a realistic enterprise split between:

- UI and request/response services
- AI retrieval/generation logic
- background operational jobs

## Troubleshooting

### `react-scripts` not recognized

Run the frontend install first:

```powershell
cd frontend
npm.cmd install
```

### PowerShell blocks `npm`

If `npm` is blocked by execution policy, use:

```powershell
npm.cmd install
npm.cmd start
```

### Frontend shows `Unexpected token '<'`

That usually means the frontend hit HTML instead of API JSON.

Check that:

- backend is running on `:5000`
- Python is running on `:8000`

### `.NET SDKs were found` / `dotnet --version` fails

Install the SDK:

```powershell
winget install Microsoft.DotNet.SDK.8
```

### Reindex if answers are empty

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/reindex
```

