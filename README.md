# ERP-GenAI-Knowledge-Assistant

A multi-service ERP policy assistant built with React, Node.js, FastAPI, Chroma, Azure OpenAI, and a .NET background worker.

This repository is structured as a small full-stack RAG system:

- `frontend/`: React + TypeScript client
- `backend/`: Express + TypeScript API layer
- `python_rag/`: FastAPI retrieval and generation service
- `dotnet_worker/`: .NET 8 worker for background policy change monitoring

The main idea is straightforward:

1. A user asks a question in the React UI.
2. The Node/Express API accepts the request.
3. The Python service retrieves relevant policy chunks from Chroma.
4. Azure OpenAI generates a grounded answer from the retrieved context.
5. The .NET worker can watch for policy file changes and trigger reindexing.

## Why this project exists

This project is meant to show how a RAG application can be split into clear service boundaries instead of placing everything in a single script:

- React handles the UI
- Node handles the API surface and request orchestration
- Python handles document retrieval and LLM integration
- .NET handles background operational work

The goal is not to claim a perfect production deployment. The goal is to show architecture, service boundaries, and a working retrieval flow in a way that is easy to understand and extend.

## Current implementation status

What is implemented in this repo today:

- React frontend for asking questions and viewing sources/chunks
- Express API layer that proxies chat/search/reindex requests to Python
- Automated regression coverage for the frontend dashboard and backend API layer
- FastAPI RAG service with:
  - Azure OpenAI chat and embedding calls
  - Chroma persistence
  - PDF loading and chunking
  - retrieval
  - answer generation
  - reindex endpoint
- .NET worker scaffold that polls the policy directory and triggers Python reindexing
- Dockerfiles for each service and a `docker-compose.yml`

What to know when reading the code:

- The Python service is the real retrieval core.
- The Node service is intentionally thin.
- The .NET service is a background worker, not the primary API.
- There are also older root-level Python files from an earlier single-service version. The multi-service layout under `frontend/`, `backend/`, `python_rag/`, and `dotnet_worker/` is the current direction.

## Repository layout

```text
ERP-GenAI-Knowledge-Assistant/
|-- frontend/
|   |-- public/
|   |-- src/
|   |-- package.json
|   `-- tsconfig.json
|-- backend/
|   |-- src/
|   |   |-- config/
|   |   |-- routes/
|   |   |-- services/
|   |   `-- types/
|   |-- package.json
|   `-- tsconfig.json
|-- python_rag/
|   |-- utils/
|   |-- main.py
|   |-- config.py
|   |-- rag_pipeline.py
|   |-- embed_index.py
|   `-- requirements.txt
|-- dotnet_worker/
|   |-- Services/
|   |-- Workers/
|   |-- Program.cs
|   `-- Worker.csproj
|-- data/
|   |-- policies/
|   `-- chroma_db/
|-- docker-compose.yml
|-- .env.example
`-- README.md
```

## Architecture

```text
Browser
  ->
React frontend (:3000)
  ->
Express API (:5000)
  ->
FastAPI RAG service (:8000)
  ->
Azure OpenAI + Chroma

Policy document changes
  ->
.NET worker
  ->
Python /reindex
```

## Service responsibilities

### Frontend

Location: `frontend/`

Responsibilities:

- render the chat UI
- call backend `/api/*` endpoints
- display answers, cited sources, and retrieved evidence
- show basic system status

### Backend

Location: `backend/`

Responsibilities:

- expose a stable API to the frontend
- forward chat/search/reindex requests to the Python service
- keep the browser isolated from direct Python service details

Main routes:

- `GET /health`
- `GET /api/health`
- `GET /api/chunks`
- `POST /api/chat`
- `POST /api/search`
- `POST /api/reindex`

### Python RAG service

Location: `python_rag/`

Responsibilities:

- load and chunk PDF policy documents
- create embeddings with Azure OpenAI
- store and query vectors in Chroma
- retrieve relevant chunks for a question
- generate grounded answers
- rebuild the index when documents change

Main routes:

- `GET /health`
- `GET /chunks`
- `POST /ask`
- `POST /ask/stream`
- `POST /search`
- `POST /reindex`

### .NET worker

Location: `dotnet_worker/`

Responsibilities:

- monitor the policy directory
- detect file set changes using a directory snapshot
- call the Python service reindex endpoint

This is included to demonstrate a common enterprise pattern: keeping background operational tasks outside the main request/response path.

## Tech stack

- Frontend: React 18, TypeScript, react-scripts
- Backend: Node.js, Express, TypeScript, Axios, Helmet, CORS
- RAG: Python 3.11+, FastAPI, Chroma, Azure OpenAI SDK, pypdf, tiktoken
- Worker: .NET 8 Worker Service, C#
- Observability: optional Langfuse integration
- Containers: Docker, Docker Compose

## Environment variables

Copy the template first:

```powershell
Copy-Item .env.example .env
```

Typical values:

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

## Local development

### Prerequisites

- Node.js 20+
- Python 3.11+
- .NET 8 SDK
- Azure OpenAI credentials

Optional:

- Docker Desktop

### Install dependencies

From the repo root:

```powershell
cd C:\Users\heman\Desktop\Project
```

Python:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r python_rag\requirements.txt
```

Backend:

```powershell
cd backend
npm.cmd install
cd ..
```

Frontend:

```powershell
cd frontend
npm.cmd install
cd ..
```

.NET worker:

```powershell
cd dotnet_worker
dotnet restore
cd ..
```

## Running the stack locally

Start each service in its own terminal.

### 1. Python RAG service

```powershell
cd C:\Users\heman\Desktop\Project
.\.venv\Scripts\Activate.ps1
cd python_rag
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Backend API

```powershell
cd C:\Users\heman\Desktop\Project\backend
npm.cmd run dev
```

### 3. Frontend

```powershell
cd C:\Users\heman\Desktop\Project\frontend
npm.cmd start
```

### 4. .NET worker

```powershell
cd C:\Users\heman\Desktop\Project\dotnet_worker
dotnet run
```

## Automated checks

### Backend

Run the backend regression suite:

```powershell
cd C:\Users\heman\Desktop\Project\backend
npm.cmd test
```

Build the backend:

```powershell
cd C:\Users\heman\Desktop\Project\backend
npm.cmd run build
```

### Frontend

Run the frontend regression suite:

```powershell
cd C:\Users\heman\Desktop\Project\frontend
npm.cmd test
```

Type-check the frontend:

```powershell
cd C:\Users\heman\Desktop\Project\frontend
node .\node_modules\typescript\bin\tsc --noEmit
```

Notes:

- The frontend test script runs Jest in-band so it behaves well in constrained environments.
- Automated tests for the Python RAG service and the .NET worker are still future work.

## Local URLs

- Frontend: `http://localhost:3000`
- Backend health: `http://localhost:5000/health`
- Python health: `http://localhost:8000/health`

## Docker

Build and run:

```powershell
docker compose build
docker compose up
```

Stop:

```powershell
docker compose down
```

## Example requests

### Chat

```json
{
  "message": "What is the expense reimbursement policy?",
  "conversationId": "session-1"
}
```

### Search

```json
{
  "query": "expense reimbursement",
  "topK": 3
}
```

## Troubleshooting

### `react-scripts` is not recognized

Install frontend dependencies first:

```powershell
cd frontend
npm.cmd install
```

### PowerShell blocks `npm`

Use `npm.cmd` instead of `npm`:

```powershell
npm.cmd install
npm.cmd start
```

### Frontend returns `Unexpected token '<'`

That usually means the frontend received HTML instead of API JSON.

Check that:

- backend is running on port `5000`
- Python is running on port `8000`

### `.NET SDKs were found` / `dotnet --version` fails

Install the SDK:

```powershell
winget install Microsoft.DotNet.SDK.8
```

### Answers are empty or stale

Trigger a reindex:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/reindex
```

## Notes

- `.env` should not be committed
- `node_modules`, `bin`, and `obj` should not be committed
- `python_rag/` currently contains the retrieval core
- the backend is intentionally simple and mostly acts as a proxy/orchestration layer
- the frontend and backend now include small automated regression suites
- the .NET worker is meant to show background job separation, not replace the Python retrieval service

## Future improvements

- expand automated tests to the Python RAG service and .NET worker
- add authentication
- improve typed shared contracts between frontend and backend
- add stronger document ingestion and metadata handling
- replace polling in the worker with a more robust file-watching or event-based mechanism
- add CI for linting/build validation

This project is public as a code sample and learning artifact. It is intentionally transparent about service roles and current implementation state.
