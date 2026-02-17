# ERP RAG Assistant

Production-style ERP Knowledge Assistant built with Azure OpenAI, ChromaDB, and FastAPI.

## Features
- PDF policy ingestion from `data/policies`
- Text cleaning and token-aware chunking (500 tokens, 50 overlap)
- Azure OpenAI embeddings (`text-embedding-3-small`)
- Persistent Chroma vector database (`data/chroma_db`)
- Grounded RAG answers using strict prompt constraints
- FastAPI endpoints for Q&A and health checks

## Project Structure
```text
erp-rag-assistant/
├── main.py
├── rag_pipeline.py
├── embed_index.py
├── config.py
├── requirements.txt
├── .env.example
├── README.md
├── data/
│   ├── policies/
│   └── chroma_db/
└── utils/
    ├── loader.py
    └── chunker.py
```

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy environment template:
   ```bash
   cp .env.example .env
   ```
3. Fill `.env` with Azure OpenAI credentials.
4. Add ERP policy PDFs into `data/policies`.

## Build Vector Index
```bash
python embed_index.py
```

## Run API
```bash
uvicorn main:app --reload
```

Open docs: `http://localhost:8000/docs`
Open web UI: `http://localhost:8000/ui`
Open analytics UI: `http://localhost:8000/ui/analytics`

## API Endpoints
- `GET /` -> welcome message
- `GET /health` -> service and vector DB status
- `GET /chunks?limit=20` -> chunk preview from Chroma
- `GET /analytics/data` -> query usage + source + similarity analytics payload
- `POST /reindex` -> rebuild index from PDFs and reload pipeline
- `POST /ask`
  - Request:
    ```json
    { "question": "Why was my expense rejected?" }
    ```
  - Response:
    ```json
    {
      "question": "Why was my expense rejected?",
      "answer": "...",
      "sources": ["expense_policy.pdf"]
    }
    ```

## Notes
- Answers are grounded and instructed to return `"I don't know"` if context is missing.
- Run `python embed_index.py` whenever policy PDFs change.
- The custom web UI at `/ui` supports:
  - health check
  - reindex trigger
  - grounded ask flow
  - source chips and retrieved evidence preview
- Analytics dashboard at `/ui/analytics` uses Plotly charts for:
  - daily query volume
  - top retrieved policy files
  - similarity score distribution
