from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel, Field

from analytics_store import AnalyticsStore
from embed_index import build_index
from rag_pipeline import RAGPipeline


logger = logging.getLogger("erp_rag_assistant")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="ERP GenAI Knowledge Assistant")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

pipeline: RAGPipeline | None = None
analytics_store: AnalyticsStore | None = None
_startup_executor = ThreadPoolExecutor(max_workers=1)


def _init_pipeline() -> None:
    global pipeline, analytics_store
    try:
        pipeline = RAGPipeline()
        analytics_store = AnalyticsStore()
        logger.info("RAG pipeline initialized. Collection count: %s", pipeline.collection_count())
    except Exception as e:
        logger.exception("Startup init failed: %s", e)


@app.on_event("startup")
def startup_event() -> None:
    """Start server immediately; init pipeline in background so app doesn't block."""
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_startup_executor, _init_pipeline)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Employee question about ERP policies")


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    chunks: list[dict]


@app.get("/")
def root() -> dict:
    return {"message": "Welcome to ERP GenAI Knowledge Assistant"}


@app.get("/ui", response_class=HTMLResponse)
def ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/ui/analytics", response_class=HTMLResponse)
def ui_analytics(request: Request):
    return templates.TemplateResponse("analytics.html", {"request": request})


@app.get("/health")
def health() -> dict:
    if pipeline is None:
        return {
            "status": "starting",
            "vector_db_loaded": False,
            "indexed_chunks": 0,
        }
    try:
        count = pipeline.collection_count()
        return {
            "status": "ok",
            "vector_db_loaded": True,
            "indexed_chunks": count,
        }
    except Exception as exc:  # pragma: no cover
        return {"status": "error", "detail": str(exc)}


@app.get("/chunks")
def chunks(limit: int = 20) -> dict:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    capped_limit = max(1, min(limit, 100))
    return {"chunks": pipeline.list_chunks(limit=capped_limit)}


@app.post("/reindex")
def reindex() -> dict:
    global pipeline
    try:
        result = build_index()
        pipeline = RAGPipeline()
        return {
            "status": "ok",
            "index_result": result,
            "indexed_chunks": pipeline.collection_count(),
        }
    except Exception as exc:
        logger.exception("Reindex failed")
        raise HTTPException(status_code=500, detail=f"Reindex failed: {exc}") from exc


def _ask_stream_gen(question: str):
    """Generator for SSE: yields 'event: start' first, then meta, token events, done."""
    if pipeline is None:
        yield "event: error\ndata: " + json.dumps({"detail": "RAG pipeline not initialized"}) + "\n\n"
        return
    question = question.strip()
    if not question:
        yield "event: error\ndata: " + json.dumps({"detail": "Question cannot be empty"}) + "\n\n"
        return
    yield "event: start\ndata: {}\n\n"
    sources, top_similarity = [], None
    try:
        for kind, payload in pipeline.answer_stream(question):
            if kind == "meta":
                sources = payload.get("sources", [])
                chunks = payload.get("chunks", [])
                if chunks:
                    top_similarity = max(float(c.get("similarity", 0)) for c in chunks)
                yield "event: meta\ndata: " + json.dumps(payload) + "\n\n"
            elif kind == "token":
                yield "event: token\ndata: " + json.dumps(payload) + "\n\n"
        yield "event: done\ndata: {}\n\n"
        if analytics_store is not None:
            analytics_store.log_query(question=question, sources=sources, top_similarity=top_similarity)
    except Exception as exc:
        logger.exception("Stream ask failed")
        yield "event: error\ndata: " + json.dumps({"detail": str(exc)}) + "\n\n"


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """Stream answer as Server-Sent Events: meta (sources + chunks), then token events, then done."""
    return StreamingResponse(
        _ask_stream_gen(req.question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = pipeline.answer(question)
        if analytics_store is not None:
            top_similarity = None
            if result["chunks"]:
                top_similarity = max(float(chunk.get("similarity", 0)) for chunk in result["chunks"])
            analytics_store.log_query(
                question=question,
                sources=result["sources"],
                top_similarity=top_similarity,
            )
        return AskResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
            chunks=result["chunks"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to answer question")
        raise HTTPException(status_code=500, detail=f"Failed to process question: {exc}") from exc


@app.get("/analytics/data")
def analytics_data() -> dict:
    if analytics_store is None:
        raise HTTPException(status_code=503, detail="Analytics store not initialized")
    logs = analytics_store.fetch_logs()

    by_day_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    similarities: list[float] = []
    for row in logs:
        created_at = row.get("created_at")
        try:
            day = datetime.fromisoformat(created_at).date().isoformat()
            by_day_counter[day] += 1
        except Exception:
            pass
        for src in row.get("sources", []):
            source_counter[src] += 1
        sim = row.get("top_similarity")
        if isinstance(sim, (float, int)):
            similarities.append(float(sim))

    by_day = [{"date": k, "count": v} for k, v in sorted(by_day_counter.items())]
    top_sources = [{"source": k, "count": v} for k, v in source_counter.most_common(10)]
    return {
        "total_queries": len(logs),
        "by_day": by_day,
        "top_sources": top_sources,
        "similarities": similarities,
    }
