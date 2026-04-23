from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.requests import Request

from embed_index import build_index
from memory_store import ConversationMemoryStore
from observability import observer
from rag_pipeline import RAGPipeline
from request_id import assign_request_id


logger = logging.getLogger("python_rag")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="ERP-GenAI-Knowledge-Assistant Python Service")


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = assign_request_id(request.headers.get("x-request-id"))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response

pipeline: RAGPipeline | None = None
memory_store = ConversationMemoryStore(max_turns=6)
startup_executor = ThreadPoolExecutor(max_workers=1)


def _init_pipeline() -> None:
    global pipeline
    try:
        pipeline = RAGPipeline()
        logger.info("Python RAG service initialized. Collection count: %s", pipeline.collection_count())
    except Exception as exc:
        logger.exception("Python RAG startup failed: %s", exc)


@app.on_event("startup")
def startup_event() -> None:
    loop = asyncio.get_event_loop()
    loop.run_in_executor(startup_executor, _init_pipeline)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str | None = Field(default=None, min_length=1)


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    chunks: list[dict]
    session_id: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10, alias="topK")

    model_config = {"populate_by_name": True}


class SearchResponse(BaseModel):
    results: list[dict]
    count: int


def _require_pipeline() -> RAGPipeline:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    return pipeline


@app.get("/")
def root() -> dict:
    return {"message": "Python RAG service is running"}


@app.get("/health")
def health() -> dict:
    if pipeline is None:
        return {"status": "starting", "vector_db_loaded": False, "indexed_chunks": 0}
    try:
        count = pipeline.collection_count()
        return {"status": "ok", "vector_db_loaded": True, "indexed_chunks": count}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


@app.get("/chunks")
def chunks(limit: int = 20) -> dict:
    rag = _require_pipeline()
    capped_limit = max(1, min(limit, 100))
    return {"chunks": rag.list_chunks(limit=capped_limit)}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    rag = _require_pipeline()
    retrieved = rag.retrieve(req.query.strip(), top_k=req.top_k)
    return SearchResponse(results=retrieved.chunks, count=len(retrieved.chunks))


@app.post("/reindex")
def reindex() -> dict:
    global pipeline
    trace = observer.start_trace(name="reindex", metadata={"endpoint": "/reindex"})
    span = observer.start_span(trace, name="build_index")
    try:
        result = build_index()
        observer.end_span(span, output_data=result)
        pipeline = RAGPipeline()
        response = {
            "status": "ok",
            "index_result": result,
            "indexed_chunks": pipeline.collection_count(),
        }
        observer.update_trace(trace, output_data=response)
        return response
    except Exception as exc:
        observer.end_span(span, output_data={"status": "error", "detail": str(exc)})
        observer.update_trace(trace, output_data={"status": "error"}, metadata={"error": str(exc)})
        logger.exception("Reindex failed")
        raise HTTPException(status_code=500, detail=f"Reindex failed: {exc}") from exc
    finally:
        observer.flush()


def _ask_stream_gen(question: str, session_id: str | None = None):
    if pipeline is None:
        yield "event: error\ndata: " + json.dumps({"detail": "RAG pipeline not initialized"}) + "\n\n"
        return
    question = question.strip()
    if not question:
        yield "event: error\ndata: " + json.dumps({"detail": "Question cannot be empty"}) + "\n\n"
        return
    history = memory_store.get_history(session_id) if session_id else []
    yield "event: start\ndata: {}\n\n"
    trace = observer.start_trace(
        name="ask_stream",
        input_data={"question": question},
        metadata={"endpoint": "/ask/stream", "session_id": session_id},
    )
    retrieval_span = observer.start_span(trace, name="retrieval", input_data={"question": question})
    answer_tokens: list[str] = []
    try:
        for kind, payload in pipeline.answer_stream(question, conversation_history=history):
            if kind == "meta":
                observer.end_span(
                    retrieval_span,
                    output_data={
                        "source_count": len(payload.get("sources", [])),
                        "chunk_count": len(payload.get("chunks", [])),
                    },
                )
                yield "event: meta\ndata: " + json.dumps(payload) + "\n\n"
            elif kind == "token":
                answer_tokens.append(payload)
                yield "event: token\ndata: " + json.dumps(payload) + "\n\n"
        yield "event: done\ndata: {}\n\n"
        answer_text = "".join(answer_tokens).strip()
        if session_id and answer_text:
            memory_store.append_turn(session_id=session_id, question=question, answer=answer_text)
        observer.update_trace(trace, output_data={"answer": answer_text})
    except Exception as exc:
        observer.end_span(retrieval_span, output_data={"status": "error"}, metadata={"error": str(exc)})
        observer.update_trace(trace, output_data={"status": "error"}, metadata={"error": str(exc)})
        logger.exception("Stream ask failed")
        yield "event: error\ndata: " + json.dumps({"detail": str(exc)}) + "\n\n"
    finally:
        observer.flush()


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    return StreamingResponse(
        _ask_stream_gen(req.question, req.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    rag = _require_pipeline()
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    session_id = req.session_id.strip() if req.session_id else None
    history = memory_store.get_history(session_id) if session_id else []

    trace = observer.start_trace(
        name="ask",
        input_data={"question": question},
        metadata={"endpoint": "/ask", "session_id": session_id},
    )
    rag_span = observer.start_span(trace, name="rag_answer", input_data={"question": question})
    try:
        result = rag.answer(question, conversation_history=history)
        observer.end_span(
            rag_span,
            output_data={"source_count": len(result["sources"]), "chunk_count": len(result["chunks"])},
        )
        observer.update_trace(trace, output_data={"answer": result["answer"], "sources": result["sources"]})
        if session_id and result["answer"]:
            memory_store.append_turn(session_id=session_id, question=question, answer=result["answer"])
        return AskResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
            chunks=result["chunks"],
            session_id=session_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        observer.end_span(rag_span, output_data={"status": "error"}, metadata={"error": str(exc)})
        observer.update_trace(trace, output_data={"status": "error"}, metadata={"error": str(exc)})
        logger.exception("Ask failed")
        raise HTTPException(status_code=500, detail=f"Failed to process question: {exc}") from exc
    finally:
        observer.flush()


@app.delete("/memory/{session_id}")
def clear_memory(session_id: str) -> dict:
    cleared = memory_store.clear_session(session_id.strip())
    return {"session_id": session_id, "cleared": cleared}
