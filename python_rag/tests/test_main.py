from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient


PYTHON_RAG_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_RAG_DIR))

import main as rag_main
from memory_store import ConversationMemoryStore


rag_main.app.router.on_startup = []


class DummyObserver:
    def __init__(self) -> None:
        self.last_trace_metadata = None

    def start_trace(self, *args, **kwargs):
        self.last_trace_metadata = kwargs.get("metadata")
        return None

    def start_span(self, *args, **kwargs):
        return None

    def end_span(self, *args, **kwargs) -> None:
        return None

    def update_trace(self, *args, **kwargs) -> None:
        return None

    def flush(self) -> None:
        return None


class ChunkPipeline:
    def __init__(self) -> None:
        self.received_limit: int | None = None

    def list_chunks(self, limit: int = 20) -> list[dict]:
        self.received_limit = limit
        return [{"id": "chunk-1", "source": "policy.pdf", "chunk_id": "c-1", "text_preview": "Preview"}]


class SearchPipeline:
    def __init__(self) -> None:
        self.received_query: str | None = None
        self.received_top_k: int | None = None

    def retrieve(self, query: str, top_k: int = 3):
        self.received_query = query
        self.received_top_k = top_k
        return SimpleNamespace(chunks=[{"source": "policy.pdf", "chunk_id": "c-2", "text": "Expense policy"}])


class AskPipeline:
    def __init__(self) -> None:
        self.received_question: str | None = None
        self.received_history: list[dict] | None = None

    def answer(self, question: str, conversation_history: list[dict] | None = None) -> dict:
        self.received_question = question
        self.received_history = conversation_history
        return {
            "question": question,
            "answer": "Travel must be approved.",
            "sources": ["travel-policy.pdf"],
            "chunks": [{"source": "travel-policy.pdf", "chunk_id": "c-3", "text": "Manager approval required."}],
        }


class FailingAskPipeline(AskPipeline):
    def answer(self, question: str, conversation_history: list[dict] | None = None) -> dict:
        raise RuntimeError("simulated RAG failure")


class StreamPipeline:
    def __init__(self) -> None:
        self.received_question: str | None = None
        self.received_history: list[dict] | None = None

    def answer_stream(self, question: str, conversation_history: list[dict] | None = None):
        self.received_question = question
        self.received_history = conversation_history
        yield ("meta", {"sources": ["travel-policy.pdf"], "chunks": [{"chunk_id": "c-4"}]})
        yield ("token", "Streamed")
        yield ("token", " answer")


class CountPipeline:
    def __init__(self, count: int) -> None:
        self.count = count

    def collection_count(self) -> int:
        return self.count


class PythonRagApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(rag_main.app)
        self.original_observer = rag_main.observer
        self.observer = DummyObserver()
        rag_main.observer = self.observer
        rag_main.pipeline = None
        rag_main.memory_store = ConversationMemoryStore(max_turns=6)

    def tearDown(self) -> None:
        self.client.close()
        rag_main.observer = self.original_observer
        rag_main.pipeline = None
        rag_main.memory_store = ConversationMemoryStore(max_turns=6)

    def test_health_reports_starting_without_pipeline(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.headers.get("x-request-id"))
        self.assertGreater(len(response.headers.get("x-request-id", "")), 0)
        self.assertEqual(
            response.json(),
            {"status": "starting", "vector_db_loaded": False, "indexed_chunks": 0},
        )

    def test_echoes_valid_incoming_x_request_id(self) -> None:
        response = self.client.get("/health", headers={"X-Request-Id": "gw-trace-99"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("x-request-id"), "gw-trace-99")

    def test_ask_trace_metadata_uses_incoming_request_id(self) -> None:
        pipeline = AskPipeline()
        rag_main.pipeline = pipeline

        response = self.client.post(
            "/ask",
            json={"question": "What is policy?"},
            headers={"X-Request-Id": "trace-abc-123"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.observer.last_trace_metadata["request_id"], "trace-abc-123")

    def test_ask_failure_logs_request_id(self) -> None:
        rag_main.pipeline = FailingAskPipeline()

        with patch.object(rag_main.logger, "exception") as mock_exc:
            response = self.client.post(
                "/ask",
                json={"question": "What breaks?"},
                headers={"X-Request-Id": "err-trace-xyz"},
            )

        self.assertEqual(response.status_code, 500)
        mock_exc.assert_called()
        fmt, msg, rid = mock_exc.call_args[0]
        self.assertEqual(fmt, "%s request_id=%s")
        self.assertEqual(msg, "Ask failed")
        self.assertEqual(rid, "err-trace-xyz")

    def test_chunks_caps_limit_before_calling_pipeline(self) -> None:
        pipeline = ChunkPipeline()
        rag_main.pipeline = pipeline

        response = self.client.get("/chunks?limit=500")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(pipeline.received_limit, 100)
        self.assertEqual(
            response.json(),
            {"chunks": [{"id": "chunk-1", "source": "policy.pdf", "chunk_id": "c-1", "text_preview": "Preview"}]},
        )

    def test_search_trims_query_and_maps_results(self) -> None:
        pipeline = SearchPipeline()
        rag_main.pipeline = pipeline

        response = self.client.post("/search", json={"query": " expense reimbursement ", "topK": 2})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(pipeline.received_query, "expense reimbursement")
        self.assertEqual(pipeline.received_top_k, 2)
        self.assertEqual(
            response.json(),
            {"results": [{"source": "policy.pdf", "chunk_id": "c-2", "text": "Expense policy"}], "count": 1},
        )

    def test_ask_trims_inputs_and_persists_memory(self) -> None:
        pipeline = AskPipeline()
        rag_main.pipeline = pipeline

        response = self.client.post("/ask", json={"question": "  What is the travel policy?  ", "session_id": "  session-7  "})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(pipeline.received_question, "What is the travel policy?")
        self.assertEqual(pipeline.received_history, [])
        self.assertEqual(
            response.json(),
            {
                "question": "What is the travel policy?",
                "answer": "Travel must be approved.",
                "sources": ["travel-policy.pdf"],
                "chunks": [{"source": "travel-policy.pdf", "chunk_id": "c-3", "text": "Manager approval required."}],
                "session_id": "session-7",
            },
        )
        self.assertEqual(
            rag_main.memory_store.get_history("session-7"),
            [{"question": "What is the travel policy?", "answer": "Travel must be approved."}],
        )
        self.assertEqual(self.observer.last_trace_metadata["request_id"], response.headers["x-request-id"])

    def test_ask_stream_emits_events_and_persists_answer(self) -> None:
        pipeline = StreamPipeline()
        rag_main.pipeline = pipeline

        response = self.client.post("/ask/stream", json={"question": "  Follow up?  ", "session_id": "stream-1"})

        self.assertEqual(response.status_code, 200)
        body = response.text
        self.assertIn("event: start", body)
        self.assertIn('event: meta\ndata: {"sources": ["travel-policy.pdf"], "chunks": [{"chunk_id": "c-4"}]}', body)
        self.assertIn('event: token\ndata: "Streamed"', body)
        self.assertIn('event: token\ndata: " answer"', body)
        self.assertIn("event: done", body)
        self.assertEqual(pipeline.received_question, "Follow up?")
        self.assertEqual(pipeline.received_history, [])
        self.assertEqual(
            rag_main.memory_store.get_history("stream-1"),
            [{"question": "Follow up?", "answer": "Streamed answer"}],
        )

    def test_reindex_rebuilds_pipeline_and_returns_new_count(self) -> None:
        rebuilt_pipeline = CountPipeline(count=9)

        with patch.object(rag_main, "build_index", return_value={"indexed_files": 2}), patch.object(
            rag_main, "RAGPipeline", return_value=rebuilt_pipeline
        ):
            response = self.client.post("/reindex")

        self.assertEqual(response.status_code, 200)
        self.assertIs(rag_main.pipeline, rebuilt_pipeline)
        self.assertEqual(
            response.json(),
            {"status": "ok", "index_result": {"indexed_files": 2}, "indexed_chunks": 9},
        )


if __name__ == "__main__":
    unittest.main()
