from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import chromadb

from config import generate_chat, generate_chat_stream, get_embedding, settings


STRICT_RAG_PROMPT = """You are an ERP Knowledge Assistant.
Answer ONLY from the provided context.
If the answer is not present, say 'I don't know'.
Do not invent information.

Conversation History:
{history}

Context:
{context}

Question:
{question}

Answer:"""

BALANCED_RAG_PROMPT = """You are an ERP Knowledge Assistant.
Use ONLY the provided context.
If relevant policy statements exist, provide a concise direct answer and cite source tags in brackets.
If evidence is missing, say 'I don't know'.
Do not invent information.

Conversation History:
{history}

Context:
{context}

Question:
{question}

Answer:"""


@dataclass
class RetrievalResult:
    context: str
    sources: list[str]
    chunks: list[dict]


class RAGPipeline:
    def __init__(self, persist_dir: str | None = None, collection_name: str | None = None) -> None:
        self.persist_dir = persist_dir or settings.chroma_path
        self.collection_name = collection_name or settings.chroma_collection
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def collection_count(self) -> int:
        return self.collection.count()

    def list_chunks(self, limit: int = 20) -> list[dict]:
        payload = self.collection.get(
            limit=limit,
            include=["documents", "metadatas"],
        )
        ids = payload.get("ids", [])
        docs = payload.get("documents", [])
        metadatas = payload.get("metadatas", [])
        rows: list[dict] = []
        for chunk_id, doc, metadata in zip(ids, docs, metadatas):
            meta = metadata or {}
            rows.append(
                {
                    "id": chunk_id,
                    "source": meta.get("source", "unknown.pdf"),
                    "chunk_id": meta.get("chunk_id", "unknown"),
                    "text_preview": (doc or "")[:220],
                }
            )
        return rows

    def retrieve(self, question: str, top_k: int | None = None) -> RetrievalResult:
        top_k = top_k or settings.top_k
        query_vector = get_embedding(question)
        result = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        chunks: list[dict] = []
        sources: list[str] = []
        context_parts: list[str] = []

        for doc, metadata, distance in zip(docs, metadatas, distances):
            source = (metadata or {}).get("source", "unknown.pdf")
            chunk_id = (metadata or {}).get("chunk_id", "unknown")
            # Convert distance to stable [0,1] score across metrics.
            similarity = round(1.0 / (1.0 + max(0.0, float(distance))), 4)
            sources.append(source)
            chunks.append(
                {
                    "source": source,
                    "chunk_id": chunk_id,
                    "similarity": similarity,
                    "text": doc,
                }
            )
            context_parts.append(f"[{source}::{chunk_id}] {doc}")

        unique_sources = sorted(set(sources))
        context = "\n\n".join(context_parts).strip()
        return RetrievalResult(context=context, sources=unique_sources, chunks=chunks)

    @staticmethod
    def _format_history(conversation_history: list[dict] | None) -> str:
        if not conversation_history:
            return "None"
        lines: list[str] = []
        for i, turn in enumerate(conversation_history[-4:], start=1):
            q = (turn.get("question") or "").strip()
            a = (turn.get("answer") or "").strip()
            if q:
                lines.append(f"Turn {i} User: {q}")
            if a:
                lines.append(f"Turn {i} Assistant: {a}")
        return "\n".join(lines) if lines else "None"

    @staticmethod
    def _build_retrieval_query(question: str, conversation_history: list[dict] | None) -> str:
        """Build a slightly enriched query for follow-up questions."""
        if not conversation_history:
            return question
        lines: list[str] = []
        for turn in conversation_history[-2:]:
            prev_q = (turn.get("question") or "").strip()
            prev_a = (turn.get("answer") or "").strip()
            if prev_q:
                lines.append(f"Previous question: {prev_q}")
            if prev_a:
                lines.append(f"Previous answer: {prev_a[:280]}")
        lines.append(f"Follow-up question: {question}")
        query = "\n".join(lines).strip()
        return query[:1400] if len(query) > 1400 else query

    @staticmethod
    def _extract_cited_sources(answer_text: str) -> set[str]:
        sources: set[str] = set()
        for block in re.findall(r"\[([^\[\]]+)\]", answer_text or ""):
            for part in block.split(","):
                token = part.strip()
                if not token:
                    continue
                source = token.split("::", 1)[0].strip()
                if source.lower().endswith(".pdf"):
                    sources.add(source.lower())
        return sources

    def _is_grounded_to_retrieved(self, answer_text: str, retrieved_sources: list[str]) -> bool:
        cited = self._extract_cited_sources(answer_text)
        if not cited:
            return True
        allowed = {src.lower() for src in retrieved_sources}
        return cited.issubset(allowed)

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        conversation_history: list[dict] | None = None,
    ) -> dict:
        retrieval_query = self._build_retrieval_query(question, conversation_history)
        retrieved = self.retrieve(retrieval_query, top_k=top_k)
        if not retrieved.context:
            return {
                "question": question,
                "answer": "I don't know",
                "sources": [],
                "chunks": [],
            }

        context = retrieved.context[: settings.context_max_chars]
        if len(retrieved.context) > settings.context_max_chars:
            context = context.rsplit("\n\n", 1)[0] + "\n\n[...truncated]"
        history = self._format_history(conversation_history)
        prompt = BALANCED_RAG_PROMPT.format(context=context, history=history, question=question)
        answer_text = generate_chat(prompt=prompt)
        if not answer_text:
            answer_text = "I don't know."

        normalized = answer_text.strip().lower()
        if normalized in {"i don't know", "i don't know.", "i dont know", "i dont know."} and retrieved.chunks:
            top_chunks = retrieved.chunks[:2]
            extract = " ".join(chunk["text"] for chunk in top_chunks)
            extract = " ".join(extract.split())
            extract = extract[:520].strip()
            answer_text = f"{extract} [{top_chunks[0]['source']}]"

        # Guardrail: if cited sources are not part of retrieved evidence, force grounded fallback.
        if not self._is_grounded_to_retrieved(answer_text, retrieved.sources):
            if retrieved.chunks:
                top = retrieved.chunks[0]
                extract = " ".join((top.get("text") or "").split())[:520].strip()
                answer_text = f"{extract} [{top['source']}::{top['chunk_id']}]"
            else:
                answer_text = "I don't know."

        return {
            "question": question,
            "answer": answer_text,
            "sources": retrieved.sources,
            "chunks": retrieved.chunks,
        }

    def answer_stream(
        self,
        question: str,
        top_k: int | None = None,
        conversation_history: list[dict] | None = None,
    ):
        """Retrieve context, then stream chat tokens. Yields (sources, chunks) once, then content deltas."""
        retrieval_query = self._build_retrieval_query(question, conversation_history)
        retrieved = self.retrieve(retrieval_query, top_k=top_k)
        if not retrieved.context:
            yield ("meta", {"sources": [], "chunks": []})
            yield ("token", "I don't know.")
            return
        yield ("meta", {"sources": retrieved.sources, "chunks": retrieved.chunks})
        context = retrieved.context[: settings.context_max_chars]
        if len(retrieved.context) > settings.context_max_chars:
            context = context.rsplit("\n\n", 1)[0] + "\n\n[...truncated]"
        history = self._format_history(conversation_history)
        prompt = STRICT_RAG_PROMPT.format(context=context, history=history, question=question)
        for delta in generate_chat_stream(prompt=prompt):
            yield ("token", delta)
