from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb

from config import generate_chat, generate_chat_stream, get_embedding, settings


STRICT_RAG_PROMPT = """You are an ERP Knowledge Assistant.
Answer ONLY from the provided context.
If the answer is not present, say 'I don't know'.
Do not invent information.

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

    def answer(self, question: str, top_k: int | None = None) -> dict:
        retrieved = self.retrieve(question, top_k=top_k)
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
        prompt = BALANCED_RAG_PROMPT.format(context=context, question=question)
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

        return {
            "question": question,
            "answer": answer_text,
            "sources": retrieved.sources,
            "chunks": retrieved.chunks,
        }

    def answer_stream(self, question: str, top_k: int | None = None):
        """Retrieve context, then stream chat tokens. Yields (sources, chunks) once, then content deltas."""
        retrieved = self.retrieve(question, top_k=top_k)
        if not retrieved.context:
            yield ("meta", {"sources": [], "chunks": []})
            yield ("token", "I don't know.")
            return
        yield ("meta", {"sources": retrieved.sources, "chunks": retrieved.chunks})
        context = retrieved.context[: settings.context_max_chars]
        if len(retrieved.context) > settings.context_max_chars:
            context = context.rsplit("\n\n", 1)[0] + "\n\n[...truncated]"
        prompt = STRICT_RAG_PROMPT.format(context=context, question=question)
        for delta in generate_chat_stream(prompt=prompt):
            yield ("token", delta)
