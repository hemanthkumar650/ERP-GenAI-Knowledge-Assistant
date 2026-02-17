from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from hashlib import md5
from typing import Sequence

try:
    import chromadb
except ImportError:  # pragma: no cover - optional dependency at runtime
    chromadb = None

from openai import AzureOpenAI, OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.core.config import settings
from src.ingestion.doc_loader import DocumentChunk


@dataclass
class RAGResult:
    answer: str
    sources: list[str]
    scores: list[float]
    backend: str
    confidence: str
    citations: list[dict]
    retrieved_context: list[str]


class RAGAssistant:
    def __init__(self, chunks: Sequence[DocumentChunk]) -> None:
        self.chunks = list(chunks)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        corpus = [c.text for c in self.chunks] or [""]
        self.matrix = self.vectorizer.fit_transform(corpus)
        self.client = None
        self.chat_target = ""
        self.embedding_target = ""
        self.backend = "tfidf"
        self.chroma_client = None
        self.collection = None
        self._init_llm_clients()
        self._init_vector_db()
        self.index_documents()

    def _init_llm_clients(self) -> None:
        provider = settings.llm_provider
        if provider == "azure":
            if not (
                settings.azure_openai_api_key
                and settings.azure_openai_endpoint
                and settings.azure_openai_chat_deployment
                and settings.azure_openai_embedding_deployment
            ):
                return
            self.client = AzureOpenAI(
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
                azure_endpoint=settings.azure_openai_endpoint,
            )
            self.chat_target = settings.azure_openai_chat_deployment
            self.embedding_target = settings.azure_openai_embedding_deployment
            return

        if provider == "openai" and settings.openai_api_key:
            self.client = OpenAI(api_key=settings.openai_api_key)
            self.chat_target = settings.openai_model
            self.embedding_target = settings.openai_embedding_model

    def _init_vector_db(self) -> None:
        if self.client is None or chromadb is None:
            return
        self.chroma_client = chromadb.PersistentClient(path=settings.chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(name=settings.chroma_collection)
        self.backend = "chroma"

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.client is None or not self.embedding_target:
            return []
        response = self.client.embeddings.create(model=self.embedding_target, input=texts)
        return [item.embedding for item in response.data]

    def index_documents(self, force_reindex: bool = False) -> None:
        if self.collection is None or not self.chunks:
            return
        if force_reindex:
            if self.chroma_client is None:
                return
            self.chroma_client.delete_collection(name=settings.chroma_collection)
            self.collection = self.chroma_client.get_or_create_collection(name=settings.chroma_collection)

        count = self.collection.count()
        if count > 0 and not force_reindex:
            return

        docs = [chunk.text for chunk in self.chunks]
        embeddings = self._embed_texts(docs)
        if not embeddings:
            return
        ids = [f"{chunk.source}:{md5(chunk.text.encode('utf-8')).hexdigest()[:12]}" for chunk in self.chunks]
        metadatas = [
            {
                "source": chunk.source,
                "source_file": chunk.source_file,
                "section_id": chunk.section_id,
                "chunk_len": int(chunk.chunk_len),
            }
            for chunk in self.chunks
        ]
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metadatas)

    @staticmethod
    def _dedupe_hits(hits: list[dict]) -> list[dict]:
        deduped: list[dict] = []
        seen_signatures: set[str] = set()
        for hit in hits:
            text = hit["text"]
            signature = md5(" ".join(text.lower().split())[:180].encode("utf-8")).hexdigest()
            if signature in seen_signatures:
                continue
            is_near_duplicate = any(
                SequenceMatcher(None, text[:300], existing["text"][:300]).ratio() > 0.92
                for existing in deduped
            )
            if is_near_duplicate:
                continue
            seen_signatures.add(signature)
            deduped.append(hit)
        return deduped

    def _retrieve_with_chroma(self, question: str, top_k: int) -> list[dict]:
        query_embeddings = self._embed_texts([question])
        if not query_embeddings:
            return []
        query_result = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=settings.retrieve_candidates,
            include=["documents", "metadatas", "distances"],
        )
        docs = query_result.get("documents", [[]])[0]
        metadatas = query_result.get("metadatas", [[]])[0]
        distances = query_result.get("distances", [[]])[0]

        raw_hits: list[dict] = []
        for doc, metadata, distance in zip(docs, metadatas, distances):
            source = metadata.get("source", "unknown") if metadata else "unknown"
            source_file = metadata.get("source_file", source) if metadata else source
            section_id = metadata.get("section_id", "section-unknown") if metadata else "section-unknown"
            chunk_len = int(metadata.get("chunk_len", len(doc))) if metadata else len(doc)
            score = round(max(0.0, float(1 - distance)), 4)
            raw_hits.append(
                {
                    "source": source,
                    "source_file": source_file,
                    "section_id": section_id,
                    "text": doc,
                    "score": score,
                    "chunk_len": chunk_len,
                    "snippet": f"{doc[:200]}..." if len(doc) > 200 else doc,
                }
            )

        filtered = [
            hit
            for hit in raw_hits
            if hit["score"] >= settings.min_retrieval_score and hit["chunk_len"] >= 60
        ]
        deduped = self._dedupe_hits(filtered)
        return deduped[:top_k]

    def _retrieve_with_tfidf(self, question: str, top_k: int) -> list[dict]:
        if not self.chunks:
            return []
        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        ranked_idx = scores.argsort()[::-1][: settings.retrieve_candidates]
        raw_hits = [
            {
                "source": self.chunks[i].source,
                "source_file": self.chunks[i].source_file,
                "section_id": self.chunks[i].section_id,
                "text": self.chunks[i].text,
                "score": round(float(scores[i]), 4),
                "chunk_len": self.chunks[i].chunk_len,
                "snippet": f"{self.chunks[i].text[:200]}..." if len(self.chunks[i].text) > 200 else self.chunks[i].text,
            }
            for i in ranked_idx
            if float(scores[i]) >= settings.min_retrieval_score and self.chunks[i].chunk_len >= 60
        ]
        return self._dedupe_hits(raw_hits)[:top_k]

    def retrieve(self, question: str, top_k: int = 4) -> list[dict]:
        if self.collection is not None and self.client is not None:
            return self._retrieve_with_chroma(question, top_k=top_k)
        return self._retrieve_with_tfidf(question, top_k=top_k)

    @staticmethod
    def _confidence_from_score(top_score: float) -> str:
        if top_score >= 0.75:
            return "high"
        if top_score >= 0.55:
            return "medium"
        return "low"

    @staticmethod
    def _fallback_answer(question: str, hits: list[dict]) -> str:
        best = hits[0]["text"] if hits else ""
        short = best[:550].strip()
        return (
            "Answer draft from retrieved ERP docs (LLM not configured):\n"
            f"- Question: {question}\n"
            f"- Most relevant context: {short}"
        )

    def answer(self, question: str, top_k: int = 4) -> RAGResult:
        hits = self.retrieve(question, top_k=top_k)
        if not hits:
            return RAGResult(
                answer=(
                    "Insufficient context in indexed ERP documents to answer this accurately.\n"
                    "Add more relevant documents or reindex after cleaning."
                ),
                sources=[],
                scores=[],
                backend=self.backend,
                confidence="low",
                citations=[],
                retrieved_context=[],
            )

        sources = [h["source"] for h in hits]
        scores = [h["score"] for h in hits]
        top_score = max(scores)
        confidence = self._confidence_from_score(top_score)
        context = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)

        if top_score < settings.min_retrieval_score:
            return RAGResult(
                answer=(
                    "Not in docs: retrieved context is too weak to answer reliably.\n"
                    "Try a more specific question or expand the ERP knowledge base."
                ),
                sources=sources,
                scores=scores,
                backend=self.backend,
                confidence="low",
                citations=hits,
                retrieved_context=[h["text"] for h in hits],
            )

        if self.client and self.chat_target:
            prompt = (
                "You are an ERP Knowledge Assistant.\n"
                "Use only the provided context.\n"
                "Return concise output with this structure:\n"
                "1) Answer (2-4 short sentences)\n"
                "2) Action Steps (bullets only if the question is procedural)\n"
                "3) Missing Info (write 'Not in docs' if information is unavailable)\n"
                "4) Citations (source ids in square brackets)\n\n"
                f"Question: {question}\n\n"
                f"Context:\n{context}"
            )
            completion = self.client.chat.completions.create(
                model=self.chat_target,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            answer = completion.choices[0].message.content or "No answer generated."
        else:
            answer = self._fallback_answer(question, hits)

        return RAGResult(
            answer=answer,
            sources=sources,
            scores=scores,
            backend=self.backend,
            confidence=confidence,
            citations=hits,
            retrieved_context=[h["text"] for h in hits],
        )
