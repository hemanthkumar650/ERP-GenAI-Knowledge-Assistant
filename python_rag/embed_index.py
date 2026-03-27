from __future__ import annotations

from hashlib import sha1
import time

import chromadb
from openai import RateLimitError

from config import get_embeddings_batch, settings
from utils.chunker import chunk_documents
from utils.loader import load_policy_documents


def _stable_chunk_uid(chunk: dict) -> str:
    payload = f"{chunk['source']}|{chunk['chunk_id']}|{chunk['text']}".encode("utf-8")
    return sha1(payload).hexdigest()


METADATA_KEYS = ("policy_type", "effective_date", "department", "version")


def _chunk_metadata(chunk: dict) -> dict:
    meta = {
        "source": chunk["source"],
        "chunk_id": chunk["chunk_id"],
    }
    for key in METADATA_KEYS:
        value = str(chunk.get(key, "unknown")).strip()
        meta[key] = value if value else "unknown"
    return meta


def _embed_with_retry(texts: list[str], max_retries: int = 6) -> list[list[float]]:
    delay_seconds = 10
    attempt = 0
    while True:
        try:
            return get_embeddings_batch(texts)
        except RateLimitError:
            attempt += 1
            if attempt > max_retries:
                raise
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 2, 120)


def build_index() -> dict:
    docs = load_policy_documents(settings.policies_path)
    if not docs:
        return {"status": "no_documents", "indexed": 0, "skipped": 0}

    chunks = chunk_documents(docs, chunk_size_tokens=500, overlap_tokens=50)
    if not chunks:
        return {"status": "no_chunks", "indexed": 0, "skipped": 0}

    client = chromadb.PersistentClient(path=settings.chroma_path)
    desired_metadata = {"hnsw:space": "cosine"}
    collection = client.get_or_create_collection(name=settings.chroma_collection, metadata=desired_metadata)

    if (collection.metadata or {}).get("hnsw:space") != "cosine":
        client.delete_collection(name=settings.chroma_collection)
        collection = client.get_or_create_collection(name=settings.chroma_collection, metadata=desired_metadata)

    # One-time metadata schema migration: if existing chunks are missing new policy fields,
    # rebuild collection so all records carry consistent metadata.
    sample = collection.get(limit=1, include=["metadatas"])
    sample_meta = (sample.get("metadatas") or [None])[0] or {}
    if sample_meta and any(key not in sample_meta for key in METADATA_KEYS):
        client.delete_collection(name=settings.chroma_collection)
        collection = client.get_or_create_collection(name=settings.chroma_collection, metadata=desired_metadata)

    existing = collection.get(include=[])
    existing_ids = set(existing.get("ids", []))

    to_index = []
    for chunk in chunks:
        uid = _stable_chunk_uid(chunk)
        if uid in existing_ids:
            continue
        to_index.append((uid, chunk))

    if not to_index:
        return {"status": "up_to_date", "indexed": 0, "skipped": len(chunks)}

    batch_size = 8
    indexed = 0
    for index in range(0, len(to_index), batch_size):
        batch = to_index[index : index + batch_size]
        ids = [item[0] for item in batch]
        docs_text = [item[1]["text"] for item in batch]
        metadatas = [_chunk_metadata(item[1]) for item in batch]
        vectors = _embed_with_retry(docs_text)
        collection.upsert(ids=ids, documents=docs_text, embeddings=vectors, metadatas=metadatas)
        indexed += len(batch)
        time.sleep(1.0)

    return {
        "status": "ok",
        "indexed": indexed,
        "skipped": len(chunks) - indexed,
        "total_chunks_generated": len(chunks),
    }


if __name__ == "__main__":
    print(build_index())
