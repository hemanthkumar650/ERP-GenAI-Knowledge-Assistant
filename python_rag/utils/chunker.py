from __future__ import annotations

import re
from typing import Iterable

import tiktoken


def _split_into_paragraphs(text: str) -> list[str]:
    pieces = re.split(r"\n\s*\n+", text)
    return [" ".join(piece.strip().split()) for piece in pieces if piece and piece.strip()]


def _token_len(text: str, encoder: tiktoken.Encoding) -> int:
    return len(encoder.encode(text))


def split_text_into_chunks(
    text: str,
    chunk_size_tokens: int = 500,
    overlap_tokens: int = 50,
    model_encoding: str = "cl100k_base",
) -> list[str]:
    if not text.strip():
        return []
    if overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_size_tokens")

    encoder = tiktoken.get_encoding(model_encoding)
    paragraphs = _split_into_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        paragraph_tokens = _token_len(paragraph, encoder)
        if paragraph_tokens == 0:
            continue

        if paragraph_tokens > chunk_size_tokens and not current_parts:
            token_ids = encoder.encode(paragraph)
            step = chunk_size_tokens - overlap_tokens
            for start in range(0, len(token_ids), step):
                window = token_ids[start : start + chunk_size_tokens]
                if not window:
                    continue
                chunk_text = encoder.decode(window).strip()
                if chunk_text:
                    chunks.append(chunk_text)
            continue

        if current_tokens + paragraph_tokens <= chunk_size_tokens:
            current_parts.append(paragraph)
            current_tokens += paragraph_tokens
            continue

        if current_parts:
            chunks.append("\n\n".join(current_parts).strip())
            previous_tokens = encoder.encode(chunks[-1])
            overlap_ids = previous_tokens[-overlap_tokens:] if overlap_tokens > 0 else []
            overlap_text = encoder.decode(overlap_ids).strip()
            current_parts = [overlap_text, paragraph] if overlap_text else [paragraph]
            current_tokens = _token_len("\n\n".join(current_parts), encoder)
        else:
            chunks.append(paragraph)
            current_parts = []
            current_tokens = 0

    if current_parts:
        chunks.append("\n\n".join(current_parts).strip())

    return [chunk for chunk in chunks if _token_len(chunk, encoder) >= 60]


def chunk_documents(
    documents: Iterable[dict],
    chunk_size_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[dict]:
    chunks: list[dict] = []
    for doc in documents:
        source = doc["source"]
        text = doc["text"]
        parts = split_text_into_chunks(
            text=text,
            chunk_size_tokens=chunk_size_tokens,
            overlap_tokens=overlap_tokens,
        )
        for index, chunk_text in enumerate(parts, start=1):
            chunk_id = f"{source}::chunk-{index}"
            chunks.append({"chunk_id": chunk_id, "text": chunk_text, "source": source})
    return chunks
