"""Keyword (BM25) + dense vector fusion via reciprocal rank fusion (RRF)."""

from __future__ import annotations

import re

# Keeps acronyms, policy-style numbers (e.g. 2.21, 409A), and words.
TOKEN_RE = re.compile(
    r"[A-Za-z]{2,}|[A-Za-z0-9]+(?:[./-][A-Za-z0-9]+)*|\d+(?:\.\d+)+",
    re.UNICODE,
)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """
    Merge ranked lists of chunk ids. Same id may appear in multiple lists; scores add up.
    Returns (chroma_id, normalized_rrf_score) sorted by score descending.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    if not scores:
        return []
    max_score = max(scores.values())
    denom = max_score if max_score > 0 else 1.0
    ordered = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [(i, scores[i] / denom) for i in ordered]


def build_bm25_index(corpus_texts: list[str]):
    """Build BM25 over tokenized chunks. Empty docs get a minimal token to satisfy BM25Okapi."""
    from rank_bm25 import BM25Okapi

    tokenized: list[list[str]] = []
    for raw in corpus_texts:
        toks = tokenize(raw)
        tokenized.append(toks if toks else ["_"])
    return BM25Okapi(tokenized), tokenized
