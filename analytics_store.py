from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class AnalyticsStore:
    def __init__(self, db_path: str = "data/analytics.db") -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    top_similarity REAL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def log_query(self, question: str, sources: list[str], top_similarity: float | None) -> None:
        payload = json.dumps(sources)
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO query_logs(question, sources_json, top_similarity, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (question, payload, top_similarity, created_at),
            )

    def fetch_logs(self, limit: int = 2000) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT question, sources_json, top_similarity, created_at
                FROM query_logs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        output: list[dict] = []
        for question, sources_json, top_similarity, created_at in rows:
            try:
                sources = json.loads(sources_json)
            except Exception:
                sources = []
            output.append(
                {
                    "question": question,
                    "sources": sources,
                    "top_similarity": top_similarity,
                    "created_at": created_at,
                }
            )
        return output
