import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable


class Database:
    def __init__(self, db_path: str) -> None:
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
                    answer TEXT NOT NULL,
                    top_sources TEXT,
                    retrieval_backend TEXT,
                    confidence TEXT,
                    top_score REAL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._ensure_query_log_columns(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS anomaly_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    txn_id TEXT NOT NULL,
                    account TEXT NOT NULL,
                    amount REAL NOT NULL,
                    reason TEXT NOT NULL,
                    score REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def _ensure_query_log_columns(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute("PRAGMA table_info(query_logs)").fetchall()
        existing = {row[1] for row in rows}
        additions = [
            ("retrieval_backend", "TEXT"),
            ("confidence", "TEXT"),
            ("top_score", "REAL"),
        ]
        for name, col_type in additions:
            if name not in existing:
                conn.execute(f"ALTER TABLE query_logs ADD COLUMN {name} {col_type}")

    def log_query(
        self,
        question: str,
        answer: str,
        sources: Iterable[str],
        retrieval_backend: str = "",
        confidence: str = "",
        top_score: float | None = None,
    ) -> None:
        joined_sources = " | ".join(sources)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO query_logs(
                    question, answer, top_sources, retrieval_backend, confidence, top_score, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    question,
                    answer,
                    joined_sources,
                    retrieval_backend,
                    confidence,
                    top_score,
                    datetime.utcnow().isoformat(),
                ),
            )

    def log_anomalies(self, rows: list[dict]) -> None:
        if not rows:
            return
        values = [
            (
                row["txn_id"],
                row["account"],
                float(row["amount"]),
                row["reason"],
                float(row["score"]),
                datetime.utcnow().isoformat(),
            )
            for row in rows
        ]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO anomaly_logs(txn_id, account, amount, reason, score, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                values,
            )

    def fetch_query_logs(self):
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT question, top_sources, retrieval_backend, confidence, top_score, created_at
                FROM query_logs
                ORDER BY id DESC
                """
            ).fetchall()

    def fetch_anomaly_logs(self):
        with self._connect() as conn:
            return conn.execute(
                "SELECT txn_id, account, amount, reason, score, created_at FROM anomaly_logs ORDER BY id DESC"
            ).fetchall()
