from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock


@dataclass
class ConversationTurn:
    question: str
    answer: str


class ConversationMemoryStore:
    def __init__(self, max_turns: int = 6) -> None:
        self.max_turns = max_turns
        self._sessions: dict[str, deque[ConversationTurn]] = {}
        self._lock = Lock()

    def get_history(self, session_id: str) -> list[dict]:
        with self._lock:
            turns = list(self._sessions.get(session_id, deque()))
        return [{"question": turn.question, "answer": turn.answer} for turn in turns]

    def append_turn(self, session_id: str, question: str, answer: str) -> None:
        question = (question or "").strip()
        answer = (answer or "").strip()
        if not session_id or not question or not answer:
            return
        with self._lock:
            bucket = self._sessions.get(session_id)
            if bucket is None:
                bucket = deque(maxlen=self.max_turns)
                self._sessions[session_id] = bucket
            bucket.append(ConversationTurn(question=question, answer=answer))

    def clear_session(self, session_id: str) -> bool:
        if not session_id:
            return False
        with self._lock:
            existed = session_id in self._sessions
            if existed:
                del self._sessions[session_id]
        return existed
