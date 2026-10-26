"""
rag/history.py
---------------
Persists chat history (question/answer/sources) to a JSON file on disk,
so conversations survive server restarts and page refreshes.

Kept intentionally simple (single JSON file, single conversation) — matches
the "one document at a time" scope of the rest of the app. Swap for a real
DB (SQLite/Postgres) if you need multi-user or multi-session history later.
"""

import json
import os
import time
from typing import List, Optional


class ChatHistoryManager:
    def __init__(self, persist_path: str):
        self.persist_path = persist_path
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        if not os.path.exists(self.persist_path):
            self._write([])

    def _read(self) -> List[dict]:
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _write(self, history: List[dict]) -> None:
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def add(self, question: str, answer: str, sources: Optional[list] = None,
            index: Optional[dict] = None) -> None:
        """`index` records the chunk settings the answer was produced under."""
        history = self._read()
        history.append({
            "question": question,
            "answer": answer,
            "sources": sources or [],
            "index": index,
            "timestamp": time.time(),
        })
        self._write(history)

    def get_all(self) -> List[dict]:
        return self._read()

    def get_recent(self, n: int = 5) -> List[dict]:
        """Most recent n turns, oldest first — useful for feeding conversational context to the LLM."""
        return self._read()[-n:]

    def clear(self) -> None:
        self._write([])
