from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


class LLMCache:
    """SQLite-кэш ответов LLM: повторный прогон по тем же карточкам бесплатен и воспроизводим."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()

    def get(self, k: str) -> dict | None:
        row = self.db.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, k: str, v: dict) -> None:
        self.db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (k, json.dumps(v, ensure_ascii=False)))
        self.db.commit()
