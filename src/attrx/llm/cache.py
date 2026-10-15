from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path


class LLMCache:
    """SQLite-кэш ответов LLM: повторный прогон по тем же карточкам бесплатен и воспроизводим.

    Там же — суточный счётчик вызовов и токенов (у ключа организаторов дневной лимит).
    Одно соединение на процесс, доступ под замком: клиент LLM вызывается из нескольких потоков.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        with self._lock:
            self.db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
            self.db.execute("CREATE TABLE IF NOT EXISTS usage (day TEXT PRIMARY KEY, calls INTEGER, "
                            "prompt_tokens INTEGER, completion_tokens INTEGER)")
            self.db.commit()

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()

    def get(self, k: str) -> dict | None:
        with self._lock:
            row = self.db.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, k: str, v: dict) -> None:
        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (k, json.dumps(v, ensure_ascii=False)))
            self.db.commit()

    def add_usage(self, day: str, calls: int, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self.db.execute(
                "INSERT INTO usage VALUES (?, ?, ?, ?) ON CONFLICT(day) DO UPDATE SET calls=calls+excluded.calls, "
                "prompt_tokens=prompt_tokens+excluded.prompt_tokens, "
                "completion_tokens=completion_tokens+excluded.completion_tokens",
                (day, calls, prompt_tokens, completion_tokens),
            )
            self.db.commit()

    def usage(self, day: str) -> dict:
        with self._lock:
            row = self.db.execute("SELECT calls, prompt_tokens, completion_tokens FROM usage WHERE day=?", (day,)).fetchone()
        calls, pt, ct = row or (0, 0, 0)
        return {"day": day, "calls": calls, "prompt_tokens": pt, "completion_tokens": ct}
