"""Клиент для ЛЮБОГО OpenAI-совместимого API: ключ организаторов, GigaChat, YandexGPT, локальные vLLM/Ollama.

Провайдер меняется только переменными окружения (LLM_BASE_URL / LLM_MODEL / LLM_API_KEY),
поэтому в контуре М.Видео можно поднять модель on-prem — без VPN и внешних прокси.
"""
from __future__ import annotations

import json
import logging
import re
import threading

from ..config import Settings, get_settings
from .cache import LLMCache

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    pass


class LLMClient:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.cache = LLMCache(self.s.cache_path)
        self.calls = 0
        self._lock = threading.Lock()
        self._client = None

    @property
    def enabled(self) -> bool:
        return self.s.llm_enabled

    def _openai(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.s.llm_base_url, api_key=self.s.llm_api_key,
                timeout=self.s.llm_timeout, max_retries=self.s.llm_max_retries,
            )
        return self._client

    def complete_json(self, system: str, user: str, schema: dict, name: str = "result") -> dict:
        key = LLMCache.key(self.s.llm_model, system, user, json.dumps(schema, sort_keys=True))
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        with self._lock:
            if self.calls >= self.s.llm_max_calls:
                raise LLMUnavailable("Превышен лимит LLM_MAX_CALLS")
            self.calls += 1
        kwargs: dict = {}
        if self.s.llm_reasoning_effort:  # "none" отключает «размышления» у qwen3.x в Ollama — иначе ответ в разы дольше
            kwargs["reasoning_effort"] = self.s.llm_reasoning_effort
        fmt = self.s.llm_response_format
        if fmt == "json_schema":
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}
        elif fmt == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            resp = self._openai().chat.completions.create(model=self.s.llm_model, messages=messages, temperature=0, **kwargs)
        except Exception as e:  # провайдеры по-разному поддерживают response_format — откатываемся на текст
            if _is_error(e, "APITimeoutError"):
                raise LLMUnavailable(
                    f"LLM не ответила за {self.s.llm_timeout:g} с — увеличьте LLM_TIMEOUT или возьмите модель полегче"
                ) from e
            if _is_error(e, "APIConnectionError"):
                raise LLMUnavailable(f"нет соединения с LLM по адресу {self.s.llm_base_url} ({e})") from e
            if fmt == "none":
                raise LLMUnavailable(str(e)) from e
            log.warning("response_format=%s не поддержан (%s), повтор без него", fmt, e)
            try:
                kwargs.pop("response_format", None)
                resp = self._openai().chat.completions.create(model=self.s.llm_model, messages=messages, temperature=0, **kwargs)
            except Exception as e2:
                raise LLMUnavailable(str(e2)) from e2
        data = parse_json(resp.choices[0].message.content or "")
        self.cache.set(key, data)
        return data


def _is_error(e: Exception, name: str) -> bool:
    """Сеть/таймаут (openai.APIConnectionError, APITimeoutError): повтор без response_format не поможет."""
    return any(c.__name__ == name for c in type(e).__mro__)


def parse_json(text: str) -> dict:
    text = re.sub(r"^\x60{3}(?:json)?|\x60{3}$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise LLMUnavailable(f"Ответ LLM не JSON: {text[:200]}") from None
        return json.loads(m.group(0))


_client: LLMClient | None = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
