"""Клиент для ЛЮБОГО OpenAI-совместимого API: ключ организаторов на ВМ, GigaChat, YandexGPT, vLLM/Ollama on-prem.

Провайдер меняется только переменными окружения, код не меняется (см. config.py):
LLM_BASE_URL/LLM_API_KEY/LLM_MODEL → иначе OPENAI_BASE_URL/OPENAI_API_KEY (так ключ организаторов задан на ВМ)
→ иначе LLM_LOCAL_URL (локальная Ollama, если запущена). LLM_MODEL=auto — модель из GET {base}/models.
Поэтому в контуре М.Видео модель можно поднять on-prem — без VPN и внешних прокси.

Что гарантирует клиент:
- строгий JSON: response_format json_schema → json_object → без него (провайдеры поддерживают по-разному);
  рабочий вариант запоминается, лишние запросы не повторяются;
- temperature 0 и SQLite-кэш: повторный прогон бесплатен и воспроизводим;
- учёт стоимости: вызовы, попадания в кэш, токены (из usage, иначе оценка по длине), латентность — stats();
- параллельные запросы с ограничением LLM_CONCURRENCY, лимиты вызовов на процесс и на сутки;
- понятные ошибки и «предохранитель»: если сервер недоступен, карточки не ждут таймаут по очереди.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from typing import Any, TypeVar
from urllib.parse import urlparse

from ..config import DEFAULT_LOCAL_MODEL, Settings, get_settings
from .cache import LLMCache

log = logging.getLogger(__name__)
T = TypeVar("T")
R = TypeVar("R")

FORMATS = ("json_schema", "json_object", "none")
# оценка токенов, если провайдер не вернул usage: замер на реальных карточках (qwen3, Ollama) — docs/LLM.md
CHARS_PER_TOKEN = 3.0
NON_CHAT_MODELS = ("embed", "whisper", "tts", "dall-e", "image", "moderation", "rerank", "audio", "transcri",
                   "speech", "ocr", "guard")
FALLBACK_MODELS = {"ollama": DEFAULT_LOCAL_MODEL, "gigachat": "GigaChat"}
FALLBACK_MODEL = "gpt-4o-mini"


class LLMUnavailable(RuntimeError):
    pass


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / CHARS_PER_TOKEN)) if text else 0


def provider_of(base_url: str, source: str = "") -> str:
    """Провайдер по адресу: ollama | gigachat | yandexgpt | openai | organizers | on-prem | openai-compatible."""
    u = urlparse(base_url)
    host, port = (u.hostname or "").lower(), u.port
    if port == 11434 or "ollama" in host:
        return "ollama"
    if "gigachat" in host or "sberbank" in host:
        return "gigachat"
    if "yandex" in host:
        return "yandexgpt"
    if host == "api.openai.com":
        return "openai"
    if source == "OPENAI_*":
        return "organizers"  # OPENAI_BASE_URL пришёл из окружения — на ВМ хакатона это шлюз организаторов
    if host in {"localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal"} or host.startswith(("10.", "192.168.")):
        return "on-prem"
    return "openai-compatible"


def ipv4_loopback(url: str) -> str:
    """localhost → 127.0.0.1. Ollama/vLLM по умолчанию слушают IPv4, а httpx (внутри openai SDK) без «happy eyeballs»
    сначала идёт на ::1 и, если IPv6-loopback фильтруется, ждёт полный таймаут: замерено +60 с на первый запрос."""
    u = urlparse(url)
    return u._replace(netloc=u.netloc.replace("localhost", "127.0.0.1", 1)).geturl() if u.hostname == "localhost" else url


def choose_model(models: Iterable[str], preference: str | Iterable[str]) -> str | None:
    """Модель для LLM_MODEL=auto: первая по списку предпочтений, иначе первая «чатовая» из списка провайдера."""
    prefs = preference.split(",") if isinstance(preference, str) else list(preference)
    prefs = [p.strip().lower() for p in prefs if p.strip()]
    chat = [m for m in models if m and not any(x in m.lower() for x in NON_CHAT_MODELS)]
    for p in prefs:
        for m in chat:
            if p in m.lower():
                return m
    return chat[0] if chat else None


@dataclass
class _Stats:
    requests: int = 0  # обращений к complete_json (включая кэш)
    calls: int = 0  # вызовов API (без кэша) — за них платим
    http_requests: int = 0  # HTTP-запросов, включая повторы при откате формата
    cache_hits: int = 0
    errors: int = 0
    invalid_json: int = 0
    format_fallbacks: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_usage: int = 0  # вызовов без usage от провайдера: токены оценены по длине текста
    latencies: list[float] = field(default_factory=list)


class LLMClient:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.cache = LLMCache(self.s.cache_path)
        self.calls = 0  # вызовов API за жизнь клиента (совместимость: pipeline/report/cli читают client.calls)
        self._lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._client = None
        self._model: str | None = None
        self._model_source = ""
        self._model_checked = 0.0
        self._fmt = FORMATS.index(self.s.llm_response_format) if self.s.llm_response_format in FORMATS else 0
        self._dropped: set[str] = set()  # параметры, которые провайдер отверг (reasoning_effort, temperature, ...)
        self._max_tokens_param = "max_tokens"
        self._down_until = 0.0
        self._down_reason = ""
        self._timeouts_in_row = 0
        self._probe: dict | None = None
        self._probe_at = 0.0
        self._st = _Stats()
        self._sem = threading.BoundedSemaphore(self.concurrency)

    # ------------------------------------------------------------------ свойства
    @property
    def provider(self) -> str:
        return provider_of(self.s.llm_base_url, self.s.llm_source)

    @property
    def concurrency(self) -> int:
        """Сколько запросов держать в полёте: LLM_CONCURRENCY, иначе 1 для Ollama (одна GPU) и 4 для API."""
        if self.s.llm_concurrency > 0:
            return self.s.llm_concurrency
        return 1 if self.provider == "ollama" else 4

    @property
    def response_format(self) -> str:
        return FORMATS[self._fmt]

    @property
    def enabled(self) -> bool:
        if not self.s.llm_enabled:
            return False
        if self.s.llm_source == "LLM_LOCAL_URL":  # «локальная LLM, если запущена»: проверяем без лишних ожиданий
            return bool(self.ping(max_age=60.0).get("reachable"))
        return True

    @property
    def model(self) -> str:
        return self._resolve_model()

    @property
    def model_source(self) -> str:
        """env — задана явно, discovered — выбрана из /models, fallback — запасное имя, '' — ещё не выбиралась."""
        if (self.s.llm_model or "").strip().lower() not in {"", "auto"}:
            return "env"
        return self._model_source

    def down_for(self) -> float:
        """Сколько секунд ещё действует «предохранитель» после ошибки соединения (0 — LLM доступна)."""
        return max(0.0, self._down_until - time.monotonic())

    # ------------------------------------------------------------------ сеть
    def _openai(self):
        with self._lock:
            if self._client is None:
                from openai import OpenAI

                self._client = OpenAI(
                    base_url=ipv4_loopback(self.s.llm_base_url), api_key=self.s.llm_api_key,
                    timeout=self.s.llm_timeout, max_retries=self.s.llm_max_retries,
                )
            return self._client

    def list_models(self, timeout: float = 5.0) -> list[str]:
        """GET {base}/models — id доступных моделей."""
        client = self._openai().with_options(timeout=timeout, max_retries=0)
        return [m.id for m in client.models.list()]

    def ping(self, max_age: float = 30.0, timeout: float = 2.0) -> dict:
        """Доступность сервера (GET /models) с кэшем на max_age секунд: для /api/health и автоопределения."""
        now = time.monotonic()
        if self._probe is not None and now - self._probe_at < max_age:
            return self._probe
        t: float | None = None
        try:
            self._openai()  # импорт SDK и создание клиента — не сеть, в задержку не входят
            t = time.perf_counter()
            models = self.list_models(timeout=timeout)
            res = {"reachable": True, "error": None, "models": models[:50], "n_models": len(models)}
        except Exception as e:  # статус не должен падать
            # сервер ответил HTTP-ошибкой (401/404 ...) — он доступен, просто не отдаёт список моделей
            responded = _is_error(e, "APIStatusError")
            res = {"reachable": responded, "error": _short_error(e), "models": [], "n_models": 0}
        res["latency_ms"] = None if t is None else round((time.perf_counter() - t) * 1000, 1)
        self._probe, self._probe_at = res, now
        return res

    def _resolve_model(self) -> str:
        configured = (self.s.llm_model or "").strip()
        if configured and configured.lower() != "auto":
            return configured
        with self._model_lock:
            now = time.monotonic()
            if self._model and (self._model_source == "discovered" or now - self._model_checked < 60 or self.down_for()):
                return self._model
            if self.down_for():  # сервер недоступен — не ждём /models, вызов всё равно отклонит предохранитель
                return FALLBACK_MODELS.get(self.provider, FALLBACK_MODEL)
            self._model_checked = now
            fresh = self._probe if self._probe and now - self._probe_at < 60 and self._probe.get("models") else None
            try:
                models = fresh["models"] if fresh else self.list_models(timeout=min(self.s.llm_timeout, 10.0))
                pick = choose_model(models, self.s.llm_model_preference)
            except Exception as e:
                log.warning("LLM_MODEL=auto: список моделей недоступен (%s), беру запасную", _short_error(e))
                pick = None
            if pick:
                self._model, self._model_source = pick, "discovered"
            else:
                self._model = FALLBACK_MODELS.get(self.provider, FALLBACK_MODEL)
                self._model_source = "fallback"
            log.info("LLM: модель %s (%s), провайдер %s", self._model, self._model_source, self.provider)
            return self._model

    # ------------------------------------------------------------------ основной вызов
    def complete_json(self, system: str, user: str, schema: dict, name: str = "result", *,
                      max_tokens: int | None = None) -> dict:
        """Один JSON-ответ модели по схеме. Ошибки провайдера → LLMUnavailable с понятным текстом."""
        model = self._resolve_model()
        with self._lock:
            self._st.requests += 1
        key = LLMCache.key(model, system, user, json.dumps(schema, sort_keys=True, ensure_ascii=False))
        if self.s.llm_cache:
            cached = self.cache.get(key)
            if cached is not None:
                with self._lock:
                    self._st.cache_hits += 1
                return cached
        if (left := self.down_for()) > 0:
            raise LLMUnavailable(f"{self._down_reason} (повтор через {left:.0f} с)")
        with self._lock:
            if self.calls >= self.s.llm_max_calls:
                raise LLMUnavailable("Превышен лимит LLM_MAX_CALLS")
            if self.s.llm_daily_max_calls and self.usage_today()["calls"] >= self.s.llm_daily_max_calls:
                raise LLMUnavailable("Исчерпан суточный лимит LLM_DAILY_MAX_CALLS")
            self.calls += 1
            self._st.calls += 1
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        limit = max_tokens or self.s.llm_max_tokens
        try:
            with self._sem:
                resp, latency = self._create(model, messages, schema, name, limit)
            if not getattr(resp, "choices", None) or resp.choices[0].message is None:
                raise LLMUnavailable("провайдер вернул ответ без choices (фильтр контента или сбой шлюза)")
            choice = resp.choices[0]
            content = choice.message.content or ""
            self._account(resp, system + user, content, latency)
            if not content.strip():
                raise LLMUnavailable("модель вернула пустой ответ — если это «думающая» модель, задайте "
                                     "LLM_REASONING_EFFORT=none или увеличьте LLM_MAX_TOKENS")
            try:
                data = parse_json(content)
            except LLMUnavailable:
                with self._lock:
                    self._st.invalid_json += 1
                if choice.finish_reason == "length":
                    raise LLMUnavailable(f"ответ LLM обрезан на {limit} токенах — увеличьте LLM_MAX_TOKENS") from None
                raise
        except LLMUnavailable:
            with self._lock:
                self._st.errors += 1
            raise
        if self.s.llm_cache:
            self.cache.set(key, data)
        return data

    def _kwargs(self, fmt: str, schema: dict, name: str, max_tokens: int) -> dict:
        kwargs: dict[str, Any] = {}
        if "temperature" not in self._dropped:
            kwargs["temperature"] = 0
        if max_tokens and "max_tokens" not in self._dropped:
            kwargs[self._max_tokens_param] = max_tokens
        # "none" отключает «размышления» у qwen3.x в Ollama — иначе ответ в разы дольше
        if self.s.llm_reasoning_effort and "reasoning_effort" not in self._dropped:
            kwargs["reasoning_effort"] = self.s.llm_reasoning_effort
        if fmt == "json_schema":
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}
        elif fmt == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def _create(self, model: str, messages: list[dict], schema: dict, name: str, max_tokens: int):
        for _ in range(8):  # формат и необязательные параметры откатываются максимум по разу
            kwargs = self._kwargs(FORMATS[self._fmt], schema, name, max_tokens)
            t = time.perf_counter()
            with self._lock:
                self._st.http_requests += 1
            try:
                resp = self._openai().chat.completions.create(model=model, messages=messages, **kwargs)
                self._timeouts_in_row = 0
                return resp, time.perf_counter() - t
            except Exception as e:  # классифицируем ниже
                self._raise_if_fatal(e, model)
                if not self._downgrade(e, kwargs):
                    raise LLMUnavailable(_short_error(e)) from e
        raise LLMUnavailable("провайдер отклоняет запрос при любом формате ответа")

    def _raise_if_fatal(self, e: Exception, model: str) -> None:
        """Сеть, ключ, лимиты, нет модели — повтор с другим форматом не поможет."""
        url = self.s.llm_base_url
        if _is_error(e, "APITimeoutError"):
            self._timeouts_in_row += 1
            if self._timeouts_in_row >= 2:  # один долгий ответ — не повод отключать LLM, два подряд — повод
                self._trip(f"LLM не отвечает за {self.s.llm_timeout:g} с")
            raise LLMUnavailable(
                f"LLM не ответила за {self.s.llm_timeout:g} с — увеличьте LLM_TIMEOUT или возьмите модель полегче"
            ) from e
        if _is_error(e, "APIConnectionError"):
            self._trip(f"нет соединения с LLM по адресу {url}")
            raise LLMUnavailable(f"нет соединения с LLM по адресу {url} ({e})") from e
        if _is_error(e, "AuthenticationError") or _is_error(e, "PermissionDeniedError"):
            self._trip("ключ LLM не принят", seconds=max(300.0, self.s.llm_retry_after))
            raise LLMUnavailable(f"ключ LLM не принят ({_short_error(e)}) — проверьте LLM_API_KEY / OPENAI_API_KEY") from e
        if _is_error(e, "RateLimitError"):
            self._trip("лимит запросов провайдера (429)", seconds=_retry_after(e, self.s.llm_retry_after))
            raise LLMUnavailable(f"лимит запросов провайдера (429): {_short_error(e)} — у ключа организаторов "
                                 "лимит дневной, кэш и LLM_DAILY_MAX_CALLS помогают его не исчерпать") from e
        if _is_error(e, "NotFoundError"):
            with self._model_lock:
                if self._model_source == "fallback":
                    self._model_checked = 0.0  # при следующем вызове снова спросим /models
            raise LLMUnavailable(f"модель {model!r} не найдена у провайдера ({_short_error(e)}) — задайте LLM_MODEL "
                                 "или LLM_MODEL=auto") from e
        if _is_error(e, "InternalServerError"):
            raise LLMUnavailable(f"ошибка сервера LLM: {_short_error(e)}") from e
        msg = str(e).lower()
        if "context" in msg and ("length" in msg or "window" in msg or "too long" in msg):
            raise LLMUnavailable(f"карточка не помещается в контекст модели — уменьшите LLM_MAX_INPUT_CHARS "
                                 f"({_short_error(e)})") from e

    def _downgrade(self, e: Exception, kwargs: dict) -> bool:
        """Провайдер отверг запрос (400/422 и т.п.): убираем то, что он не поддерживает, и пробуем снова."""
        msg = str(e).lower()
        with self._lock:
            if "reasoning" in msg and "reasoning_effort" in kwargs:
                self._dropped.add("reasoning_effort")
            elif "max_completion_tokens" in msg and "max_completion_tokens" in kwargs:
                self._dropped.add("max_tokens")
            elif "max_tokens" in msg and "max_tokens" in kwargs:
                self._max_tokens_param = "max_completion_tokens"  # o1/gpt-5: только max_completion_tokens
            elif "temperature" in msg and "temperature" in kwargs:
                self._dropped.add("temperature")
            elif self._fmt < len(FORMATS) - 1:
                log.warning("response_format=%s не поддержан (%s), пробую %s",
                            FORMATS[self._fmt], _short_error(e), FORMATS[self._fmt + 1])
                self._fmt += 1
                self._st.format_fallbacks += 1
            else:
                return False
        return True

    def _trip(self, reason: str, seconds: float | None = None) -> None:
        with self._lock:
            self._down_until = time.monotonic() + (self.s.llm_retry_after if seconds is None else seconds)
            self._down_reason = reason

    def _account(self, resp: Any, prompt_text: str, content: str, latency: float) -> None:
        usage = getattr(resp, "usage", None)
        pt = getattr(usage, "prompt_tokens", None)
        ct = getattr(usage, "completion_tokens", None)
        estimated = not isinstance(pt, int) or not isinstance(ct, int)
        pt = pt if isinstance(pt, int) else estimate_tokens(prompt_text) + 8
        ct = ct if isinstance(ct, int) else estimate_tokens(content)
        with self._lock:
            self._st.prompt_tokens += pt
            self._st.completion_tokens += ct
            self._st.estimated_usage += int(estimated)
            self._st.latencies.append(latency)
        try:
            self.cache.add_usage(date.today().isoformat(), 1, pt, ct)
        except Exception as e:  # учёт не должен ломать извлечение
            log.debug("usage: %s", e)

    # ------------------------------------------------------------------ параллельность и учёт
    def map(self, fn: Callable[[T], R], items: Iterable[T], concurrency: int | None = None) -> list[R]:
        """fn по элементам в concurrency потоках (по умолчанию self.concurrency), порядок сохраняется."""
        items = list(items)
        n = max(1, concurrency or self.concurrency)
        if n == 1 or len(items) <= 1:
            return [fn(x) for x in items]
        with ThreadPoolExecutor(max_workers=min(n, len(items))) as pool:
            return list(pool.map(fn, items))

    def usage_today(self) -> dict:
        try:
            return self.cache.usage(date.today().isoformat())
        except Exception:
            return {"day": date.today().isoformat(), "calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def stats(self) -> dict:
        """Счётчики для отчёта о стоимости: вызовы, кэш, токены, латентность (attrx.llm.cost переводит в ₽)."""
        with self._lock:
            st = _Stats(**{k: (list(v) if isinstance(v, list) else v) for k, v in self._st.__dict__.items()})
        lat = sorted(st.latencies)

        def q(p: float) -> float | None:
            return round(lat[min(len(lat) - 1, int(p * len(lat)))], 3) if lat else None

        return {
            "provider": self.provider, "model": self._model or self.s.llm_model, "response_format": self.response_format,
            "concurrency": self.concurrency,
            "requests": st.requests, "calls": st.calls, "http_requests": st.http_requests,
            "cache_hits": st.cache_hits, "cache_hit_rate": round(st.cache_hits / st.requests, 3) if st.requests else 0.0,
            "errors": st.errors, "invalid_json": st.invalid_json, "format_fallbacks": st.format_fallbacks,
            "prompt_tokens": st.prompt_tokens, "completion_tokens": st.completion_tokens,
            "total_tokens": st.prompt_tokens + st.completion_tokens, "estimated_usage": st.estimated_usage,
            "latency_s": {"total": round(sum(lat), 3), "avg": round(sum(lat) / len(lat), 3) if lat else None,
                          "p50": q(0.5), "p95": q(0.95), "max": round(lat[-1], 3) if lat else None},
        }

    def reset_stats(self) -> None:
        with self._lock:
            self._st = _Stats()


def _is_error(e: Exception, name: str) -> bool:
    """Класс ошибки openai по имени (APIConnectionError, APITimeoutError ...) — не зависит от версии пакета."""
    return any(c.__name__ == name for c in type(e).__mro__)


def _short_error(e: Exception) -> str:
    code = getattr(e, "status_code", None)
    text = re.sub(r"\s+", " ", str(e)).strip()[:300]
    return f"HTTP {code}: {text}" if code else f"{type(e).__name__}: {text}"


def _retry_after(e: Exception, default: float) -> float:
    try:
        return float(e.response.headers.get("retry-after"))  # type: ignore[attr-defined]
    except Exception:
        return default


_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def parse_json(text: str) -> dict:
    """JSON-объект из ответа модели: без <think>-блоков и ```-ограждений; иначе LLMUnavailable."""
    text = _THINK.sub("", text or "").strip()
    text = re.sub(r"^\x60{3}(?:json)?|\x60{3}$", "", text, flags=re.M).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        try:
            data = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            data = None
        if data is None:
            raise LLMUnavailable(f"Ответ LLM не JSON: {text[:200]}") from None
    if not isinstance(data, dict):
        raise LLMUnavailable(f"Ответ LLM — не JSON-объект: {text[:200]}")
    return data


_client: LLMClient | None = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
