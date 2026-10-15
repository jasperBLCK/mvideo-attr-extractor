"""Состояние LLM для /api/health и UI: провайдер, модель, доступность. Никогда не бросает и не раскрывает ключ."""
from __future__ import annotations

from urllib.parse import urlparse

from .client import LLMClient, get_client

PROVIDER_TITLES = {
    "organizers": "Шлюз организаторов (OPENAI_BASE_URL на ВМ)",
    "ollama": "Ollama on-prem (локальная GPU)",
    "on-prem": "On-prem OpenAI-совместимый сервер (vLLM и т.п.)",
    "gigachat": "GigaChat (Сбер)",
    "yandexgpt": "YandexGPT (Yandex AI Studio)",
    "openai": "OpenAI API",
    "openai-compatible": "OpenAI-совместимый API",
}


def llm_status(client: LLMClient | None = None, probe: bool = True, max_age: float = 30.0) -> dict:
    """Короткий JSON о LLM-слое.

    probe=True — проверить сервер (GET /models, таймаут 2 с, результат кэшируется на max_age секунд) и,
    при LLM_MODEL=auto, выбрать модель; probe=False — только конфигурация, без сети.
    Поля: enabled, mode (auto|on|off), source (LLM_*|OPENAI_*|LLM_LOCAL_URL|default), provider, provider_title,
    base_host, model, model_source (env|discovered|fallback), reachable (True/False/None — не проверялось),
    latency_ms, error, response_format, concurrency, circuit_open_s, calls, today.
    """
    try:
        c = client or get_client()
        s = c.s
        u = urlparse(s.llm_base_url)
        out: dict = {
            "enabled": bool(s.llm_enabled), "mode": s.llm_mode, "source": s.llm_source,
            "provider": c.provider, "provider_title": PROVIDER_TITLES.get(c.provider, c.provider),
            "base_host": f"{u.hostname}:{u.port}" if u.port else u.hostname,
            "model": s.llm_model, "model_source": c.model_source or None,
            "reachable": None, "latency_ms": None, "error": None,
            "response_format": c.response_format, "concurrency": c.concurrency,
            "circuit_open_s": round(c.down_for(), 1), "calls": c.calls, "today": c.usage_today(),
        }
        if probe and s.llm_enabled:
            p = c.ping(max_age=max_age)
            out.update(reachable=p["reachable"], latency_ms=p["latency_ms"], error=p["error"])
            if p["reachable"]:
                out["model"], out["model_source"] = c.model, c.model_source
            # «auto» с LLM_LOCAL_URL: если локальная LLM не запущена, LLM фактически выключена
            out["enabled"] = c.enabled
        return out
    except Exception as e:  # health-check не должен падать из-за LLM
        return {"enabled": False, "reachable": None, "error": f"llm_status: {type(e).__name__}: {e}"}
