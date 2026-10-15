"""Клиент LLM с подменённым OpenAI SDK: фолбэки формата, учёт токенов и кэша, лимиты, предохранитель, статус."""
import json
import threading
import time
from types import SimpleNamespace

import pytest

from attrx.config import Settings
from attrx.llm.client import LLMClient, LLMUnavailable, parse_json
from attrx.llm.status import llm_status


class BadRequestError(Exception):  # имя как у openai.BadRequestError: клиент классифицирует ошибки по имени класса
    status_code = 400


class APIConnectionError(Exception):
    pass


def reply(content, pt=100, ct=20, finish="stop"):
    usage = None if pt is None else SimpleNamespace(prompt_tokens=pt, completion_tokens=ct)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)],
                           usage=usage)


class FakeSDK:
    """Минимальный двойник openai.OpenAI: chat.completions.create, models.list, with_options."""

    def __init__(self, handler, models=("qwen3:4b-instruct",)):
        self.handler, self.requests, self._models = handler, [], list(models)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.models = SimpleNamespace(list=lambda: [SimpleNamespace(id=m) for m in self._models])

    def with_options(self, **_):
        return self

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.handler(kwargs)


def make_client(tmp_path, handler, **overrides):
    params = {"llm_enabled": True, "llm_base_url": "https://gw.example/v1", "llm_model": "test-model",
              "cache_path": tmp_path / "cache.sqlite", **overrides}
    s = Settings(**params)
    c = LLMClient(s)
    c._client = FakeSDK(handler)
    return c


SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"], "additionalProperties": False}


def test_json_schema_to_json_object_to_none_fallback_chain(tmp_path):
    def handler(kw):
        rf = kw.get("response_format")
        if rf is not None:
            raise BadRequestError(f"response_format {rf['type']} is not supported by this model")
        return reply('Вот ответ:\n```json\n{"x": "ok"}\n```')

    c = make_client(tmp_path, handler)
    assert c.complete_json("sys JSON", "user", SCHEMA) == {"x": "ok"}
    sent = [r.get("response_format", {}).get("type", "none") for r in c._client.requests]
    assert sent == ["json_schema", "json_object", "none"]
    assert c.response_format == "none"
    st = c.stats()
    assert (st["calls"], st["http_requests"], st["format_fallbacks"]) == (1, 3, 2)
    # рабочий формат запомнен: следующий вызов — сразу без response_format, одним запросом
    c.complete_json("sys JSON", "другая карточка", SCHEMA)
    assert c.stats()["http_requests"] == 4 and "response_format" not in c._client.requests[-1]


def test_strict_json_schema_is_sent_with_temperature_zero(tmp_path):
    c = make_client(tmp_path, lambda kw: reply('{"x": "ok"}'))
    c.complete_json("sys", "user", SCHEMA, "attributes")
    req = c._client.requests[0]
    assert req["temperature"] == 0 and req["model"] == "test-model"
    assert req["response_format"] == {"type": "json_schema",
                                      "json_schema": {"name": "attributes", "schema": SCHEMA, "strict": True}}


def test_unsupported_reasoning_effort_is_dropped_not_format(tmp_path):
    def handler(kw):
        if "reasoning_effort" in kw:
            raise BadRequestError("Unrecognized request argument supplied: reasoning_effort")
        return reply('{"x": "ok"}')

    c = make_client(tmp_path, handler, llm_reasoning_effort="none")
    assert c.complete_json("sys", "u", SCHEMA) == {"x": "ok"}
    assert c.response_format == "json_schema"  # формат не понижали
    c.complete_json("sys", "u2", SCHEMA)
    assert "reasoning_effort" not in c._client.requests[-1]


def test_stats_tokens_cache_and_latency(tmp_path):
    def handler(kw):
        time.sleep(0.01)
        return reply('{"x": "ok"}', pt=1200, ct=80)

    c = make_client(tmp_path, handler)
    c.complete_json("sys", "card 1", SCHEMA)
    c.complete_json("sys", "card 2", SCHEMA)
    c.complete_json("sys", "card 1", SCHEMA)  # повтор — из кэша, бесплатно
    st = c.stats()
    assert (st["requests"], st["calls"], st["cache_hits"]) == (3, 2, 1)
    assert (st["prompt_tokens"], st["completion_tokens"], st["total_tokens"]) == (2400, 160, 2560)
    assert st["cache_hit_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert st["latency_s"]["avg"] >= 0.01 and st["latency_s"]["p95"] is not None
    assert c.calls == 2  # совместимость: pipeline/report читают client.calls
    assert c.usage_today()["calls"] == 2 and c.usage_today()["prompt_tokens"] == 2400
    c.reset_stats()
    assert c.stats()["calls"] == 0


def test_tokens_estimated_when_provider_has_no_usage(tmp_path):
    c = make_client(tmp_path, lambda kw: reply('{"x": "ok"}', pt=None))
    c.complete_json("s" * 300, "u" * 300, SCHEMA)
    st = c.stats()
    assert st["estimated_usage"] == 1 and st["prompt_tokens"] > 100 and st["completion_tokens"] > 0


def test_daily_and_process_limits(tmp_path):
    c = make_client(tmp_path, lambda kw: reply('{"x": "ok"}'), llm_daily_max_calls=1)
    c.complete_json("sys", "a", SCHEMA)
    with pytest.raises(LLMUnavailable, match="суточный лимит"):
        c.complete_json("sys", "b", SCHEMA)
    c2 = make_client(tmp_path / "p", lambda kw: reply('{"x": "ok"}'), llm_max_calls=0)
    with pytest.raises(LLMUnavailable, match="LLM_MAX_CALLS"):
        c2.complete_json("sys", "a", SCHEMA)


def test_connection_error_opens_circuit(tmp_path):
    def handler(kw):
        raise APIConnectionError("Connection refused")

    c = make_client(tmp_path, handler, llm_retry_after=30)
    with pytest.raises(LLMUnavailable, match="нет соединения"):
        c.complete_json("sys", "a", SCHEMA)
    with pytest.raises(LLMUnavailable, match="повтор через"):
        c.complete_json("sys", "b", SCHEMA)
    assert len(c._client.requests) == 1  # вторая карточка не ждала сеть
    assert c.down_for() > 0


def test_invalid_json_and_truncation_are_reported(tmp_path):
    c = make_client(tmp_path, lambda kw: reply("извините, не могу", finish="stop"), llm_response_format="none")
    with pytest.raises(LLMUnavailable, match="не JSON"):
        c.complete_json("sys", "a", SCHEMA)
    c2 = make_client(tmp_path / "t", lambda kw: reply('{"x": "о', finish="length"), llm_response_format="none")
    with pytest.raises(LLMUnavailable, match="LLM_MAX_TOKENS"):
        c2.complete_json("sys", "a", SCHEMA)
    assert c.stats()["invalid_json"] == 1 and c.stats()["errors"] == 1


def test_parse_json_strips_think_and_fences():
    assert parse_json('<think>рассуждаю...</think>\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Ответ: {"a": {"b": null}} — готово') == {"a": {"b": None}}
    with pytest.raises(LLMUnavailable):
        parse_json("[1, 2]")


def test_auto_model_discovered_from_models_endpoint(tmp_path):
    c = make_client(tmp_path, lambda kw: reply('{"x": "ok"}'), llm_model="auto")
    c._client._models = ["text-embedding-3-small", "mistral:latest", "qwen2.5:7b-instruct"]
    c.complete_json("sys", "a", SCHEMA)
    assert c._client.requests[0]["model"] == "qwen2.5:7b-instruct" and c.model_source == "discovered"


def test_auto_model_fallback_when_models_endpoint_fails(tmp_path):
    c = make_client(tmp_path, lambda kw: reply('{"x": "ok"}'), llm_model="auto")

    def broken():
        raise BadRequestError("not implemented")

    c._client.models = SimpleNamespace(list=broken)
    assert c.model == "gpt-4o-mini" and c.model_source == "fallback"


def test_map_runs_in_parallel_and_keeps_order(tmp_path):
    c = make_client(tmp_path, lambda kw: reply("{}"), llm_concurrency=4)
    active, peak, lock = [0], [0], threading.Lock()

    def work(i):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        return i * 10

    assert c.map(work, range(8)) == [i * 10 for i in range(8)]
    assert peak[0] > 1


def test_status_is_safe_and_hides_key(tmp_path):
    c = make_client(tmp_path, lambda kw: reply("{}"), llm_api_key="test-secret-value", llm_model="auto")
    st = llm_status(c, probe=True, max_age=0)
    assert st["reachable"] is True and st["model"] == "qwen3:4b-instruct" and st["model_source"] == "discovered"
    assert st["provider"] == "openai-compatible" and st["base_host"] == "gw.example"
    assert "test-secret-value" not in json.dumps(st, ensure_ascii=False)
    off = llm_status(make_client(tmp_path / "off", lambda kw: reply("{}"), llm_enabled=False), probe=True)
    assert off["enabled"] is False and off["reachable"] is None


def test_empty_choices_is_a_clean_error(tmp_path):
    c = make_client(tmp_path, lambda kw: SimpleNamespace(choices=[], usage=None))
    with pytest.raises(LLMUnavailable, match="без choices"):
        c.complete_json("sys", "a", SCHEMA)
