"""Настройки LLM из окружения: фолбэк на OPENAI_* (ВМ хакатона), LLM_ENABLED=auto, выбор провайдера и модели."""
import pytest

from attrx.config import Settings
from attrx.llm.client import LLMClient, choose_model, provider_of

LLM_VARS = ["LLM_ENABLED", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_LOCAL_URL", "LLM_CONCURRENCY",
            "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"]


@pytest.fixture
def env(monkeypatch):
    for v in LLM_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("LLM_ENABLED", "auto")
    return monkeypatch


def test_auto_without_any_config_is_disabled(env):
    s = Settings()
    assert (s.llm_mode, s.llm_source, s.llm_enabled) == ("auto", "default", False)
    assert s.llm_base_url == "http://127.0.0.1:11434/v1"


def test_vm_openai_env_enables_llm(env):
    env.setenv("OPENAI_API_KEY", "test-vm-key")
    env.setenv("OPENAI_BASE_URL", "https://llm.hackathon.example/v1/")
    s = Settings()
    assert s.llm_enabled and s.llm_source == "OPENAI_*"
    assert s.llm_base_url == "https://llm.hackathon.example/v1"
    assert s.llm_api_key == "test-vm-key"
    assert s.llm_model == "auto"  # модель организаторов неизвестна — выбирается из GET /models
    assert "test-vm-key" not in repr(s)  # ключ не попадает в логи через repr
    assert provider_of(s.llm_base_url, s.llm_source) == "organizers"


def test_llm_vars_override_openai_and_key_is_not_leaked(env):
    env.setenv("OPENAI_API_KEY", "test-vm-key")
    env.setenv("OPENAI_BASE_URL", "https://llm.hackathon.example/v1")
    env.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    env.setenv("LLM_MODEL", "qwen3:4b-instruct")
    s = Settings()
    assert s.llm_source == "LLM_*" and s.llm_base_url == "http://localhost:11434/v1"
    assert s.llm_api_key == "not-needed"  # ключ организаторов не уходит на чужой адрес
    assert s.llm_model == "qwen3:4b-instruct"


def test_explicit_off_wins_over_vm_key(env):
    env.setenv("OPENAI_API_KEY", "test-vm-key")
    env.setenv("LLM_ENABLED", "false")
    assert Settings().llm_enabled is False


def test_explicit_on_without_config_uses_local_ollama(env):
    env.setenv("LLM_ENABLED", "true")
    s = Settings()
    assert s.llm_enabled and s.llm_base_url == "http://127.0.0.1:11434/v1"


@pytest.mark.parametrize("key", ["changeme", "", "  "])
def test_placeholder_or_empty_key_does_not_enable(env, key):
    env.setenv("LLM_API_KEY", key)
    env.setenv("OPENAI_API_KEY", "")  # docker compose пробрасывает пустую строку, если на хосте ключа нет
    assert Settings().llm_enabled is False


def test_openai_key_without_url_goes_to_official_api(env):
    env.setenv("OPENAI_API_KEY", "test-openai-key")
    env.setenv("OPENAI_MODEL", "gpt-4o-mini")
    s = Settings()
    assert s.llm_base_url == "https://api.openai.com/v1" and s.llm_model == "gpt-4o-mini"


def test_local_url_enabled_only_if_reachable(env, tmp_path):
    env.setenv("LLM_LOCAL_URL", "http://127.0.0.1:9/v1")  # порт discard — никто не слушает
    s = Settings(cache_path=tmp_path / "c.sqlite")
    assert s.llm_source == "LLM_LOCAL_URL" and s.llm_enabled  # настроено ...
    assert LLMClient(s).enabled is False  # ... но сервер не запущен — работаем без LLM, без ошибок на карточках


def test_concurrency_defaults(env, tmp_path):
    assert LLMClient(Settings(llm_base_url="http://localhost:11434/v1", cache_path=tmp_path / "a")).concurrency == 1
    assert LLMClient(Settings(llm_base_url="https://gw.example/v1", cache_path=tmp_path / "b")).concurrency == 4
    env.setenv("LLM_CONCURRENCY", "3")
    assert LLMClient(Settings(cache_path=tmp_path / "c")).concurrency == 3


def test_provider_detection():
    assert provider_of("http://localhost:11434/v1") == "ollama"
    assert provider_of("http://ollama:11434/v1") == "ollama"
    assert provider_of("https://gigachat.devices.sberbank.ru/api/v1") == "gigachat"
    assert provider_of("https://llm.api.cloud.yandex.net/v1") == "yandexgpt"
    assert provider_of("https://api.openai.com/v1") == "openai"
    assert provider_of("http://10.0.0.5:8000/v1") == "on-prem"


def test_choose_model_prefers_list_and_skips_embeddings():
    models = ["text-embedding-3-small", "whisper-1", "mistral:latest", "qwen3:4b-instruct"]
    assert choose_model(models, "qwen3:4b-instruct,mistral") == "qwen3:4b-instruct"
    assert choose_model(["text-embedding-3-small", "my-llm"], "qwen") == "my-llm"
    assert choose_model(["text-embedding-3-small"], "qwen") is None


def test_localhost_is_dialed_as_ipv4():
    from attrx.llm.client import ipv4_loopback

    assert ipv4_loopback("http://localhost:11434/v1") == "http://127.0.0.1:11434/v1"
    assert ipv4_loopback("https://gw.example/v1") == "https://gw.example/v1"
