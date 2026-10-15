from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _find_root() -> Path:
    """Корень проекта (config/, web/, data/): ATTRX_ROOT, иначе репо рядом с пакетом, иначе текущая папка."""
    if env := os.getenv("ATTRX_ROOT"):
        return Path(env).resolve()
    pkg = Path(__file__).resolve().parents[2]
    for cand in (pkg, Path.cwd()):
        if (cand / "config").is_dir() and (cand / "web").is_dir():
            return cand
    return pkg


ROOT = _find_root()


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(ROOT / ".env")


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: str) -> Path:
    p = Path(os.getenv(name, default))
    return p if p.is_absolute() else ROOT / p


# ---------------------------------------------------------------- LLM ---------------------------------------------
# Источники настроек по приоритету:
#   1) LLM_BASE_URL / LLM_API_KEY / LLM_MODEL      — явная настройка (GigaChat, YandexGPT, vLLM, Ollama ...);
#   2) OPENAI_BASE_URL / OPENAI_API_KEY             — так ключ и шлюз организаторов заданы на ВМ хакатона;
#   3) LLM_LOCAL_URL                                — «попробовать локальную LLM, если она запущена» (docker compose);
#   4) ничего                                       — LLM выключена, всё работает на правилах и NER.
# LLM_ENABLED=auto (по умолчанию) включает LLM, только если есть 1–3. Пустая переменная = не задана.
DEFAULT_LOCAL_URL = "http://127.0.0.1:11434/v1"  # Ollama (IPv4: см. llm.client.ipv4_loopback)
OPENAI_DEFAULT_URL = "https://api.openai.com/v1"
# модель по умолчанию для локальной Ollama — выбрана бенчмарком (docs/LLM.md)
DEFAULT_LOCAL_MODEL = "qwen3:4b-instruct"
# порядок выбора модели при LLM_MODEL=auto из GET {base}/models (подстроки id, без учёта регистра):
# сначала проверенные замером локальные, затем быстрые «mini» облачные (дешевле для дневного лимита), затем семейства
DEFAULT_MODEL_PREFERENCE = (
    "qwen3:4b-instruct,qwen2.5:7b-instruct,gpt-4.1-mini,gpt-4o-mini,gpt-5-mini,"
    "qwen3,qwen2.5,gigachat,yandexgpt,gpt-4o,gpt-4.1,deepseek,llama,mistral,gemma"
)
_PLACEHOLDER_KEYS = {"changeme", "change-me", "changeit", "your-key", "your-api-key", "sk-xxx", "none", "null",
                     "dummy", "ollama", "empty", "not-needed", "-"}


def _env(*names: str) -> str | None:
    """Первое непустое значение из переменных окружения (пустая строка считается «не задано»)."""
    for name in names:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return None


def _real_key(key: str | None) -> bool:
    return bool(key) and key.strip().lower() not in _PLACEHOLDER_KEYS


def llm_source() -> str:
    """Откуда взята настройка LLM: 'LLM_*', 'OPENAI_*' (ВМ), 'LLM_LOCAL_URL' или 'default' (ничего не задано)."""
    if _env("LLM_BASE_URL") or _real_key(_env("LLM_API_KEY")):
        return "LLM_*"
    if _env("OPENAI_BASE_URL") or _real_key(_env("OPENAI_API_KEY")):
        return "OPENAI_*"
    if _env("LLM_LOCAL_URL"):
        return "LLM_LOCAL_URL"
    return "default"


def _llm_mode() -> str:
    raw = (os.getenv("LLM_ENABLED") or "auto").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return "on"
    if raw in {"0", "false", "no", "off"}:
        return "off"
    return "auto"


def _llm_enabled() -> bool:
    mode = _llm_mode()
    return mode == "on" or (mode == "auto" and llm_source() != "default")


def _llm_base_url() -> str:
    if url := _env("LLM_BASE_URL", "OPENAI_BASE_URL"):
        return url.rstrip("/")
    if _real_key(_env("OPENAI_API_KEY")):
        return OPENAI_DEFAULT_URL  # ключ OpenAI без адреса — официальный API
    return (_env("LLM_LOCAL_URL") or DEFAULT_LOCAL_URL).rstrip("/")


def _llm_api_key() -> str:
    # ключ OPENAI_API_KEY не отправляем на чужой LLM_BASE_URL — только в паре с OPENAI_BASE_URL/официальным API
    key = _env("LLM_API_KEY") or (None if _env("LLM_BASE_URL") else _env("OPENAI_API_KEY"))
    return key or "not-needed"  # Ollama/vLLM ключ не проверяют, а клиенту openai нужна непустая строка


@dataclass
class Settings:
    # True — LLM будет вызываться для хвоста; режим и источник — в llm_mode / llm_source
    llm_enabled: bool = field(default_factory=_llm_enabled)
    llm_mode: str = field(default_factory=_llm_mode)  # auto | on | off  (LLM_ENABLED)
    llm_source: str = field(default_factory=llm_source)
    llm_base_url: str = field(default_factory=_llm_base_url)
    llm_api_key: str = field(default_factory=_llm_api_key, repr=False)
    # auto — первая подходящая модель из GET {base}/models (порядок — LLM_MODEL_PREFERENCE), иначе фолбэк
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "OPENAI_MODEL") or "auto")
    llm_model_preference: str = field(default_factory=lambda: _env("LLM_MODEL_PREFERENCE") or DEFAULT_MODEL_PREFERENCE)
    llm_response_format: str = field(default_factory=lambda: os.getenv("LLM_RESPONSE_FORMAT", "json_schema"))
    llm_timeout: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT", "60")))
    llm_max_retries: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "0")))
    llm_reasoning_effort: str = field(default_factory=lambda: os.getenv("LLM_REASONING_EFFORT", ""))
    llm_max_calls: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_CALLS", "5000")))
    # 0 — без суточного лимита; у ключа организаторов лимит дневной — счётчик по дням хранится в кэше
    llm_daily_max_calls: int = field(default_factory=lambda: int(os.getenv("LLM_DAILY_MAX_CALLS", "0")))
    # 0 — авто: 1 для локальной Ollama (одна GPU), 4 для удалённых API
    llm_concurrency: int = field(default_factory=lambda: int(os.getenv("LLM_CONCURRENCY", "0")))
    llm_max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "700")))
    llm_max_input_chars: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_INPUT_CHARS", "3500")))
    llm_enum_values_max: int = field(default_factory=lambda: int(os.getenv("LLM_ENUM_VALUES_MAX", "60")))
    llm_few_shot: bool = field(default_factory=lambda: _bool("LLM_FEW_SHOT", False))
    # true — зашить справочник в json_schema как enum. По замеру хуже: модель вынуждена выбрать значение из списка,
    # даже если в тексте его нет (POCO → «OPPO»); по умолчанию модель пишет значение как в тексте, а к справочнику
    # его приводит детерминированная нормализация (неизвестное — в заметки для пополнения справочника)
    llm_schema_enum: bool = field(default_factory=lambda: _bool("LLM_SCHEMA_ENUM", False))
    # не спрашивать LLM об атрибутах, о которых в тексте нет ни слова (ни синонима, ни единицы, ни значения
    # из справочника): заземлённо извлечь их невозможно, а вызов стоит денег
    llm_prefilter: bool = field(default_factory=lambda: _bool("LLM_PREFILTER", True))
    llm_cache: bool = field(default_factory=lambda: _bool("LLM_CACHE", True))
    # после ошибки соединения/таймаута не дёргать LLM N секунд (иначе каждая карточка ждёт таймаут)
    llm_retry_after: float = field(default_factory=lambda: float(os.getenv("LLM_RETRY_AFTER", "30")))
    llm_prices_path: Path = field(default_factory=lambda: _path("LLM_PRICES_PATH", "config/llm_prices.yaml"))
    # порог автопринятия: CONFIDENCE_THRESHOLD, иначе выбранный на dev при калибровке (config/calibration.json), иначе 0.7
    confidence_threshold: float = field(default_factory=lambda: _default_threshold())
    calibration_path: Path = field(default_factory=lambda: _path("CALIBRATION_PATH", "config/calibration.json"))
    llm_route_threshold: float = field(default_factory=lambda: float(os.getenv("LLM_ROUTE_THRESHOLD", "0.6")))
    ml_enabled: bool = field(default_factory=lambda: _bool("ML_ENABLED", True))
    ml_model_path: Path = field(default_factory=lambda: _path("ML_MODEL_PATH", "models/ner"))
    taxonomy_path: Path = field(default_factory=lambda: _path("TAXONOMY_PATH", "config/taxonomy.yaml"))
    dataset_config: Path = field(default_factory=lambda: _path("DATASET_CONFIG", "config/dataset.yaml"))
    cache_path: Path = field(default_factory=lambda: _path("CACHE_PATH", "outputs/llm_cache.sqlite"))
    feedback_path: Path = field(default_factory=lambda: _path("FEEDBACK_PATH", "outputs/feedback.jsonl"))
    predictions_path: Path = field(default_factory=lambda: _path("PREDICTIONS_PATH", "outputs/predictions.jsonl"))


def _default_threshold() -> float:
    if env := _env("CONFIDENCE_THRESHOLD"):
        return float(env)
    p = _path("CALIBRATION_PATH", "config/calibration.json")
    try:
        import json

        t = json.loads(p.read_text(encoding="utf-8")).get("threshold") if p.exists() else None
    except (OSError, ValueError):
        t = None
    return float(t) if t is not None else 0.7


def get_settings() -> Settings:
    return Settings()
