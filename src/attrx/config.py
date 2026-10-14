from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


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


@dataclass
class Settings:
    llm_enabled: bool = field(default_factory=lambda: _bool("LLM_ENABLED", False))
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "changeme"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "qwen2.5:7b-instruct"))
    llm_response_format: str = field(default_factory=lambda: os.getenv("LLM_RESPONSE_FORMAT", "json_schema"))
    llm_timeout: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT", "60")))
    llm_max_calls: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_CALLS", "5000")))
    confidence_threshold: float = field(default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.7")))
    llm_route_threshold: float = field(default_factory=lambda: float(os.getenv("LLM_ROUTE_THRESHOLD", "0.6")))
    ml_enabled: bool = field(default_factory=lambda: _bool("ML_ENABLED", True))
    ml_model_path: Path = field(default_factory=lambda: _path("ML_MODEL_PATH", "models/ner"))
    taxonomy_path: Path = field(default_factory=lambda: _path("TAXONOMY_PATH", "config/taxonomy.yaml"))
    dataset_config: Path = field(default_factory=lambda: _path("DATASET_CONFIG", "config/dataset.yaml"))
    cache_path: Path = field(default_factory=lambda: _path("CACHE_PATH", "outputs/llm_cache.sqlite"))
    feedback_path: Path = field(default_factory=lambda: _path("FEEDBACK_PATH", "outputs/feedback.jsonl"))
    predictions_path: Path = field(default_factory=lambda: _path("PREDICTIONS_PATH", "outputs/predictions.jsonl"))


def get_settings() -> Settings:
    return Settings()
