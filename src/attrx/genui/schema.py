"""Словарь компонентов generative UI. LLM выбирает ТОЛЬКО из этого списка и не видит/не пишет HTML/JS.

Данные (значения, уверенность, цитаты) подставляет фронтенд из ExtractionResult по ссылкам,
поэтому модель не может «нарисовать» несуществующую характеристику.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

ComponentType = Literal["summary", "alert", "attribute_table", "confidence_chart", "evidence", "review_queue",
                        "conflicts", "metrics"]
Filter = Literal["all", "accepted", "review", "missing"]
Level = Literal["info", "success", "warning", "danger"]

# Что делает каждый компонент — общий источник для промпта LLM и документации (docs/GENUI.md).
COMPONENTS: dict[str, str] = {
    "summary": "сводка: сколько значений принято, на проверке и не найдено в тексте",
    "alert": "короткое сообщение (level: info|success|warning|danger), текст — в text",
    "attribute_table": "таблица характеристик с уверенностью, статусом, источниками и цитатой (filter, attributes)",
    "confidence_chart": "уверенность по характеристикам относительно порога автопринятия (filter, attributes)",
    "evidence": "цитаты из текста карточки, на которых основано каждое значение (attributes)",
    "review_queue": "значения на проверке с кнопками «Принять / Исправить / Отклонить» (filter)",
    "conflicts": "спорные значения: альтернативы от разных источников и отброшенные LLM-значения (attributes)",
    "metrics": "качество на эталоне: точность, полнота, F1, покрытие",
}


class Component(BaseModel):
    type: ComponentType
    title: str | None = Field(default=None, max_length=80)
    text: str | None = Field(default=None, max_length=300)
    filter: Filter | None = None
    attributes: list[str] | None = None
    level: Level | None = None

    @field_validator("title", "text")
    @classmethod
    def no_markup(cls, v: str | None) -> str | None:
        return None if v is None else v.replace("<", "‹").replace(">", "›")


class UISpec(BaseModel):
    version: Literal[1] = 1
    title: str = Field(max_length=100)
    layout: Literal["column", "grid"] = "column"
    components: list[Component] = Field(min_length=1, max_length=8)
    generated_by: Literal["rules", "llm"] = "rules"


# Плоская strict-совместимая JSON Schema для response_format (OpenAI / GigaChat / vLLM guided decoding)
COMPONENT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": list(ComponentType.__args__)},
        "title": {"type": ["string", "null"]},
        "text": {"type": ["string", "null"]},
        "filter": {"type": ["string", "null"], "enum": [*Filter.__args__, None]},
        "attributes": {"type": ["array", "null"], "items": {"type": "string"}},
        "level": {"type": ["string", "null"], "enum": [*Level.__args__, None]},
    },
    "required": ["type", "title", "text", "filter", "attributes", "level"],
    "additionalProperties": False,
}
UISPEC_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "layout": {"type": "string", "enum": ["column", "grid"]},
        "components": {"type": "array", "items": COMPONENT_JSON_SCHEMA},
    },
    "required": ["title", "layout", "components"],
    "additionalProperties": False,
}
