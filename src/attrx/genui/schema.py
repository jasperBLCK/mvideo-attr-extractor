"""Словарь компонентов generative UI. LLM выбирает ТОЛЬКО из этого списка и не видит/не пишет HTML/JS.

Данные (значения, уверенность, цитаты) подставляет фронтенд из ExtractionResult по ссылкам,
поэтому модель не может «нарисовать» несуществующую характеристику.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

ComponentType = Literal[
    "summary", "alert", "kpi", "attribute_table", "attribute_cards", "confidence_chart", "evidence", "highlight",
    "conflicts", "source_breakdown", "review_queue", "checklist", "metrics",
]
Filter = Literal["all", "accepted", "review", "missing", "low_confidence"]
Level = Literal["info", "success", "warning", "danger"]
Sort = Literal["confidence_asc", "confidence_desc", "name"]
Variant = Literal["compact", "detailed"]

# Описание для промпта и README — единственный источник правды о том, что умеет каждый компонент
CATALOG: dict[str, str] = {
    "summary": "категория, уверенность в ней и счётчики принято/проверить/нет",
    "alert": "цветная плашка с коротким текстом (level: info|success|warning|danger)",
    "kpi": "крупные цифры: заполнено, на проверку, нет в тексте, средняя уверенность",
    "attribute_table": "таблица «характеристика — значение» как на странице товара",
    "attribute_cards": "плитки по одной характеристике: значение, статус, цитата",
    "confidence_chart": "горизонтальные бары уверенности по атрибутам",
    "evidence": "список цитат из текста, откуда взято каждое значение",
    "highlight": "исходный текст карточки с подсвеченными цитатами",
    "conflicts": "расхождения между источниками (продавец vs текст) и альтернативы",
    "source_breakdown": "сколько значений дали продавец / правила / ML / LLM",
    "review_queue": "очередь модератора: поле ввода и кнопка «Подтвердить» для спорных значений",
    "checklist": "чеклист модератора по спорным и отсутствующим характеристикам",
    "metrics": "ссылка на сводные метрики качества",
}


class Component(BaseModel):
    type: ComponentType
    title: str | None = Field(default=None, max_length=80)
    text: str | None = Field(default=None, max_length=300)
    filter: Filter | None = None
    attributes: list[str] | None = None
    level: Level | None = None
    sort: Sort | None = None
    limit: int | None = Field(default=None, ge=1, le=50)
    span: Literal[1, 2] | None = None
    variant: Variant | None = None

    @field_validator("title", "text")
    @classmethod
    def no_markup(cls, v: str | None) -> str | None:
        return None if v is None else v.replace("<", "‹").replace(">", "›")


class UISpec(BaseModel):
    version: Literal[1] = 1
    title: str = Field(max_length=100)
    subtitle: str | None = Field(default=None, max_length=160)
    layout: Literal["column", "grid"] = "column"
    components: list[Component] = Field(min_length=1, max_length=8)
    generated_by: Literal["rules", "llm"] = "rules"

    @field_validator("title", "subtitle")
    @classmethod
    def no_markup(cls, v: str | None) -> str | None:
        return None if v is None else v.replace("<", "‹").replace(">", "›")


def _nullable(t: str, **extra) -> dict:
    return {"type": [t, "null"], **extra}


# Плоская strict-совместимая JSON Schema для response_format (OpenAI / Groq / GigaChat / vLLM guided decoding)
COMPONENT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": list(ComponentType.__args__)},
        "title": _nullable("string"),
        "text": _nullable("string"),
        "filter": _nullable("string", enum=[*Filter.__args__, None]),
        "attributes": _nullable("array", items={"type": "string"}),
        "level": _nullable("string", enum=[*Level.__args__, None]),
        "sort": _nullable("string", enum=[*Sort.__args__, None]),
        "limit": _nullable("integer"),
        "span": _nullable("integer", enum=[1, 2, None]),
        "variant": _nullable("string", enum=[*Variant.__args__, None]),
    },
    "required": ["type", "title", "text", "filter", "attributes", "level", "sort", "limit", "span", "variant"],
    "additionalProperties": False,
}
UISPEC_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "subtitle": _nullable("string"),
        "layout": {"type": "string", "enum": ["column", "grid"]},
        "components": {"type": "array", "items": COMPONENT_JSON_SCHEMA},
    },
    "required": ["title", "subtitle", "layout", "components"],
    "additionalProperties": False,
}
