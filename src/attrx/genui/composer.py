"""Генерация интерфейса по данным: детерминированный шаблон + LLM-композиция под намерение пользователя.

Идея как в Agentic OS (UI собирается на лету под задачу), но безопасно для прода:
LLM возвращает декларативную спецификацию, мы её валидируем; при любой ошибке — детерминированный фолбэк.
"""
from __future__ import annotations

import json

from pydantic import ValidationError

from ..llm.client import LLMClient, LLMUnavailable
from ..schema import ExtractionResult
from .schema import UISPEC_JSON_SCHEMA, Component, UISpec

SYSTEM = (
    "Ты проектируешь интерфейс для модератора каталога М.Видео. По результату извлечения характеристик "
    "и запросу пользователя собери экран из компонентов: summary, alert, attribute_table (filter: all|accepted|review|missing), "
    "confidence_chart, evidence (attributes: список ключей), review_queue, metrics. "
    "Не пересказывай значения характеристик в text — данные подставит интерфейс. "
    "Используй только ключи атрибутов из результата. Не больше 6 компонентов. Ответ — строго JSON."
)


def compose_default(r: ExtractionResult) -> UISpec:
    n_review = sum(a.status == "review" for a in r.attributes)
    n_missing = sum(a.status == "missing" for a in r.attributes)
    comps: list[Component] = []
    if n_review or n_missing or r.warnings:
        comps.append(Component(type="alert", level="warning",
                               text=f"На проверку: {n_review}, не найдено в тексте: {n_missing}"))
    else:
        comps.append(Component(type="alert", level="success", text="Все обязательные характеристики заполнены уверенно"))
    comps.append(Component(type="summary"))
    comps.append(Component(type="attribute_table", title="Характеристики", filter="all"))
    if n_review:
        comps.append(Component(type="review_queue", title="Требует подтверждения"))
    comps.append(Component(type="confidence_chart", title="Уверенность модели"))
    return UISpec(title=f"Карточка {r.card_id}", layout="column", components=comps, generated_by="rules")


def sanitize(spec: UISpec, r: ExtractionResult) -> UISpec:
    known = {a.attribute for a in r.attributes}
    for c in spec.components:
        if c.attributes is not None:
            c.attributes = [a for a in c.attributes if a in known] or None
    return spec


def compose_llm(r: ExtractionResult, intent: str, client: LLMClient) -> tuple[UISpec, str | None]:
    """Возвращает (spec, предупреждение). Без LLM или при невалидном ответе — compose_default."""
    if not client.enabled:
        return compose_default(r), "LLM выключена — показан шаблонный интерфейс"
    payload = {
        "intent": intent[:500],
        "category": r.category,
        "attributes": [{"key": a.attribute, "title": a.title, "status": a.status, "confidence": a.confidence} for a in r.attributes],
        "warnings": r.warnings[:10],
    }
    try:
        data = client.complete_json(SYSTEM, json.dumps(payload, ensure_ascii=False), UISPEC_JSON_SCHEMA, "ui_spec")
        spec = UISpec.model_validate({**data, "generated_by": "llm"})
        return sanitize(spec, r), None
    except LLMUnavailable as e:
        return compose_default(r), f"LLM недоступна, показан шаблон: {str(e)[:160]}"
    except (ValidationError, TypeError, ValueError) as e:
        return compose_default(r), f"LLM вернула невалидный UI, показан шаблон: {str(e)[:120]}"
