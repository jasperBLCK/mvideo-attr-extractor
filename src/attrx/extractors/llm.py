"""LLM-экстрактор: вызывается ТОЛЬКО для атрибутов, которые правила не закрыли уверенно.

Модель обязана вернуть дословную цитату (evidence). Значение без подтверждения в тексте отбрасывается
в verify.grounded — так мы не выдумываем характеристики.
"""
from __future__ import annotations

import json

from ..llm.client import LLMClient, LLMUnavailable
from ..normalize import normalize_value
from ..schema import Candidate, Card
from ..taxonomy import CategorySpec, Taxonomy

LLM_BASE_SCORE = 0.75

SYSTEM = (
    "Ты извлекаешь характеристики товара из карточки интернет-магазина электроники. "
    "Используй ТОЛЬКО информацию из карточки. Если характеристики нет в тексте — верни null. "
    "Ничего не додумывай по знаниям о модели. Для каждого значения верни evidence — "
    "ДОСЛОВНУЮ цитату из карточки (3–60 символов), из которой следует значение. Ответ — строго JSON."
)


def build_schema(attrs: list[str]) -> dict:
    item = {
        "type": "object",
        "properties": {"value": {"type": ["string", "null"]}, "evidence": {"type": ["string", "null"]}},
        "required": ["value", "evidence"],
        "additionalProperties": False,
    }
    return {"type": "object", "properties": {a: item for a in attrs}, "required": attrs, "additionalProperties": False}


def build_prompt(card: Card, category: CategorySpec, attrs: list[str], tax: Taxonomy) -> str:
    spec_lines = []
    for a in attrs:
        s = category.attributes[a]
        line = f"- {a}: {s.title}"
        if s.unit:
            line += f" (число, единица: {s.unit})"
        if s.dictionary and s.dictionary in tax.dictionaries and len(tax.dictionaries[s.dictionary].canon) <= 40:
            line += f"; допустимые значения: {', '.join(tax.dictionaries[s.dictionary].canon)}"
        spec_lines.append(line)
    card_json = json.dumps({"title": card.title, "description": card.description[:3000], "seller_attrs": card.seller_attrs},
                           ensure_ascii=False)
    return f"Категория: {category.title}\nНужно извлечь:\n" + "\n".join(spec_lines) + f"\n\nКарточка:\n{card_json}"


def extract(card: Card, category: CategorySpec, tax: Taxonomy, attrs: list[str], client: LLMClient) -> tuple[list[Candidate], list[str]]:
    if not attrs or not client.enabled:
        return [], []
    try:
        data = client.complete_json(SYSTEM, build_prompt(card, category, attrs, tax), build_schema(attrs), "attributes")
    except LLMUnavailable as e:
        return [], [f"llm: {e}"]
    out: list[Candidate] = []
    for a in attrs:
        item = data.get(a) or {}
        if not isinstance(item, dict) or not item.get("value"):
            continue
        res = normalize_value(str(item["value"]), category.attributes[a], tax.dictionaries)
        if res is None:
            continue
        value, quality = res
        out.append(Candidate(attribute=a, value=value, raw=str(item.get("evidence") or item["value"]), source="llm",
                             score=LLM_BASE_SCORE * quality))
    return out, []
