"""Генерация интерфейса по данным: шаблон, подбор по ключевым словам запроса и LLM-композиция под намерение.

Идея как в Agentic OS (UI собирается на лету под задачу), но безопасно для прода:
LLM возвращает декларативную спецификацию, мы её валидируем; при любой ошибке — детерминированный фолбэк.
Без LLM запрос всё равно учитывается: compose_intent подбирает компоненты по ключевым словам.
"""
from __future__ import annotations

import json

from pydantic import ValidationError

from ..llm.client import LLMClient, LLMUnavailable
from ..schema import ExtractionResult
from .schema import COMPONENTS, UISPEC_JSON_SCHEMA, Component, UISpec

SYSTEM = (
    "Ты проектируешь экран для модератора каталога М.Видео. По результату извлечения характеристик "
    "и запросу пользователя собери экран только из этих компонентов: "
    + "; ".join(f"{k} — {v}" for k, v in COMPONENTS.items())
    + ". filter: all|accepted|review|missing. В attributes — только ключи атрибутов из результата. "
    "Не пересказывай значения характеристик в text — данные подставит интерфейс. "
    "Не больше 6 компонентов. Ответ — строго JSON."
)


def _norm(s: str) -> str:
    return " ".join(s.lower().replace("ё", "е").split())


def _disputed(r: ExtractionResult) -> list[str]:
    """Атрибуты с альтернативами или конфликтом источников."""
    keys = [a.attribute for a in r.attributes if a.alternatives]
    keys += [w.split(":", 1)[0] for w in r.warnings if "конфликт" in w]
    known = {a.attribute for a in r.attributes}
    return [k for k in dict.fromkeys(keys) if k in known]


def _has_rejected(r: ExtractionResult) -> bool:
    return any("отброшено" in w for w in r.warnings)


def _category_uncertain(r: ExtractionResult) -> bool:
    """Пайплайн отправил карточку на проверку из-за категории (порог классификатора, models/category/meta.json;
    у ключевых слов — 0.6): предупреждение пишет pipeline.Extractor, своего порога здесь нет."""
    return any(w.startswith("категория определена неуверенно") for w in r.warnings)


def compose_default(r: ExtractionResult) -> UISpec:
    comps: list[Component] = []
    if _category_uncertain(r):
        comps.append(Component(type="alert", level="warning",
                               text="Категория определена неуверенно: проверьте, к какой категории относится товар"))
    comps.append(Component(type="summary"))
    comps.append(Component(type="attribute_table", title="Характеристики", filter="all"))
    if _disputed(r) or _has_rejected(r):
        comps.append(Component(type="conflicts", title="Спорные и отброшенные значения"))
    return UISpec(title=f"Карточка {r.card_id}", layout="column", components=comps, generated_by="rules")


# Ключевые слова запроса -> компоненты. Порядок списка = порядок на экране.
_INTENTS: list[tuple[tuple[str, ...], str]] = [
    (("сводк", "кратк", "итог", "обзор", "менеджер", "главное"), "summary"),
    (("провер", "сомнит", "модерац", "ручн", "review"), "review"),
    (("конфликт", "спорн", "расход", "альтернатив", "отброш", "галлюц", "противореч", "источник"), "conflicts"),
    (("цитат", "доказ", "откуда", "обоснов", "evidence", "подсвет"), "evidence"),
    (("уверен", "вероятн", "график", "диаграм", "confidence", "порог"), "confidence"),
    (("пропуск", "не найд", "нет в тексте", "незаполн", "пуст", "missing"), "missing"),
    (("принят", "готов", "надежн", "accepted"), "accepted"),
    (("метрик", "качеств", "точност", "полнот", "f1", "эталон"), "metrics"),
]


def compose_intent(r: ExtractionResult, intent: str) -> UISpec:
    """Детерминированная сборка экрана по ключевым словам запроса (работает без LLM)."""
    t = _norm(intent)
    wanted = [kind for stems, kind in _INTENTS if any(s in t for s in stems)]
    if not wanted:
        spec = compose_default(r)
        return spec.model_copy(update={"title": intent.strip()[:100] or spec.title})
    review = [a.attribute for a in r.attributes if a.status == "review"]
    comps: list[Component] = []
    for kind in wanted:
        if kind == "summary":
            comps.append(Component(type="summary"))
        elif kind == "review":
            if review:
                comps.append(Component(type="review_queue", title="Решения по значениям на проверке", filter="review"))
                if "evidence" not in wanted:
                    comps.append(Component(type="evidence", title="Откуда взяты значения", attributes=review))
            else:
                comps.append(Component(type="alert", level="success",
                                       text="Проверять нечего: все найденные значения приняты автоматически"))
        elif kind == "conflicts":
            if _disputed(r) or _has_rejected(r):
                comps.append(Component(type="conflicts", title="Спорные и отброшенные значения"))
            else:
                comps.append(Component(type="alert", level="info",
                                       text="Источники не спорят: альтернатив и отброшенных значений нет"))
        elif kind == "evidence":
            only = review if "review" in wanted and review else None
            comps.append(Component(type="evidence", title="Откуда взяты значения", attributes=only))
        elif kind == "confidence":
            comps.append(Component(type="confidence_chart", title="Уверенность относительно порога", filter="all"))
        elif kind == "missing":
            comps.append(Component(type="attribute_table", title="Не найдено в тексте", filter="missing"))
        elif kind == "accepted":
            comps.append(Component(type="attribute_table", title="Принятые значения", filter="accepted"))
        elif kind == "metrics":
            comps.append(Component(type="metrics", title="Качество на эталоне"))
    return UISpec(title=intent.strip()[:100], layout="column", components=comps[:6], generated_by="rules")


def sanitize(spec: UISpec, r: ExtractionResult) -> UISpec:
    known = {a.attribute for a in r.attributes}
    for c in spec.components:
        if c.attributes is not None:
            c.attributes = [a for a in c.attributes if a in known] or None
    return spec


def compose_llm(r: ExtractionResult, intent: str, client: LLMClient) -> tuple[UISpec, str | None]:
    """Возвращает (spec, предупреждение). Без LLM или при невалидном ответе — сборка по ключевым словам."""
    if not client.enabled:
        return compose_intent(r, intent), "LLM не подключена: экран собран по ключевым словам запроса"
    payload = {
        "intent": intent[:500],
        "category": r.category,
        "attributes": [{"key": a.attribute, "title": a.title, "status": a.status, "confidence": a.confidence,
                        "alternatives": len(a.alternatives)} for a in r.attributes],
        "warnings": r.warnings[:10],
    }
    try:
        data = client.complete_json(SYSTEM, json.dumps(payload, ensure_ascii=False), UISPEC_JSON_SCHEMA, "ui_spec")
        spec = UISpec.model_validate({**data, "generated_by": "llm"})
        return sanitize(spec, r), None
    except LLMUnavailable as e:
        return compose_intent(r, intent), f"LLM недоступна, экран собран по ключевым словам: {str(e)[:160]}"
    except (ValidationError, TypeError, ValueError) as e:
        return compose_intent(r, intent), f"LLM вернула невалидный экран, собран по ключевым словам: {str(e)[:120]}"
