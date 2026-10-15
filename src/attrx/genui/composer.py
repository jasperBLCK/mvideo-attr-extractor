"""Генерация интерфейса по данным: детерминированный шаблон + LLM-композиция под намерение пользователя.

Идея как в Agentic OS (UI собирается на лету под задачу), но безопасно для прода:
LLM возвращает декларативную спецификацию, мы её валидируем; при любой ошибке — детерминированный фолбэк.
Без LLM намерение разбирается по ключевым словам, поэтому экран меняется от запроса и в офлайне.
"""
from __future__ import annotations

import json
import re

from pydantic import ValidationError

from ..llm.client import LLMClient, LLMUnavailable
from ..schema import ExtractionResult
from .schema import CATALOG, UISPEC_JSON_SCHEMA, Component, UISpec

SYSTEM = (
    "Ты продуктовый дизайнер: собираешь экран для сотрудника каталога М.Видео под его запрос. "
    "Экран — JSON из компонентов (только из каталога ниже). Данные подставит интерфейс, поэтому в text пиши "
    "короткие подводки или выводы, а не значения характеристик.\n\n"
    "Каталог компонентов:\n" + "\n".join(f"- {k}: {v}" for k, v in CATALOG.items()) + "\n\n"
    "Параметры: filter (all|accepted|review|missing|low_confidence — какие атрибуты показать), "
    "attributes (ключи атрибутов из результата, чтобы сузить), sort (confidence_asc|confidence_desc|name), "
    "limit (сколько строк), span (2 — на всю ширину в layout=grid), variant (compact|detailed), "
    "level для alert.\n\n"
    "Правила композиции:\n"
    "- Экран должен отвечать именно на запрос: разные запросы — разный набор, порядок и раскладка компонентов. "
    "Не добавляй всё подряд; 2–5 компонентов обычно достаточно.\n"
    "- Модератору: сомнительное наверх (sort=confidence_asc, filter=review или low_confidence), цитаты (evidence или highlight), "
    "review_queue или checklist для действий.\n"
    "- Менеджеру: kpi, source_breakdown, короткий alert с выводом, без таблиц целиком.\n"
    "- Про доверие/уверенность: confidence_chart с сортировкой, alert с интерпретацией.\n"
    "- Про конфликты/расхождения: conflicts, затем highlight по спорным атрибутам.\n"
    "- Про полный обзор: attribute_table или attribute_cards, layout=grid.\n"
    "- title экрана и заголовки компонентов — по-русски, по делу, без слов «JSON» и «компонент».\n"
    "- Используй только ключи атрибутов из результата. Ответ — строго JSON."
)

_TOPIC_RULES: list[tuple[str, str]] = [
    ("manager", r"менеджер|сводк|руководител|отчёт|отчет|kpi|сколько"),
    ("conflicts", r"конфликт|расхожд|продав[ец|ца]|противореч|сомнит"),
    ("evidence", r"цитат|откуда|доказ|подтвержд|источник|текст"),
    ("confidence", r"уверен|довер|график|диаграм"),
    ("review", r"модератор|провер|очеред|подтверд|чеклист|исправ"),
    ("missing", r"нет в тексте|отсутств|пуст|не найден|чего нет"),
    ("overview", r"все|всё|таблиц|полн|характеристик|обзор|карточк"),
]


def detect_topics(intent: str | None) -> list[str]:
    if not intent:
        return []
    low = intent.lower()
    return [t for t, rx in _TOPIC_RULES if re.search(rx, low)]


def compose_default(r: ExtractionResult, intent: str | None = None) -> UISpec:
    n_review = sum(a.status == "review" for a in r.attributes)
    n_missing = sum(a.status == "missing" for a in r.attributes)
    topics = detect_topics(intent)
    comps: list[Component] = []
    title, subtitle, layout = f"Карточка {r.card_id}", None, "column"

    if "manager" in topics:
        title, subtitle, layout = "Сводка для менеджера", f"Категория: {r.category}", "grid"
        comps += [Component(type="kpi", span=2),
                  Component(type="alert", span=2, level="warning" if n_review or n_missing else "success",
                            text=(f"Нужно внимание модератора: {n_review} на проверку, {n_missing} не найдено в тексте"
                                  if n_review or n_missing else "Карточка готова к публикации без ручной проверки")),
                  Component(type="source_breakdown", title="Откуда взяты значения"),
                  Component(type="confidence_chart", title="Самые слабые места", sort="confidence_asc", limit=5)]
    elif "conflicts" in topics:
        title, subtitle = "Расхождения источников", "Продавец против текста карточки"
        comps += [Component(type="conflicts", title="Конфликты и альтернативы"),
                  Component(type="highlight", title="Где это в тексте", filter="low_confidence"),
                  Component(type="review_queue", title="Решение модератора", sort="confidence_asc")]
    elif "review" in topics or "evidence" in topics:
        title, subtitle, layout = "Экран модератора", "Сомнительное сверху, с цитатами", "grid"
        n_bad = n_review + n_missing
        comps += [Component(type="alert", span=2, level="warning" if n_bad else "success",
                            text=f"Требуют решения: {n_bad} из {len(r.attributes)} характеристик" if n_bad
                            else "Спорных значений нет — можно только сверить цитаты")]
        if "review" in topics:
            comps += [Component(type="review_queue", title="Очередь проверки", sort="confidence_asc"),
                      Component(type="evidence", title="Цитаты из текста", sort="confidence_asc")]
        else:
            comps += [Component(type="highlight", title="Текст карточки с цитатами", span=2),
                      Component(type="evidence", title="Откуда взято каждое значение", sort="confidence_asc", span=2)]
        if "missing" in topics or n_missing:
            comps.append(Component(type="checklist", title="Чего нет в тексте", filter="missing"))
    elif "confidence" in topics:
        title, subtitle = "Уверенность модели", "Чем ниже бар — тем нужнее человек"
        comps += [Component(type="confidence_chart", title="По атрибутам", sort="confidence_asc"),
                  Component(type="alert", level="info",
                            text="Порог принятия задаётся CONFIDENCE_THRESHOLD; всё ниже уходит модератору"),
                  Component(type="attribute_table", title="Ниже порога", filter="low_confidence", variant="compact")]
    elif "missing" in topics:
        title = "Пробелы в карточке"
        comps += [Component(type="checklist", title="Не найдено в тексте", filter="missing"),
                  Component(type="attribute_table", title="Что уже заполнено", filter="accepted", variant="compact")]
    elif "overview" in topics:
        title, layout = f"Все характеристики · {r.card_id}", "grid"
        comps += [Component(type="summary", span=2), Component(type="attribute_cards", title="Характеристики", span=2)]
    else:
        if n_review or n_missing or r.warnings:
            comps.append(Component(type="alert", level="warning", text=f"На проверку: {n_review}, не найдено в тексте: {n_missing}"))
        else:
            comps.append(Component(type="alert", level="success", text="Все обязательные характеристики заполнены уверенно"))
        comps.append(Component(type="summary"))
        comps.append(Component(type="attribute_table", title="Характеристики", filter="all"))
        if n_review:
            comps.append(Component(type="review_queue", title="Требует подтверждения"))
        comps.append(Component(type="confidence_chart", title="Уверенность модели"))
    return UISpec(title=title, subtitle=subtitle, layout=layout, components=comps, generated_by="rules")


def sanitize(spec: UISpec, r: ExtractionResult) -> UISpec:
    known = {a.attribute for a in r.attributes}
    for c in spec.components:
        if c.attributes is not None:
            c.attributes = [a for a in c.attributes if a in known] or None
    return spec


def compose_llm(r: ExtractionResult, intent: str, client: LLMClient) -> tuple[UISpec, str | None]:
    """Возвращает (spec, предупреждение). Без LLM или при невалидном ответе — compose_default(intent)."""
    if not client.enabled:
        return compose_default(r, intent), "LLM выключена — экран собран по ключевым словам запроса"
    payload = {
        "intent": intent[:500],
        "category": r.category,
        "attributes": [{"key": a.attribute, "title": a.title, "status": a.status, "confidence": a.confidence,
                        "sources": a.sources} for a in r.attributes],
        "warnings": r.warnings[:10],
    }
    try:
        data = client.complete_json(SYSTEM, json.dumps(payload, ensure_ascii=False), UISPEC_JSON_SCHEMA, "ui_spec")
        spec = UISpec.model_validate({**data, "generated_by": "llm"})
        return sanitize(spec, r), None
    except LLMUnavailable as e:
        return compose_default(r, intent), f"LLM недоступна, показан шаблон: {str(e)[:160]}"
    except (ValidationError, TypeError, ValueError) as e:
        return compose_default(r, intent), f"LLM вернула невалидный UI, показан шаблон: {str(e)[:120]}"
