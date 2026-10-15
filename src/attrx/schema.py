from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

Source = Literal["seller", "rules", "ml", "llm", "human"]
Status = Literal["accepted", "review", "missing"]


def card_auto_id(title: Any = "", description: Any = "", seller_attrs: Any = None) -> str:
    """Id карточки, если клиент его не передал: детерминированный хэш текста («auto-» + 12 hex).
    Одна и та же карточка получает тот же id — решения модератора по ней не расходятся между запросами."""
    raw = json.dumps([str(title or ""), str(description or ""), seller_attrs or {}], ensure_ascii=False,
                     sort_keys=True, default=str)
    return "auto-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


class Card(BaseModel):
    """Текстовая карточка товара — единственный вход решения.

    id необязателен: без него (или при пустом) генерируется card_auto_id() по тексту; числовой id приводится к строке."""

    id: str = ""
    title: str = ""
    description: str = ""
    seller_attrs: dict[str, Any] = Field(default_factory=dict)
    category: str | None = None
    category_id: str | None = None  # id категории в дереве каталога (YML), если известен
    seller: str | None = None  # продавец (М.Видео или партнёр маркетплейса) — для сегментов справедливости

    @model_validator(mode="before")
    @classmethod
    def _fill_id(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        cid = data.get("id")
        if isinstance(cid, int | float) and not isinstance(cid, bool):  # id из PIM часто числовой
            return {**data, "id": str(int(cid)) if float(cid).is_integer() else str(cid)}
        if cid is None or (isinstance(cid, str) and not cid.strip()):
            return {**data, "id": card_auto_id(data.get("title"), data.get("description"), data.get("seller_attrs"))}
        return data

    def full_text(self) -> str:
        seller = "; ".join(f"{k}: {v}" for k, v in self.seller_attrs.items())
        return "\n".join(x for x in (self.title, self.description, seller) if x)


class Candidate(BaseModel):
    """Сырое предложение значения от одного из экстракторов."""

    attribute: str
    value: str | float
    raw: str
    source: Source
    score: float
    span: tuple[int, int] | None = None


class AttributeValue(BaseModel):
    attribute: str
    title: str
    value: str | float | None
    unit: str | None = None
    display: str | None = None
    confidence: float
    status: Status
    sources: list[Source] = Field(default_factory=list)
    evidence: str | None = None
    # [start, end) цитаты в Card.full_text() — для подсветки в UI; None, если цитату не удалось локализовать
    evidence_span: tuple[int, int] | None = None
    # проигравшие кандидаты при конфликте: [{"value": ..., "confidence": ..., "sources": [...]}]
    alternatives: list[dict[str, Any]] = Field(default_factory=list)
    required: bool = True  # обязательный атрибут категории
    traffic_weight: float = 1.0  # вес атрибута по поисковому спросу (query_clicks)


class ExtractionResult(BaseModel):
    card_id: str
    category: str
    category_confidence: float
    attributes: list[AttributeValue]
    llm_used: bool = False
    warnings: list[str] = Field(default_factory=list)
    category_title: str | None = None
    category_path: list[str] = Field(default_factory=list)  # путь в дереве каталога: ["Смартфоны и гаджеты", "Смартфоны"]
    timings_ms: dict[str, float] = Field(default_factory=dict)  # время по этапам: {"rules": 0.4, "llm": 820.0}

    def as_flat(self) -> dict[str, Any]:
        return {a.attribute: a.value for a in self.attributes if a.status == "accepted"}
