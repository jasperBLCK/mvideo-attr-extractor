from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Source = Literal["seller", "rules", "ml", "llm", "human"]
Status = Literal["accepted", "review", "missing"]


class Card(BaseModel):
    """Текстовая карточка товара — единственный вход решения."""

    id: str
    title: str = ""
    description: str = ""
    seller_attrs: dict[str, Any] = Field(default_factory=dict)
    category: str | None = None

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


class ExtractionResult(BaseModel):
    card_id: str
    category: str
    category_confidence: float
    attributes: list[AttributeValue]
    llm_used: bool = False
    warnings: list[str] = Field(default_factory=list)

    def as_flat(self) -> dict[str, Any]:
        return {a.attribute: a.value for a in self.attributes if a.status == "accepted"}
