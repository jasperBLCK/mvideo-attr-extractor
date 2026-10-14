"""Экстрактор по полям продавца: сопоставляем ключи продавца с атрибутами таксономии через aliases."""
from __future__ import annotations

from ..normalize import normalize_value
from ..schema import Candidate, Card
from ..taxonomy import CategorySpec, Taxonomy, norm

SELLER_BASE_SCORE = 0.8  # продавец часто прав, но бывает мусор/ошибки — поэтому не 1.0


def match_key(key: str, category: CategorySpec) -> str | None:
    k = norm(key).strip(" :")
    for name, spec in category.attributes.items():
        if k == name or k in (norm(a) for a in spec.aliases) or k == norm(spec.title):
            return name
    for name, spec in category.attributes.items():
        if any(norm(a) in k for a in spec.aliases if len(a) > 3):
            return name
    return None


def extract(card: Card, category: CategorySpec, tax: Taxonomy) -> list[Candidate]:
    out: list[Candidate] = []
    for key, raw in card.seller_attrs.items():
        attr = match_key(str(key), category)
        if attr is None or raw in (None, ""):
            continue
        res = normalize_value(str(raw), category.attributes[attr], tax.dictionaries)
        if res is None:
            continue
        value, quality = res
        out.append(Candidate(attribute=attr, value=value, raw=f"{key}: {raw}", source="seller",
                             score=SELLER_BASE_SCORE * quality))
    return out
