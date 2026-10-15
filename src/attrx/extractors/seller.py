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


def seller_spans(card: Card) -> dict[str, tuple[int, int]]:
    """Смещения «ключ: значение» каждого поля продавца в Card.full_text() (блок после названия и описания)."""
    head = "\n".join(x for x in (card.title, card.description) if x)
    pos = len(head) + 1 if head else 0
    out: dict[str, tuple[int, int]] = {}
    for k, v in card.seller_attrs.items():
        seg = f"{k}: {v}"
        out[str(k)] = (pos, pos + len(seg))
        pos += len(seg) + 2  # "; "
    return out


def extract(card: Card, category: CategorySpec, tax: Taxonomy) -> list[Candidate]:
    out: list[Candidate] = []
    spans = seller_spans(card) if card.seller_attrs else {}
    for key, raw in card.seller_attrs.items():
        attr = match_key(str(key), category)
        if attr is None or raw in (None, ""):
            continue
        res = normalize_value(str(raw), category.attributes[attr], tax.dictionaries)
        if res is None:
            continue
        value, quality = res
        out.append(Candidate(attribute=attr, value=value, raw=f"{key}: {raw}", source="seller",
                             score=SELLER_BASE_SCORE * quality, span=spans.get(str(key))))
    return out
