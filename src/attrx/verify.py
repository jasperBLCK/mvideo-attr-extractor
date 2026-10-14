"""Проверка «заземлённости»: значение должно подтверждаться текстом карточки. Главная защита от галлюцинаций LLM."""
from __future__ import annotations

import re

from .normalize import parse_number
from .schema import Candidate, Card
from .taxonomy import AttrSpec, Dictionary, norm


def _compact(s: str) -> str:
    return re.sub(r"[\s\-_\"'«»]+", "", norm(s))


def grounded(cand: Candidate, card: Card, spec: AttrSpec, dictionaries: dict[str, Dictionary]) -> bool:
    text = norm(card.full_text())
    if cand.source in {"seller", "human"}:
        return True
    raw = norm(cand.raw)
    if raw and raw not in text and _compact(raw) not in _compact(text):
        return False
    # значение должно следовать из процитированного фрагмента, а не просто «где-то в тексте есть цитата»
    if spec.type == "number":
        parsed = parse_number(raw, spec.kind)
        if parsed and abs(parsed[0] - float(cand.value)) < 1e-6:
            return True
        nums = re.findall(r"\d+(?:[.,]\d+)?", raw)
        v = float(cand.value)
        return any(abs(float(n.replace(",", ".")) - v) < 1e-6 or abs(float(n.replace(",", ".")) * 1024 - v) < 1e-6 for n in nums)
    if spec.type == "enum" and spec.dictionary in dictionaries:
        d = dictionaries[spec.dictionary]
        return any(h[0] == cand.value for h in d.find_all(raw)) or d.get(raw) == cand.value
    return _compact(str(cand.value)) in _compact(text)
