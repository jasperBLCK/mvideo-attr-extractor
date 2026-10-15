"""Петля обратной связи: решения модератора -> feedback.jsonl -> новые синонимы в learned.yaml (attrx learn).

Запись: {ts, card_id, category, attribute, predicted, corrected, raw}; category — категория таксономии карточки
(в старых записях её нет), raw — формулировка значения в тексте карточки (UI шлёт фрагмент evidence_span; у поля
продавца — только значение, без «Ключ:»).
Учатся только исправления (модератор заменил значение): «принято» ничего нового не говорит, «отклонено» не даёт
правильного значения — такие решения остаются в журнале (очередь, аудит), но в справочники не попадают.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import yaml

from .config import get_settings
from .taxonomy import Taxonomy, load_taxonomy, norm


def log_correction(card_id: str, attribute: str, predicted, corrected, raw: str | None = None, path: Path | None = None,
                   category: str | None = None) -> dict:
    rec = {"ts": int(time.time()), "card_id": card_id, "category": category, "attribute": attribute,
           "predicted": predicted, "corrected": corrected, "raw": raw}
    p = path or get_settings().feedback_path
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def _same_value(a, b) -> bool:
    """«6000» = 6000.0, «Panasonic» = «panasonic»: перепечатанное то же значение — не исправление."""
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a).strip().casefold() == str(b).strip().casefold()


def decision_action(rec: dict) -> str:
    """Что сделал модератор: accepted (подтвердил), corrected (заменил значение) или rejected (значения нет)."""
    corrected, predicted = rec.get("corrected"), rec.get("predicted")
    if corrected in (None, ""):
        return "rejected"
    return "accepted" if predicted is not None and _same_value(corrected, predicted) else "corrected"


# «Цвет: графитовый» -> «графитовый»: ключ поля продавца (с буквы, до двоеточия и пробела) — не часть значения;
# «16:9» и «2:1» не трогаем (после двоеточия нет пробела)
_KEY_PREFIX = re.compile(r"^\s*[^\W\d_][^:;\n]{0,60}?:\s+(?=\S)")


def value_text(raw: str | None) -> str:
    """Формулировка значения для синонима: цитата без «Ключ:» поля продавца."""
    return _KEY_PREFIX.sub("", str(raw or ""), count=1).strip()


def dictionary_for(tax: Taxonomy, attribute: str, category: str | None) -> str | None:
    """Справочник атрибута по схеме категории карточки. Старые записи без категории — только если у атрибута
    один справочник во всех категориях (color, brand); type у наушников, пылесосов и стиралок — разные справочники."""
    if category:
        cat = tax.categories.get(category)
        spec = cat.attributes.get(attribute) if cat else None
        return spec.dictionary if spec is not None and spec.type == "enum" else None
    found = {c.attributes[attribute].dictionary for c in tax.categories.values()
             if attribute in c.attributes and c.attributes[attribute].type == "enum"}
    return found.pop() if len(found) == 1 else None


def learn_synonyms(feedback: Path | None = None, out: Path | None = None, stats: dict | None = None) -> dict:
    """Исправления модератора по значениям из справочника -> синонимы {справочник: {формулировка: канон}}.

    stats — необязательный словарь: сколько решений учтено и почему остальные пропущены (для attrx learn)."""
    s = get_settings()
    tax = load_taxonomy(str(s.taxonomy_path))
    feedback = feedback or s.feedback_path
    out = out or s.taxonomy_path.parent / "dictionaries" / "learned.yaml"
    learned: dict = yaml.safe_load(out.read_text(encoding="utf-8")) if out.exists() else {}
    learned = learned or {}
    st = stats if stats is not None else {}
    for k in ("records", "learned", "accepted", "rejected", "no_text", "no_dictionary"):
        st.setdefault(k, 0)
    if not feedback.exists():
        return learned
    for line in feedback.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        st["records"] += 1
        action = decision_action(r)
        if action != "corrected":  # подтверждение и отклонение не учат справочник
            st[action] += 1
            continue
        raw = norm(value_text(r.get("raw")))
        if not raw:
            st["no_text"] += 1
            continue
        dname = dictionary_for(tax, r.get("attribute", ""), r.get("category"))
        if dname is None or dname not in tax.dictionaries:
            st["no_dictionary"] += 1
            continue
        corrected = str(r["corrected"]).strip()
        canon = tax.dictionaries[dname].get(corrected) or corrected  # «черный» -> канон справочника «чёрный»
        bucket = learned.get(dname) or {}
        bucket[raw] = canon
        learned[dname] = bucket
        st["learned"] += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(learned, allow_unicode=True, sort_keys=True), encoding="utf-8")
    load_taxonomy.cache_clear()
    return learned
