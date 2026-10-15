"""Петля обратной связи: решения модератора -> feedback.jsonl -> новые синонимы в learned.yaml (attrx learn).

Запись: {ts, card_id, category, attribute, predicted, corrected, raw}; category — категория таксономии карточки
(в старых записях её нет), raw — формулировка значения в тексте карточки (UI шлёт фрагмент evidence_span; у поля
продавца — только значение, без «Ключ:»).
Учатся только исправления (модератор заменил значение): «принято» ничего нового не говорит, «отклонено» не даёт
правильного значения — такие решения остаются в журнале (очередь, аудит), но в справочники не попадают.

Защита живого справочника от ошибок модератора (опечатка «чорный» не должна стать каноном после одного клика):
синоним «формулировка -> значение» попадает в learned.yaml, только если
  (а) исправленное значение уже есть в справочнике (канон или известный синоним — тогда пишется канон), или
  (б) это же исправление сделано минимум на MIN_AGREE разных карточках.
Остальное, а также расходящиеся правки одной формулировки (поровну голосов за разные значения), — «кандидаты»:
они пишутся в learned_candidates.yaml рядом с feedback.jsonl и НЕ загружаются таксономией, пока их не подтвердят
повторные правки (или человек не перенесёт значение в справочник config/dictionaries/*.yaml).
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

import yaml

from .config import get_settings
from .taxonomy import Dictionary, Taxonomy, load_taxonomy, norm

MIN_AGREE = 2  # согласных исправлений (на разных карточках), чтобы значение вне справочника стало каноном
CANDIDATES_FILE = "learned_candidates.yaml"


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


def _suggest(d: Dictionary, value: str) -> str | None:
    """Ближайшее значение справочника к значению вне его («чорный» -> «чёрный») — подсказка в кандидатах."""
    try:
        from rapidfuzz import fuzz, process
    except ImportError:  # rapidfuzz — базовая зависимость, но подсказка не обязательна
        return None
    canon = [str(c) for c in d.canon]
    hit = process.extractOne(norm(value), [norm(c) for c in canon], scorer=fuzz.ratio, score_cutoff=80)
    return canon[hit[2]] if hit else None


def learn(feedback: Path | None = None, out: Path | None = None, candidates_out: Path | None = None,
          min_agree: int = MIN_AGREE) -> dict:
    """Исправления модератора -> {"learned": {справочник: {формулировка: канон}}, "candidates": {...}, "stats": {...},
    "paths": {...}}. learned.yaml дополняется (выученное раньше сохраняется), кандидаты пересчитываются по всему журналу."""
    s = get_settings()
    tax = load_taxonomy(str(s.taxonomy_path))
    feedback = feedback or s.feedback_path
    out = out or s.taxonomy_path.parent / "dictionaries" / "learned.yaml"
    cand_path = candidates_out or feedback.with_name(CANDIDATES_FILE)
    learned: dict = (yaml.safe_load(out.read_text(encoding="utf-8")) if out.exists() else {}) or {}
    st = {k: 0 for k in ("records", "corrections", "learned", "candidates", "accepted", "rejected", "no_text",
                         "no_dictionary")}
    res = {"learned": learned, "candidates": {}, "stats": st,
           "paths": {"learned": str(out), "candidates": str(cand_path)}, "min_agree": min_agree}
    if not feedback.exists():
        return res
    # (справочник, формулировка) -> {значение: {"cards": {card_id}, "known": в справочнике?, "last": № строки}}
    votes: dict[tuple[str, str], dict[str, dict]] = {}
    for i, line in enumerate(feedback.read_text(encoding="utf-8").splitlines()):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        st["records"] += 1
        action = decision_action(r)
        if action != "corrected":  # подтверждение и отклонение не учат справочник
            st[action] += 1
            continue
        st["corrections"] += 1
        raw = norm(value_text(r.get("raw")))
        if not raw:
            st["no_text"] += 1
            continue
        dname = dictionary_for(tax, r.get("attribute", ""), r.get("category"))
        if dname is None or dname not in tax.dictionaries:
            st["no_dictionary"] += 1
            continue
        typed = " ".join(str(r["corrected"]).split())
        known = tax.dictionaries[dname].get(typed)  # «черный» -> канон справочника «чёрный»; «чорный» -> None
        # одинаковые правки — с точностью до регистра, пробелов и ё/е; пишется канон или самое частое написание
        v = votes.setdefault((dname, raw), {}).setdefault(known or norm(typed), {"cards": set(), "forms": Counter(),
                                                                                  "known": known is not None})
        v["cards"].add(str(r.get("card_id") or f"#{i}"))  # повторный клик по той же карточке — не второй голос
        v["forms"][known or typed] += 1
        v["last"] = i

    candidates: dict[str, dict] = {}
    for (dname, raw), options in sorted(votes.items()):
        ranked = sorted(options.values(), key=lambda o: (-len(o["cards"]), -o["last"]))
        best = ranked[0]
        value = best["forms"].most_common(1)[0][0]
        n = len(best["cards"])
        tie = len(ranked) > 1 and len(ranked[1]["cards"]) == n
        if not tie and (best["known"] or n >= min_agree):
            learned.setdefault(dname, {})
            learned[dname] = learned[dname] or {}
            learned[dname][raw] = value
            st["learned"] += 1
            continue
        item: dict = {"value": value, "votes": n, "need": min_agree, "in_dictionary": best["known"],
                      "cards": sorted(best["cards"])[:5]}
        if tie:
            opts = {o["forms"].most_common(1)[0][0]: len(o["cards"]) for o in ranked}
            item["reason"] = "правки расходятся: " + ", ".join(f"{k} ({v})" for k, v in opts.items())
            item["options"] = opts
        else:
            item["reason"] = (f"значения нет в справочнике {dname}: одинаковых исправлений на разных карточках "
                              f"{n} из {min_agree}")
            if hint := _suggest(tax.dictionaries[dname], value):
                item["suggest"] = hint
        candidates.setdefault(dname, {})[raw] = item
        st["candidates"] += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(learned, allow_unicode=True, sort_keys=True), encoding="utf-8")
    cand_path.parent.mkdir(parents=True, exist_ok=True)
    cand_path.write_text("# Кандидаты в синонимы из правок модератора (attrx learn). В справочники НЕ загружаются:\n"
                         f"# станут синонимами после {min_agree} одинаковых исправлений на разных карточках или после "
                         "переноса значения в config/dictionaries/*.yaml.\n"
                         + yaml.safe_dump(candidates, allow_unicode=True, sort_keys=True), encoding="utf-8")
    res["candidates"] = candidates
    load_taxonomy.cache_clear()
    return res


def learn_synonyms(feedback: Path | None = None, out: Path | None = None, stats: dict | None = None,
                   candidates_out: Path | None = None, min_agree: int = MIN_AGREE) -> dict:
    """Исправления модератора по значениям из справочника -> синонимы {справочник: {формулировка: канон}}.

    stats — необязательный словарь: сколько решений учтено и почему остальные пропущены (для attrx learn);
    кандидаты (не прошедшие правило MIN_AGREE) — в learned_candidates.yaml, см. learn()."""
    res = learn(feedback, out, candidates_out, min_agree)
    if stats is not None:
        stats.update(res["stats"])
    return res["learned"]
