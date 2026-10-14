"""Петля обратной связи: правки модераторов -> feedback.jsonl -> новые синонимы в learned.yaml и gold для метрик."""
from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

from .config import get_settings
from .taxonomy import load_taxonomy, norm


def log_correction(card_id: str, attribute: str, predicted, corrected, raw: str | None = None, path: Path | None = None) -> dict:
    rec = {"ts": int(time.time()), "card_id": card_id, "attribute": attribute, "predicted": predicted,
           "corrected": corrected, "raw": raw}
    p = path or get_settings().feedback_path
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def learn_synonyms(feedback: Path | None = None, out: Path | None = None) -> dict:
    """Если модератор исправил enum-значение и есть исходная формулировка (raw) — запоминаем синоним."""
    s = get_settings()
    tax = load_taxonomy(str(s.taxonomy_path))
    feedback = feedback or s.feedback_path
    out = out or s.taxonomy_path.parent / "dictionaries" / "learned.yaml"
    learned: dict = yaml.safe_load(out.read_text(encoding="utf-8")) if out.exists() else {}
    learned = learned or {}
    if not feedback.exists():
        return learned
    for line in feedback.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if not r.get("raw") or not r.get("corrected"):
            continue
        for cat in tax.categories.values():
            spec = cat.attributes.get(r["attribute"])
            if spec and spec.dictionary:
                learned.setdefault(spec.dictionary, {})[norm(r["raw"])] = r["corrected"]
                break
    out.write_text(yaml.safe_dump(learned, allow_unicode=True, sort_keys=True), encoding="utf-8")
    load_taxonomy.cache_clear()
    return learned
