"""Отчёт для сдачи: сравнение подходов, качество по сегментам, разбор ошибок, файлы предсказаний."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from ..io import write_predictions
from ..schema import Card, ExtractionResult
from ..taxonomy import Taxonomy
from .metrics import _as_list, evaluate, gold_matches, match_key, threshold_sweep

MODES: dict[str, frozenset[str]] = {
    "правила": frozenset({"rules"}),
    "продавец + правила": frozenset({"seller", "rules"}),
    "+ ML (NER)": frozenset({"seller", "rules", "ml"}),
    "+ LLM": frozenset({"seller", "rules", "ml", "llm"}),
}


def available_modes(ex) -> dict[str, frozenset[str]]:
    modes = dict(MODES)
    if ex.ml is None:
        modes.pop("+ ML (NER)")
    if not ex.client.enabled:
        modes.pop("+ LLM")
    return modes


def run_mode(ex, cards: list[Card], sources: frozenset[str]) -> tuple[list[ExtractionResult], dict]:
    calls0 = ex.client.calls
    t = time.perf_counter()
    results = list(ex.extract_many(cards, sources=set(sources)))
    dt = time.perf_counter() - t
    return results, {"seconds": round(dt, 3), "cards_per_sec": round(len(cards) / dt, 1) if dt else None,
                     "llm_calls": ex.client.calls - calls0}


def compare(ex, cards: list[Card], gold: dict, tax: Taxonomy, modes: dict | None = None) -> tuple[list[dict], dict]:
    """Метрики по каждому режиму; возвращает таблицу и результаты по режимам."""
    rows, by_mode = [], {}
    for name, sources in (modes or available_modes(ex)).items():
        results, perf = run_mode(ex, cards, sources)
        m = evaluate(results, gold, tax)
        by_mode[name] = results
        rows.append({"mode": name, "sources": sorted(sources), **perf, "precision": m["micro"]["precision"],
                     "recall": m["micro"]["recall"], "f1": m["micro"]["f1"], "macro_f1": m["macro_f1"],
                     "traffic_weighted_f1": m["traffic_weighted_f1"], "coverage": m["coverage"],
                     "review_rate": m["review_rate"]})
    return rows, by_mode


SEGMENTS = {
    "есть атрибуты продавца": lambda c: bool(c.seller_attrs),
    "нет атрибутов продавца": lambda c: not c.seller_attrs,
    "короткое описание (<80 симв.)": lambda c: len(c.description or "") < 80,
    "длинное описание (>=80 симв.)": lambda c: len(c.description or "") >= 80,
}


def segments(results: list[ExtractionResult], cards: list[Card], gold: dict, tax: Taxonomy) -> dict[str, dict]:
    """Контроль покрытия по сегментам продавцов/карточек (риск «несправедливость к мелким продавцам»)."""
    by_id = {c.id: c for c in cards}
    out = {}
    for name, pred in SEGMENTS.items():
        sub = [r for r in results if r.card_id in by_id and pred(by_id[r.card_id])]
        if not sub:
            continue
        m = evaluate(sub, gold, tax)
        out[name] = {"n_cards": m["n_cards"], "precision": m["micro"]["precision"], "recall": m["micro"]["recall"],
                     "f1": m["micro"]["f1"], "coverage": m["coverage"]}
    return out


def errors(results: list[ExtractionResult], gold: dict, tax: Taxonomy) -> list[dict]:
    """Каждая ошибка: wrong (неверное значение), extra (значение есть, в gold нет), missed (не нашли)."""
    rows = []
    for r in results:
        if r.card_id not in gold:
            continue
        cat = tax.categories[r.category]
        g = gold[r.card_id]
        brand = (_as_list(g.get("brand")) or [None])[0]
        for a in r.attributes:
            spec = cat.attributes[a.attribute]
            pred = match_key(a.value, spec, tax, brand) if a.status == "accepted" else None
            truth = _as_list(g.get(a.attribute))
            if (pred is None and not truth) or (pred is not None and truth
                                                 and gold_matches(a.value, truth, spec, tax, brand)):
                continue
            # extra — значение там, где в каталоге пусто (на неполном эталоне это new_fill, а не обязательно ошибка)
            kind = "missed" if pred is None else "extra" if not truth else "wrong"
            rows.append({"card_id": r.card_id, "category": r.category, "attribute": a.attribute, "kind": kind,
                         "predicted": a.value, "gold": gold[r.card_id].get(a.attribute), "status": a.status,
                         "confidence": a.confidence, "sources": ",".join(a.sources), "evidence": a.evidence or ""})
    return rows


def _md_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(t for _, t in cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join(str(r.get(k, "")) for k, _ in cols) + " |\n" for r in rows)


def write_report(ex, cards: list[Card], gold: dict, tax: Taxonomy, out_dir: Path, note: str = "") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, by_mode = compare(ex, cards, gold, tax)
    # в сдачу идёт лучший режим по traffic-weighted F1 (на реальных данных выбирать по holdout, см. attrx split)
    best = max(rows, key=lambda r: (r["traffic_weighted_f1"], r["f1"]))["mode"]
    results = by_mode[best]
    write_predictions(results, out_dir / "predictions.jsonl")
    m = evaluate(results, gold, tax)
    m["threshold_sweep"] = threshold_sweep(results, gold, tax)
    m["mode"] = best
    seg = segments(results, cards, gold, tax)
    errs = errors(results, gold, tax)
    (out_dir / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "compare.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "segments.json").write_text(json.dumps(seg, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "errors.csv").open("w", encoding="utf-8", newline="") as f:
        cols = ["card_id", "category", "attribute", "kind", "predicted", "gold", "status", "confidence", "sources", "evidence"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(errs)

    attr_rows = [{"attribute": k, **v} for k, v in sorted(m["per_attribute"].items())]
    seg_rows = [{"segment": k, **v} for k, v in seg.items()]
    err_counts: dict[str, int] = {}
    for e in errs:
        err_counts[e["attribute"] + " / " + e["kind"]] = err_counts.get(e["attribute"] + " / " + e["kind"], 0) + 1
    md = ["# Отчёт: извлечение характеристик\n", note + "\n" if note else "",
          f"Карточек: **{m['n_cards']}**, режим в сдаче: **{best}**.\n",
          "## Сравнение подходов\n",
          _md_table(rows, [("mode", "Режим"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
                           ("traffic_weighted_f1", "Traffic-weighted F1"), ("coverage", "Coverage"),
                           ("review_rate", "На проверку"), ("cards_per_sec", "Карточек/с"), ("llm_calls", "LLM-вызовов")]),
          "\n## По атрибутам\n",
          _md_table(attr_rows, [("attribute", "Атрибут"), ("precision", "P"), ("recall", "R"), ("f1", "F1"),
                                ("tp", "TP"), ("fp", "FP"), ("fn", "FN")]),
          "\n## По сегментам карточек\n",
          _md_table(seg_rows, [("segment", "Сегмент"), ("n_cards", "Карточек"), ("precision", "P"), ("recall", "R"),
                               ("f1", "F1"), ("coverage", "Coverage")]),
          "\n## Ошибки\n",
          ("Нет ошибок на этом наборе.\n" if not errs else
           "".join(f"- {k}: {v}\n" for k, v in sorted(err_counts.items(), key=lambda x: -x[1]))
           + "\nПострочно: errors.csv (wrong — неверное значение, extra — лишнее, missed — пропуск).\n"),
          "\n## Файлы\n",
          "- predictions.jsonl — значения, уверенность, статус, источники и цитата по каждому атрибуту\n"
          "- predictions.csv — то же в плоском виде\n- metrics.json, compare.json, segments.json, errors.csv\n"]
    (out_dir / "REPORT.md").write_text("".join(md), encoding="utf-8")
    return {"compare": rows, "metrics": m, "segments": seg, "errors": len(errs)}
