"""Метрики кейса: P/R/F1 по атрибутам, coverage, traffic-weighted quality, разрез по категориям, подбор порога."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..normalize import normalize_value
from ..schema import ExtractionResult
from ..taxonomy import Taxonomy


def _canon(v: Any, spec, tax: Taxonomy):
    if v is None or v == "":
        return None
    res = normalize_value(str(v), spec, tax.dictionaries)
    val = res[0] if res else str(v)
    return round(float(val), 1) if isinstance(val, float) else str(val).strip().lower()


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}


def evaluate(results: list[ExtractionResult], gold: dict[str, dict], tax: Taxonomy, threshold: float | None = None,
             include_review: bool = False) -> dict:
    """threshold=None — используем статусы пайплайна; иначе принимаем всё с confidence >= threshold."""
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])  # (category, attr) -> tp, fp, fn
    filled = total = review = 0
    for r in results:
        if r.card_id not in gold:
            continue
        g = gold[r.card_id]
        cat = tax.categories[r.category]
        for a in r.attributes:
            spec = cat.attributes[a.attribute]
            total += 1
            if threshold is None:
                accepted = a.status == "accepted" or (include_review and a.status == "review")
            else:
                accepted = a.value is not None and a.confidence >= threshold
            review += a.status == "review"
            pred = _canon(a.value, spec, tax) if accepted else None
            truth = _canon(g.get(a.attribute), spec, tax)
            c = counts[(r.category, a.attribute)]
            if pred is not None:
                filled += 1
                if truth is not None and pred == truth:
                    c[0] += 1
                else:
                    c[1] += 1
                    if truth is not None:
                        c[2] += 1
            elif truth is not None:
                c[2] += 1

    per_attr: dict[str, dict] = {}
    per_cat: dict[str, dict] = defaultdict(dict)
    agg_attr: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for (cat, attr), (tp, fp, fn) in counts.items():
        per_cat[cat][attr] = _prf(tp, fp, fn)
        for i, x in enumerate((tp, fp, fn)):
            agg_attr[attr][i] += x
    for attr, (tp, fp, fn) in agg_attr.items():
        per_attr[attr] = _prf(tp, fp, fn)

    weights = {}
    for cat_name, attrs in per_cat.items():
        for attr in attrs:
            weights[(cat_name, attr)] = tax.categories[cat_name].attributes[attr].traffic_weight
    wsum = sum(weights.values()) or 1.0
    tw_f1 = sum(per_cat[c][a]["f1"] * w for (c, a), w in weights.items()) / wsum
    tw_p = sum(per_cat[c][a]["precision"] * w for (c, a), w in weights.items()) / wsum
    tp, fp, fn = (sum(v[i] for v in counts.values()) for i in range(3))
    return {
        "micro": _prf(tp, fp, fn),
        "macro_f1": round(sum(v["f1"] for v in per_attr.values()) / max(1, len(per_attr)), 4),
        "traffic_weighted_f1": round(tw_f1, 4),
        "traffic_weighted_precision": round(tw_p, 4),
        "coverage": round(filled / total, 4) if total else 0.0,
        "review_rate": round(review / total, 4) if total else 0.0,
        "per_attribute": per_attr,
        "per_category": dict(per_cat),
        "n_cards": sum(1 for r in results if r.card_id in gold),
    }


def threshold_sweep(results: list[ExtractionResult], gold: dict, tax: Taxonomy, steps: int = 20) -> list[dict]:
    """Кривая precision/coverage от порога — по ней выбираем CONFIDENCE_THRESHOLD под целевую точность бизнеса."""
    rows = []
    for i in range(steps + 1):
        t = round(i / steps, 3)
        m = evaluate(results, gold, tax, threshold=t)
        rows.append({"threshold": t, "precision": m["micro"]["precision"], "recall": m["micro"]["recall"],
                     "f1": m["micro"]["f1"], "coverage": m["coverage"]})
    return rows
