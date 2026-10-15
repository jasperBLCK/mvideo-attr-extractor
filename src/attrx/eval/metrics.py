"""Метрики кейса на эталоне каталога: P/R/F1 по атрибутам, извлекаемый recall, покрытие, новые заполнения,
качество с учётом трафика, разрезы по категориям и сегментам справедливости, подбор порога.

Эталон каталога НЕПОЛНЫЙ (не у каждого товара заполнена каждая характеристика), поэтому:
- P/R/F1 считаются только на парах (карточка, атрибут), где эталон есть;
- предсказание там, где эталона нет, — не ошибка, а `new_fill` (потенциальное дозаполнение каталога);
- `recall_extractable` — recall на эталонных значениях, которые вообще есть в тексте карточки (потолок «только из текста»).
Правила сравнения: числа — с относительным допуском 1% в канонической единице; строки — без учёта регистра, ё/е,
пробелов, дефисов и пунктуации; эталон с несколькими допустимыми значениями — совпадение с любым.
Подробно: docs/EVALUATION.md.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..dictgen import is_placeholder
from ..normalize import compact, cpu_key, normalize_bool, normalize_colors, normalize_enum, normalize_value
from ..schema import ExtractionResult
from ..taxonomy import AttrSpec, Taxonomy

REL_TOL = 0.01  # относительный допуск для чисел


def _as_list(v: Any) -> list:
    if v is None or v == "":
        return []
    return [x for x in v if x not in (None, "")] if isinstance(v, (list, tuple)) else [v]


def match_key(v: Any, spec: AttrSpec, tax: Taxonomy, brand: str | None = None) -> float | str | None:
    """Канонический ключ значения для сравнения (тот же нормализатор для эталона и предсказаний)."""
    if v is None or v == "":
        return None
    if spec.type == "number":
        if isinstance(v, (int, float)):
            return float(v)
        res = normalize_value(str(v), spec, tax.dictionaries)
        return float(res[0]) if res else None
    if spec.type == "bool":
        return normalize_bool(str(v), spec.keywords)
    if spec.kind == "cpu":
        return cpu_key(str(v)) or None
    if spec.kind == "model":
        from .gold import model_key

        return model_key(v, brand) or None
    if spec.type == "enum" and spec.dictionary in tax.dictionaries:
        d = tax.dictionaries[spec.dictionary]
        if spec.dictionary == "brands" and is_placeholder(str(v)):
            return None  # «нет бренда», «No Name», «###» — бренд не указан: не эталон и не предсказание
        if spec.dictionary == "colors":
            cs = normalize_colors(str(v), d)
            return compact(cs[0]) if cs else compact(str(v))
        res = normalize_enum(str(v), d)
        return compact(res[0] if res else str(v))
    return compact(str(v)) or None


def is_match(pred: float | str, gold_keys: list, spec: AttrSpec) -> bool:
    for g in gold_keys:
        if isinstance(pred, float) and isinstance(g, float):
            if abs(pred - g) <= REL_TOL * abs(g) + 1e-9:
                return True
        elif pred == g:
            return True
        elif (spec.kind == "cpu" and g[-1:].isdigit() and pred.startswith(g) and pred[len(g):len(g) + 1].isdigit()
              and not any(k != g and isinstance(k, str) and k.startswith(g) for k in gold_keys)):
            return True  # в эталоне только семейство («Core i5»), предсказана модель этого семейства (i5-1235U)
    return False


def gold_matches(pred: Any, gold_value: Any, spec: AttrSpec, tax: Taxonomy, brand: str | None = None) -> bool:
    """Совпадает ли предсказание с эталоном (эталон — значение или список допустимых значений)."""
    pk = match_key(pred, spec, tax, brand)
    gk = [k for k in (match_key(g, spec, tax, brand) for g in _as_list(gold_value)) if k is not None]
    return pk is not None and bool(gk) and is_match(pk, gk, spec)


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}


def _cell_metrics(c: Counter) -> dict:
    """Счётчики ячейки -> метрики. tp/wrong/missed — только где есть эталон; new_fill — где эталона нет."""
    tp, wrong, gold, n = c["tp"], c["wrong"], c["gold"], c["n"]
    m = _prf(tp, wrong, gold - tp)
    m |= {
        "n_cards": n, "n_gold": gold, "n_pred": c["pred"], "new_fills": c["new_fill"],
        "recall_extractable": round(c["tp_ext"] / c["gold_ext"], 4) if c["gold_ext"] else None,
        "extractable_share": round(c["gold_ext"] / c["gold_known"], 4) if c["gold_known"] else None,
        # покрытие: насколько полнее становится каталог
        "catalog_fill": round(gold / n, 4) if n else 0.0,
        "pred_fill": round(c["pred"] / n, 4) if n else 0.0,
        "union_fill": round((gold + c["new_fill"]) / n, 4) if n else 0.0,
    }
    return m


@lru_cache(maxsize=4)
def _clicks_by_category(processed: str) -> dict[str, float]:
    p = Path(processed) / "clicks_by_category.parquet"
    if not p.exists():
        return {}
    import pyarrow.parquet as pq

    return {r["category_id"]: float(r["clicks"] or 0) for r in pq.read_table(p).to_pylist()}


def traffic_weights(tax: Taxonomy, weights: dict | None = None, clicks: dict[str, float] | None = None,
                    ) -> dict[tuple[str, str], float]:
    """Вес ячейки (категория, атрибут) = доля кликов категории × вес атрибута по спросу внутри категории.

    weights — config/traffic_weights.yaml: {category_id: {имя param | vendor | model: вес}}; без него — traffic_weight
    из таксономии. clicks — клики поиска по категориям YML; без них все категории равны."""
    weights = weights or {}
    clicks = clicks or {}
    covered = {cid for cat in tax.categories.values() for cid in cat.category_ids}
    total = sum(clicks.values())
    out: dict[tuple[str, str], float] = {}
    for cname, cat in tax.categories.items():
        ids = cat.category_ids
        cat_clicks = (sum(clicks.get(i, 0.0) for i in ids) if ids else total - sum(clicks.get(i, 0.0) for i in covered))
        if not ids:  # generic: спрос на атрибут — по всем категориям вне таксономии, где он измерен (взвешенно по кликам)
            ids = [i for i in weights if not str(i).startswith("_") and i not in covered]
        share = cat_clicks / total if total else 1.0
        for a, spec in cat.attributes.items():
            per_id = []
            for i in ids:
                w = [float(weights[i][g]) for g in spec.gold if g in (weights.get(i) or {})]
                if w:
                    per_id.append((clicks.get(i, 1.0), max(w)))
            aw = (sum(c * w for c, w in per_id) / (sum(c for c, _ in per_id) or 1.0)) if per_id else spec.traffic_weight
            out[(cname, a)] = share * aw
    return out


def _meta_clicks(raw: Any) -> dict[str, float]:
    """Клики по категориям из _meta того же YAML (attrx traffic, в git): категории с весами + остаток всех кликов
    поиска одной строкой (он вне таксономии, т.е. достаётся generic) — доли категорий те же, что по parquet."""
    meta = raw.get("_meta") if isinstance(raw, dict) else None
    cats = meta.get("categories") if isinstance(meta, dict) else None
    if not isinstance(cats, dict):
        return {}
    out = {str(k): float(v.get("clicks") or 0) for k, v in cats.items() if isinstance(v, dict)}
    rest = float(meta.get("total_clicks") or 0) - sum(out.values())
    if rest > 0:
        out["_other"] = rest
    return out


def load_traffic_weights(tax: Taxonomy, root: Path | None = None) -> dict[tuple[str, str], float]:
    """config/traffic_weights.yaml (поток EDA) + клики по категориям: data/processed/clicks_by_category.parquet, если
    есть, иначе _meta.categories того же YAML (свежий клон, CI, attrx demo — те же веса без data/processed)."""
    import yaml

    from ..config import ROOT

    root = root or ROOT
    p = root / "config" / "traffic_weights.yaml"
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
    w = {str(k): v for k, v in (raw.get("weights", raw) if isinstance(raw, dict) else {}).items() if isinstance(v, dict)}
    clicks = _clicks_by_category(str(root / "data" / "processed")) or _meta_clicks(raw)
    return traffic_weights(tax, w, clicks)


def evaluate(results: list[ExtractionResult], gold: dict[str, dict], tax: Taxonomy, threshold: float | None = None,
             include_review: bool = False, meta: dict[str, dict] | None = None,
             weights: dict[tuple[str, str], float] | None = None) -> dict:
    """threshold=None — используем статусы пайплайна; иначе принимаем всё с confidence >= threshold.

    gold: {card_id: {attr: значение | [допустимые значения]}}; meta: {card_id: {"extractable": {attr: bool}, ...}}
    (eval/gold.load_*); weights: {(категория, атрибут): вес} для traffic-weighted метрик."""
    cells: dict[tuple[str, str], Counter] = defaultdict(Counter)
    review = 0
    for r in results:
        if r.card_id not in gold:
            continue
        g = gold[r.card_id]
        ext = ((meta or {}).get(r.card_id) or {}).get("extractable") or {}
        cat = tax.categories[r.category]
        brand_gold = _as_list(g.get("brand"))
        brand = str(brand_gold[0]) if brand_gold else None
        for a in r.attributes:
            spec = cat.attributes.get(a.attribute)
            if spec is None:
                continue
            if threshold is None:
                accepted = a.status == "accepted" or (include_review and a.status == "review")
            else:
                accepted = a.value is not None and a.confidence >= threshold
            review += a.status == "review"
            gk = [k for k in (match_key(v, spec, tax, brand) for v in _as_list(g.get(a.attribute))) if k is not None]
            pk = match_key(a.value, spec, tax, brand) if accepted else None
            c = cells[(r.category, a.attribute)]
            c["n"] += 1
            c["pred"] += pk is not None
            if gk:
                c["gold"] += 1
                hit = pk is not None and is_match(pk, gk, spec)
                c["tp"] += hit
                c["wrong"] += pk is not None and not hit
                if a.attribute in ext:
                    c["gold_known"] += 1
                    c["gold_ext"] += bool(ext[a.attribute])
                    c["tp_ext"] += bool(ext[a.attribute]) and hit
            elif pk is not None:
                c["new_fill"] += 1

    per_cat: dict[str, dict] = defaultdict(dict)
    by_attr: dict[str, Counter] = defaultdict(Counter)
    total: Counter = Counter()
    for (cat, attr), c in sorted(cells.items()):
        per_cat[cat][attr] = _cell_metrics(c)
        by_attr[attr].update(c)
        total.update(c)
    # P/R/F1 атрибута осмысленны только при наличии эталона; ячейки без эталона остаются в per_category (new_fills)
    per_attr = {a: _cell_metrics(c) for a, c in sorted(by_attr.items()) if c["gold"]}

    w = weights or {k: tax.categories[k[0]].attributes[k[1]].traffic_weight for k in cells}
    scored = [(k, w.get(k, 0.0)) for k, c in cells.items() if c["gold"]]
    wsum = sum(x for _, x in scored) or 1.0

    def tw(metric: str) -> float:
        return round(sum(per_cat[c][a][metric] * x for (c, a), x in scored) / wsum, 4)

    micro = _cell_metrics(total)
    slots = total["n"]
    return {
        "micro": {k: micro[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn")},
        "macro_f1": round(sum(v["f1"] for v in per_attr.values()) / max(1, len(per_attr)), 4),
        "macro_f1_cells": round(sum(per_cat[c][a]["f1"] for (c, a), _ in scored) / max(1, len(scored)), 4),
        "traffic_weighted_f1": tw("f1"),
        "traffic_weighted_precision": tw("precision"),
        "traffic_weighted_recall": tw("recall"),
        "recall_extractable": micro["recall_extractable"],
        "extractable_share": micro["extractable_share"],
        "coverage": micro["pred_fill"],  # доля слотов (карточка × атрибут), заполненных предсказаниями
        "catalog_fill": micro["catalog_fill"],
        "union_fill": micro["union_fill"],
        "new_fills": micro["new_fills"],
        "review_rate": round(review / slots, 4) if slots else 0.0,
        "per_attribute": per_attr,
        "per_category": dict(per_cat),
        "n_cards": sum(1 for r in results if r.card_id in gold),
    }


def segment_metrics(results: list[ExtractionResult], gold: dict, tax: Taxonomy, meta: dict[str, dict],
                    key: str, weights: dict | None = None) -> dict[str, dict]:
    """Метрики по сегментам справедливости: key = seller_type | size_band | desc_band | category_id."""
    groups: dict[str, list[ExtractionResult]] = defaultdict(list)
    for r in results:
        if r.card_id in gold:
            groups[str((meta.get(r.card_id) or {}).get(key, "?"))].append(r)
    out = {}
    for seg, rs in sorted(groups.items()):
        m = evaluate(rs, gold, tax, meta=meta, weights=weights)
        out[seg] = {"n_cards": m["n_cards"], **{k: m["micro"][k] for k in ("precision", "recall", "f1")},
                    "recall_extractable": m["recall_extractable"], "macro_f1": m["macro_f1"],
                    "coverage": m["coverage"], "catalog_fill": m["catalog_fill"], "union_fill": m["union_fill"],
                    "new_fills": m["new_fills"]}
    return out


def threshold_sweep(results: list[ExtractionResult], gold: dict, tax: Taxonomy, steps: int = 20,
                    meta: dict | None = None) -> list[dict]:
    """Кривая precision/coverage от порога — по ней выбираем CONFIDENCE_THRESHOLD под целевую точность бизнеса."""
    rows = []
    for i in range(steps + 1):
        t = round(i / steps, 3)
        m = evaluate(results, gold, tax, threshold=t, meta=meta)
        rows.append({"threshold": t, "precision": m["micro"]["precision"], "recall": m["micro"]["recall"],
                     "f1": m["micro"]["f1"], "coverage": m["coverage"], "new_fills": m["new_fills"]})
    return rows
