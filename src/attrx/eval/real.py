"""Оценка пайплайна на реальном эталоне каталога: выборка карточек сплита -> извлечение -> метрики -> JSON + таблица.

`attrx build-gold` строит эталон (data/processed/gold.parquet), `attrx eval-real` меряет любой набор источников
(seller, rules, ml, llm) на dev — так каждый шаг гибрида сравнивается на одних и тех же карточках.
Holdout не трогаем при настройке правил: он для финального отчёта.
"""
from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path

from ..schema import Card, ExtractionResult
from ..taxonomy import Taxonomy
from .metrics import evaluate, load_traffic_weights, segment_metrics, threshold_sweep


def stratified_ids(meta: dict[str, dict], n_tax: int, n_generic: int, seed: int = 0, min_per_group: int = 150,
                   ) -> list[str]:
    """Карточки категорий таксономии — пропорционально размеру категории (не меньше min_per_group, если есть),
    generic — по полосам размера категории (head/torso/tail), чтобы хвост был представлен."""
    rnd = random.Random(seed)
    tax_groups: dict[str, list[str]] = defaultdict(list)
    gen_groups: dict[str, list[str]] = defaultdict(list)
    for sku, m in sorted(meta.items()):
        (gen_groups[m.get("size_band", "?")] if m["category"] == "generic" else tax_groups[m["category"]]).append(sku)

    def take(groups: dict[str, list[str]], n: int) -> list[str]:
        total = sum(len(v) for v in groups.values())
        if total <= n:
            return [s for v in groups.values() for s in v]
        out = []
        for _, ids in sorted(groups.items()):
            k = min(len(ids), max(min_per_group, round(n * len(ids) / total)))
            out += rnd.sample(ids, k)
        return out

    return take(tax_groups, n_tax) + take(gen_groups, n_generic)


def load_cards_by_id(ids: list[str], processed: Path) -> list[Card]:
    import pyarrow.parquet as pq

    cols = ["sku_id", "title", "description_clean", "seller_attrs", "category_id", "category_title", "seller"]
    t = pq.read_table(processed / "cards.parquet", columns=cols, filters=[("sku_id", "in", ids)])
    order = {s: i for i, s in enumerate(ids)}
    cards = [Card(id=r["sku_id"], title=r["title"] or "", description=r["description_clean"] or "",
                  seller_attrs=json.loads(r["seller_attrs"] or "{}"), category=r["category_title"],
                  category_id=r["category_id"], seller=r["seller"]) for r in t.to_pylist()]
    return sorted(cards, key=lambda c: order[c.id])


def run(ex, cards: list[Card], sources: set[str]) -> tuple[list[ExtractionResult], dict]:
    t = time.perf_counter()
    results = list(ex.extract_many(cards, use_llm="llm" in sources, sources=sources))
    dt = time.perf_counter() - t
    return results, {"seconds": round(dt, 2), "cards_per_sec": round(len(cards) / dt, 1) if dt else None}


def report(results: list[ExtractionResult], gold: dict, meta: dict, tax: Taxonomy, perf: dict, **info) -> dict:
    weights = load_traffic_weights(tax)
    m = evaluate(results, gold, tax, meta=meta, weights=weights)
    m["segments"] = {k: segment_metrics(results, gold, tax, meta, k, weights)
                     for k in ("seller_type", "size_band", "desc_band")}
    m["threshold_sweep"] = threshold_sweep(results, gold, tax, meta=meta)
    m["perf"] = perf
    m["info"] = info
    return m


def table(m: dict) -> str:
    """Компактная таблица категория × атрибут: P, R, R_extractable, доля извлекаемых, покрытие каталог/пред/объединение."""
    head = (f"{'категория':<16}{'атрибут':<14}{'n_gold':>7}{'P':>7}{'R':>7}{'R_ext':>7}{'ext%':>6}"
            f"{'кат.':>6}{'пред':>6}{'объед':>7}{'new':>6}")
    lines = [head, "-" * len(head)]

    def fmt(x):
        return f"{x:.2f}" if isinstance(x, float) else "  —"

    for cat, attrs in m["per_category"].items():
        for a, v in attrs.items():
            prec = v["precision"] if v["tp"] + v["fp"] else None  # нет предсказаний — точность не определена
            lines.append(f"{cat:<16}{a:<14}{v['n_gold']:>7}{fmt(prec):>7}{fmt(v['recall']):>7}"
                         f"{fmt(v['recall_extractable']):>7}{fmt(v['extractable_share']):>6}{fmt(v['catalog_fill']):>6}"
                         f"{fmt(v['pred_fill']):>6}{fmt(v['union_fill']):>7}{v['new_fills']:>6}")
    mi = m["micro"]
    lines.append("-" * len(head))
    lines.append(f"micro P={mi['precision']:.3f} R={mi['recall']:.3f} F1={mi['f1']:.3f} R_ext={m['recall_extractable']} "
                 f"macro-F1={m['macro_f1']} traffic-F1={m['traffic_weighted_f1']} coverage {m['catalog_fill']}->"
                 f"{m['union_fill']} (+{m['new_fills']} новых), карточек {m['n_cards']}, {m['perf'].get('cards_per_sec')} карт/с")
    return "\n".join(lines)
