"""Калибровка уверенности на dev: сырая оценка источников -> P(значение верно), и выбор порога автопринятия.

`attrx calibrate` (только dev; holdout не трогаем):
1. Пайплайн прогоняется на выборке dev (как eval-real) БЕЗ калибровки — получаем сырые уверенности.
2. Для каждого предсказанного значения, где есть эталон: (группа, сырая оценка, верно ли).
   Группы — иерархия от частной к общей: «категория/атрибут/источники[/конфликт]» -> «источники|тип атрибута
   [|конфликт]» -> «*|тип атрибута»; у группы своя кривая, если в ней >= MIN_N точек, иначе используется общая.
3. Кривая — изотоническая регрессия (PAV, неубывающая), блоки сглажены к средней точности родительской группы.
4. Порог выбирается ЧЕСТНО: 2-fold кросс-фиттинг по карточкам (кривые учатся на одной половине dev, применяются
   к другой) — наименьший порог, при котором точность принятых значений >= target на отложенных половинах
   (нижняя граница 95% ДИ) и ни одна ячейка «категория/атрибут» не принимается с точностью ниже cell_floor.
   Итоговые кривые учатся на всём dev. Метрики кросс-фиттинга сохраняются в config/calibration.json.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..confidence import CONFLICT_ACCEPT, Curve, group_keys
from ..schema import ExtractionResult
from ..taxonomy import Taxonomy
from .metrics import _as_list, is_match, match_key

MIN_N = 80  # точек на собственную кривую группы (подобрано по ECE кросс-фиттинга на dev)
MIN_LEVEL = 25  # меньше MIN_N, но не меньше: форма — у родительской кривой, уровень — точность самой группы
SHRINK = 3.0  # псевдонаблюдений родительской точности в каждом блоке PAV
# CONFLICT_ACCEPT (confidence.py) — та же граница принятия при конфликте, что в pipeline.Extractor._finalize


@dataclass
class Point:
    card: str
    keys: list[str]  # группы от частной к общей
    raw: float
    ok: bool
    conflict: bool
    llm_only: bool
    cross: bool = False  # альтернатива от другого источника — pipeline всегда отправляет на проверку


def collect(results: list[ExtractionResult], gold: dict, tax: Taxonomy) -> list[Point]:
    """Точки калибровки: предсказанные значения (любой статус) там, где есть эталон."""
    pts: list[Point] = []
    for r in results:
        g = gold.get(r.card_id)
        if g is None:
            continue
        cat = tax.categories[r.category]
        brand = (_as_list(g.get("brand")) or [None])[0]
        for a in r.attributes:
            if a.value is None:
                continue
            spec = cat.attributes.get(a.attribute)
            gk = [k for k in (match_key(v, spec, tax, brand) for v in _as_list(g.get(a.attribute))) if k is not None]
            if spec is None or not gk:
                continue
            pk = match_key(a.value, spec, tax, brand)
            ok = pk is not None and is_match(pk, gk, spec)
            conflict = bool(a.alternatives)
            cross = any(set(alt.get("sources") or []) - set(a.sources) for alt in a.alternatives)
            pts.append(Point(r.card_id, group_keys(r.category, a.attribute, spec, a.sources, conflict),
                             float(a.confidence), ok, conflict, set(a.sources) <= {"llm"}, cross))
    return pts


def pav_curve(xs: list[float], ys: list[float], prior: float) -> Curve:
    """Изотоническая регрессия по уникальным x (веса = число точек), блоки сглажены к prior."""
    agg: dict[float, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for x, y in zip(xs, ys, strict=True):
        agg[round(x, 3)][0] += y
        agg[round(x, 3)][1] += 1
    blocks: list[list[float]] = []  # [x_min, sum_y, n]
    for x in sorted(agg):
        sy, n = agg[x]
        blocks.append([x, sy, n])
        while len(blocks) > 1 and blocks[-2][1] / blocks[-2][2] >= blocks[-1][1] / blocks[-1][2]:
            b = blocks.pop()
            blocks[-1][1] += b[1]
            blocks[-1][2] += b[2]
    ys_s = [(sy + SHRINK * prior) / (n + SHRINK) for _, sy, n in blocks]
    for i in range(1, len(ys_s)):  # сглаживание могло нарушить монотонность
        ys_s[i] = max(ys_s[i], ys_s[i - 1])
    return Curve([b[0] for b in blocks], [round(y, 4) for y in ys_s], int(sum(b[2] for b in blocks)))


def fit(points: list[Point]) -> dict[str, Curve]:
    """Кривые для всех групп с >= MIN_N точками; сглаживание блоков — к точности родительской (более общей) группы."""
    by: dict[str, list[Point]] = defaultdict(list)
    for p in points:
        for k in p.keys:
            by[k].append(p)
    overall = sum(p.ok for p in points) / max(1, len(points))
    curves: dict[str, Curve] = {}
    # от общих к частным (depth — расстояние от самой общей группы «*|тип»): у частной группы prior —
    # точность её ближайшей более общей группы, у которой уже есть кривая
    for depth in range(0, 5):
        for k, ps in by.items():
            if len(ps) < MIN_N:
                continue
            key_depth = ps[0].keys.index(k) if k in ps[0].keys else 0
            if len(ps[0].keys) - 1 - key_depth != depth:
                continue
            parent = next((ps[0].keys[i] for i in range(key_depth + 1, len(ps[0].keys)) if ps[0].keys[i] in curves), None)
            prior = (sum(p.ok for p in by[parent]) / len(by[parent])) if parent else overall
            curves[k] = pav_curve([p.raw for p in ps], [float(p.ok) for p in ps], prior)
    # маленькие группы (ячейка «категория/атрибут» с 25–79 точками): своей кривой не учим (переобучение), но и не
    # отдаём ей оптимистичную общую — родительская кривая масштабируется к сглаженной точности самой ячейки
    for k, ps in by.items():
        if k in curves or not MIN_LEVEL <= len(ps) < MIN_N:
            continue
        i0 = ps[0].keys.index(k)
        parent = next((ps[0].keys[i] for i in range(i0 + 1, len(ps[0].keys)) if ps[0].keys[i] in curves), None)
        if parent is None:
            continue
        pp = sum(p.ok for p in by[parent]) / len(by[parent])
        pc = (sum(p.ok for p in ps) + SHRINK * pp) / (len(ps) + SHRINK)
        ratio = pc / pp if pp else 1.0
        base = curves[parent]
        curves[k] = Curve(list(base.x), [round(min(1.0, y * ratio), 4) for y in base.y], len(ps))
    return curves


def apply(curves: dict[str, Curve], p: Point) -> float:
    for k in p.keys:
        if k in curves:
            return curves[k](p.raw)
    return p.raw


def _accepted(p: Point, conf: float, t: float) -> bool:
    """То же правило статуса, что в pipeline.Extractor._finalize."""
    return not p.llm_only and not p.cross and conf >= t and not (p.conflict and conf < CONFLICT_ACCEPT)


CELL_MIN_ACCEPTED = 20  # ячейка «категория/атрибут/источники» учитывается в «худшей ячейке» от стольких принятых


def _cell(p: Point) -> str:
    return (p.keys[1] if p.conflict else p.keys[0]).split("|")[0]


def sweep(points: list[Point], confs: list[float]) -> list[dict]:
    """Порог -> точность и покрытие принятых + худшая ячейка (точность принятых в самой слабой ячейке)."""
    rows = []
    n_all = len(points)
    for i in range(30, 100):
        t = i / 100
        acc = [(p, c) for p, c in zip(points, confs, strict=True) if _accepted(p, c, t)]
        tp = sum(p.ok for p, _ in acc)
        cells: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for p, _ in acc:
            cells[_cell(p)][0] += p.ok
            cells[_cell(p)][1] += 1
        worst = min(((k_ok / n, k) for k, (k_ok, n) in cells.items() if n >= CELL_MIN_ACCEPTED), default=(None, None))
        rows.append({"threshold": t, "precision": round(tp / len(acc), 4) if acc else None,
                     "coverage": round(len(acc) / n_all, 4) if n_all else 0.0, "accepted": len(acc), "tp": tp,
                     "worst_cell": worst[1], "worst_cell_precision": round(worst[0], 4) if worst[0] is not None else None})
    return rows


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    """Нижняя граница доверительного интервала Уилсона для доли k/n."""
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return (c - r) / d


def _half(card_id: str) -> int:
    return int(hashlib.md5(f"calib:{card_id}".encode()).hexdigest()[:8], 16) % 2


def calibrate(results: list[ExtractionResult], gold: dict, tax: Taxonomy, target: float = 0.95, cell_floor: float = 0.75,
              info: dict[str, Any] | None = None) -> dict[str, Any]:
    """-> содержимое config/calibration.json (кривые на всём dev + порог и проверка кросс-фиттингом)."""
    t0 = time.time()
    pts = collect(results, gold, tax)
    folds = [[p for p in pts if _half(p.card) == h] for h in (0, 1)]
    oof_pts, oof_conf = [], []
    for h in (0, 1):
        curves = fit(folds[1 - h])
        oof_pts += folds[h]
        oof_conf += [apply(curves, p) for p in folds[h]]
    oof = sweep(oof_pts, oof_conf)
    # порог с запасом на шум выборки: нижняя граница 95% доверительного интервала (Уилсон) точности принятых
    # на отложенных половинах dev — не ниже целевой; иначе порог «впритык» к 0.95 на holdout может оказаться ниже
    # и без «тихо плохих» ячеек: в каждой ячейке с >= CELL_MIN_ACCEPTED принятых точность >= cell_floor
    # (общая точность 0.95 может прятать атрибут с точностью 0.5 — среднее маскирует хвост)
    ok_rows = [r for r in oof if r["accepted"] and wilson_lower(r["tp"], r["accepted"]) >= target
               and (r["worst_cell_precision"] is None or r["worst_cell_precision"] >= cell_floor)]
    threshold = min((r["threshold"] for r in ok_rows), default=0.95)
    raw_sweep = sweep(pts, [p.raw for p in pts])
    # ошибка калибровки (ECE, 10 бинов) на отложенных половинах — насколько уверенность = точность
    bins: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for p, c in zip(oof_pts, oof_conf, strict=True):
        bins[min(9, int(c * 10))].append((c, p.ok))
    ece = sum(abs(sum(c for c, _ in b) / len(b) - sum(ok for _, ok in b) / len(b)) * len(b) for b in bins.values())
    ece /= max(1, len(oof_pts))
    curves = fit(pts)
    at = next(r for r in oof if r["threshold"] == threshold)
    raw_at = next((r for r in raw_sweep if r["threshold"] == round(threshold, 2)), None)
    return {
        "version": 1,
        "method": "isotonic (PAV) по группам «категория/атрибут/источники» -> «источники|тип атрибута» -> «*|тип», "
                  f"блоки сглажены к точности родительской группы (вес {SHRINK:g}); группа — от {MIN_N} точек",
        "fit_split": "dev",
        "target_precision": target,
        "threshold": threshold,
        "cell_floor": cell_floor,
        "threshold_rule": "наименьший порог, при котором на отложенных половинах dev (2-fold кросс-фиттинг) нижняя "
                          "граница 95% ДИ Уилсона точности принятых значений >= target_precision и в каждой ячейке "
                          f"«категория/атрибут/источники» с >= {CELL_MIN_ACCEPTED} принятыми точность >= cell_floor",
        "cross_fit": {"folds": 2, "points": len(pts), "precision_at_threshold": at["precision"],
                      "precision_lower95_at_threshold": round(wilson_lower(at["tp"], at["accepted"]), 4),
                      "worst_cell_at_threshold": [at["worst_cell"], at["worst_cell_precision"]],
                      "coverage_at_threshold": at["coverage"], "ece": round(ece, 4),
                      "raw_precision_at_same_threshold": raw_at["precision"] if raw_at else None,
                      "raw_coverage_at_same_threshold": raw_at["coverage"] if raw_at else None,
                      "sweep": oof[::5] + [at]},
        "info": (info or {}) | {"seconds": round(time.time() - t0, 1)},
        "groups": {k: {"x": c.x, "y": c.y, "n": c.n} for k, c in sorted(curves.items())},
    }


def write(cal: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(cal, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
