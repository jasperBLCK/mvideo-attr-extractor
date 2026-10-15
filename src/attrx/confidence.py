"""Агрегация кандидатов от разных источников в одно значение + калиброванная уверенность.

1. aggregate_full: кандидаты группируются по значению; поддержка группы — noisy-OR лучших оценок независимых
   источников (согласие повышает уверенность), конфликт снижает её на половину поддержки альтернативы.
2. Calibrator: «сырая» уверенность -> вероятность того, что значение верно (P(correct)), отдельно для каждой
   группы «набор источников × тип атрибута» (rules|brand, seller+rules|number, llm|enum, ...). Изотоническая
   регрессия (PAV), обученная на dev (`attrx calibrate`, config/calibration.json); порог автопринятия выбирается
   на dev под целевую точность принятых значений. Без файла калибровки уверенность = сырая оценка.
"""
from __future__ import annotations

import json
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any

from .schema import Candidate
from .taxonomy import AttrSpec

SOURCE_WEIGHT = {"human": 1.0, "seller": 1.0, "rules": 1.0, "ml": 0.9, "llm": 0.85}
# при конфликте источников значение принимается только с калиброванной уверенностью не ниже этой
# (одна константа для pipeline.Extractor._finalize и подбора порога в eval/calibrate.py)
CONFLICT_ACCEPT = 0.85


def _key(v) -> str:
    return str(v).lower() if isinstance(v, str) else f"{float(v):g}"


def support_of(group: list[Candidate]) -> float:
    """Поддержка группы кандидатов одного значения: noisy-OR лучших оценок разных источников."""
    return _support(group)


def _support(group: list[Candidate]) -> float:
    best_by_source: dict[str, float] = {}
    for c in group:
        s = min(0.99, c.score * SOURCE_WEIGHT.get(c.source, 0.8))
        best_by_source[c.source] = max(best_by_source.get(c.source, 0.0), s)
    miss = 1.0
    for s in best_by_source.values():
        miss *= 1 - s
    return 1 - miss


@dataclass
class Aggregate:
    value: str | float | None
    confidence: float  # сырая (до калибровки)
    support: list[Candidate]
    conflict: bool  # есть альтернативное значение от какого-либо источника
    alternatives: list[dict[str, Any]] = field(default_factory=list)  # [{value, confidence, sources}]

    @property
    def sources(self) -> list[str]:
        return sorted({c.source for c in self.support})

    @property
    def cross_conflict(self) -> bool:
        """Альтернативу предлагает другой источник (продавец написал одно, текст говорит другое) — решает человек."""
        own = set(self.sources)
        return any(set(a["sources"]) - own for a in self.alternatives)


NUM_TOL = 0.01  # числа в пределах 1% — одно значение («43 дюйма (109 см)» = 43 и 42.9), как в метриках


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _same(a, b, key: Callable[[str], str] | None = None) -> bool:
    if _is_num(a) and _is_num(b):
        return abs(float(a) - float(b)) <= NUM_TOL * max(abs(float(b)), 1e-9)
    if _is_num(a) or _is_num(b):
        return False
    if _key(a) == _key(b):
        return True
    if key is not None:  # «одно и то же значение» по ключу атрибута: модель «Redmi Note 13» = «Redmi Note 13 8/256»
        ka, kb = key(str(a)), key(str(b))
        return bool(ka) and ka == kb
    return False


def aggregate_full(cands: list[Candidate], key: Callable[[str], str] | None = None) -> Aggregate:
    """key — ключ сравнения строковых значений атрибута (модель, процессор): согласные источники не спорят,
    даже если записали значение по-разному."""
    if not cands:
        return Aggregate(None, 0.0, [], False)
    groups: list[list[Candidate]] = []  # значение группы — у самого уверенного кандидата
    for c in sorted(cands, key=lambda c: -c.score):
        g = next((g for g in groups if _same(c.value, g[0].value, key)), None)
        if g is None:
            groups.append([c])
        else:
            g.append(c)
    ranked = sorted(groups, key=_support, reverse=True)
    top = _support(ranked[0])
    conflict = len(ranked) > 1
    conf = top - 0.5 * _support(ranked[1]) if conflict else top
    alts = [{"value": g[0].value, "confidence": round(_support(g), 3), "sources": sorted({c.source for c in g})}
            for g in ranked[1:4]]
    return Aggregate(ranked[0][0].value, round(max(0.0, min(1.0, conf)), 3), ranked[0], conflict, alts)


# ------------------------------------------------------------------ калибровка -------------------------------------
def attr_group(spec: AttrSpec) -> str:
    """Тип атрибута для калибровки: у бренда, модели, цвета и процессора своя «кривая» точности."""
    if spec.dictionary == "brands":
        return "brand"
    if spec.kind == "model":
        return "model"
    if spec.dictionary == "colors":
        return "color"
    if spec.kind == "cpu":
        return "cpu"
    return {"enum": "enum", "number": "number", "bool": "bool"}.get(spec.type, "string")


def group_keys(category: str, attr: str, spec: AttrSpec, sources: list[str] | set[str], conflict: bool) -> list[str]:
    """Группы калибровки от частной к общей: «категория/атрибут/источники[|conflict]» -> «источники|тип[|conflict]»
    -> «*|тип». Одна функция для обучения (eval/calibrate.py) и применения (Calibrator)."""
    src = "+".join(sorted(sources))
    g = attr_group(spec)
    c = "|conflict" if conflict else ""
    keys = [f"{category}/{attr}/{src}{c}", f"{category}/{attr}/{src}"] if conflict else [f"{category}/{attr}/{src}"]
    keys += [f"{src}|{g}{c}", f"{src}|{g}"] if conflict else [f"{src}|{g}"]
    return keys + [f"*|{g}"]


def pav(xs: list[float], ys: list[float], weights: list[float] | None = None) -> tuple[list[float], list[float]]:
    """Изотоническая регрессия (pool adjacent violators): неубывающая y(x). -> (границы блоков x, значения y)."""
    pts = sorted(zip(xs, ys, weights or [1.0] * len(xs), strict=True))
    blocks: list[list[float]] = []  # [x_min, x_max, sum_wy, sum_w]
    for x, y, w in pts:
        blocks.append([x, x, y * w, w])
        while len(blocks) > 1 and blocks[-2][2] / blocks[-2][3] >= blocks[-1][2] / blocks[-1][3]:
            b = blocks.pop()
            blocks[-1][1] = b[1]
            blocks[-1][2] += b[2]
            blocks[-1][3] += b[3]
    return [b[0] for b in blocks], [round(b[2] / b[3], 4) for b in blocks]


@dataclass
class Curve:
    x: list[float]  # нижние границы блоков сырой уверенности
    y: list[float]  # точность блока
    n: int = 0

    def __call__(self, raw: float) -> float:
        i = bisect_right(self.x, raw) - 1
        return self.y[max(0, i)]


class Calibrator:
    """Сырая уверенность -> P(значение верно) по самой частной группе (group_keys), для которой есть кривая;
    нет ни одной (новый источник/категория) — сырая оценка без изменений."""

    def __init__(self, data: dict | None = None):
        data = data or {}
        self.meta = {k: v for k, v in data.items() if k != "groups"}
        self.curves = {k: Curve(v["x"], v["y"], v.get("n", 0)) for k, v in (data.get("groups") or {}).items()}
        self.threshold: float | None = data.get("threshold")

    @property
    def enabled(self) -> bool:
        return bool(self.curves)

    def curve_for(self, category: str, attr: str, spec: AttrSpec, sources: list[str],
                  conflict: bool) -> tuple[str, Curve] | None:
        for k in group_keys(category, attr, spec, sources, conflict):
            if k in self.curves:
                return k, self.curves[k]
        return None

    def __call__(self, raw: float, sources: list[str], spec: AttrSpec, conflict: bool = False,
                 category: str = "", attr: str = "") -> float:
        hit = self.curve_for(category, attr or spec.name, spec, sources, conflict)
        return round(hit[1](raw), 3) if hit else raw

    def has_own(self, category: str, attr: str, spec: AttrSpec, sources: list[str], conflict: bool) -> bool:
        """Есть ли кривая именно для этого набора источников (а не общая «*|тип»)."""
        keys = group_keys(category, attr, spec, sources, conflict)
        return any(k in self.curves for k in keys[:-1])

    def monotone(self, agg: Aggregate, spec: AttrSpec, category: str, attr: str) -> float:
        """Калиброванная уверенность, не убывающая при согласии ещё одного источника. Для набора источников без своей
        кривой (новое сочетание: «правила + LLM», «продавец + NER»…) — максимум из общей кривой и калиброванной
        уверенности того же значения без «лишних» источников: подтверждение может только поднять уверенность."""
        conf = self(agg.confidence, agg.sources, spec, agg.conflict, category, attr)
        srcs = agg.sources
        if len(srcs) < 2 or self.has_own(category, attr, spec, srcs, agg.conflict):
            return conf
        penalty = 0.5 * agg.alternatives[0]["confidence"] if agg.conflict and agg.alternatives else 0.0
        for k in range(1, len(srcs)):
            for sub in combinations(srcs, k):
                raw = max(0.0, support_of([c for c in agg.support if c.source in sub]) - penalty)
                conf = max(conf, self(round(raw, 3), list(sub), spec, agg.conflict, category, attr))
        return conf


@lru_cache(maxsize=4)
def load_calibrator(path: str) -> Calibrator:
    """config/calibration.json (attrx calibrate); пустой путь или нет файла — без калибровки."""
    p = Path(path) if path else None
    if p is None or not p.is_file():
        return Calibrator()
    return Calibrator(json.loads(p.read_text(encoding="utf-8")))
