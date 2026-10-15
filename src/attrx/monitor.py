"""Мониторинг дрейфа по предсказаниям: `attrx monitor -i <predictions.jsonl> [--ref submission/predictions.jsonl]`.

Что меряется — по всем карточкам и по каждой категории таксономии:
- доли слотов «карточка × атрибут»: принято / на проверке (review) / пусто (missing); средняя уверенность найденных
  значений и доля значений с конфликтом источников;
- доля карточек с неуверенной категорией (предупреждение пайплайна «категория определена неуверенно»);
- доля generic — карточки вне таксономии: сюда уходят новые категории каталога;
- сдвиг распределения категорий относительно эталонного прогона (по умолчанию — итоговые предсказания holdout
  в submission/): PSI и дивергенция Йенсена — Шеннона; новые листья каталога, которых в эталоне не было;
- доля карточек с вызовом LLM (стоимость) и значения LLM, отброшенные проверкой заземления (галлюцинации).
Пороги — DEFAULT_THRESHOLDS (переопределяются `--threshold имя=значение`), сработавшие пороги — alerts с подсказкой,
что делать (петля дообучения: attrx learn, calibrate, train-category, train-ner --distant). Подробно — docs/MONITORING.md.
Модуль читает только JSONL предсказаний: без data/processed, torch и LLM.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # модуль импортирует `attrx --help` (пороги в справке) — pandas и pydantic не тянем
    from .schema import ExtractionResult

DEFAULT_THRESHOLDS: dict[str, float] = {
    "min_cards": 50,  # в категории меньше карточек — алерты по ней не поднимаются (шум); сводка по всем — всегда
    "review_share": 0.20,  # доля слотов на проверке выше — растёт ручная работа модераторов
    "missing_share": 0.60,  # доля пустых слотов выше — правила и NER перестали находить значения
    "mean_confidence": 0.75,  # средняя уверенность найденных значений ниже
    "category_unsure_share": 0.10,  # доля карточек с неуверенной категорией выше
    "generic_share": 0.60,  # доля карточек вне таксономии выше (только по всем карточкам)
    "llm_share": 0.50,  # доля карточек с вызовом LLM выше (бюджет)
    "delta": 0.05,  # рост доли review / missing / неуверенной категории / generic относительно эталона, в долях
    "delta_confidence": 0.05,  # падение средней уверенности относительно эталона
    "delta_llm_share": 0.10,  # рост доли карточек с вызовом LLM относительно эталона
    "psi": 0.25,  # сдвиг распределения категорий: < 0,1 — нет, 0,1–0,25 — умеренный, выше — сильный (алерт)
}
THRESHOLD_HELP = {
    "min_cards": "минимум карточек в категории для алертов по ней",
    "review_share": "доля слотов на проверке, выше — алерт",
    "missing_share": "доля пустых слотов, выше — алерт",
    "mean_confidence": "средняя уверенность найденных значений, ниже — алерт",
    "category_unsure_share": "доля карточек с неуверенной категорией, выше — алерт",
    "generic_share": "доля карточек вне таксономии (по всем), выше — алерт",
    "llm_share": "доля карточек с вызовом LLM, выше — алерт",
    "delta": "рост review / missing / неуверенной категории / generic против эталона",
    "delta_confidence": "падение средней уверенности против эталона",
    "delta_llm_share": "рост доли карточек с LLM против эталона",
    "psi": "PSI распределения категорий против эталона, выше — алерт",
}
_UNSURE = re.compile(r"^категория определена неуверенно")
_LLM_REJECTED = "llm: отброшено"
EPS = 1e-4  # сглаживание нулевых долей в PSI

# что делать, если сработал порог: так мониторинг замыкает петлю дообучения
ACTIONS = {
    "review_share": "разобрать очередь проверки (UI «Очередь»), затем attrx learn; устойчивый рост — attrx calibrate "
                    "на свежем dev",
    "missing_share": "посмотреть пустые слоты категории: новые формулировки и единицы — правила и справочники; "
                     "attrx train-ner --distant на свежем train",
    "mean_confidence": "attrx calibrate на свежем dev (уверенность разошлась с точностью), проверить порог",
    "category_unsure_share": "новые или смешанные категории: attrx prepare -> build-gold -> train-category",
    "generic_share": "новые категории вне таксономии: добавить в config/taxonomy.yaml (список new_leaves), "
                     "затем build-gold -> build-dicts -> calibrate",
    "psi": "сменился ассортимент: проверить new_leaves и category_unsure, при необходимости train-category и "
           "пересчёт эталона (build-gold) и калибровки",
    "llm_share": "LLM вызывается чаще: проверить LLM_ROUTE_THRESHOLD и бюджет (LLM_DAILY_MAX_CALLS), стоимость — "
                 "docs/LLM.md; рост обычно следует за ростом missing",
}
METRIC_TITLES = {
    "review_share": "доля значений на проверке", "missing_share": "доля пустых слотов",
    "mean_confidence": "средняя уверенность", "category_unsure_share": "доля неуверенной категории",
    "generic_share": "доля generic (вне таксономии)", "llm_share": "доля карточек с LLM", "psi": "PSI категорий",
}


# ------------------------------------------------------------------ статистика ---------------------------------------
def group_stats(results: list[ExtractionResult]) -> dict:
    """Сводка по набору карточек: доли слотов по статусам, уверенность, категория, LLM."""
    n = len(results)
    slots = acc = rev = miss = found = conflicts = unsure = llm = rejected = 0
    conf_sum = cat_conf = 0.0
    for r in results:
        unsure += any(_UNSURE.match(w) for w in r.warnings)
        rejected += sum(w.startswith(_LLM_REJECTED) for w in r.warnings)
        llm += bool(r.llm_used)
        cat_conf += r.category_confidence
        for a in r.attributes:
            slots += 1
            if a.status == "missing" or a.value is None:
                miss += 1
                continue
            found += 1
            conf_sum += a.confidence
            acc += a.status == "accepted"
            rev += a.status == "review"
            conflicts += bool(a.alternatives)

    def share(x: float, d: int) -> float | None:
        return round(x / d, 4) if d else None

    return {"n_cards": n, "slots": slots, "accepted_share": share(acc, slots), "review_share": share(rev, slots),
            "missing_share": share(miss, slots), "mean_confidence": share(conf_sum, found),
            "conflict_share": share(conflicts, found), "category_unsure_share": share(unsure, n),
            "mean_category_confidence": share(cat_conf, n), "llm_share": share(llm, n),
            "llm_rejected_per_card": share(rejected, n)}


def distribution(results: list[ExtractionResult]) -> Counter:
    return Counter(r.category for r in results)


def psi(cur: Counter, ref: Counter, eps: float = EPS) -> float:
    """Population Stability Index распределений категорий (доли сглажены снизу eps: новая категория — большой вклад)."""
    nc, nr = sum(cur.values()), sum(ref.values())
    if not nc or not nr:
        return 0.0
    total = 0.0
    for k in set(cur) | set(ref):
        p, q = max(cur[k] / nc, eps), max(ref[k] / nr, eps)
        total += (p - q) * math.log(p / q)
    return round(total, 4)


def js_divergence(cur: Counter, ref: Counter) -> float:
    """Дивергенция Йенсена — Шеннона (log2, от 0 — одинаковые распределения — до 1)."""
    nc, nr = sum(cur.values()), sum(ref.values())
    if not nc or not nr:
        return 0.0

    def kl(a: dict, b: dict) -> float:
        return sum(v * math.log2(v / b[k]) for k, v in a.items() if v > 0)

    keys = set(cur) | set(ref)
    p = {k: cur[k] / nc for k in keys}
    q = {k: ref[k] / nr for k in keys}
    m = {k: (p[k] + q[k]) / 2 for k in keys}
    return round(0.5 * kl(p, m) + 0.5 * kl(q, m), 4)


def psi_level(x: float) -> str:
    return "нет сдвига" if x < 0.1 else "умеренный" if x < 0.25 else "сильный"


def _leaf(r: ExtractionResult) -> str:
    return (r.category_path[-1] if r.category_path else None) or r.category_title or r.category


# ------------------------------------------------------------------ алерты -------------------------------------------
def _alert(scope: str, metric: str, value: float, threshold: float, kind: str, ref: float | None = None) -> dict:
    title = METRIC_TITLES.get(metric, metric)
    if kind == "max":
        msg = f"{title} {value:.3f} > {threshold:g}"
    elif kind == "min":
        msg = f"{title} {value:.3f} < {threshold:g}"
    elif kind == "drop":
        msg = f"{title} упала на {ref - value:.3f} (эталон {ref:.3f}, сейчас {value:.3f}; порог {threshold:g})"
    else:  # rise
        msg = f"{title} выросла на {value - ref:.3f} (эталон {ref:.3f}, сейчас {value:.3f}; порог {threshold:g})"
    return {"scope": scope, "metric": metric, "kind": kind, "value": value, "ref": ref, "threshold": threshold,
            "message": f"{scope}: {msg}", "action": ACTIONS.get(metric, "")}


def check(scope: str, cur: dict, ref: dict | None, t: dict[str, float], overall: bool = False) -> list[dict]:
    """Абсолютные пороги + дрейф относительно эталона для одной группы карточек."""
    out: list[dict] = []
    for m in ("review_share", "missing_share", "category_unsure_share", "llm_share") + (("generic_share",) if overall else ()):
        v = cur.get(m)
        if v is not None and v > t[m]:
            out.append(_alert(scope, m, v, t[m], "max"))
    if (v := cur.get("mean_confidence")) is not None and v < t["mean_confidence"]:
        out.append(_alert(scope, "mean_confidence", v, t["mean_confidence"], "min"))
    if not ref:
        return out
    for m in ("review_share", "missing_share", "category_unsure_share") + (("generic_share",) if overall else ()):
        v, r = cur.get(m), ref.get(m)
        if v is not None and r is not None and v - r > t["delta"]:
            out.append(_alert(scope, m, v, t["delta"], "rise", r))
    v, r = cur.get("llm_share"), ref.get("llm_share")
    if v is not None and r is not None and v - r > t["delta_llm_share"]:
        out.append(_alert(scope, "llm_share", v, t["delta_llm_share"], "rise", r))
    v, r = cur.get("mean_confidence"), ref.get("mean_confidence")
    if v is not None and r is not None and r - v > t["delta_confidence"]:
        out.append(_alert(scope, "mean_confidence", v, t["delta_confidence"], "drop", r))
    return out


def parse_thresholds(items: list[str] | None) -> dict[str, float]:
    """['psi=0.2', 'min_cards=100'] -> пороги поверх DEFAULT_THRESHOLDS; неизвестное имя — ValueError."""
    t = dict(DEFAULT_THRESHOLDS)
    for item in items or []:
        name, sep, value = item.partition("=")
        name = name.strip()
        if not sep or name not in t:
            raise ValueError(f"порог {item!r}: ожидается имя=значение, имена: {', '.join(t)}")
        t[name] = float(value)
    return t


# ------------------------------------------------------------------ отчёт --------------------------------------------
def monitor(results: list[ExtractionResult], ref: list[ExtractionResult] | None = None,
            thresholds: dict[str, float] | None = None, info: dict | None = None) -> dict:
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    by_cat: dict[str, list[ExtractionResult]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)
    ref_by_cat: dict[str, list[ExtractionResult]] = defaultdict(list)
    for r in ref or []:
        ref_by_cat[r.category].append(r)
    n, n_ref = len(results), len(ref or [])
    cur_d, ref_d = distribution(results), distribution(ref or [])

    overall = group_stats(results) | {"generic_share": round(cur_d.get("generic", 0) / n, 4) if n else None}
    ref_overall = (group_stats(ref) | {"generic_share": round(ref_d.get("generic", 0) / n_ref, 4)}) if n_ref else None
    alerts = check("все карточки", overall, ref_overall, t, overall=True)

    per_cat: dict[str, dict] = {}
    for cat in sorted(by_cat, key=lambda c: -len(by_cat[c])):
        cur = group_stats(by_cat[cat])
        rs = group_stats(ref_by_cat[cat]) if ref_by_cat.get(cat) else None
        item = {**cur, "share": round(len(by_cat[cat]) / n, 4),
                "ref_share": round(len(ref_by_cat[cat]) / n_ref, 4) if n_ref else None, "ref": rs}
        if rs:
            item["delta"] = {k: round(cur[k] - rs[k], 4) for k in ("review_share", "missing_share", "mean_confidence",
                                                                    "category_unsure_share", "llm_share")
                             if cur.get(k) is not None and rs.get(k) is not None}
        item["alerts"] = check(cat, cur, rs, t) if cur["n_cards"] >= t["min_cards"] else []
        alerts += item["alerts"]
        per_cat[cat] = item

    shift = None
    if n_ref:
        ref_leaves = {_leaf(r) for r in ref}
        new_leaves = Counter(_leaf(r) for r in results if r.category == "generic" and _leaf(r) not in ref_leaves)
        p = psi(cur_d, ref_d)
        shift = {"psi": p, "js": js_divergence(cur_d, ref_d), "level": psi_level(p),
                 "new_categories": sorted(set(cur_d) - set(ref_d)),
                 "missing_categories": sorted(set(ref_d) - set(cur_d)),
                 "distribution": {k: {"n": cur_d.get(k, 0), "share": round(cur_d.get(k, 0) / n, 4) if n else 0.0,
                                      "ref_share": round(ref_d.get(k, 0) / n_ref, 4)}
                                  for k in sorted(set(cur_d) | set(ref_d), key=lambda k: -cur_d.get(k, 0))},
                 "new_leaves": [{"title": k, "n": v} for k, v in new_leaves.most_common(20)]}
        if p > t["psi"]:
            alerts.append(_alert("все карточки", "psi", p, t["psi"], "max"))

    actions = []
    for a in alerts:
        if a["action"] and a["action"] not in actions:
            actions.append(a["action"])
    return {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **(info or {}), "thresholds": t,
            "n_cards": n, "ref_n_cards": n_ref or None, "overall": overall, "reference_overall": ref_overall,
            "category_shift": shift, "per_category": per_cat, "alerts": alerts,
            "status": "alerts" if alerts else "ok", "next_steps": actions}


def feedback_stats(path: Path) -> dict | None:
    """Петля обратной связи: решения модератора из журнала (outputs/feedback.jsonl) — подтверждено / исправлено /
    отклонено по категориям — и сколько синонимов-кандидатов ждут подтверждения (attrx learn)."""
    if not path.exists():
        return None
    import json

    import yaml

    from .feedback import CANDIDATES_FILE, decision_action

    total: Counter = Counter()
    by_cat: dict[str, Counter] = defaultdict(Counter)
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        a = decision_action(r)
        total[a] += 1
        by_cat[r.get("category") or "?"][a] += 1
    n = sum(total.values())
    cand = path.with_name(CANDIDATES_FILE)
    pending = None
    if cand.exists():
        data = yaml.safe_load(cand.read_text(encoding="utf-8")) or {}
        pending = sum(len(v or {}) for v in data.values())
    return {"path": str(path), "decisions": n, **{k: total[k] for k in ("accepted", "corrected", "rejected")},
            "corrected_share": round(total["corrected"] / n, 4) if n else None,
            "by_category": {k: dict(v) for k, v in sorted(by_cat.items())}, "pending_synonym_candidates": pending}


def run(input_path: Path, ref_path: Path | None = None, thresholds: dict[str, float] | None = None,
        feedback_path: Path | None = None) -> dict:
    from .io import read_predictions

    results = list(read_predictions(input_path))
    ref = list(read_predictions(ref_path)) if ref_path is not None and ref_path.exists() else None
    info = {"input": str(input_path), "reference": str(ref_path) if ref else None}
    if ref_path is not None and ref is None:
        info["note"] = f"эталонного прогона {ref_path} нет — только абсолютные пороги"
    rep = monitor(results, ref, thresholds, info)
    rep["feedback"] = feedback_stats(feedback_path) if feedback_path is not None else None
    return rep


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def table(rep: dict) -> str:
    """Короткая сводка для терминала."""
    head = f"{'категория':<14}{'карт.':>7}{'доля':>8}{'эталон':>8}{'review':>8}{'missing':>9}{'увер.':>7}" \
           f"{'кат.?':>7}{'LLM':>7}{'алерты':>8}"
    lines = [head, "-" * len(head)]
    for cat, v in rep["per_category"].items():
        conf = "—" if v["mean_confidence"] is None else f"{v['mean_confidence']:.3f}"
        lines.append(f"{cat:<14}{v['n_cards']:>7}{_pct(v['share']):>8}{_pct(v['ref_share']):>8}"
                     f"{_pct(v['review_share']):>8}{_pct(v['missing_share']):>9}{conf:>7}"
                     f"{_pct(v['category_unsure_share']):>7}{_pct(v['llm_share']):>7}{len(v['alerts']):>8}")
    o = rep["overall"]
    lines.append("-" * len(head))
    conf = "—" if o["mean_confidence"] is None else f"{o['mean_confidence']:.3f}"
    lines.append(f"{'все':<14}{o['n_cards']:>7}{'':>8}{'':>8}{_pct(o['review_share']):>8}{_pct(o['missing_share']):>9}"
                 f"{conf:>7}{_pct(o['category_unsure_share']):>7}{_pct(o['llm_share']):>7}")
    if (s := rep["category_shift"]) is not None:
        lines.append(f"Сдвиг категорий против эталона ({rep['ref_n_cards']} карт.): PSI {s['psi']} ({s['level']}), "
                     f"JS {s['js']}; generic {_pct(o['generic_share'])} (эталон "
                     f"{_pct(rep['reference_overall']['generic_share'])})")
        if s["new_leaves"]:
            lines.append("Новые листья в generic (нет в эталоне): "
                         + ", ".join(f"{x['title']} ({x['n']})" for x in s["new_leaves"][:8]))
    elif rep.get("note"):
        lines.append(rep["note"])
    if fb := rep.get("feedback"):
        pend = fb["pending_synonym_candidates"]
        lines.append(f"Решения модератора: {fb['decisions']} (подтверждено {fb['accepted']}, исправлено {fb['corrected']}, "
                     f"отклонено {fb['rejected']})" + (f"; кандидатов в синонимы ждут подтверждения: {pend}"
                                                       if pend is not None else "") + " — attrx learn")
    return "\n".join(lines)
