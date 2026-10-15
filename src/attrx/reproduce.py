"""Сверка итоговых цифр: `attrx verify` (`make verify`).

Пересчитывает строки «правила» и «продавец + правила» из submission/compare.json на закоммиченном наборе holdout
(data/sample/holdout_eval_cards.jsonl + holdout_eval_gold.jsonl) ТЕМ ЖЕ кодом, что и итоговый отчёт
(`attrx report-real`: eval.final.run_mode -> summarize -> compare_row), и печатает рядом с числами из compare.json.
Расхождение больше допуска (0,002) — код возврата 1. Эти режимы не используют NER и LLM, поэтому сверка идёт офлайн:
без data/processed, torch и сети, за секунды. Строки с NER и LLM так не проверить (нужны веса NER и LLM) — они
воспроизводятся `make report` на данных организаторов.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import ROOT

SAMPLE = ROOT / "data" / "sample"
DEFAULT_CARDS = SAMPLE / "holdout_eval_cards.jsonl"
DEFAULT_GOLD = SAMPLE / "holdout_eval_gold.jsonl"
DEFAULT_COMPARE = ROOT / "submission" / "compare.json"
VERIFY_MODES = ("правила", "продавец + правила")
METRICS = ("precision", "recall", "f1", "recall_extractable", "macro_f1", "traffic_weighted_f1", "coverage",
           "catalog_fill", "union_fill", "review_rate", "accepted_precision")
TOL = 0.002


class VerifyError(RuntimeError):
    """Сверку нельзя выполнить (нет файлов, нет строки в compare.json) — текст для пользователя."""


def _rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


def read_cards(path: Path) -> list:
    """Карточки из JSONL (формат Card: id, title, description, seller_attrs, category, category_id, seller;
    допускаются имена data/processed: sku_id, description_clean). Порядок — как в файле."""
    from .schema import Card

    cards = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        if "id" not in r and "sku_id" in r:
            r["id"] = r["sku_id"]
        if "description" not in r and "description_clean" in r:
            r["description"] = r["description_clean"]
        if isinstance(r.get("seller_attrs"), str):
            r["seller_attrs"] = json.loads(r["seller_attrs"] or "{}")
        cards.append(Card.model_validate(r))
    return cards


def recompute(cards_path: Path = DEFAULT_CARDS, gold_path: Path = DEFAULT_GOLD,
              modes: tuple[str, ...] = VERIFY_MODES) -> tuple[list[dict], dict]:
    """Строки compare.json для режимов modes на наборе из JSONL -> (строки, сведения о прогоне)."""
    for p in (cards_path, gold_path):
        if not p.exists():
            raise VerifyError(
                f"нет {_rel(p)}: набор holdout для сверки выгружается вместе с итоговым отчётом (make report, поток "
                f"оценки) и коммитится в data/sample/. Без него цифры submission/compare.json сверить нельзя.")
    from .config import Settings
    from .eval import final
    from .eval.gold import load_jsonl
    from .pipeline import Extractor

    if bad := [m for m in modes if m not in final.MODES or {"ml", "llm"} & set(final.MODES[m])]:
        raise VerifyError(f"режимы {bad} нельзя сверить офлайн (нужны NER или LLM); доступны: {', '.join(VERIFY_MODES)}")
    cards = read_cards(cards_path)
    gold, meta = load_jsonl(gold_path)
    settings = Settings(llm_enabled=False, ml_enabled=False)
    ex = Extractor(settings=settings)
    rows = []
    for name in modes:
        sources = final.MODES[name]
        res, perf = final.run_mode(ex, cards, sources)
        m = final.summarize(res, gold, meta, ex.tax, perf, mode=name, set="full", sources=list(sources))
        rows.append(final.compare_row(name, "full", sources, m))
    info = {"cards": _rel(cards_path), "gold": _rel(gold_path), "n_cards": len(cards), "n_gold_cards": len(gold),
            "threshold": settings.confidence_threshold}
    return rows, info


SET_PREFERENCE = ("stratified", "full", "full_holdout")  # набор, выгруженный в data/sample, — стратифицированный


def base_mode(mode: str) -> str:
    """«правила · стратифицированный набор» -> «правила» (суффикс набора в compare.json)."""
    return str(mode).split(" · ")[0].strip()


def reference_row(reference: list[dict], mode: str, n_cards: int, set_name: str = "auto") -> dict | None:
    """Строка compare.json для режима: тот же режим (без суффикса набора), не подвыборка LLM и не срез; из наборов —
    с тем же числом карточек, что в выгрузке (так выгрузка однозначно сопоставляется со своим набором), затем по
    SET_PREFERENCE. set_name — выбрать набор явно."""
    rows = [r for r in reference if base_mode(r.get("mode", "")) == mode and r.get("set") != "llm_subsample"]
    if set_name != "auto":
        rows = [r for r in rows if r.get("set") == set_name]
    if not rows:
        return None

    def rank(r: dict) -> tuple:
        s = r.get("set", "full")
        return (r.get("n_cards") != n_cards, SET_PREFERENCE.index(s) if s in SET_PREFERENCE else len(SET_PREFERENCE))

    return min(rows, key=rank)


def compare(rows: list[dict], reference: list[dict], tol: float = TOL, set_name: str = "auto") -> list[dict]:
    """Сравнение пересчитанных строк с compare.json: [{mode, set, metric, expected, actual, delta, ok}]."""
    checks = []
    for row in rows:
        exp = reference_row(reference, row["mode"], row["n_cards"], set_name)
        if exp is None:
            raise VerifyError(f"в compare.json нет строки «{row['mode']}»"
                              + (f" (set={set_name})" if set_name != "auto" else ""))
        row["matched"] = {"mode": exp.get("mode"), "set": exp.get("set")}
        checks.append({"mode": row["mode"], "set": exp.get("set"), "metric": "n_cards", "expected": exp.get("n_cards"),
                       "actual": row["n_cards"], "delta": None, "ok": exp.get("n_cards") == row["n_cards"]})
        for k in METRICS:
            e, a = exp.get(k), row.get(k)
            if e is None and a is None:
                continue
            d = None if e is None or a is None else round(a - e, 4)
            checks.append({"mode": row["mode"], "set": exp.get("set"), "metric": k, "expected": e, "actual": a,
                           "delta": d, "ok": d is not None and abs(d) <= tol + 1e-9})
    return checks


def verify(cards_path: Path = DEFAULT_CARDS, gold_path: Path = DEFAULT_GOLD, compare_path: Path = DEFAULT_COMPARE,
           tol: float = TOL, modes: tuple[str, ...] = VERIFY_MODES, set_name: str = "auto") -> dict:
    if not compare_path.exists():
        raise VerifyError(f"нет {_rel(compare_path)} — его пишет итоговый отчёт (make report)")
    reference = json.loads(compare_path.read_text(encoding="utf-8"))
    rows, info = recompute(cards_path, gold_path, modes)
    checks = compare(rows, reference, tol, set_name)
    return {"ok": all(c["ok"] for c in checks), "tol": tol, "compare": _rel(compare_path), **info,
            "rows": rows, "checks": checks}


def table(rep: dict) -> str:
    def f(x) -> str:
        return "—" if x is None else f"{x:.4f}" if isinstance(x, float) else str(x)

    head = f"{'режим':<22}{'набор':<13}{'метрика':<22}{'compare.json':>13}{'пересчёт':>10}{'Δ':>9}"
    lines = [head, "-" * (len(head) + 13)]
    for c in rep["checks"]:
        lines.append(f"{c['mode']:<22}{str(c.get('set')):<13}{c['metric']:<22}{f(c['expected']):>13}"
                     f"{f(c['actual']):>10}{f(c['delta']):>9}  {'ok' if c['ok'] else 'РАСХОЖДЕНИЕ'}")
    return "\n".join(lines)
