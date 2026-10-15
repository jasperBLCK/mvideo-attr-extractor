"""Финальный отчёт на holdout: `attrx report-real` (`make report`) -> submission/. Holdout считается ОДИН раз.

Набор (как dev): все карточки сплита с эталоном в категориях таксономии + 1000 карточек хвоста (generic) по полосам
размера категории (`real.stratified_ids`, seed 0) — КРОМЕ 400 SKU `data/sample/real_cards.jsonl`: на них показывали
демо и мерили LLM-бенчмарк, поэтому в честную оценку они не входят.

Режимы на всём наборе: «правила» -> «продавец + правила» -> «+ NER». LLM медленная (секунды на вызов на одной GPU),
поэтому она меряется на стратифицированной подвыборке (по умолчанию 600 карточек, из них >= 150 смартфонов и ноутбуков),
и на ЭТОЙ ЖЕ подвыборке прогоняются все четыре режима — прирост от LLM сравнивается честно.

Режим в сдаче выбирается заранее заданным правилом (до прогона): «+ NER», если его точность не ниже «продавец +
правила» больше чем на 0,005 и F1 не ниже; иначе «продавец + правила». LLM в сдачу не идёт (только подвыборка, и
значение от одной LLM всё равно уходит на проверку) — её вклад показан отдельной строкой.
"""
from __future__ import annotations

import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..io import write_predictions
from ..schema import Card, ExtractionResult
from ..taxonomy import Taxonomy
from . import real
from .calibrate import collect, wilson_lower
from .gold import load_parquet
from .metrics import load_traffic_weights
from .report import errors as error_rows

MODES: dict[str, tuple[str, ...]] = {
    "правила": ("rules",),
    "продавец + правила": ("seller", "rules"),
    "+ NER": ("seller", "rules", "ml"),
    "+ LLM": ("seller", "rules", "ml", "llm"),
}
FULL_MODES = ("правила", "продавец + правила", "+ NER")
SUB_SUFFIX = " · подвыборка LLM"
TARGET_PRECISION = 0.95
ERRORS_CAP = 5000
JSONL_CAP_MB = 25.0


# ------------------------------------------------------------------ набор карточек ---------------------------------
def sample_ids(root: Path) -> set[str]:
    p = root / "data" / "sample" / "real_cards.jsonl"
    if not p.exists():
        return set()
    return {str(json.loads(x)["id"]) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()}


def eval_set(processed: Path, split: str, exclude: set[str], seed: int = 0,
             ) -> tuple[list[str], dict[str, dict], dict[str, dict], int]:
    """(ids, эталон, мета, сколько SKU сэмпла исключено) — как dev в eval-real, без SKU сэмпла."""
    g, meta = load_parquet(processed / "gold.parquet", splits={split})
    excluded = sum(1 for k in meta if k in exclude)
    meta_ex = {k: v for k, v in meta.items() if k not in exclude}
    return real.stratified_ids(meta_ex, 5000, 1000, seed=seed), g, meta, excluded


def llm_subsample(ids: list[str], meta: dict[str, dict], n: int = 600, core_min: int = 150, seed: int = 0) -> list[str]:
    """Стратифицированная подвыборка для LLM: смартфоны+ноутбуки (>= core_min), остальные категории таксономии
    (не меньше 12 на категорию), хвост generic по полосам размера (не меньше 40 на полосу). Порядок — как в ids."""
    rnd = random.Random(f"llm-subsample:{seed}")
    groups: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        m = meta[i]
        groups[m["category"] if m["category"] != "generic" else f"generic/{m.get('size_band', '?')}"].append(i)
    core = [k for k in ("smartphone", "laptop") if groups.get(k)]
    gen = sorted(k for k in groups if k.startswith("generic/"))
    other = sorted(k for k in groups if k not in core and k not in gen)
    n_core = max(core_min, round(n * 0.28))
    n_gen = round(n * 0.33)
    n_other = max(0, n - n_core - n_gen)

    def alloc(keys: list[str], total: int, floor: int) -> dict[str, int]:
        avail = {k: len(groups[k]) for k in keys}
        s = sum(avail.values()) or 1
        return {k: min(avail[k], max(floor, round(total * avail[k] / s))) for k in keys}

    quotas = alloc(core, n_core, 60) | alloc(other, n_other, 12) | alloc(gen, n_gen, 40)
    chosen: set[str] = set()
    for k in sorted(quotas):
        chosen |= set(rnd.sample(sorted(groups[k]), quotas[k]))
    return [i for i in ids if i in chosen]


def cold_start_cards(processed: Path, split: str, n: int, exclude: set[str], seed: int = 0) -> list[Card]:
    """Карточки сплита БЕЗ оффера (вне фида: эталона нет, 75% каталога) — для покрытия на cold start."""
    import duckdb

    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")
    rows = con.execute(f"""
        SELECT sku_id, title, description_clean, seller_attrs, category_id, category_title, seller
        FROM read_parquet('{processed / 'cards.parquet'}') WHERE split = '{split}' AND NOT has_offer
        ORDER BY md5('cold:' || sku_id) LIMIT {int(n) + len(exclude)}""").fetchall()
    out = [Card(id=s, title=t or "", description=d or "", seller_attrs=json.loads(sa or "{}"), category=ct,
                category_id=cid, seller=sel) for s, t, d, sa, cid, ct, sel in rows if s not in exclude]
    return out[:n]


# ------------------------------------------------------------------ прогон и метрики -------------------------------
def run_mode(ex, cards: list[Card], sources: tuple[str, ...]) -> tuple[list[ExtractionResult], dict]:
    llm = "llm" in sources
    if llm:
        ex.client.reset_stats()
    t = time.perf_counter()
    results = list(ex.extract_many(cards, use_llm=llm, sources=set(sources)))
    dt = time.perf_counter() - t
    n = max(1, len(cards))
    perf: dict[str, Any] = {"seconds": round(dt, 2), "cards_per_sec": round(n / dt, 1) if dt else None,
                            "ms_per_card": round(1000 * dt / n, 2), "llm_calls": 0, "llm_share": 0.0}
    if llm:
        st = ex.client.stats()
        perf |= {"llm_stats": st, "llm_calls": st["requests"], "llm_api_calls": st["calls"],
                 "llm_share": round(sum(r.llm_used for r in results) / n, 4),
                 "llm_rejected_values": sum(w.startswith("llm: отброшено") for r in results for w in r.warnings)}
    perf["llm_calls_per_1000"] = round(1000 * perf["llm_calls"] / n, 1)
    return results, perf


def reliability(results: list[ExtractionResult], gold: dict, tax: Taxonomy, n_bins: int = 10) -> dict:
    """Диаграмма надёжности на holdout: значения с эталоном (любой статус) по бинам уверенности -> доля верных; ECE."""
    pts = collect(results, gold, tax)
    bins: dict[int, list] = defaultdict(list)
    for p in pts:
        bins[min(n_bins - 1, int(p.raw * n_bins))].append(p)
    rows, ece = [], 0.0
    for b in range(n_bins):
        ps = bins.get(b, [])
        if not ps:
            continue
        conf = sum(p.raw for p in ps) / len(ps)
        acc = sum(p.ok for p in ps) / len(ps)
        ece += abs(conf - acc) * len(ps)
        rows.append({"lo": round(b / n_bins, 2), "hi": round((b + 1) / n_bins, 2), "confidence": round(conf, 4),
                     "accuracy": round(acc, 4), "n": len(ps)})
    return {"ece": round(ece / max(1, len(pts)), 4), "n": len(pts), "bins": rows}


def accepted_precision(results: list[ExtractionResult], gold: dict, tax: Taxonomy) -> dict:
    """Точность принятых значений с 95% доверительным интервалом (Уилсон) и доля принятых среди предсказанных."""
    pts = collect(results, gold, tax)
    status = {(r.card_id, a.attribute): a.status for r in results for a in r.attributes}
    acc = [p for p in pts if status.get((p.card, p.keys[0].split("/")[1])) == "accepted"]
    tp = sum(p.ok for p in acc)
    return {"precision": round(tp / len(acc), 4) if acc else None, "lower95": round(wilson_lower(tp, len(acc)), 4),
            "accepted": len(acc), "predicted_with_gold": len(pts),
            "accepted_share": round(len(acc) / len(pts), 4) if pts else None}


def traffic_coverage(m: dict, weights: dict[tuple[str, str], float]) -> dict:
    """«Доля поискового спроса, по которой атрибут заполнен»: Σ W(c,a)·заполненность / Σ W(c,a) по ячейкам
    (W — доля кликов категории × спрос на атрибут), в каталоге и после дозаполнения пайплайном."""
    num_b = num_a = den = 0.0
    for cat, attrs in m["per_category"].items():
        for a, v in attrs.items():
            w = weights.get((cat, a), 0.0)
            if not v["n_cards"] or not w:
                continue
            den += w
            num_b += w * v["catalog_fill"]
            num_a += w * v["union_fill"]
    return {"before": round(num_b / den, 4) if den else None, "after": round(num_a / den, 4) if den else None}


def summarize(results: list[ExtractionResult], gold: dict, meta: dict, tax: Taxonomy, perf: dict, **info) -> dict:
    m = real.report(results, gold, meta, tax, perf, **info)
    m["accepted_precision"] = accepted_precision(results, gold, tax)
    m["traffic_weighted_coverage"] = traffic_coverage(m, load_traffic_weights(tax))
    return m


def llm_review_value(res_llm: list[ExtractionResult], res_ner: list[ExtractionResult], gold: dict, meta: dict,
                     tax: Taxonomy, threshold: float) -> dict:
    """Что LLM даёт модератору: значения ТОЛЬКО от LLM (по инварианту доверия — на проверке) — сколько их, сколько
    закрывают пустой без LLM слот, их точность по эталону; и контрфакт «принимать их автоматически при уверенности
    выше порога» — сколько стоил бы отказ от инварианта."""
    from .metrics import _as_list, evaluate, is_match, match_key

    before = {(r.card_id, a.attribute): a.value for r in res_ner for a in r.attributes}
    n = new_slot = with_gold = ok = 0
    cf: list[ExtractionResult] = []
    for r in res_llm:
        cat = tax.categories[r.category]
        g = gold.get(r.card_id) or {}
        brand = (_as_list(g.get("brand")) or [None])[0]
        attrs = []
        for a in r.attributes:
            if a.value is not None and a.sources == ["llm"]:
                n += 1
                new_slot += before.get((r.card_id, a.attribute)) is None
                spec = cat.attributes[a.attribute]
                gk = [k for k in (match_key(v, spec, tax, brand) for v in _as_list(g.get(a.attribute))) if k is not None]
                if gk:
                    with_gold += 1
                    pk = match_key(a.value, spec, tax, brand)
                    ok += pk is not None and is_match(pk, gk, spec)
                if a.confidence >= threshold:
                    a = a.model_copy(update={"status": "accepted"})
            attrs.append(a)
        cf.append(r.model_copy(update={"attributes": attrs}))
    m = evaluate(cf, gold, tax, meta=meta, weights=load_traffic_weights(tax))
    return {"llm_only_values": n, "new_slots": new_slot, "with_gold": with_gold,
            "precision": round(ok / with_gold, 4) if with_gold else None,
            "counterfactual_accept_llm_only": {k: m["micro"][k] for k in ("precision", "recall", "f1")}
            | {"traffic_weighted_f1": m["traffic_weighted_f1"], "review_rate": m["review_rate"]}}


def compare_row(name: str, set_name: str, sources: tuple[str, ...], m: dict) -> dict:
    p = m["perf"]
    return {"mode": name, "set": set_name, "n_cards": m["n_cards"], "sources": list(sources),
            "precision": m["micro"]["precision"], "recall": m["micro"]["recall"], "f1": m["micro"]["f1"],
            "recall_extractable": m["recall_extractable"], "macro_f1": m["macro_f1"],
            "traffic_weighted_f1": m["traffic_weighted_f1"], "coverage": m["coverage"],
            "union_fill": m["union_fill"], "catalog_fill": m["catalog_fill"], "new_fills": m["new_fills"],
            "review_rate": m["review_rate"], "accepted_precision": m["accepted_precision"]["precision"],
            "seconds": p["seconds"], "cards_per_sec": p["cards_per_sec"], "ms_per_card": p["ms_per_card"],
            "llm_calls": p["llm_calls"], "llm_calls_per_1000": p["llm_calls_per_1000"], "llm_share": p["llm_share"]}


def choose_final(rows: dict[str, dict]) -> str:
    """Правило, заданное ДО прогона holdout: «+ NER», если не теряет точность (> 0,005) и не хуже по F1."""
    base, ner = rows.get("продавец + правила"), rows.get("+ NER")
    if ner and base and ner["precision"] >= base["precision"] - 0.005 and ner["f1"] >= base["f1"]:
        return "+ NER"
    return "продавец + правила"


# ------------------------------------------------------------------ стоимость --------------------------------------
def cost_block(compare: list[dict], llm_perf: dict | None, n_sub: int, prices: dict) -> dict:
    """cost.json: ₽ на 1000 карточек по режимам. Свой стенд — по времени: (амортизация + электричество) за час ×
    время пайплайна на карточку (LLM на нём же, последовательно). API-провайдеры — по токенам замера."""
    from ..llm.cost import cost_per_call, hourly_cost

    provs = prices.get("providers") or {}
    usd = float(prices.get("usd_rub", 90))
    stand = provs.get("onprem-gtx1060", {})
    rub_h = hourly_cost(stand.get("hourly", {})) if stand else 0.0
    rows = []
    for r in compare:
        if r["set"] != "full" and r["mode"] != "+ LLM" + SUB_SUFFIX:
            continue
        rows.append({"mode": r["mode"], "basis": "замер", "provider": "свой стенд (GTX 1060)",
                     "cost_per_1000": round(rub_h * r["ms_per_card"] / 3600, 4), "ms_per_card": r["ms_per_card"],
                     "cards_per_sec": r["cards_per_sec"], "llm_share": r["llm_share"],
                     "llm_calls_per_1000": r["llm_calls_per_1000"]})
    refs: list[dict] = []
    if llm_perf and llm_perf.get("llm_stats", {}).get("requests"):
        st = llm_perf["llm_stats"]
        calls = max(1, st["calls"])
        pt, ct = st["prompt_tokens"] / calls, st["completion_tokens"] / calls
        per_card = llm_perf["llm_calls"] / max(1, n_sub)
        for pid in ("gigachat-lite", "gigachat-pro", "yandexgpt-lite", "yandexgpt-pro"):
            p = provs.get(pid)
            c = cost_per_call(p, pt, ct, usd) if p else None
            if c is not None:
                refs.append({"mode": f"+ LLM через {p['title'].split(' (')[0]} (оценка)", "basis": "оценка по токенам замера",
                             "provider": p["title"], "cost_per_1000": round(1000 * c * per_card, 2),
                             "llm_share": llm_perf["llm_share"], "llm_calls_per_1000": llm_perf["llm_calls_per_1000"]})
    # «LLM на каждую карточку» — замер бенчмарка (attrx llm-bench, docs/llm_bench.json): все атрибуты категории,
    # 1102 + 199 токенов, 6,9 с на карточку на свободном стенде
    every_pt, every_ct, every_s = 1102, 199, 6.9
    refs.append({"mode": "LLM на каждую карточку: свой стенд (оценка)", "basis": "бенчмарк llm-bench", "provider": "свой стенд",
                 "cost_per_1000": round(rub_h * every_s * 1000 / 3600, 2), "ms_per_card": every_s * 1000,
                 "cards_per_sec": round(1 / every_s, 3), "llm_share": 1.0, "llm_calls_per_1000": 1000})
    for pid in ("gigachat-lite", "yandexgpt-lite"):
        p = provs.get(pid)
        c = cost_per_call(p, every_pt, every_ct, usd) if p else None
        if c is not None:
            refs.append({"mode": f"LLM на каждую карточку: {p['title'].split(' (')[0]} (оценка)", "basis": "бенчмарк llm-bench",
                         "provider": p["title"], "cost_per_1000": round(1000 * c, 2), "llm_share": 1.0,
                         "llm_calls_per_1000": 1000})
    note = (f"Свой стенд (GTX 1060 6 ГБ, Ollama qwen3:4b-instruct): {rub_h:.2f} ₽/ч = амортизация 35 000 ₽ за 3 года 24/7 "
            "+ 0,2 кВт × 8 ₽/кВт·ч (config/llm_prices.yaml); стоимость режима = ₽/ч × время пайплайна на карточку "
            "(LLM на той же машине, последовательно, LLM_CONCURRENCY=1). Режимы без LLM — время CPU/GPU того же стенда. "
            "API-провайдеры (GigaChat, YandexGPT) — по тарифам с НДС на 14.10.2026 и токенам этого замера; "
            "«LLM на каждую карточку» — по бенчмарку attrx llm-bench (все атрибуты категории, 1102 + 199 токенов, "
            "6,9 с на карточку на свободном стенде). Гибрид зовёт LLM только для слабых атрибутов и только если они "
            "упомянуты в тексте (префильтр); значение от одной LLM всё равно уходит на проверку модератору.")
    return {"note": note, "rub_per_hour_stand": round(rub_h, 3), "modes": rows + refs}


# ------------------------------------------------------------------ файлы сдачи ------------------------------------
def write_errors(results: list[ExtractionResult], gold: dict, meta: dict, tax: Taxonomy, cards: list[Card],
                 path: Path, cap: int = ERRORS_CAP) -> dict:
    rows = error_rows(results, gold, tax)
    by_id = {c.id: c for c in cards}
    for r in rows:
        m = meta.get(r["card_id"]) or {}
        r["extractable"] = (m.get("extractable") or {}).get(r["attribute"])
        r["seller_type"], r["size_band"] = m.get("seller_type"), m.get("size_band")
        r["title"] = (by_id[r["card_id"]].title if r["card_id"] in by_id else "")[:140]
    counts = Counter(r["kind"] for r in rows)
    missed = [r for r in rows if r["kind"] == "missed"]
    stats = {"missed_not_in_text": sum(1 for r in missed if r["extractable"] is False),
             "missed_in_review": sum(1 for r in missed if r["extractable"] is not False and r["status"] == "review"),
             "missed_not_found": sum(1 for r in missed if r["extractable"] is not False and r["status"] == "missing"),
             "top_wrong": Counter(f"{r['category']}/{r['attribute']}" for r in rows
                                  if r["kind"] == "wrong").most_common(8),
             "top_missed_found_in_text": Counter(f"{r['category']}/{r['attribute']}" for r in missed
                                                 if r["extractable"] and r["status"] == "missing").most_common(8)}
    if len(rows) > cap:  # стратифицированно по типу ошибки: неверные значения — целиком, остальное пропорционально
        rnd = random.Random("errors")
        keep = [r for r in rows if r["kind"] == "wrong"][:cap]
        rest = [r for r in rows if r["kind"] != "wrong"]
        room = cap - len(keep)
        by_kind = defaultdict(list)
        for r in rest:
            by_kind[r["kind"]].append(r)
        for _kind, rs in sorted(by_kind.items()):
            q = round(room * len(rs) / max(1, len(rest)))
            keep += rnd.sample(rs, min(len(rs), q))
        rows = sorted(keep, key=lambda r: (r["kind"] != "wrong", r["category"], r["attribute"], r["card_id"]))
    cols = ["card_id", "category", "attribute", "kind", "predicted", "gold", "status", "confidence", "sources",
            "evidence", "extractable", "seller_type", "size_band", "title"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return {"total": sum(counts.values()), "by_kind": dict(counts), "written": len(rows), **stats}


def write_predictions_capped(results: list[ExtractionResult], out_dir: Path) -> dict:
    """predictions.jsonl (полные ExtractionResult) + predictions.csv (плоско, все карточки); jsonl > 25 МБ —
    только категории таксономии (csv остаётся полным)."""
    write_predictions(results, out_dir / "predictions.jsonl")
    mb = (out_dir / "predictions.jsonl").stat().st_size / 2**20
    trimmed = False
    if mb > JSONL_CAP_MB:
        with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
            for r in results:
                if r.category != "generic":
                    f.write(r.model_dump_json() + "\n")
        trimmed = True
    return {"jsonl_mb": round((out_dir / "predictions.jsonl").stat().st_size / 2**20, 2), "jsonl_trimmed": trimmed,
            "csv_mb": round((out_dir / "predictions.csv").stat().st_size / 2**20, 2)}


def _dump(obj: Any, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def cold_start_stats(ex, cards: list[Card], sources: tuple[str, ...]) -> dict:
    """Карточки вне фида (без эталона): сколько слотов схемы заполняется автоматически и как определялась категория."""
    if not cards:
        return {}
    t = time.perf_counter()
    results = list(ex.extract_many(cards, use_llm=False, sources=set(sources)))
    dt = time.perf_counter() - t
    slots = sum(len(r.attributes) for r in results)
    acc = sum(a.status == "accepted" for r in results for a in r.attributes)
    rev = sum(a.status == "review" for r in results for a in r.attributes)
    cats = Counter(r.category for r in results)
    unsure = sum(any(w.startswith("категория определена неуверенно") for w in r.warnings) for r in results)
    with_id = sum(1 for c in cards if c.category_id not in (None, ""))
    return {"n_cards": len(cards), "slots": slots, "accepted_share": round(acc / slots, 4) if slots else None,
            "review_share": round(rev / slots, 4) if slots else None,
            "cards_with_category_id": with_id, "category_unsure_share": round(unsure / len(cards), 4),
            "taxonomy_share": round(1 - cats.get("generic", 0) / len(cards), 4), "categories": dict(cats.most_common()),
            "cards_per_sec": round(len(cards) / dt, 1) if dt else None}


def run_final(root: Path, processed: Path, out_dir: Path, *, split: str = "holdout", llm_n: int = 600,
              cold_n: int = 1000, seed: int = 0, log=print) -> dict:
    """Весь финальный прогон: набор -> режимы -> подвыборка LLM -> cold start -> файлы submission/ + REPORT.md."""
    from ..confidence import load_calibrator
    from ..config import Settings, get_settings
    from ..llm.cost import load_prices
    from ..pipeline import Extractor

    t_all = time.perf_counter()
    exclude = sample_ids(root)
    ids, g, meta, n_excl = eval_set(processed, split, exclude, seed)
    cards = real.load_cards_by_id(ids, processed)
    ex = Extractor(settings=Settings(ml_enabled=True))
    tax = ex.tax
    log(f"{split}: {len(cards)} карточек (исключено SKU сэмпла: {n_excl}); NER: {ex.ml is not None}; "
        f"LLM: {ex.client.enabled}")
    full: dict[str, dict] = {}
    results: dict[str, list[ExtractionResult]] = {}
    for name in FULL_MODES:
        if "ml" in MODES[name] and ex.ml is None:
            continue
        res, perf = run_mode(ex, cards, MODES[name])
        full[name] = summarize(res, g, meta, tax, perf, mode=name, set="full", split=split, sources=list(MODES[name]))
        results[name] = res
        log(f"  {name}: P={full[name]['micro']['precision']} R={full[name]['micro']['recall']} "
            f"F1={full[name]['micro']['f1']} ({perf['cards_per_sec']} карт/с)")
    rows = [compare_row(n, "full", MODES[n], m) for n, m in full.items()]
    final = choose_final({r["mode"]: r for r in rows})
    sub: dict[str, dict] = {}
    sub_rows: list[dict] = []
    llm_perf = None
    sub_ids: list[str] = []
    sub_res: dict[str, list[ExtractionResult]] = {}
    llm_value: dict = {}
    if llm_n and ex.client.enabled:
        sub_ids = llm_subsample(ids, meta, llm_n, seed=seed)
        keep = set(sub_ids)
        sub_cards = [c for c in cards if c.id in keep]
        for name in ("правила", "продавец + правила", "+ NER", "+ LLM"):
            if "ml" in MODES[name] and ex.ml is None:
                continue
            res, perf = run_mode(ex, sub_cards, MODES[name])
            sub[name] = summarize(res, g, meta, tax, perf, mode=name + SUB_SUFFIX, set="llm_subsample", split=split,
                                  sources=list(MODES[name]))
            sub_rows.append(compare_row(name + SUB_SUFFIX, "llm_subsample", MODES[name], sub[name]))
            sub_res[name] = res
            if name == "+ LLM":
                llm_perf = perf
                if "+ NER" in sub_res:
                    llm_value = llm_review_value(res, sub_res["+ NER"], g, meta, tax,
                                                 get_settings().confidence_threshold)
            log(f"  [подвыборка {len(sub_cards)}] {name}: F1={sub[name]['micro']['f1']} ({perf['seconds']} с, "
                f"LLM-запросов {perf['llm_calls']})")
    cold = cold_start_stats(ex, cold_start_cards(processed, split, cold_n, exclude, seed), MODES[final]) if cold_n else {}

    out_dir.mkdir(parents=True, exist_ok=True)
    fm = full[final]
    cal = load_calibrator(str(get_settings().calibration_path))
    rel = reliability(results[final], g, tax)
    threshold = get_settings().confidence_threshold
    metrics = {**{k: v for k, v in fm.items() if k not in ("segments",)}, "mode": final,
               "threshold": threshold, "calibration_ece_holdout": rel["ece"],
               "eval_set": {"split": split, "n_cards": len(cards), "excluded_sample_skus": n_excl,
                            "taxonomy_cards": sum(1 for i in ids if meta[i]["category"] != "generic"),
                            "generic_cards": sum(1 for i in ids if meta[i]["category"] == "generic")}}
    _dump(metrics, out_dir / "metrics.json")
    compare = rows + sub_rows
    _dump(compare, out_dir / "compare.json")
    _dump({k: {sk: sv for sk, sv in v.items() if sv["n_cards"] >= 5} for k, v in fm["segments"].items()},
          out_dir / "segments.json")
    _dump({"mode": final, "split": split, "threshold": threshold, **rel,
           "fit": {k: cal.meta.get(k) for k in ("fit_split", "method", "threshold_rule", "cross_fit") if k in cal.meta},
           "note": "уверенность пайплайна (калибровка выучена на dev) против доли верных на holdout; "
                   "значения с эталоном, любой статус"}, out_dir / "calibration.json")
    at = min(fm["threshold_sweep"], key=lambda r: abs(r["threshold"] - threshold))
    _dump({"target_precision": TARGET_PRECISION, "chosen": threshold, "mode": final, "split": split,
           "chosen_on": "dev: 2-fold кросс-фиттинг калибровки (attrx calibrate), на holdout не подбирался",
           "at_chosen": {**at, "accepted_precision": fm["accepted_precision"]},
           "points": fm["threshold_sweep"]}, out_dir / "threshold.json")
    cost = cost_block(compare, llm_perf, len(sub_ids), load_prices())
    _dump(cost, out_dir / "cost.json")
    errs = write_errors(results[final], g, meta, tax, cards, out_dir / "errors.csv")
    preds = write_predictions_capped(results[final], out_dir)
    info = {"split": split, "n_cards": len(cards), "excluded": n_excl, "final": final, "threshold": threshold,
            "n_tax": metrics["eval_set"]["taxonomy_cards"], "n_gen": metrics["eval_set"]["generic_cards"],
            "llm_value": llm_value,
            "llm_n": len(sub_ids), "llm_stats": (llm_perf or {}).get("llm_stats"), "cold": cold, "errors": errs,
            "predictions": preds, "seconds": round(time.perf_counter() - t_all, 1), "calibration": rel,
            "calibration_meta": cal.meta}
    write_report_md(out_dir / "REPORT.md", full, sub, final, cost, info, tax)
    _dump({"full": {k: {kk: vv for kk, vv in v.items() if kk not in ("per_category", "threshold_sweep")}
                    for k, v in full.items()},
           "llm_subsample": {k: {kk: vv for kk, vv in v.items() if kk not in ("per_category", "threshold_sweep")}
                             for k, v in sub.items()},
           "info": {k: v for k, v in info.items() if k != "calibration_meta"}},
          processed / f"final_{split}.json")
    return info


# ------------------------------------------------------------------ REPORT.md --------------------------------------
def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, int) and not isinstance(x, bool):
        return f"{x:,}".replace(",", " ")
    return f"{x:.{nd}f}".replace(".", ",")


def _pct(x: Any, nd: int = 1) -> str:
    return "—" if x is None else f"{100 * x:.{nd}f}".replace(".", ",") + "%"


def _table(head: list[str], rows: list[list[Any]]) -> str:
    return ("| " + " | ".join(head) + " |\n|" + "|".join("---" if i == 0 else "---:" for i in range(len(head))) + "|\n"
            + "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows))


def write_report_md(path: Path, full: dict, sub: dict, final: str, cost: dict, info: dict, tax: Taxonomy) -> None:
    fm = full[final]
    base = full.get("продавец + правила")
    ap = fm["accepted_precision"]
    twc = fm["traffic_weighted_coverage"]
    cm = info.get("calibration_meta") or {}
    cf = cm.get("cross_fit") or {}
    L: list[str] = []
    L.append("# Отчёт: извлечение характеристик на holdout каталога М.Видео\n")
    if info["split"] == "holdout":
        L.append("Реальный каталог М.Видео, отложенный сплит holdout: при настройке правил, словарей, калибровки и порога "
                 "не использовался и посчитан один раз. Эталон — характеристики YML-фида, вход — только текст карточки. "
                 f"Режим в сдаче — «{final}».\n")
    else:
        L.append(f"ПРОБНЫЙ прогон отчёта на сплите {info['split']} (на нём настраивались правила и калибровка — это не "
                 "итоговая оценка). Эталон — характеристики YML-фида, вход — только текст карточки.\n")
    ho = info["split"]
    L.append(f"Карточек: **{_f(info['n_cards'])}** (категории таксономии — {_f(info['n_tax'])}, хвост каталога — "
             f"{_f(info['n_gen'])}), режим в сдаче: **{final}**, порог автопринятия **{_f(info['threshold'], 2)}**.\n")
    # главное
    L.append("## Главное\n")
    mi = fm["micro"]
    L.append(f"- **Точность принятых значений {_f(ap['precision'])}** (95% ДИ от {_f(ap['lower95'])}) при пороге "
             f"{_f(info['threshold'], 2)}, выбранном на dev под точность ≥ 0,95; автоматически принимается "
             f"{_pct(ap['accepted_share'])} найденных значений, остальное — модератору.")
    L.append(f"- micro P / R / F1 = **{_f(mi['precision'])} / {_f(mi['recall'])} / {_f(mi['f1'])}**; recall на значениях, "
             f"которые есть в тексте карточки, — **{_f(fm['recall_extractable'])}** (в тексте есть только "
             f"{_pct(fm['extractable_share'])} эталона: это потолок «только из текста»).")
    L.append(f"- F1 с учётом поискового спроса **{_f(fm['traffic_weighted_f1'])}**, macro-F1 по атрибутам "
             f"{_f(fm['macro_f1'])}, по ячейкам «категория × атрибут» {_f(fm['macro_f1_cells'])}.")
    L.append(f"- Каталог становится полнее: слоты схемы {_pct(fm['catalog_fill'])} → **{_pct(fm['union_fill'])}** "
             f"(+{_f(fm['new_fills'])} значений там, где в каталоге пусто); доля поискового спроса, по которой атрибут "
             f"заполнен, {_pct(twc['before'])} → **{_pct(twc['after'])}**.")
    seg = fm["segments"]
    st, sb = seg["seller_type"], seg["size_band"]
    L.append(f"- Справедливость: F1 М.Видео {_f(st.get('mvideo', {}).get('f1'))} и маркетплейс "
             f"{_f(st.get('marketplace', {}).get('f1'))}; head / torso / tail — {_f(sb.get('head', {}).get('f1'))} / "
             f"{_f(sb.get('torso', {}).get('f1'))} / {_f(sb.get('tail', {}).get('f1'))}.")
    if base:
        L.append(f"- Скорость: {_f(base['perf']['cards_per_sec'], 0)} карточек/с на одном ядре без моделей; "
                 f"с NER на GPU — {_f(fm['perf']['cards_per_sec'], 0)} карточек/с.")
    if sub.get("+ LLM"):
        lm, nm = sub["+ LLM"], sub.get("+ NER")
        L.append(f"- LLM (локальная qwen3:4b-instruct) на подвыборке {_f(lm['n_cards'])} карточек: F1 "
                 f"{_f(nm['micro']['f1']) if nm else '—'} → {_f(lm['micro']['f1'])}, LLM вызывается для "
                 f"{_pct(lm['perf']['llm_share'])} карточек; значения только от LLM идут на проверку, а не в каталог.")
    L.append("")
    # 1. набор
    L.append("## 1. Набор и протокол\n")
    L.append(f"- **Набор:** все карточки сплита `{info['split']}` с эталоном в 12 категориях таксономии и 1 000 карточек хвоста "
             "каталога (generic), стратифицированных по размеру категории — так же, как на dev (`attrx eval-real`).")
    L.append(f"- **Исключены {info['excluded']} SKU** `data/sample/real_cards.jsonl`: на них показывали демо и мерили "
             "LLM-бенчмарк; в честную оценку они не входят.")
    L.append("- **Holdout посчитан один раз** после того, как правила, словари (бренды, соответствие справочников, стиль "
             "моделей — по train), NER (train) и калибровка с порогом (dev) были заморожены. Режим в сдаче выбран "
             "правилом, заданным до прогона: «+ NER», если он не теряет точность больше 0,005 и не хуже по F1.")
    L.append("- **Метрики:** P/R/F1 — только по слотам, где эталон есть; значение там, где в каталоге пусто, — "
             "дозаполнение (`new_fills`), а не ошибка; числа сравниваются с допуском 1%, бренды и модели — без регистра "
             "и пунктуации, цвета — по базовой палитре (docs/EVALUATION.md).")
    L.append(f"- **LLM — на подвыборке** {_f(info['llm_n'])} карточек (≥ 150 смартфонов и ноутбуков, остальные категории "
             "и хвост пропорционально): на этой же подвыборке прогнаны все четыре режима, поэтому прирост от LLM честный.\n")
    # 2. лестница
    L.append("## 2. Что даёт каждый шаг (весь набор)\n")
    head = ["Режим", "P", "R", "F1", "R_extr", "F1 по спросу", "macro-F1", "Покрытие", "На проверку", "Карт./с", "LLM / 1000"]
    L.append(_table(head, [[n, _f(m["micro"]["precision"]), _f(m["micro"]["recall"]), _f(m["micro"]["f1"]),
                            _f(m["recall_extractable"]), _f(m["traffic_weighted_f1"]), _f(m["macro_f1"]),
                            _pct(m["coverage"]), _pct(m["review_rate"]), _f(m["perf"]["cards_per_sec"], 0),
                            _f(m["perf"]["llm_calls_per_1000"], 0)] for n, m in full.items()]))
    L.append("Покрытие — доля слотов «карточка × атрибут схемы», заполненных принятыми значениями. «На проверку» — "
             "доля слотов со статусом review (очередь модератора).\n")
    # 3. LLM
    if sub:
        n_sub = next(iter(sub.values()))["n_cards"]
        L.append(f"## 3. LLM на подвыборке ({_f(n_sub)} карточек, все режимы на одних и тех же карточках)\n")
        L.append(_table(head, [[n, _f(m["micro"]["precision"]), _f(m["micro"]["recall"]), _f(m["micro"]["f1"]),
                                _f(m["recall_extractable"]), _f(m["traffic_weighted_f1"]), _f(m["macro_f1"]),
                                _pct(m["coverage"]), _pct(m["review_rate"]),
                                _f(m["perf"]["cards_per_sec"], 0 if (m["perf"]["cards_per_sec"] or 0) >= 10 else 2),
                                _f(m["perf"]["llm_calls_per_1000"], 0)] for n, m in sub.items()]))
        st_ = info.get("llm_stats") or {}
        lat = st_.get("latency_s") or {}
        lm = sub.get("+ LLM")
        if lm:
            L.append(f"Модель — {st_.get('model')} ({st_.get('provider')}, локальная GPU, по одному запросу): запросов "
                     f"{_f(st_.get('requests'))}, из кэша {_f(st_.get('cache_hits'))}, ошибок {_f(st_.get('errors'))}; "
                     f"{_f((st_.get('prompt_tokens') or 0) / max(1, st_.get('calls') or 1), 0)} + "
                     f"{_f((st_.get('completion_tokens') or 0) / max(1, st_.get('calls') or 1), 0)} токенов на вызов; "
                     f"латентность {_f(lat.get('avg'), 1)} с (p95 {_f(lat.get('p95'), 1)} с). LLM спрашивают только о "
                     "слабых атрибутах (нет значения, низкая уверенность, конфликт) и только если о них что-то сказано "
                     f"в тексте; незаземлённых ответов отброшено {_f(lm['perf'].get('llm_rejected_values'))}. Значение, "
                     "которое нашла только LLM, автоматически не принимается — оно попадает в очередь модератора "
                     "с цитатой, поэтому вклад LLM в принятые значения — через подтверждение других источников.\n")
        lv = info.get("llm_value") or {}
        if lv:
            cfx = lv["counterfactual_accept_llm_only"]
            L.append(f"**Что LLM даёт модератору.** Значений только от LLM (на проверке): {_f(lv['llm_only_values'])}, "
                     f"из них {_f(lv['new_slots'])} закрывают слот, пустой без LLM; где есть эталон ({_f(lv['with_gold'])}), "
                     f"верны {_pct(lv['precision'])}. Если бы такие значения принимались автоматически (без инварианта "
                     f"доверия), было бы P / R / F1 = {_f(cfx['precision'])} / {_f(cfx['recall'])} / {_f(cfx['f1'])} "
                     f"против {_f(lm['micro']['precision'])} / {_f(lm['micro']['recall'])} / {_f(lm['micro']['f1'])}: "
                     "полнота растёт, но ценой точности каталога — поэтому одна LLM только предлагает, решает человек.\n")
    # 4. атрибуты
    L.append(f"## 4. По атрибутам («{final}»)\n")
    pa = sorted(fm["per_attribute"].items(), key=lambda kv: kv[1]["f1"])
    L.append(_table(["Атрибут", "P", "R", "F1", "R_extr", "Эталона", "TP", "FP", "FN"],
                    [[a, _f(v["precision"]), _f(v["recall"]), _f(v["f1"]), _f(v["recall_extractable"]),
                      _f(v["n_gold"]), _f(v["tp"]), _f(v["fp"]), _f(v["fn"])] for a, v in pa]))
    # 5. категории
    L.append(f"## 5. По категориям («{final}»; в скобках — «продавец + правила»)\n")
    rows5 = []
    for cat, attrs in sorted(fm["per_category"].items()):
        cells = [v for v in attrs.values() if v["n_gold"]]
        if not cells:
            continue
        tp, fp, fn = (sum(v[k] for v in cells) for k in ("tp", "fp", "fn"))
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        b = base["per_category"].get(cat, {}) if base else {}
        btp, bfp, bfn = (sum(v[k] for v in b.values() if v["n_gold"]) for k in ("tp", "fp", "fn"))
        bf1 = 2 * btp / (2 * btp + bfp + bfn) if btp else 0.0
        mf = sum(v["f1"] for v in cells) / len(cells)
        rows5.append([tax.categories[cat].title if cat in tax.categories else cat, _f(cells[0]["n_cards"]),
                      _f(tp / (tp + fp) if tp + fp else None), _f(tp / (tp + fn) if tp + fn else None),
                      f"{_f(f1)} ({_f(bf1)})", _f(mf)])
    L.append(_table(["Категория", "Карточек", "P", "R", "micro-F1", "macro-F1 по атрибутам"], rows5))
    L.append("Полная таблица «категория × атрибут» — `metrics.json` (`per_category`) и страница «Качество» в интерфейсе.\n")
    # 6. сегменты
    L.append("## 6. Справедливость: сегменты\n")
    names = {"seller_type": "Продавец", "size_band": "Размер категории", "desc_band": "Длина описания"}
    rows6 = []
    for k, title in names.items():
        for s, v in sorted(fm["segments"].get(k, {}).items()):
            if v["n_cards"] < 5:
                continue
            bv = (base["segments"].get(k, {}).get(s) if base else None) or {}
            rows6.append([f"{title}: {s}", _f(v["n_cards"]), _f(v["precision"]), _f(v["recall"]),
                          f"{_f(v['f1'])} ({_f(bv.get('f1'))})", _f(v["recall_extractable"]), _pct(v["coverage"])])
    L.append(_table(["Сегмент", "Карточек", "P", "R", "F1 (продавец + правила)", "R_extr", "Покрытие"], rows6))
    L.append("head — листовая категория с ≥ 1000 офферов, torso — 100–999, tail — < 100; описание short < 500 символов, "
             "long ≥ 1500.\n")
    # 7. калибровка
    rel = info["calibration"]
    L.append(f"## 7. Уверенность: калибровка (выучена на dev, проверена на {ho})\n")
    L.append(f"ECE на {ho} — **{_f(rel['ece'])}** ({_f(rel['n'])} значений с эталоном, любой статус). Калибровка: "
             f"{cm.get('method', 'изотоническая регрессия по группам')}; на dev (2-fold кросс-фиттинг) точность "
             f"принятых {_f(cf.get('precision_at_threshold'))}, покрытие {_f(cf.get('coverage_at_threshold'))}, "
             f"ECE {_f(cf.get('ece'))}.\n")
    L.append(_table(["Уверенность", "Средняя уверенность", "Доля верных", "Значений"],
                    [[f"{_f(b['lo'], 1)}–{_f(b['hi'], 1)}", _f(b["confidence"]), _f(b["accuracy"]), _f(b["n"])]
                     for b in rel["bins"]]))
    # 8. порог
    L.append("## 8. Порог автопринятия\n")
    L.append(f"Порог **{_f(info['threshold'], 2)}** выбран на dev (правило: нижняя граница 95% ДИ точности принятых ≥ 0,95 "
             "на кросс-фиттинге и ни одна ячейка «категория/атрибут» не принимается с точностью ниже 0,75), на holdout "
             f"не подбирался. На {ho} при этом пороге точность принятых {_f(ap['precision'])} "
             f"(нижняя граница {_f(ap['lower95'])}), принято {_f(ap['accepted'])} из {_f(ap['predicted_with_gold'])} "
             "найденных значений с эталоном.\n")
    pts = [r for r in fm["threshold_sweep"] if r["threshold"] in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)]
    L.append(_table(["Порог", "Точность", "Полнота", "F1", "Покрытие слотов", "Новых значений"],
                    [[_f(r["threshold"], 2), _f(r["precision"]), _f(r["recall"]), _f(r["f1"]), _pct(r["coverage"]),
                      _f(r["new_fills"])] for r in pts]))
    # 9. покрытие
    L.append("## 9. Покрытие каталога\n")
    rows9 = [[n, _pct(m["catalog_fill"]), _pct(m["union_fill"]), _f(m["new_fills"]),
              _pct(m["traffic_weighted_coverage"]["before"]), _pct(m["traffic_weighted_coverage"]["after"])]
             for n, m in full.items()]
    L.append(_table(["Режим", "Слоты: в каталоге", "+ пайплайн", "Новых значений", "Спрос: в каталоге", "+ пайплайн"], rows9))
    L.append("«Спрос» — доля поискового спроса, по которой атрибут заполнен: слоты взвешены долей кликов категории × "
             f"долей запросов, где покупатели называют этот атрибут (config/traffic_weights.yaml). В наборе ({ho}) у всех "
             "карточек есть оффер в фиде, поэтому каталог уже заполнен на ~90%; основной эффект — на карточках вне фида.\n")
    cold = info.get("cold") or {}
    if cold:
        L.append(f"**Cold start — карточки вне фида** ({_f(cold['n_cards'])} карточек того же сплита без оффера, эталона нет; "
                 f"таких в каталоге 75%): категория из таксономии у {_pct(cold['taxonomy_share'])} (классификатор по "
                 f"тексту; неуверенная категория — {_pct(cold['category_unsure_share'])} карточек), автоматически "
                 f"заполняется {_pct(cold['accepted_share'])} слотов схемы, на проверку — {_pct(cold['review_share'])}; "
                 f"{_f(cold['cards_per_sec'], 0)} карточек/с. Точность на них не измерить (эталона нет) — значения "
                 "уходят в каталог по тому же порогу и попадают в выборочную проверку модератором.\n")
    # 10. стоимость
    L.append("## 10. Стоимость и скорость (₽ на 1000 карточек)\n")
    L.append(_table(["Режим", "Основа", "₽ / 1000 карточек", "мс на карточку", "Доля карточек с LLM", "LLM-запросов на 1000"],
                    [[c["mode"], c["basis"], _f(c["cost_per_1000"], 2), _f(c.get("ms_per_card"), 1),
                      _pct(c.get("llm_share")), _f(c.get("llm_calls_per_1000"), 0)] for c in cost["modes"]]))
    L.append(cost["note"] + "\n")
    # 11. ошибки
    errs = info["errors"]
    L.append("## 11. Разбор ошибок\n")
    L.append(f"Всего строк: {_f(errs['total'])} — неверных значений {_f(errs['by_kind'].get('wrong', 0))}, пропусков "
             f"{_f(errs['by_kind'].get('missed', 0))}, значений там, где в каталоге пусто (дозаполнение, не ошибка) "
             f"{_f(errs['by_kind'].get('extra', 0))}; в `errors.csv` — {_f(errs['written'])} строк (колонка `extractable` "
             "— есть ли эталонное значение в тексте карточки).\n")
    L.append(f"Из {_f(errs['by_kind'].get('missed', 0))} пропусков **{_f(errs['missed_not_in_text'])} — значения нет в "
             f"тексте карточки** (потолок «только из текста»), {_f(errs['missed_in_review'])} — значение найдено, но "
             f"уверенность ниже порога (очередь модератора), {_f(errs['missed_not_found'])} — значение в тексте есть, "
             "но не найдено.\n")
    L.append("Больше всего неверных принятых значений: "
             + ", ".join(f"{k} — {v}" for k, v in errs["top_wrong"]) + ".")
    L.append("Чаще всего не найдено (значение в тексте есть): "
             + ", ".join(f"{k} — {v}" for k, v in errs["top_missed_found_in_text"]) + ".\n")
    L.append("Классы ошибок (подробно — docs/EVALUATION.md §8.5 и §9): значения нет в тексте карточки (основная часть "
             "пропусков — потолок «только из текста»); значение найдено, но уверенность ниже порога (очередь модератора); "
             "соглашения каталога, которых нет в тексте (тип наушников, цвет узора), и ошибки самого эталона.\n")
    # 12. ограничения
    L.append("## 12. Ограничения и честность оценки\n")
    L.append("- **Эталон — каталог, а не ручная разметка:** контент-команда М.Видео иногда ошибается (модель «Телевизор "
             "смарт», процессор «AMD HX 370» без линейки, MacBook 13\" при эталоне 13,6); такие расхождения засчитаны как "
             "ошибки, поэтому реальная точность выше измеренной.")
    L.append(f"- **Потолок «только из текста»:** {_pct(fm['extractable_share'])} эталонных значений вообще есть в тексте "
             f"карточки; recall на них — {_f(fm['recall_extractable'])}, общий recall ограничен этим потолком. Выдумывать "
             "отсутствующие характеристики кейс запрещает.")
    L.append("- **Holdout использован один раз**, после заморозки правил, словарей, NER, калибровки и порога; после "
             "прогона ничего не подстраивалось.")
    L.append(f"- **Исключены {info['excluded']} SKU сэмпла** (демо и бенчмарк LLM), поэтому карточек категорий таксономии "
             "меньше, чем в holdout целиком.")
    L.append(f"- **LLM — на подвыборке** {_f(info['llm_n'])} карточек и на одной локальной GPU (4B-модель): прирост и "
             "стоимость — оценка для этого стенда; в продакшене модель сильнее и быстрее (docs/LLM.md).")
    L.append("- **Карточки с эталоном — карточки фида:** блок «Характеристики» продавца у них редок (7%), поэтому оценка "
             "в основном меряет извлечение из названия и описания; для карточек вне фида показано только покрытие.\n")
    L.append("## Файлы\n")
    pr = info["predictions"]
    L.append(f"- `predictions.jsonl` ({_f(pr['jsonl_mb'], 1)} МБ) — полный ExtractionResult на каждую карточку набора"
             + (" (только категории таксономии: файл > 25 МБ)" if pr["jsonl_trimmed"] else "")
             + f"; `predictions.csv` ({_f(pr['csv_mb'], 1)} МБ) — то же плоско, все карточки.")
    L.append("- `metrics.json` (итоговый режим: атрибуты, категории × атрибуты, сегменты, кривая порога), `compare.json` "
             "(лестница на всём наборе + подвыборка LLM), `segments.json`, `calibration.json` (диаграмма надёжности на "
             "holdout), `threshold.json`, `cost.json`, `errors.csv`, `traffic.json` (спрос, поток EDA).")
    L.append(f"- Воспроизвести: `attrx report-real` (`make report`), {_f(info['seconds'], 0)} с на этой машине.\n")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
