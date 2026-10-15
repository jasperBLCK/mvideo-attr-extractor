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
import statistics
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
def run_mode(ex, cards: list[Card], sources: tuple[str, ...], passes: int = 1, warmup: int = 50,
             ) -> tuple[list[ExtractionResult], dict]:
    """Прогон режима. Без LLM: прогрев на warmup карточках вне замера, затем медиана passes замеров (первый
    прогон процесса медленнее: ленивая загрузка классификатора, трафик-весов, компиляция регулярок)."""
    llm = "llm" in sources
    if llm:
        ex.client.reset_stats()
        passes = 1
    elif warmup and cards:
        list(ex.extract_many(cards[:warmup], use_llm=False, sources=set(sources)))
    times: list[float] = []
    results: list[ExtractionResult] = []
    for _ in range(max(1, passes)):
        t = time.perf_counter()
        results = list(ex.extract_many(cards, use_llm=llm, sources=set(sources)))
        times.append(time.perf_counter() - t)
    dt = statistics.median(times)
    n = max(1, len(cards))
    perf: dict[str, Any] = {"seconds": round(dt, 2), "passes": len(times), "seconds_all": [round(x, 2) for x in times],
                            "cards_per_sec": round(n / dt, 1) if dt else None, "ms_per_card": round(1000 * dt / n, 2),
                            "llm_calls": 0, "llm_share": 0.0}
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
    primary = compare[0]["set"] if compare else "full_holdout"
    for r in compare:
        if not ((r["set"] == primary and "без category_id" not in r["mode"]) or r["mode"] == "+ LLM" + SUB_SUFFIX):
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
    note = (f"Свой стенд (GTX 1060 6 ГБ, Ollama qwen3:4b-instruct): {f'{rub_h:.2f}'.replace('.', ',')} ₽/ч = амортизация "
            "35 000 ₽ за 3 года 24/7 "
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


# ------------------------------------------------------------------ v2: провайдер LLM, срезы, экспорт -------------
LOCAL_LLM_URL = "http://127.0.0.1:11434/v1"
LOCAL_LLM_MODEL = "qwen3:4b-instruct"
SET_SUFFIX = {"full_holdout": "", "stratified": " · стратифицированный набор", "llm_subsample": SUB_SUFFIX}


def llm_settings(provider: str = "auto"):
    """Настройки LLM для отчёта (кэш ответов выключен: нужны настоящие вызовы, латентность и стоимость).

    auto  — задан OPENAI_BASE_URL (ВМ хакатона: шлюз и ключ организаторов) -> OPENAI_* целиком, LLM_* не смешиваются
            (даже если их подставил Makefile или .env); иначе задан LLM_BASE_URL или ключ LLM_API_KEY -> LLM_*;
            иначе локальная Ollama (qwen3:4b-instruct);
    env   — обычный порядок настроек пайплайна (LLM_* -> OPENAI_* -> LLM_LOCAL_URL, config.py);
    local — локальная Ollama."""
    from urllib.parse import urlparse

    from ..config import Settings, _env, _real_key

    if provider == "auto":
        choice = ("openai" if _env("OPENAI_BASE_URL") else
                  "env" if _env("LLM_BASE_URL") or _real_key(_env("LLM_API_KEY")) else "local")
    elif provider in ("local", "env"):
        choice = provider
    else:
        raise ValueError(f"--llm-provider: auto | local | env, не {provider!r}")
    common = {"llm_enabled": True, "llm_cache": False, "ml_enabled": True}
    if choice == "local":  # LLM_LOCAL_URL: клиент сначала проверяет, что Ollama отвечает
        s = Settings(llm_source="LLM_LOCAL_URL", llm_base_url=LOCAL_LLM_URL, llm_model=LOCAL_LLM_MODEL,
                     llm_api_key="not-needed", **common)
    elif choice == "openai":
        s = Settings(llm_source="OPENAI_*", llm_base_url=(_env("OPENAI_BASE_URL") or "").rstrip("/"),
                     llm_api_key=_env("OPENAI_API_KEY") or "not-needed", llm_model=_env("OPENAI_MODEL") or "auto",
                     **common)
    else:
        s = Settings(llm_cache=False, ml_enabled=True)
    return s, {"choice": choice, "requested": provider, "base_host": urlparse(s.llm_base_url).netloc,
               "model": s.llm_model, "source": s.llm_source}


def llm_probe(settings) -> tuple[Any, dict]:
    """Клиент LLM и его проверка ДО долгого прогона: (клиент или None, что выяснили). Модель при LLM_MODEL=auto
    выбирается из GET /models — её имя нужно сразу, чтобы не перезаписать отчёт с другой LLM."""
    from ..llm.client import LLMClient

    client = LLMClient(settings)
    probe = client.ping(max_age=0.0, timeout=10.0) if client.enabled else {"reachable": False, "error": "LLM выключена"}
    if not probe.get("reachable"):
        return None, {"model": None, "unavailable": probe.get("error") or "нет ответа"}
    return client, {"model": client.model, "model_source": client.model_source, "provider": client.provider}


def seen_title_ids(processed: Path, ids: list[str]) -> set[str]:
    """SKU из ids, чьё название (без регистра, ё/е и лишних пробелов) дословно встречается в train."""
    import duckdb
    import pandas as pd

    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")
    con.execute("SET threads=2")
    con.register("ids", pd.DataFrame({"sku_id": ids}))
    key = "lower(regexp_replace(replace(trim(title), 'ё', 'е'), '\\s+', ' ', 'g'))"
    rows = con.execute(f"""
        WITH tr AS (SELECT DISTINCT {key} AS t FROM read_parquet('{processed / 'cards.parquet'}') WHERE split = 'train')
        SELECT c.sku_id FROM read_parquet('{processed / 'cards.parquet'}') c JOIN ids USING (sku_id)
        WHERE {key.replace('title', 'c.title')} IN (SELECT t FROM tr)""").fetchall()
    return {r[0] for r in rows}


def export_eval_set(cards: list[Card], gold: dict, meta: dict, root: Path) -> dict:
    """data/sample/holdout_eval_{cards,gold}.jsonl: воспроизвести строки «правила» без data/processed (make verify)."""
    d = root / "data" / "sample"
    d.mkdir(parents=True, exist_ok=True)
    pc, pg = d / "holdout_eval_cards.jsonl", d / "holdout_eval_gold.jsonl"
    with pc.open("w", encoding="utf-8") as f:
        for c in cards:
            f.write(json.dumps({"id": c.id, "title": c.title, "description": c.description,
                                "seller_attrs": c.seller_attrs, "category": c.category, "category_id": c.category_id,
                                "seller": c.seller}, ensure_ascii=False).replace("\u2028", "\\u2028")
                    .replace("\u2029", "\\u2029") + "\n")
    with pg.open("w", encoding="utf-8") as f:
        for c in cards:
            m = {k: v for k, v in (meta.get(c.id) or {}).items()}
            f.write(json.dumps({"id": c.id, **m, "attributes": gold.get(c.id) or {}}, ensure_ascii=False) + "\n")
    return {"cards": str(pc.relative_to(root)), "gold": str(pg.relative_to(root)), "n": len(cards),
            "mb": round((pc.stat().st_size + pg.stat().st_size) / 2**20, 2)}


def operating_point(m: dict, threshold: float) -> dict:
    """Фактическая рабочая точка режима при выбранном пороге — статусы пайплайна, а не ближайший узел сетки."""
    return {"threshold": threshold, "precision": m["micro"]["precision"], "recall": m["micro"]["recall"],
            "f1": m["micro"]["f1"], "coverage": m["coverage"], "new_fills": m["new_fills"],
            "review_rate": m["review_rate"], "accepted_precision": m["accepted_precision"]}


def band_precision(results: list[ExtractionResult], gold: dict, tax: Taxonomy, lo: float, hi: float) -> dict:
    """Точность ПРИНЯТЫХ значений с уверенностью в [lo; hi) — проверка калибровки у порога: сколько их, какая доля
    всех принятых и из каких ячеек «категория/атрибут/источники» они в основном."""
    pts = collect(results, gold, tax)
    status = {(r.card_id, a.attribute): a.status for r in results for a in r.attributes}
    acc = [p for p in pts if status.get((p.card, p.keys[0].split("/")[1])) == "accepted"]
    sel = [p for p in acc if lo <= p.raw < hi]
    tp = sum(p.ok for p in sel)
    cells: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for p in sel:
        cells[p.keys[0]][0] += 1
        cells[p.keys[0]][1] += p.ok
    top = sorted(cells.items(), key=lambda kv: -kv[1][0])[:4]
    return {"lo": lo, "hi": hi, "n": len(sel), "precision": round(tp / len(sel), 4) if sel else None,
            "mean_confidence": round(sum(p.raw for p in sel) / len(sel), 4) if sel else None,
            "lower95": round(wilson_lower(tp, len(sel)), 4) if sel else None,
            "share_of_accepted": round(len(sel) / len(acc), 4) if acc else None,
            "wrong": len(sel) - tp, "wrong_accepted_total": sum(not p.ok for p in acc),
            "generic_share": round(sum(p.keys[0].startswith("generic/") for p in sel) / len(sel), 4) if sel else None,
            "top_cells": [{"cell": k, "n": n, "precision": round(ok / n, 4)} for k, (n, ok) in top]}


def slice_metrics(results: list[ExtractionResult], keep: set[str], gold: dict, meta: dict, tax: Taxonomy) -> dict:
    from .metrics import evaluate

    sub = [r for r in results if r.card_id in keep]
    if not sub:
        return {"n_cards": 0}
    m = evaluate(sub, gold, tax, meta=meta, weights=load_traffic_weights(tax))
    return {"n_cards": m["n_cards"], **{k: m["micro"][k] for k in ("precision", "recall", "f1")},
            "recall_extractable": m["recall_extractable"], "traffic_weighted_f1": m["traffic_weighted_f1"],
            "coverage": m["coverage"], "accepted_precision": accepted_precision(sub, gold, tax)["precision"]}


def _check_llm_change(out_dir: Path, model: str | None, force: bool) -> None:
    """Отчёт в out_dir не перезаписывается молча, если прежний посчитан с другой LLM (или с LLM, а сейчас её нет)."""
    p = out_dir / "metrics.json"
    if force or not p.exists():
        return
    try:
        prev = (json.loads(p.read_text(encoding="utf-8")).get("llm") or {}).get("model")
    except (OSError, ValueError):
        prev = None
    if prev and prev != model:
        now = f"сейчас — {model}" if model else "сейчас LLM недоступна или выключена"
        raise SystemExit(f"{out_dir.name}/ посчитан с LLM {prev}, {now}: отчёт не перезаписан. Выберите LLM явно "
                         "(--llm-provider local|env), запишите отчёт в другую папку (--out) или добавьте --force, если "
                         "смена осознанная.")


def run_final(root: Path, processed: Path, out_dir: Path, *, split: str = "holdout", llm_n: int = 600,
              cold_n: int = 1000, seed: int = 0, passes: int = 3, llm_provider: str = "auto", full: bool = True,
              force: bool = False, version: str = "v2", log=print) -> dict:
    """Финальный прогон v2: стратифицированный набор (как v1) + ВЕСЬ holdout с эталоном (основная таблица) +
    подвыборка LLM + срезы (новые названия, без category_id) + cold start -> submission/ + REPORT.md."""
    from ..confidence import load_calibrator
    from ..config import Settings, get_settings
    from ..llm.cost import load_prices
    from ..pipeline import Extractor

    t_all = time.perf_counter()
    exclude = sample_ids(root)
    ids, g, meta, n_excl = eval_set(processed, split, exclude, seed)
    cards = real.load_cards_by_id(ids, processed)
    ex = Extractor(settings=Settings(ml_enabled=True, llm_enabled=False))
    tax = ex.tax
    threshold = get_settings().confidence_threshold
    llm_s, llm_info = llm_settings(llm_provider) if llm_n else (None, {})
    client = None
    if llm_s is not None:
        client, found = llm_probe(llm_s)
        llm_info |= found
    _check_llm_change(out_dir, llm_info.get("model"), force)  # до долгого прогона, а не после
    log(f"{split}: стратифицированный набор {len(cards)} карточек (исключено SKU сэмпла: {n_excl}); NER: "
        f"{ex.ml is not None}; LLM: {llm_info or 'нет'}; порог {threshold}")

    def mode_block(name: str, cards_: list[Card], set_name: str, passes_: int) -> tuple[dict, list[ExtractionResult]]:
        res, perf = run_mode(ex, cards_, MODES[name], passes=passes_)
        m = summarize(res, g, meta, tax, perf, mode=name + SET_SUFFIX[set_name], set=set_name, split=split,
                      sources=list(MODES[name]))
        log(f"  [{set_name}] {name}: P={m['micro']['precision']} R={m['micro']['recall']} F1={m['micro']['f1']} "
            f"({perf['cards_per_sec']} карт/с, медиана {perf['passes']} замеров)")
        return m, res

    strat: dict[str, dict] = {}
    strat_res: dict[str, list[ExtractionResult]] = {}
    for name in FULL_MODES:
        if "ml" in MODES[name] and ex.ml is None:
            continue
        strat[name], strat_res[name] = mode_block(name, cards, "stratified", passes)
    fullh: dict[str, dict] = {}
    full_res: dict[str, list[ExtractionResult]] = {}
    full_cards: list[Card] = []
    if full:
        full_ids = sorted(k for k in meta if k not in exclude)
        full_cards = real.load_cards_by_id(full_ids, processed)
        log(f"весь {split} с эталоном: {len(full_cards)} карточек")
        for name in FULL_MODES:
            if "ml" in MODES[name] and ex.ml is None:
                continue
            fullh[name], full_res[name] = mode_block(name, full_cards, "full_holdout", passes)
    primary, primary_res, primary_set = (fullh, full_res, "full_holdout") if fullh else (strat, strat_res, "stratified")
    final = choose_final({n: compare_row(n, primary_set, MODES[n], m) for n, m in primary.items()})
    # срез «без category_id»: категорию выбирает классификатор по тексту (как у карточек вне фида)
    nc_cards = [c.model_copy(update={"category_id": None, "category": None}) for c in cards]
    nc_m, nc_res = mode_block(final, nc_cards, "stratified", passes)
    nc_m["category_accuracy"] = round(sum(r.category == meta[r.card_id]["category"] for r in nc_res) / len(nc_res), 4)
    # срез «новые названия»: название дословно не встречается в train
    seen = seen_title_ids(processed, [c.id for c in (full_cards or cards)])
    new_titles = {
        "stratified": slice_metrics(strat_res[final], {c.id for c in cards} - seen, g, meta, tax),
        "stratified_seen": slice_metrics(strat_res[final], {c.id for c in cards} & seen, g, meta, tax),
        "share_seen_stratified": round(len({c.id for c in cards} & seen) / len(cards), 4)}
    if full_cards:
        fid = {c.id for c in full_cards}
        new_titles |= {"full_holdout": slice_metrics(full_res[final], fid - seen, g, meta, tax),
                       "full_holdout_seen": slice_metrics(full_res[final], fid & seen, g, meta, tax),
                       "share_seen_full": round(len(fid & seen) / len(fid), 4)}
    # LLM на подвыборке (те же карточки, что в v1)
    sub: dict[str, dict] = {}
    sub_res: dict[str, list[ExtractionResult]] = {}
    llm_perf: dict | None = None
    llm_value: dict = {}
    sub_ids: list[str] = []
    if client is not None:
        ex_llm = Extractor(taxonomy=tax, settings=llm_s, client=client)
        if ex_llm.client.enabled:
            sub_ids = llm_subsample(ids, meta, llm_n, seed=seed)
            keep = set(sub_ids)
            sub_cards = [c for c in cards if c.id in keep]
            for name in ("правила", "продавец + правила", "+ NER", "+ LLM"):
                e = ex_llm if name == "+ LLM" else ex
                res, perf = run_mode(e, sub_cards, MODES[name], passes=1 if name == "+ LLM" else passes)
                sub[name] = summarize(res, g, meta, tax, perf, mode=name + SUB_SUFFIX, set="llm_subsample",
                                      split=split, sources=list(MODES[name]))
                sub_res[name] = res
                if name == "+ LLM":
                    llm_perf = perf
                    llm_info["model"] = perf["llm_stats"].get("model") or llm_info.get("model")
                    llm_info["provider"] = perf["llm_stats"].get("provider")
                    if "+ NER" in sub_res:
                        llm_value = llm_review_value(res, sub_res["+ NER"], g, meta, tax, threshold)
                log(f"  [llm_subsample {len(sub_cards)}] {name}: F1={sub[name]['micro']['f1']} ({perf['seconds']} с, "
                    f"LLM-запросов {perf['llm_calls']})")
    elif llm_s is not None:
        log(f"LLM недоступна ({llm_info}) — режим «+ LLM» пропущен")
    _check_llm_change(out_dir, llm_info.get("model"), force)  # модель, которая реально отвечала
    cold = cold_start_stats(ex, cold_start_cards(processed, split, cold_n, exclude, seed), MODES[final]) if cold_n else {}
    export = export_eval_set(cards, g, meta, root) if split == "holdout" else {}

    out_dir.mkdir(parents=True, exist_ok=True)
    fm, fres = primary[final], primary_res[final]
    cal = load_calibrator(str(get_settings().calibration_path))
    rel = reliability(fres, g, tax)
    band = band_precision(fres, g, tax, threshold, 0.8)
    rel_strat = reliability(strat_res[final], g, tax)  # тот же набор, что в v1: бин 0,7–0,8 сравним напрямую
    band_strat = band_precision(strat_res[final], g, tax, threshold, 0.8)
    rows = ([compare_row(n + SET_SUFFIX[primary_set], primary_set, MODES[n], m) for n, m in primary.items()]
            + ([compare_row(n + SET_SUFFIX["stratified"], "stratified", MODES[n], m) for n, m in strat.items()]
               if primary_set != "stratified" else [])
            + [compare_row(final + " без category_id" + SET_SUFFIX["stratified"], "stratified", MODES[final], nc_m)]
            + [compare_row(n + SUB_SUFFIX, "llm_subsample", MODES[n], m) for n, m in sub.items()])
    metrics = {**{k: v for k, v in fm.items() if k != "segments"}, "mode": final, "version": version,
               "threshold": threshold, "calibration_ece_holdout": rel["ece"], "llm": llm_info or None,
               "eval_set": {"split": split, "primary": primary_set, "n_cards": fm["n_cards"],
                            "stratified_cards": len(cards), "full_holdout_cards": len(full_cards),
                            "excluded_sample_skus": n_excl,
                            "taxonomy_cards": sum(1 for i in ids if meta[i]["category"] != "generic"),
                            "generic_cards": sum(1 for i in ids if meta[i]["category"] == "generic")}}
    _dump(metrics, out_dir / "metrics.json")
    _dump(rows, out_dir / "compare.json")
    _dump({k: {sk: sv for sk, sv in v.items() if sv["n_cards"] >= 5} for k, v in fm["segments"].items()},
          out_dir / "segments.json")
    _dump({"mode": final, "split": split, "set": primary_set, "threshold": threshold, **rel,
           "accepted_band": band, "stratified": {"ece": rel_strat["ece"], "bins": rel_strat["bins"],
                                                  "accepted_band": band_strat},
           "fit": {k: cal.meta.get(k) for k in ("fit_split", "method", "threshold_rule", "cross_fit") if k in cal.meta},
           "note": "уверенность пайплайна (калибровка выучена на dev) против доли верных на holdout; значения с "
                   "эталоном, любой статус. accepted_band — точность принятых значений с уверенностью в [порог; 0,8)"},
          out_dir / "calibration.json")
    from .metrics import evaluate

    exact = evaluate(fres, g, tax, threshold=threshold, meta=meta)
    exact_pt = {"threshold": threshold, "precision": exact["micro"]["precision"], "recall": exact["micro"]["recall"],
                "f1": exact["micro"]["f1"], "coverage": exact["coverage"], "new_fills": exact["new_fills"]}
    points = sorted([p for p in fm["threshold_sweep"] if abs(p["threshold"] - threshold) > 1e-9] + [exact_pt],
                    key=lambda p: p["threshold"])
    _dump({"target_precision": TARGET_PRECISION, "chosen": threshold, "mode": final, "split": split, "set": primary_set,
           "chosen_on": "dev: 2-fold кросс-фиттинг калибровки (attrx calibrate), на holdout не подбирался",
           "at_chosen": operating_point(fm, threshold),
           "note": "at_chosen — рабочая точка пайплайна (статусы: порог + правила конфликтов и инвариант LLM); "
                   "points — принять всё с уверенностью ≥ порога (без правил статуса), включая точку ровно на пороге",
           "points": points}, out_dir / "threshold.json")
    cost = cost_block(rows, llm_perf, len(sub_ids), load_prices())
    _dump(cost, out_dir / "cost.json")
    errs = write_errors(fres, g, meta, tax, full_cards or cards, out_dir / "errors.csv")
    write_predictions(fres, out_dir / "predictions.jsonl")  # csv — весь основной набор (плоско)
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:  # jsonl — стратифицированный набор
        for r in strat_res[final]:
            f.write(r.model_dump_json() + "\n")
    preds = {"jsonl_mb": round((out_dir / "predictions.jsonl").stat().st_size / 2**20, 2), "jsonl_cards": len(cards),
             "jsonl_set": "stratified", "csv_mb": round((out_dir / "predictions.csv").stat().st_size / 2**20, 2),
             "csv_cards": len(fres), "csv_set": primary_set, "jsonl_trimmed": False}
    info = {"split": split, "version": version, "n_cards": fm["n_cards"], "stratified_cards": len(cards),
            "full_generic_share": round(sum(meta[c.id]["category"] == "generic" for c in full_cards) / len(full_cards), 4)
            if full_cards else None,
            "full_cards": len(full_cards), "excluded": n_excl, "final": final, "threshold": threshold,
            "n_tax": metrics["eval_set"]["taxonomy_cards"], "n_gen": metrics["eval_set"]["generic_cards"],
            "llm_n": len(sub_ids), "llm": llm_info, "llm_stats": (llm_perf or {}).get("llm_stats"), "llm_value": llm_value,
            "cold": cold, "errors": errs, "predictions": preds, "calibration": rel, "band": band,
            "calibration_stratified": rel_strat, "band_stratified": band_strat,
            "calibration_meta": cal.meta, "no_category": nc_m, "new_titles": new_titles, "export": export,
            "primary_set": primary_set, "seconds": round(time.perf_counter() - t_all, 1)}
    v1 = _load_json(processed / f"final_{split}_v1.json")
    write_report_md_v2(out_dir / "REPORT.md", fullh, strat, sub, final, cost, info, tax, v1)
    _dump({"full_holdout": {k: _brief(v) for k, v in fullh.items()}, "stratified": {k: _brief(v) for k, v in strat.items()},
           "llm_subsample": {k: _brief(v) for k, v in sub.items()},
           "info": {k: v for k, v in info.items() if k != "calibration_meta"}},
          processed / f"final_{split}.json")
    return info


def _brief(m: dict) -> dict:
    return {k: v for k, v in m.items() if k not in ("per_category", "threshold_sweep")}


def _load_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (OSError, ValueError):
        return None


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


V2_FIXES = [
    "модель аксессуара не берётся из «для/for/совместим с + бренд устройства» («Baseus 20W для iPhone 15» -> модель не "
    "«15», «Чехол для Samsung Galaxy A55» -> не «A55»); «модель» из одних характеристик («20W», «12 В 2 Ач») — низкая "
    "уверенность, на проверку",
    "число в названии и другое в описании («256 ГБ» / «128 ГБ»): побеждает название, слот уходит на проверку (на dev "
    "варианты «название побеждает / на проверку / отбросить описание» различаются меньше чем на 0,0002 точности)",
    "вывод единиц с согласованием («6,1 дюйма», «55 дюймов») — только показ, метрики не затрагивает",
    "калибровка переобучена на dev после исправлений тем же правилом выбора порога: 0,69 -> 0,70 (при 0,69 на "
    "кросс-фиттинге ячейка «холодильник/общий объём» опускалась ниже пола 0,75; при 0,70 на проверку уходит и тип "
    "наушников — на dev его принятые были верны лишь на 0,73)",
    "протокол: скорость — медиана 3 замеров после прогрева на 50 карточках (в v1 первый режим платил за ленивую "
    "загрузку); основная таблица — весь holdout; срезы «новые названия» и «без category_id»; рабочая точка ровно на пороге "
    "в threshold.json; выгрузка стратифицированного набора для `attrx verify`",
]


def _mode_rows(blk: dict, cps_nd: int = 0) -> list[list[Any]]:
    return [[n, _f(m["n_cards"]), _f(m["micro"]["precision"]), _f(m["micro"]["recall"]), _f(m["micro"]["f1"]),
             _f(m["recall_extractable"]), _f(m["traffic_weighted_f1"]), _pct(m["coverage"]), _pct(m["review_rate"]),
             _f(m["perf"]["cards_per_sec"], 0 if (m["perf"]["cards_per_sec"] or 0) >= 10 else 2),
             _f(m["perf"]["llm_calls_per_1000"], 0)] for n, m in blk.items()]


# P — точность принятых значений (micro по слотам с эталоном): статус accepted и есть «принято в каталог»
MODE_HEAD = ["Режим", "Карточек", "P", "R", "F1", "R_extr", "F1 по спросу", "Покрытие", "На проверку", "Карт./с",
             "LLM / 1000"]


def write_report_md_v2(path: Path, fullh: dict, strat: dict, sub: dict, final: str, cost: dict, info: dict,
                       tax: Taxonomy, v1: dict | None) -> None:
    prim = fullh or strat
    fm = prim[final]
    sm = strat[final]
    ap = fm["accepted_precision"]
    twc = fm["traffic_weighted_coverage"]
    rel, band = info["calibration"], info["band"]
    cm = info.get("calibration_meta") or {}
    cf = cm.get("cross_fit") or {}
    seg = fm["segments"]
    L: list[str] = [f"# Отчёт {info['version']}: извлечение характеристик на holdout каталога М.Видео\n"]
    L.append(f"{info['version']} (после исправлений, найденных на dev и на демо-сэмпле вне eval-набора; eval-набор для "
             "выбора исправлений не использовался). Реальный каталог М.Видео, отложенный сплит holdout: при настройке "
             "правил, словарей, калибровки и порога не использовался. Эталон — характеристики YML-фида; вход — текст "
             f"карточки и id категории каталога. Режим в сдаче — «{final}».\n")
    L.append(f"Карточек: **{_f(fm['n_cards'])}** — весь holdout с эталоном, кроме {info['excluded']} SKU демо-сэмпла; "
             f"режим в сдаче **{final}**, порог автопринятия **{_f(info['threshold'], 2)}**.\n")
    mi = fm["micro"]
    prim_name = "Весь holdout" if fullh else "Стратифицированный набор"
    L.append("## Главное\n")
    L.append(f"- **{prim_name} ({_f(fm['n_cards'])} карточек): P / R / F1 = {_f(mi['precision'])} / {_f(mi['recall'])} / "
             f"{_f(mi['f1'])}**, recall на значениях, которые есть в тексте, {_f(fm['recall_extractable'])} (в тексте есть "
             f"{_pct(fm['extractable_share'])} эталона), F1 с учётом спроса {_f(fm['traffic_weighted_f1'])}.")
    L.append(f"- **Точность принятых значений {_f(ap['precision'])}** (95% ДИ от {_f(ap['lower95'])}) при пороге "
             f"{_f(info['threshold'], 2)}; принимается {_pct(ap['accepted_share'])} найденных значений, остальное — модератору.")
    if fullh:
        L.append(f"- Стратифицированный набор ({_f(sm['n_cards'])} карточек — те же, что в v1): P / R / F1 = "
                 f"{_f(sm['micro']['precision'])} / {_f(sm['micro']['recall'])} / {_f(sm['micro']['f1'])}, точность принятых "
                 f"{_f(sm['accepted_precision']['precision'])}. На всём holdout точность ниже: "
                 f"{_pct(info['full_generic_share'])} карточек — хвост (generic), где цвет и модель — самые трудные ячейки.")
    if strat.get("правила") and final in strat:
        L.append(f"- Скорость (медиана {sm['perf']['passes']} замеров после прогрева, одна машина): правила "
                 f"{_f(strat['правила']['perf']['cards_per_sec'], 0)} карт./с на ядро, «{final}» "
                 f"{_f(sm['perf']['cards_per_sec'], 0)} карт./с (NER на GTX 1060); стоимость — раздел 12.")
    L.append(f"- Каталог полнее: слоты {_pct(fm['catalog_fill'])} → **{_pct(fm['union_fill'])}** (+{_f(fm['new_fills'])}); "
             f"доля поискового спроса, по которой атрибут заполнен, {_pct(twc['before'])} → **{_pct(twc['after'])}**.")
    st, sb = seg["seller_type"], seg["size_band"]
    L.append(f"- Справедливость: М.Видео {_f(st.get('mvideo', {}).get('f1'))}, маркетплейс "
             f"{_f(st.get('marketplace', {}).get('f1'))}; head / torso / tail — {_f(sb.get('head', {}).get('f1'))} / "
             f"{_f(sb.get('torso', {}).get('f1'))} / {_f(sb.get('tail', {}).get('f1'))}.")
    nt = info["new_titles"]
    key_new = "full_holdout" if "full_holdout" in nt else "stratified"
    L.append(f"- Новые названия (дословно нет в train): F1 {_f(nt[key_new].get('f1'))} на {_f(nt[key_new].get('n_cards'))} "
             f"карточках против {_f(nt[key_new + '_seen'].get('f1'))} на уже встречавшихся.")
    nc = info["no_category"]
    L.append(f"- Без category_id (категорию выбирает классификатор по тексту): F1 {_f(nc['micro']['f1'])} против "
             f"{_f(sm['micro']['f1'])} с ним; категория таксономии угадана у {_pct(nc['category_accuracy'])} карточек.")
    if band.get("n"):
        L.append(f"- **Слабое место — калибровка у порога:** принятые значения с уверенностью {_f(info['threshold'], 2)}–0,80 "
                 f"({_pct(band['share_of_accepted'])} принятых) верны на {_f(band['precision'])} при обещанных "
                 f"{_f(band['mean_confidence'])}; это {_f(band['wrong'])} из {_f(band['wrong_accepted_total'])} неверных "
                 f"принятых, {_pct(band['generic_share'], 0)} слоя — ячейки хвоста (раздел 9).")
    if sub.get("+ LLM"):
        lm = sub["+ LLM"]
        L.append(f"- LLM ({(info.get('llm') or {}).get('model')}) на подвыборке {_f(lm['n_cards'])} карточек: F1 "
                 f"{_f(sub['+ NER']['micro']['f1'])} → {_f(lm['micro']['f1'])}; значения только от LLM — на проверку.")
    L.append("")
    # история
    if v1:
        L.append("## История: v1 → v2\n")
        v1s = v1.get("full", {}).get(final) or {}
        v1l = v1.get("llm_subsample", {})
        v1i = v1.get("info", {})

        def g1(m, *ks):
            for k in ks:
                m = (m or {}).get(k)
            return m

        rows = [["Стратифицированный набор: P", _f(g1(v1s, "micro", "precision")), _f(sm["micro"]["precision"])],
                ["… R", _f(g1(v1s, "micro", "recall")), _f(sm["micro"]["recall"])],
                ["… F1", _f(g1(v1s, "micro", "f1")), _f(sm["micro"]["f1"])],
                ["… R_extr", _f(g1(v1s, "recall_extractable")), _f(sm["recall_extractable"])],
                ["… F1 по спросу", _f(g1(v1s, "traffic_weighted_f1")), _f(sm["traffic_weighted_f1"])],
                ["… точность принятых", _f(g1(v1s, "accepted_precision", "precision")),
                 _f(sm["accepted_precision"]["precision"])],
                ["… покрытие / на проверку", f"{_pct(g1(v1s, 'coverage'))} / {_pct(g1(v1s, 'review_rate'))}",
                 f"{_pct(sm['coverage'])} / {_pct(sm['review_rate'])}"],
                ["Подвыборка LLM: F1 «+ NER» / «+ LLM»",
                 f"{_f(g1(v1l.get('+ NER'), 'micro', 'f1'))} / {_f(g1(v1l.get('+ LLM'), 'micro', 'f1'))}",
                 f"{_f(sub.get('+ NER', {}).get('micro', {}).get('f1'))} / {_f(sub.get('+ LLM', {}).get('micro', {}).get('f1'))}"],
                ["Порог / ECE (стратифицированный набор)",
                 f"{_f(v1i.get('threshold'), 2)} / {_f(g1(v1i, 'calibration', 'ece'))}",
                 f"{_f(info['threshold'], 2)} / {_f((info.get('calibration_stratified') or {}).get('ece'))}"],
                ["Скорость, карт./с: правила / + NER",
                 f"{_f(g1(v1.get('full', {}).get('правила'), 'perf', 'cards_per_sec'), 0)} / "
                 f"{_f(g1(v1s, 'perf', 'cards_per_sec'), 0)} (1 замер)",
                 f"{_f(g1(strat.get('правила'), 'perf', 'cards_per_sec'), 0)} / {_f(sm['perf']['cards_per_sec'], 0)} "
                 f"(медиана {sm['perf']['passes']})"]]
        L.append(_table(["Метрика (те же карточки)", "v1", "v2"], rows))
        L.append("Что изменилось v1 → v2 (найдено на dev и на демо-сэмпле `data/sample/real_cards.jsonl`, который в "
                 "eval-набор не входит; замер и выбор вариантов — только на dev). Исправления точечные: F1 на тех же "
                 "карточках почти не меняется, спорные значения чаще уходят модератору:\n")
        L += [f"- {x}" for x in V2_FIXES]
        L.append("")
    # 1. протокол
    L.append("## 1. Набор и протокол\n")
    L.append(f"- **Основной набор — весь holdout с эталоном** ({_f(info['full_cards'])} карточек, без {info['excluded']} SKU "
             "демо-сэмпла): максимум карточек на категорию. Состав — как в каталоге, поэтому хвост (generic) — "
             "большинство.")
    L.append(f"- **Стратифицированный набор** ({_f(info['stratified_cards'])} карточек: все {_f(info['n_tax'])} карточек "
             f"категорий таксономии + {_f(info['n_gen'])} карточек хвоста по полосам размера) — тот же, что в v1, для "
             "сравнения версий; выгружен в `data/sample/holdout_eval_cards.jsonl` + `holdout_eval_gold.jsonl`.")
    L.append("- **Что получает пайплайн:** название, очищенное описание, блок «Характеристики» продавца и **id категории "
             "каталога** (`category_id`, как у карточки в листинге каталога; название категории используется, только если "
             "id нет). Характеристики фида (`vendor`, `model`, `param`) в пайплайн не подаются. Имя продавца передаётся, но "
             "в извлечении не участвует (только для разрезов). Отдельный прогон без `category_id` — раздел 3.")
    L.append("- **Holdout:** правила, словари (train), NER (train), калибровка и порог (dev) заморожены до прогона; режим "
             "в сдаче — по правилу, заданному до прогона («+ NER», если не теряет > 0,005 точности и не хуже по F1).\n")
    # 2. весь holdout
    if fullh:
        L.append(f"## 2. Весь holdout ({_f(info['full_cards'])} карточек) — основная таблица\n")
        L.append(_table(MODE_HEAD, _mode_rows(fullh)))
        L.append(f"Скорость — медиана {fm['perf']['passes']} замеров после прогрева на 50 карточках (одна машина: CPU + "
                 "GTX 1060 для NER); `cost.json` и раздел 12 — по этим замерам.\n")
    # 3. стратифицированный
    L.append(f"## 3. Стратифицированный набор ({_f(info['stratified_cards'])} карточек, как v1)\n")
    rows3 = _mode_rows(strat) + _mode_rows({f"{final} без category_id": nc})
    L.append(_table(MODE_HEAD, rows3))
    L.append(f"Скорость — медиана {strat[final]['perf']['passes']} замеров после прогрева на 50 карточках. Без "
             f"`category_id` категорию таксономии выбирает классификатор по тексту: угадана у {_pct(nc['category_accuracy'])} "
             "карточек.\n")
    # 4. LLM
    if sub:
        n_sub = next(iter(sub.values()))["n_cards"]
        L.append(f"## 4. LLM на подвыборке ({_f(n_sub)} карточек — те же, что в v1; все режимы на них)\n")
        L.append(_table(MODE_HEAD, _mode_rows(sub)))
        st_ = info.get("llm_stats") or {}
        lat = st_.get("latency_s") or {}
        lv = info.get("llm_value") or {}
        L.append(f"LLM: {st_.get('model')} ({st_.get('provider')}, выбор `--llm-provider {(info.get('llm') or {}).get('requested')}` "
                 f"-> {(info.get('llm') or {}).get('choice')}), кэш выключен — настоящие вызовы: {_f(st_.get('calls'))} запросов, "
                 f"ошибок {_f(st_.get('errors'))}, {_f((st_.get('prompt_tokens') or 0) / max(1, st_.get('calls') or 1), 0)} + "
                 f"{_f((st_.get('completion_tokens') or 0) / max(1, st_.get('calls') or 1), 0)} токенов на вызов, "
                 f"{_f(lat.get('avg'), 1)} с на вызов (p95 {_f(lat.get('p95'), 1)} с).")
        if lv:
            cfx = lv["counterfactual_accept_llm_only"]
            L.append(f"Значений только от LLM (на проверке): {_f(lv['llm_only_values'])}, из них {_f(lv['new_slots'])} "
                     f"закрывают пустой слот; где есть эталон ({_f(lv['with_gold'])}), верны {_pct(lv['precision'])}. "
                     f"Если бы их принимали автоматически: P / R / F1 = {_f(cfx['precision'])} / {_f(cfx['recall'])} / "
                     f"{_f(cfx['f1'])}.\n")
    # 5. новые названия
    L.append("## 5. Новые названия\n")
    L.append(f"Название карточки дословно (без регистра, ё/е и лишних пробелов) есть в train у "
             f"{_pct(nt.get('share_seen_full', nt.get('share_seen_stratified')))} карточек основного набора — одинаковые "
             "товары разных продавцов и повторные карточки. На остальных правила и словари не могли «запомнить» ответ:\n")
    rows5 = []
    for k, title in (("full_holdout", "весь holdout"), ("stratified", "стратифицированный")):
        if k in nt:
            for kk, t2 in ((k, "новые названия"), (k + "_seen", "названия из train")):
                v = nt[kk]
                rows5.append([f"{title}: {t2}", _f(v.get("n_cards")), _f(v.get("precision")), _f(v.get("recall")),
                              _f(v.get("f1")), _f(v.get("recall_extractable")), _f(v.get("accepted_precision"))])
    L.append(_table(["Срез («" + final + "»)", "Карточек", "P", "R", "F1", "R_extr", "P принятых"], rows5))
    # 6. атрибуты
    L.append(f"## 6. По атрибутам («{final}», основной набор)\n")
    pa = sorted(fm["per_attribute"].items(), key=lambda kv: kv[1]["f1"])
    L.append(_table(["Атрибут", "P", "R", "F1", "R_extr", "Эталона", "TP", "FP", "FN"],
                    [[a, _f(v["precision"]), _f(v["recall"]), _f(v["f1"]), _f(v["recall_extractable"]),
                      _f(v["n_gold"]), _f(v["tp"]), _f(v["fp"]), _f(v["fn"])] for a, v in pa]))
    # 7. категории
    L.append(f"## 7. По категориям («{final}», основной набор)\n")
    rows7 = []
    for cat, attrs in sorted(fm["per_category"].items()):
        cells = [v for v in attrs.values() if v["n_gold"]]
        if not cells:
            continue
        tp, fp, fn = (sum(v[k] for v in cells) for k in ("tp", "fp", "fn"))
        rows7.append([tax.categories[cat].title if cat in tax.categories else cat, _f(max(v["n_cards"] for v in cells)),
                      _f(tp / (tp + fp) if tp + fp else None), _f(tp / (tp + fn) if tp + fn else None),
                      _f(2 * tp / (2 * tp + fp + fn) if tp else 0.0), _f(sum(v["f1"] for v in cells) / len(cells))])
    L.append(_table(["Категория", "Карточек", "P", "R", "micro-F1", "macro-F1 по атрибутам"], rows7))
    # 8. сегменты
    L.append("## 8. Справедливость: сегменты (основной набор)\n")
    names = {"seller_type": "Продавец", "size_band": "Размер категории", "desc_band": "Длина описания"}
    rows8 = []
    for k, title in names.items():
        for sname, v in sorted(seg.get(k, {}).items()):
            if v["n_cards"] >= 5:
                rows8.append([f"{title}: {sname}", _f(v["n_cards"]), _f(v["precision"]), _f(v["recall"]), _f(v["f1"]),
                              _f(v["recall_extractable"]), _pct(v["coverage"])])
    L.append(_table(["Сегмент", "Карточек", "P", "R", "F1", "R_extr", "Покрытие"], rows8))
    # 9. калибровка
    L.append("## 9. Калибровка уверенности (выучена на dev, проверена на holdout)\n")
    L.append(f"ECE на основном наборе — **{_f(rel['ece'])}** ({_f(rel['n'])} значений с эталоном, любой статус); на dev "
             f"кросс-фиттингом — {_f(cf.get('ece'))}.")
    v1band = None
    if v1:
        for b in (v1.get("info", {}).get("calibration") or {}).get("bins", []):
            if abs(b["lo"] - 0.7) < 1e-9:
                v1band = b
    v2band = next((b for b in (info.get("calibration_stratified") or {}).get("bins", []) if abs(b["lo"] - 0.7) < 1e-9),
                  None)
    if v1band:
        L.append(f"**Слабое место (найдено ревьюером на holdout v1):** на стратифицированном наборе в бине 0,7–0,8 "
                 f"уверенность обещала {_f(v1band['confidence'])}, а верных было {_f(v1band['accuracy'])} (n = {v1band['n']})"
                 + (f"; в v2 на тех же карточках — {_f(v2band['confidence'])} против {_f(v2band['accuracy'])} "
                    f"(n = {v2band['n']})" if v2band else "")
                 + ". На holdout ничего не подстраивалось — ниже как есть.")
    tops = "; ".join(f"{c['cell']} — {_f(c['n'])} (верны {_f(c['precision'])})" for c in band.get("top_cells", []))
    L.append(f"**Точность принятых в [порог; 0,8)** (v2, основной набор): **{_f(band['precision'])}** при средней "
             f"уверенности {_f(band['mean_confidence'])}, n = {_f(band['n'])} (95% ДИ от {_f(band['lower95'])}) — "
             f"{_pct(band['share_of_accepted'])} всех принятых, но {_f(band['wrong'])} из {_f(band['wrong_accepted_total'])} "
             f"неверных принятых. Больше всего: {tops}."
             + (f" {_pct(band['generic_share'], 0)} слоя — ячейки хвоста (generic): в наборе калибровки (dev: все "
                "карточки таксономии + 1 000 карточек хвоста) у них мало точек у порога, а в каталоге хвост — "
                "большинство карточек." if (band.get("generic_share") or 0) >= 0.5 else ""))
    bs = info.get("band_stratified") or {}
    if bs.get("n"):
        L.append(f"На стратифицированном наборе тот же слой — {_f(bs['precision'])} (n = {_f(bs['n'])}, "
                 f"{_pct(bs['share_of_accepted'])} принятых). Проверка на dev и кандидат на исправление (калибровка на всех "
                 "карточках dev, замер только на dev) — docs/EVALUATION.md §9.")
    L.append("")
    L.append(_table(["Уверенность", "Средняя уверенность", "Доля верных", "Значений"],
                    [[f"{_f(b['lo'], 1)}–{_f(b['hi'], 1)}", _f(b["confidence"]), _f(b["accuracy"]), _f(b["n"])]
                     for b in rel["bins"]]))
    # 10. порог
    op = operating_point(fm, info["threshold"])
    L.append("## 10. Порог автопринятия\n")
    L.append(f"Порог **{_f(info['threshold'], 2)}** выбран на dev (нижняя граница 95% ДИ точности принятых ≥ 0,95 на "
             "кросс-фиттинге, ни одна ячейка «категория/атрибут» не ниже 0,75). **Рабочая точка ровно при пороге** "
             f"(статусы пайплайна): P {_f(op['precision'])}, R {_f(op['recall'])}, F1 {_f(op['f1'])}, покрытие "
             f"{_pct(op['coverage'])}, новых значений {_f(op['new_fills'])}, на проверку {_pct(op['review_rate'])}. "
             "Кривая «принять всё с уверенностью ≥ t» — `threshold.json` (с точкой ровно на пороге).\n")
    # 11. покрытие
    L.append("## 11. Покрытие каталога\n")
    rows11 = [[n, _pct(m["catalog_fill"]), _pct(m["union_fill"]), _f(m["new_fills"]),
               _pct(m["traffic_weighted_coverage"]["before"]), _pct(m["traffic_weighted_coverage"]["after"])]
              for n, m in prim.items()]
    L.append(_table(["Режим", "Слоты: каталог", "+ пайплайн", "Новых значений", "Спрос: каталог", "+ пайплайн"], rows11))
    cold = info.get("cold") or {}
    if cold:
        L.append(f"Карточки вне фида ({_f(cold['n_cards'])} карточек holdout без оффера, эталона нет; таких в каталоге 75%): "
                 f"автоматически заполняется {_pct(cold['accepted_share'])} слотов схемы, на проверку — "
                 f"{_pct(cold['review_share'])}; категория из таксономии у {_pct(cold['taxonomy_share'])} карточек.\n")
    # 12. стоимость
    L.append("## 12. Стоимость и скорость (₽ на 1000 карточек)\n")
    L.append(_table(["Режим", "Основа", "₽ / 1000", "мс на карточку", "Доля карточек с LLM", "LLM-запросов на 1000"],
                    [[c["mode"], c["basis"], "< 0,01" if 0 <= (c["cost_per_1000"] or 0) < 0.005 else _f(c["cost_per_1000"], 2),
                      _f(c.get("ms_per_card"), 1),
                      _pct(c.get("llm_share")), _f(c.get("llm_calls_per_1000"), 0)] for c in cost["modes"]]))
    L.append(cost["note"] + "\n")
    # 13. ошибки
    errs = info["errors"]
    L.append("## 13. Разбор ошибок (основной набор)\n")
    L.append(f"Строк: {_f(errs['total'])} — неверных {_f(errs['by_kind'].get('wrong', 0))}, пропусков "
             f"{_f(errs['by_kind'].get('missed', 0))}, значений там, где в каталоге пусто (не ошибка) "
             f"{_f(errs['by_kind'].get('extra', 0))}; в `errors.csv` — {_f(errs['written'])} строк (неверные — все, "
             f"остальное — стратифицированно по типу). Из пропусков **{_f(errs['missed_not_in_text'])} — значения нет в "
             f"тексте**, {_f(errs['missed_in_review'])} — найдено, но ниже порога, {_f(errs['missed_not_found'])} — в тексте "
             "есть, но не найдено.")
    L.append("Больше всего неверных принятых: " + ", ".join(f"{k} — {v}" for k, v in errs["top_wrong"]) + ".\n")
    # 14. ограничения
    L.append("## 14. Ограничения и честность оценки\n")
    L.append("- **Эталон — каталог:** ошибки контент-команды засчитаны как наши ошибки (процессор «AMD HX 370» без "
             "линейки, MacBook 13\" при эталоне 13,6, модель «Телевизор смарт»).")
    L.append(f"- **Потолок «только из текста»:** в тексте есть {_pct(fm['extractable_share'])} эталона; выдумывать остальное "
             "кейс запрещает.")
    L.append("- **v2 после v1:** исправления выбраны на dev и на демо-сэмпле вне eval-набора; holdout посмотрен второй раз "
             "только для отчёта, по нему ничего не подбиралось; v1 сохранён (`data/processed/final_holdout_v1.json`).")
    L.append(f"- **Калибровка у порога оптимистична** (раздел 9): точность принятых в [порог; 0,8) — {_f(band['precision'])} "
             f"(n = {_f(band['n'])}); на holdout не исправлялась.")
    L.append(f"- **Исключены {info['excluded']} SKU демо-сэмпла.** LLM — на подвыборке и 4B-модели на одной GPU.")
    L.append("- **Карточки с эталоном — карточки фида:** для карточек вне фида показано только покрытие.\n")
    L.append("## Файлы\n")
    pr = info["predictions"]
    L.append(f"- `predictions.csv` ({_f(pr['csv_mb'], 1)} МБ) — плоско, весь основной набор ({_f(pr['csv_cards'])} карточек); "
             f"`predictions.jsonl` ({_f(pr['jsonl_mb'], 1)} МБ) — полный ExtractionResult, стратифицированный набор.")
    L.append("- `metrics.json`, `compare.json` (весь holdout, стратифицированный набор, без category_id, подвыборка LLM), "
             "`segments.json`, `calibration.json`, `threshold.json`, `cost.json`, `errors.csv`, `traffic.json`.")
    ex_ = info.get("export") or {}
    if ex_:
        L.append(f"- Стратифицированный набор для воспроизведения без data/processed: `{ex_['cards']}`, `{ex_['gold']}` "
                 f"({_f(ex_['mb'], 1)} МБ).")
    L.append(f"- Воспроизвести: `attrx report-real` ({_f(info['seconds'], 0)} с на этой машине).\n")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
