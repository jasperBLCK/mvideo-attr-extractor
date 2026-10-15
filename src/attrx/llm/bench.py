"""Бенчмарк LLM на реальных карточках М.Видео: латентность, валидность JSON, заземление цитат, токены, точность.

    attrx llm-bench --models qwen3:4b-instruct,qwen2.5:7b-instruct --n 30
    python -m attrx.llm.bench --models qwen3:4b-instruct --base-url http://localhost:11434/v1

Сценарий — «LLM на каждую карточку»: модель извлекает ВСЕ атрибуты категории (худший случай по стоимости),
запрос строится тем же кодом, что в проде (extractors/llm.build_request), кэш выключен.
Карточки: data/sample/real_cards.jsonl (+ real_params.jsonl — параметры оффера из YML-фида как эталон),
иначе первые row group data/raw/sku_desc.parquet (без эталона: только латентность, JSON и заземление).
Эталон здесь — облегчённое сопоставление параметров фида с атрибутами (GOLD_PARAMS), официальная оценка — attrx report.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

from ..config import ROOT, get_settings
from ..extractors.llm import build_request, parse_response
from ..normalize import normalize_value
from ..schema import Card
from ..taxonomy import Taxonomy, load_taxonomy, norm
from ..verify import grounded
from .client import LLMClient, LLMUnavailable, ipv4_loopback
from .cost import cost_table_md, per_1000_cards

PHONE_CATS = {"Смартфоны", "iPhone"}
LAPTOP_CATS = {"Ноутбуки"}
# параметры оффера (YML) → атрибуты таксономии; бренд — поле vendor
GOLD_PARAMS = {
    "smartphone": {"storage": ["Встроенная память (ROM)", "Встроенная память"], "screen": ["Диагональ"],
                   "battery": ["Емкость аккумулятора"], "color": ["Цвет"], "model": ["Модель"]},
    "laptop": {"cpu": ["Модель процессора", "Процессор"], "ram": ["Оперативная память (RAM)", "Оперативная память"],
               "storage": ["Общий объем накопителя", "Объём SSD"], "screen": ["Диагональ экрана"], "color": ["Цвет"]},
}


def _order(cid: str) -> str:
    return hashlib.md5(cid.encode()).hexdigest()


def _compact(s: str) -> str:
    return re.sub(r"[\s\-_\"'«»]+", "", norm(s))


def load_bench_cards(n: int = 30, root: Path = ROOT) -> tuple[list[Card], dict[str, dict]]:
    """n карточек (половина смартфоны, половина ноутбуки) и эталон {card_id: {attr: value}} — детерминированно."""
    sample = root / "data" / "sample" / "real_cards.jsonl"
    if sample.exists():
        rows = [json.loads(x) for x in sample.read_text(encoding="utf-8").splitlines() if x.strip()]
        params_path = root / "data" / "sample" / "real_params.jsonl"
        params = {}
        if params_path.exists():
            params = {str(p["id"]): p for p in map(json.loads, params_path.read_text(encoding="utf-8").splitlines())}
        phones = sorted((r for r in rows if r.get("category") in PHONE_CATS), key=lambda r: _order(str(r["id"])))
        laptops = sorted((r for r in rows if r.get("category") in LAPTOP_CATS), key=lambda r: _order(str(r["id"])))
        picked = phones[: n - n // 2] + laptops[: n // 2]
        fields = set(Card.model_fields)
        cards = [Card(**{k: v for k, v in r.items() if k in fields}) for r in picked]
        tax = load_taxonomy(str(get_settings().taxonomy_path))
        gold = {c.id: _gold(params[c.id], "smartphone" if r["category"] in PHONE_CATS else "laptop", tax)
                for c, r in zip(cards, picked, strict=True) if c.id in params}
        return cards, gold
    return _cards_from_raw(n, root), {}


def _gold(offer: dict, category: str, tax: Taxonomy | None = None) -> dict:
    """Эталон оффера: источники из таксономии (поле gold: vendor | model | имена param), иначе GOLD_PARAMS."""
    by_name = {}
    for p in offer.get("params") or []:
        by_name.setdefault(p["name"], f"{p['value']} {p['unit']}" if p.get("unit") else str(p["value"]))
    by_name["vendor"], by_name["model"] = offer.get("vendor"), offer.get("model")
    cat = tax.categories.get(category) if tax else None
    sources = {a: spec.gold for a, spec in cat.attributes.items() if getattr(spec, "gold", None)} if cat else {}
    sources = sources or {"brand": ["vendor"], **GOLD_PARAMS[category]}
    return {attr: next((by_name[x] for x in names if by_name.get(x)), None) for attr, names in sources.items()}


def _cards_from_raw(n: int, root: Path) -> list[Card]:
    """Фолбэк без подготовленных данных: первая row group sku_desc.parquet, фильтр по названию, HTML → текст."""
    import pyarrow.parquet as pq

    from ..data.text import clean_html

    f = pq.ParquetFile(root / "data" / "raw" / "sku_desc.parquet")
    df = f.read_row_group(0, columns=["sku_id", "title", "description"]).to_pandas()
    out = []
    for prefix, k in (("Смартфон", n - n // 2), ("Ноутбук", n // 2)):
        sub = df[df.title.str.startswith(prefix, na=False)]
        sub = sub.assign(o=sub.sku_id.astype(str).map(_order)).sort_values("o").head(k)
        out += [Card(id=str(r.sku_id), title=r.title, description=clean_html(r.description),
                     category="Смартфоны" if prefix == "Смартфон" else "Ноутбуки") for r in sub.itertuples()]
    return out


def _same(pred, truth, spec) -> bool:
    if isinstance(pred, float) and isinstance(truth, float):
        return abs(pred - truth) < 0.051
    if spec.type == "enum":
        return str(pred) == str(truth)
    a, b = _compact(str(pred)), _compact(str(truth))
    return bool(a) and bool(b) and (a in b or b in a)


def _in_text(truth, spec, card_text: str, tax: Taxonomy) -> bool:
    """Есть ли эталонное значение в тексте карточки (как «извлекаемость» в DATA.md: числа — по цифрам)."""
    if isinstance(truth, float):
        num = f"{truth:g}"
        variants = {num, num.replace(".", ",")}
        if truth >= 1024 and truth % 1024 == 0:  # 1024 ГБ ~ 1 ТБ
            variants.add(f"{truth / 1024:g}")
        return any(re.search(rf"(?<![\d.,]){re.escape(v)}(?![\d])", card_text) for v in variants)
    if spec.type == "enum" and spec.dictionary in tax.dictionaries:
        return any(h[0] == truth for h in tax.dictionaries[spec.dictionary].find_all(card_text))
    return _compact(str(truth)) in _compact(card_text)


def _value(item):
    return item.get("value") if isinstance(item, dict) else item


def _vram_mib(base_url: str) -> float | None:
    """Ollama: сколько VRAM занимает загруженная модель (GET /api/ps)."""
    try:
        url = ipv4_loopback(base_url).rstrip("/").removesuffix("/v1") + "/api/ps"
        with urllib.request.urlopen(url, timeout=3) as r:
            models = json.load(r).get("models") or []
        return round(sum(m.get("size_vram", 0) for m in models) / 2**20) if models else None
    except Exception:  # не Ollama — VRAM не меряем
        return None


def _new_counts() -> dict:
    return {"slots": 0, "json_ok": 0, "returned": 0, "quote_found": 0, "normalized": 0, "grounded": 0,
            "gold_slots": 0, "gold_returned": 0, "correct": 0, "gold_extractable": 0, "correct_extractable": 0}


def _rates(m: dict, n_cards: int) -> dict:
    def rate(x: int, y: int) -> float | None:
        return round(x / y, 3) if y else None

    return {"json_valid_rate": rate(m["json_ok"], n_cards),
            "values_returned": m["returned"], "quote_found_rate": rate(m["quote_found"], m["returned"]),
            "grounded_rate": rate(m["grounded"], m["returned"]), "coverage": rate(m["grounded"], m["slots"]),
            "gold_precision": rate(m["correct"], m["gold_returned"]), "gold_recall": rate(m["correct"], m["gold_slots"]),
            "gold_recall_extractable": rate(m["correct_extractable"], m["gold_extractable"])}


def score_card(card: Card, cat, attrs: list[str], data: dict, gold: dict[str, dict], tax: Taxonomy, m: dict) -> dict:
    """Ответ модели по одной карточке → счётчики m (JSON, цитаты, заземление, эталон) и строка per_card."""
    m["slots"] += len(attrs)
    text_n, text_c = norm(card.full_text()), _compact(card.full_text())
    if all(isinstance(data.get(a), dict) and {"value", "evidence"} <= set(data[a]) for a in attrs):
        m["json_ok"] += 1
    for a in attrs:
        item = data.get(a) if isinstance(data.get(a), dict) else {"value": data.get(a)}
        if item.get("value") not in (None, ""):
            m["returned"] += 1
            q = str(item.get("evidence") or "")
            m["quote_found"] += bool(q) and (norm(q) in text_n or _compact(q) in text_c)
    cands = parse_response(data, card, cat, tax, attrs)
    m["normalized"] += len(cands)
    ok = [c for c in cands if grounded(c, card, cat.attributes[c.attribute], tax.dictionaries)]
    m["grounded"] += len(ok)
    g = gold.get(card.id) or {}
    truth = {}
    for a in attrs:
        if g.get(a):
            res = normalize_value(str(g[a]), cat.attributes[a], tax.dictionaries)
            if res is not None:
                truth[a] = res[0]
    m["gold_slots"] += len(truth)
    extractable = {a for a, v in truth.items() if _in_text(v, cat.attributes[a], card.full_text(), tax)}
    m["gold_extractable"] += len(extractable)
    for c in ok:
        if c.attribute in truth:
            m["gold_returned"] += 1
            hit = _same(c.value, truth[c.attribute], cat.attributes[c.attribute])
            m["correct"] += hit
            m["correct_extractable"] += hit and c.attribute in extractable
    return {"raw": data, "values": {c.attribute: c.value for c in ok}, "gold": truth,
            "rejected": sorted({a for a in attrs if _value(data.get(a)) not in (None, "")} - {c.attribute for c in ok})}


def rescore(path: Path, n: int | None = None) -> dict:
    """Пересчитать качество по сохранённым ответам модели (после правок таксономии, эталона или verify) — без LLM.
    Латентность и токены остаются из замера."""
    r = json.loads(path.read_text(encoding="utf-8"))
    cards, gold = load_bench_cards(n or r["n_cards"])
    by_id = {c.id: c for c in cards}
    tax = load_taxonomy(str(get_settings().taxonomy_path))
    m = _new_counts()
    per_card = []
    for row in r["per_card"]:
        card = by_id.get(row["id"])
        if card is None or "raw" not in row:
            raise ValueError(f"{path}: нет сохранённого ответа для карточки {row['id']} — нужен новый прогон")
        cat = tax.categories[tax.detect_category(card)[0]]
        per_card.append({"id": card.id, "category": cat.name, "latency_s": row.get("latency_s"),
                         **score_card(card, cat, list(cat.attributes), row["raw"], gold, tax, m)})
    r.update(_rates(m, len(per_card)), counts=m, per_card=per_card)
    path.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    return r


def run_model(model: str, cards: list[Card], gold: dict[str, dict], tax: Taxonomy, *, base_url: str | None = None,
              reasoning_effort: str | None = None, few_shot: bool = False, warmup: bool = True) -> dict:
    s = dataclasses.replace(
        get_settings(), llm_enabled=True, llm_model=model, llm_cache=False, llm_concurrency=1,
        **({"llm_base_url": base_url} if base_url else {}),
        **({"llm_reasoning_effort": reasoning_effort} if reasoning_effort is not None else {}),
    )
    client = LLMClient(s)
    jobs = []
    for card in cards:
        cat = tax.categories[tax.detect_category(card)[0]]
        attrs = list(cat.attributes)
        jobs.append((card, cat, attrs, *build_request(card, cat, tax, attrs, s, few_shot=few_shot)))
    if warmup and jobs:  # загрузка модели в VRAM не должна попасть в латентность
        try:
            client.complete_json(*jobs[0][3:], "attributes")
        except LLMUnavailable:
            pass
        client.reset_stats()
    m = _new_counts()
    per_card, errors, lat = [], [], []
    ok_chars = 0
    for card, cat, attrs, system, user, schema in jobs:
        t = time.perf_counter()
        try:
            data = client.complete_json(system, user, schema, "attributes")
        except LLMUnavailable as e:
            lat.append(time.perf_counter() - t)
            m["slots"] += len(attrs)
            errors.append(f"{card.id}: {e}")
            per_card.append({"id": card.id, "error": str(e)[:200]})
            continue
        lat.append(time.perf_counter() - t)
        ok_chars += len(system) + len(user)
        per_card.append({"id": card.id, "category": cat.name, "latency_s": round(lat[-1], 2),
                         **score_card(card, cat, attrs, data, gold, tax, m)})
    st = client.stats()
    n_ok = len(cards) - len(errors)
    lat_sorted = sorted(lat)

    return {
        "model": model, "provider": client.provider, "n_cards": len(cards), "errors": len(errors),
        **_rates(m, len(cards)),
        "latency_s": {"avg": round(sum(lat) / len(lat), 2) if lat else None,
                      "p50": round(lat_sorted[len(lat) // 2], 2) if lat else None,
                      "p95": round(lat_sorted[min(len(lat) - 1, int(0.95 * len(lat)))], 2) if lat else None},
        "tokens_per_card": {"prompt": round(st["prompt_tokens"] / max(n_ok, 1)),
                            "completion": round(st["completion_tokens"] / max(n_ok, 1))},
        "chars_per_prompt_token": round(ok_chars / max(st["prompt_tokens"], 1), 2),
        "vram_mib": _vram_mib(s.llm_base_url), "response_format": st["response_format"],
        "reasoning_effort": s.llm_reasoning_effort or None, "few_shot": few_shot,
        "counts": m, "stats": st, "cost_every_card": per_1000_cards(st, len(cards), llm_share=1.0),
        "error_samples": errors[:5], "per_card": per_card,
    }


def summary_table(results: list[dict]) -> str:
    head = ("| Модель | Латентность, с (сред / p95) | JSON валиден | Цитата найдена | Заземлено | Точность по эталону "
            "| Полнота (извлекаемый эталон) | Токены вход / выход | VRAM, МиБ |\n|---|---|---|---|---|---|---|---|---|\n")

    def pct(x: float | None) -> str:
        return "—" if x is None else f"{100 * x:.0f}%"

    rows = [f"| {r['model']} | {r['latency_s']['avg']} / {r['latency_s']['p95']} | {pct(r['json_valid_rate'])} | "
            f"{pct(r['quote_found_rate'])} | {pct(r['grounded_rate'])} | {pct(r['gold_precision'])} | "
            f"{pct(r['gold_recall_extractable'])} | {r['tokens_per_card']['prompt']} / {r['tokens_per_card']['completion']} | "
            f"{r['vram_mib'] or '—'} |" for r in results]
    return head + "\n".join(rows) + "\n"


def run_bench(models: list[str], n: int = 30, base_url: str | None = None, reasoning_effort: str | None = None,
              few_shot: bool = False, out: Path | None = None) -> list[dict]:
    cards, gold = load_bench_cards(n)
    tax = load_taxonomy(str(get_settings().taxonomy_path))
    out = out or ROOT / "outputs" / "llm_bench"
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for model in models:
        print(f"== {model}: {len(cards)} карточек ...", flush=True)
        r = run_model(model, cards, gold, tax, base_url=base_url, reasoning_effort=reasoning_effort, few_shot=few_shot)
        tag = re.sub(r"[^\w.-]+", "_", model) + ("_fewshot" if few_shot else "")
        (out / f"{tag}.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(r)
        print(summary_table([r]), flush=True)
    print(summary_table(results))
    for r in results:  # «LLM на каждую карточку»: ₽ на 1000 карточек по прайсу config/llm_prices.yaml
        print(f"\n{r['model']}: стоимость 1000 карточек, если звать LLM на каждую\n{cost_table_md(r['cost_every_card'])}")
    print(f"-> {out}")
    return results


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m attrx.llm.bench", description="Бенчмарк LLM на реальных карточках")
    p.add_argument("--models", default="", help="через запятую, например qwen3:4b-instruct,qwen2.5:7b-instruct")
    p.add_argument("--n", type=int, default=30, help="карточек (половина смартфоны, половина ноутбуки)")
    p.add_argument("--base-url", default=None, help="по умолчанию из настроек (LLM_BASE_URL / OPENAI_BASE_URL)")
    p.add_argument("--reasoning-effort", default=None, help="none — отключить «размышления» (qwen3.x)")
    p.add_argument("--few-shot", action="store_true")
    p.add_argument("--out", default=str(ROOT / "outputs" / "llm_bench"))
    p.add_argument("--rescore", nargs="*", default=None, help="пересчитать качество сохранённых JSON без вызова LLM")
    a = p.parse_args(argv)
    if a.rescore is not None:
        print(summary_table([rescore(Path(x)) for x in a.rescore]))
        return
    run_bench([x.strip() for x in a.models.split(",") if x.strip()], a.n, a.base_url, a.reasoning_effort, a.few_shot,
              Path(a.out))


if __name__ == "__main__":
    main()
