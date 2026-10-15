"""Эталон (gold) из каталога М.Видео: vendor/model/param оффера -> канонические значения атрибутов таксономии.

Каталожные характеристики заполнены контент-командой М.Видео (human-curated), поэтому это честный gold по категориям.
Вход модели — только текст карточки (sku_desc), эталон — только фид: утечки нет.

Правила (подробно — docs/EVALUATION.md):
- источник атрибута — `gold:` в config/taxonomy.yaml (vendor | model | имена param по приоритету); у param сначала
  точный вариант `_search`, бакет фильтра («6.6" и более», «от 15 до 16.9») — только если он не диапазон;
- значение нормализуется тем же кодом, что и предсказания (normalize.py): числа -> каноническая единица,
  цвета -> базовая палитра (многоцветные — несколько допустимых значений), модель -> model_core, процессор -> clean_cpu;
- `extractable` — есть ли эталонное значение в тексте карточки (title + описание + атрибуты продавца) в любом
  распознаваемом написании: потолок для извлечения «только из текста».
"""
from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..dictgen import is_placeholder
from ..normalize import (
    NUM,
    UNIT_PATTERNS,
    clean_cpu,
    compact,
    cpu_key,
    fmt_number,
    is_range,
    model_core,
    normalize_bool,
    normalize_colors,
    normalize_enum,
    parse_number,
    to_float,
)
from ..taxonomy import AttrSpec, Dictionary, Taxonomy, _inflect, norm

SIZE_BANDS = ((1000, "head"), (100, "torso"), (0, "tail"))  # по числу офферов в листовой категории YML
DESC_BANDS = ((1500, "long"), (500, "medium"), (0, "short"))  # по длине очищенного описания, символов
MULTI_SPLIT = re.compile(r"\s*(?:/|;|,)\s*")
MEMORY_PAIR = re.compile(r"(?<![\w.])(\d{1,2})\s*(?:gb|гб)?\s*[/+]\s*(\d{1,4})(?![\d.,])\s*(gb|гб|tb|тб)?(?![a-zа-я])")


def band(x: float | None, bands: tuple[tuple[int, str], ...]) -> str:
    x = 0 if x is None or x != x else x  # None/NaN (категория вне дерева) -> нижняя полоса
    return next(name for edge, name in bands if x >= edge)


def model_key(value: Any, brand: str | None = None) -> str:
    """Ключ сравнения моделей: «Galaxy S24+ 8/256GB» ~ «galaxys24plus»."""
    return compact(model_core(str(value), brand).replace("+", " plus "))


def has_model_chars(s: str) -> bool:
    """Модель без латиницы и цифр («Чайник электрический стеклянный») — мусор в фиде, эталоном не считаем."""
    return bool(re.search(r"[a-z0-9]", s.lower())) and len(compact(s)) >= 2


def canon_values(spec: AttrSpec, raw: str, unit: str | None, tax: Taxonomy, brand: str | None = None,
                 ) -> tuple[list[str | float], bool]:
    """Сырое значение фида -> (канонические значения, удалось ли сопоставить со справочником)."""
    raw = str(raw or "").strip()
    if not raw:
        return [], False
    if spec.type == "number":
        text = f"{raw} {unit}" if unit and re.fullmatch(r"\d+(?:[.,]\d+)?", raw) else raw
        parsed = parse_number(text, spec.kind)
        return ([parsed[0]], True) if parsed and parsed[0] > 0 else ([], False)
    if spec.type == "bool":
        b = normalize_bool(raw, spec.keywords)
        return ([b], True) if b else ([], False)
    if spec.type == "enum":
        d = tax.dictionaries[spec.dictionary]
        if spec.dictionary == "brands":  # бренд: канон справочника или vendor как есть (vendor и есть канон каталога)
            if is_placeholder(raw):  # «нет бренда», «No Name», «###» — бренд не указан, это не эталон
                return [], False
            hit = d.get(raw)
            return [hit or re.sub(r"\s+", " ", raw)], True
        if spec.dictionary == "colors":
            vals = normalize_colors(raw, d)
            if vals:
                return list(vals), True
            return ([], False) if norm(raw) in NOT_COLOR else ([norm(raw)], False)
        out: list[str | float] = []
        for part in [raw, *MULTI_SPLIT.split(raw)] if MULTI_SPLIT.search(raw) else [raw]:
            res = normalize_enum(part, d)
            if res and res[0] not in out:
                out.append(res[0])
        return (out, True) if out else ([norm(raw)], False)
    if spec.kind == "cpu":
        c = clean_cpu(raw)
        return ([c], True) if cpu_key(c) else ([], False)
    if spec.kind == "model":
        core = model_core(raw, brand, tax.dictionaries.get("colors"))
        return ([core], True) if has_model_chars(core) else ([], False)
    s = re.sub(r"\s+", " ", raw).strip()
    return [s], True


NOT_COLOR = frozenset({"другие цвета", "другие", "другой", "другой цвет", "прочие", "комбинированный", "нет"})


def resolve(spec: AttrSpec, vendor: str | None, model: str | None, params: dict[str, list[tuple]], tax: Taxonomy,
            ) -> dict | None:
    """Эталон одного атрибута оффера: {values, raw, unit, source, mapped} или None, если в фиде его нет.

    params: {base_name: [(is_search, unit, value), ...]}. Для модели и процессора значения из всех источников
    собираются как допустимые альтернативы (модель: model, «Серия», «Модель»; процессор: полный + семейство).
    """
    collect = spec.kind in {"model", "cpu"}
    found: dict | None = None
    for src in spec.gold:
        if src in ("vendor", "model"):
            rows = [(True, None, vendor if src == "vendor" else model)]
        else:
            rows = params.get(src, [])
            exact = [r for r in rows if r[0]]
            rows = exact or [r for r in rows if not is_range(r[2])]
        vals: list[str | float] = []
        mapped, first = True, None
        for _, unit, value in rows:
            v, ok = canon_values(spec, value, unit, tax, vendor)
            if v and first is None:
                first = (value, unit)
            mapped &= ok or not v
            vals += [x for x in v if x not in vals]
        if not vals:
            continue
        if spec.kind == "model" and found is not None:  # альтернативы модели — только «конкретные» (с цифрой)
            vals = [x for x in vals if re.search(r"\d", str(x))]
        if found is None:
            found = {"values": vals, "raw": str(first[0]), "unit": first[1], "source": src, "mapped": mapped}
            if not collect:
                return found
        else:
            found["values"] += [x for x in vals if x not in found["values"]]
    return found


# ------------------------------------------------------------------ извлекаемость --------------------------------
@lru_cache(maxsize=20_000)
def _num_re(v: float) -> re.Pattern:
    """6.7 ~ «6,7»/«6.7»; 5000 ~ «5000»/«5 000»; не внутри другого числа."""
    s = f"{v:g}" if v != int(v) else str(int(v))
    whole, _, frac = s.partition(".")
    if len(whole) > 3:
        head = len(whole) % 3 or 3
        whole = whole[:head] + "".join(r"\s?" + whole[i:i + 3] for i in range(head, len(whole), 3))
    body = whole + (r"[.,]" + frac + r"0*" if frac else r"(?:[.,]0+)?")
    return re.compile(r"(?<![\d.,])" + body + r"(?![\d]|[.,]\d)")


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 0.01 * abs(b) + 1e-9


def number_in_text(v: float, kind: str | None, text: str) -> bool:
    """Число с единицей в любом написании (256 ГБ ~ «256GB», 1024 ГБ ~ «1 ТБ», «8/256», «12+512GB») или
    «узнаваемое» голое число (дробное или от 100: 6.67, 5000) где угодно в тексте."""
    for unit_re, mult in UNIT_PATTERNS.get(kind or "", []):
        for m in re.finditer(NUM + r"\s*" + unit_re + r"(?![a-zа-я])", text):
            if _close(to_float(m.group(1)) * mult, v):
                return True
    if kind in {"memory", "ram", "storage_drive"}:
        for m in MEMORY_PAIR.finditer(text):
            second = to_float(m.group(2)) * (1024 if m.group(3) in ("tb", "тб") else 1)
            if _close(to_float(m.group(1)), v) or _close(second, v):
                return True
    return (v != int(v) or v >= 100) and bool(_num_re(fmt_number(v)).search(text))


@lru_cache(maxsize=50_000)
def _canon_re(dictionary: Dictionary, canon: str) -> re.Pattern | None:
    syns = [s for s, c in dictionary.lookup.items() if c == canon] or [norm(canon)]
    alts = "|".join(_inflect(s) for s in sorted(syns, key=len, reverse=True) if s)
    return re.compile(r"(?<![\w-])(?:" + alts + r")(?![\w-])") if alts else None


@lru_cache(maxsize=100_000)
def _word_re(s: str) -> re.Pattern:
    return re.compile(r"(?<![\w-])" + re.escape(s) + r"(?![\w-])")


def is_extractable(spec: AttrSpec, gold: dict, text_n: str, text_c: str, tax: Taxonomy) -> bool:
    """Есть ли эталон в тексте (text_n = norm(full_text), text_c = compact(full_text))."""
    vals = gold["values"]
    if spec.type == "number":
        return any(number_in_text(float(v), spec.kind, text_n) for v in vals)
    if spec.type == "bool":
        return vals[0] == "да" and any(norm(k) in text_n for k in spec.keywords)
    if spec.kind == "model":
        return any(len(k) >= 2 and k in text_c for k in (model_key(v) for v in vals))
    if spec.kind == "cpu":
        keys = [cpu_key(v) for v in vals]
        return any(len(k) >= 4 and k in text_c for k in keys) or any(
            _word_re(norm(re.sub(r"(?i)^(intel|amd|apple)\s+", "", str(v)))).search(text_n) for v in vals)
    if spec.type == "enum":
        d = tax.dictionaries[spec.dictionary]
        for v in vals:
            p = _canon_re(d, str(v)) if str(v) in d.canon else None
            if (p is not None and p.search(text_n)) or _word_re(norm(str(v))).search(text_n):
                return True
        raw = norm(gold["raw"])
        return bool(raw) and _word_re(raw).search(text_n) is not None
    return compact(str(vals[0])) in text_c


# ------------------------------------------------------------------ сборка -----------------------------------------
def _attrs_needed(tax: Taxonomy) -> tuple[dict[str, set[str]], set[str]]:
    """Имена param, нужные эталону: {category_id: names} для категорий таксономии + имена generic для остальных."""
    by_id: dict[str, set[str]] = {}
    for cat in tax.categories.values():
        names = {n for spec in cat.attributes.values() for n in spec.gold if n not in ("vendor", "model")}
        for cid in cat.category_ids:
            by_id[cid] = names
    generic = {n for spec in tax.categories["generic"].attributes.values() for n in spec.gold
               if n not in ("vendor", "model")}
    return by_id, generic


def _jsonable(v: Any) -> Any:
    return fmt_number(v) if isinstance(v, float) else v


def gold_for_card(card: dict, offer: dict, params: dict, tax: Taxonomy) -> list[dict]:
    """Строки эталона одной карточки: по одной на атрибут, для которого в фиде есть значение."""
    cat_name = tax.category_of_id(card["category_id"]) or "generic"
    cat = tax.categories[cat_name]
    attrs = card.get("seller_attrs") or "{}"
    sa = json.loads(attrs) if isinstance(attrs, str) else attrs
    text = "\n".join(x for x in (card.get("title") or "", card.get("description_clean") or "",
                                 "; ".join(f"{k}: {v}" for k, v in sa.items())) if x)
    text_n, text_c = norm(text), compact(text)
    rows = []
    for a, spec in cat.attributes.items():
        g = resolve(spec, offer.get("vendor"), offer.get("model"), params, tax)
        if g is None:
            continue
        vals = g["values"]
        rows.append({
            "sku_id": card["sku_id"], "split": card["split"], "category": cat_name, "category_id": card["category_id"],
            "attribute": a, "value": str(_jsonable(vals[0])), "value_num": float(vals[0]) if spec.type == "number" else None,
            "alternatives": json.dumps([_jsonable(v) for v in vals], ensure_ascii=False),
            "raw_value": g["raw"], "raw_unit": g["unit"], "unit": spec.unit, "source": g["source"], "mapped": g["mapped"],
            "extractable": is_extractable(spec, g, text_n, text_c, tax),
        })
    return rows


def build(processed: Path, tax: Taxonomy, *, sku_ids: list[str] | None = None, batch: int = 20_000,
          log=print) -> tuple[Any, dict]:
    """Эталон для всех карточек с оффером (или для sku_ids) -> (DataFrame строк эталона, сводка)."""
    import duckdb
    import pandas as pd

    t0 = time.time()
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")
    con.execute("SET threads=2")
    by_id, generic_names = _attrs_needed(tax)
    need = pd.DataFrame([(cid, n) for cid, names in by_id.items() for n in names], columns=["category_id", "name"])
    con.register("need", need)
    ids = pd.DataFrame({"sku_id": [str(s) for s in sku_ids]}) if sku_ids is not None else None
    if ids is not None:
        con.register("ids", ids)
    where = "c.has_offer" + (" AND c.sku_id IN (SELECT sku_id FROM ids)" if ids is not None else "")
    con.execute(f"""CREATE TEMP TABLE gc AS
        SELECT c.sku_id, c.split, c.category_id, c.seller_type, c.description_len, k.n_offers
        FROM read_parquet('{processed / 'cards.parquet'}') c
        LEFT JOIN read_parquet('{processed / 'categories.parquet'}') k ON k.id = c.category_id WHERE {where}""")
    tax_ids = ",".join(f"'{c}'" for c in by_id) or "''"
    gen = ",".join("'" + n.replace("'", "''") + "'" for n in generic_names) or "''"
    params: dict[str, dict[str, list[tuple]]] = {}
    for sku, name, is_search, unit, value in con.execute(f"""
            SELECT p.sku_id, p.name, p.is_search, p.unit, p.value
            FROM read_parquet('{processed / 'params.parquet'}') p JOIN gc USING (sku_id)
            LEFT JOIN need n ON n.category_id = gc.category_id AND n.name = p.name
            WHERE NOT p.is_service AND p.value IS NOT NULL
              AND (n.name IS NOT NULL OR (gc.category_id NOT IN ({tax_ids}) AND p.name IN ({gen})))
            """).fetchall():
        rows = params.setdefault(sku, {}).setdefault(name, [])
        if (is_search, unit, value) not in rows:  # в фиде бывают дубли param
            rows.append((is_search, unit, value))
    offers = {r[0]: {"vendor": r[1], "model": r[2]} for r in con.execute(f"""
        SELECT o.sku_id, o.vendor, o.model FROM read_parquet('{processed / 'offers.parquet'}') o JOIN gc USING (sku_id)
        """).fetchall()}
    log(f"gold: параметры {sum(len(v) for v in params.values())} имён у {len(params)} SKU, {time.time() - t0:.0f} с")
    meta = con.execute("SELECT sku_id, seller_type, description_len, n_offers FROM gc").df()
    reader = con.execute(f"""SELECT c.sku_id, c.split, c.category_id, c.title, c.description_clean, c.seller_attrs
        FROM read_parquet('{processed / 'cards.parquet'}') c JOIN gc USING (sku_id)""").fetch_record_batch(batch)
    out: list[dict] = []
    n_cards = 0
    for rb in reader:
        for card in rb.to_pylist():
            n_cards += 1
            out += gold_for_card(card, offers.get(card["sku_id"], {}), params.get(card["sku_id"], {}), tax)
        log(f"gold: {n_cards} карточек, {len(out)} значений, {time.time() - t0:.0f} с")
    df = pd.DataFrame(out)
    meta["seller_type"] = meta["seller_type"].map(lambda x: x if isinstance(x, str) else "none")
    meta["size_band"] = meta["n_offers"].map(lambda x: band(x, SIZE_BANDS))
    meta["desc_band"] = meta["description_len"].map(lambda x: band(x, DESC_BANDS))
    df = df.merge(meta[["sku_id", "seller_type", "size_band", "desc_band"]], on="sku_id", how="left")
    summary = {"cards": n_cards, "values": len(df), "seconds": round(time.time() - t0, 1)}
    return df, summary


def to_jsonl_rows(df) -> list[dict]:
    """Строки parquet -> формат {"id", "attributes": {...}} (io.load_gold) + extractable и сегменты для метрик."""
    out: dict[str, dict] = {}
    for r in df.itertuples(index=False):
        row = out.setdefault(r.sku_id, {"id": r.sku_id, "category": r.category, "category_id": r.category_id,
                                        "split": r.split, "seller_type": r.seller_type, "size_band": r.size_band,
                                        "desc_band": r.desc_band, "attributes": {}, "extractable": {}})
        alts = json.loads(r.alternatives)
        row["attributes"][r.attribute] = alts[0] if len(alts) == 1 else alts
        row["extractable"][r.attribute] = bool(r.extractable)
    return list(out.values())


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:  # U+2028/2029 экранируем: str.splitlines() считает их переводом строки
            f.write(json.dumps(r, ensure_ascii=False).replace("\u2028", "\\u2028").replace("\u2029", "\\u2029") + "\n")


def load_jsonl(path: Path, splits: set[str] | None = None) -> tuple[dict[str, dict], dict[str, dict]]:
    """-> (gold {id: {attr: value | [альтернативы]}}, meta {id: {category_id, split, сегменты, extractable}})."""
    gold, meta = {}, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if splits and r.get("split") not in splits:
            continue
        gold[str(r["id"])] = r.get("attributes") or {}
        meta[str(r["id"])] = {k: v for k, v in r.items() if k not in ("id", "attributes")}
    return gold, meta


def load_parquet(path: Path, splits: set[str] | None = None, sku_ids: set[str] | None = None,
                 ) -> tuple[dict[str, dict], dict[str, dict]]:
    """То же из data/processed/gold.parquet (с фильтром по сплитам/SKU)."""
    import pyarrow.parquet as pq

    filters = [("split", "in", sorted(splits))] if splits else None
    rows = pq.read_table(path, filters=filters).to_pylist()
    gold: dict[str, dict] = {}
    meta: dict[str, dict] = {}
    for r in rows:
        if sku_ids is not None and r["sku_id"] not in sku_ids:
            continue
        alts = json.loads(r["alternatives"])
        gold.setdefault(r["sku_id"], {})[r["attribute"]] = alts[0] if len(alts) == 1 else alts
        m = meta.setdefault(r["sku_id"], {k: r[k] for k in ("category", "category_id", "split", "seller_type",
                                                             "size_band", "desc_band")} | {"extractable": {}})
        m["extractable"][r["attribute"]] = bool(r["extractable"])
    return gold, meta
