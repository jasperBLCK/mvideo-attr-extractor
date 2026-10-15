"""Кандидаты в таксономию: какие параметры фида заполнены в категории и насколько они «извлекаемы» из текста.

Для топ-категорий по кликам (+ обязательные: смартфоны, iPhone, ноутбуки) по каждому параметру считаем
заполненность среди офферов, наличие точного варианта `_search`, единицы, частые значения и долю офферов,
у которых значение буквально (или в нормализованной числовой форме) встречается в тексте карточки
(title + описание + seller_attrs — то же, что Card.full_text()). Это вход для выбора обязательных атрибутов и gold.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import atomic_path

MUST_CATEGORIES = ("761", "914", "987")  # Смартфоны, iPhone, Ноутбуки

_NUM = re.compile(r"^\d+(?:[.,]\d+)?$")
_NUM_UNIT = re.compile(r"^(\d+(?:[.,]\d+)?)\s*([^\d\s(]{0,8}?)\.?\s*(?:\([^)]*\))?$")  # «256 гб», «15.6"(39.6 см)»
_NON_WORD = re.compile(r"[\W_]+")
_SPLIT_PARTS = re.compile(r"\s*[,;/]\s*")
# написания единиц в текстах карточек (после norm(): нижний регистр, ё -> е)
UNIT_SYNONYMS = {
    "гб": ("гб", "gb", "гбайт"), "тб": ("тб", "tb"), "мб": ("мб", "mb"),
    "мач": ("мач", "mah", "ма·ч", "ма*ч", "ма ч"), '"': ('"', "”", "″", "''", "дюйм", "inch"),
    "гц": ("гц", "hz"), "ггц": ("ггц", "ghz"), "мгц": ("мгц", "mhz"), "мпикс": ("мп", "mp", "мегапикс"),
    "вт": ("вт", "w"), "квт": ("квт", "kw"), "мм": ("мм", "mm"), "см": ("см", "cm"), "м": ("м", "m"),
    "кг": ("кг", "kg"), "г": ("г", "гр", "g"), "л": ("л", "литр", "l"), "мл": ("мл", "ml"), "дб": ("дб", "db"),
    "в": ("в", "v"), "ч": ("ч", "час"), "шт": ("шт",),
}


def norm(s: str | None) -> str:
    return " ".join(str(s or "").lower().replace("ё", "е").split())


@lru_cache(maxsize=100_000)
def _number_pattern(v: str) -> re.Pattern:
    """6.1 ~ «6,1» / «6.1»; 5000 ~ «5000» / «5 000»; не внутри другого числа."""
    whole, _, frac = v.replace(",", ".").partition(".")
    frac = frac.rstrip("0")
    if len(whole) > 3:  # разряды через пробел: 5 000
        head, tail = whole[: len(whole) % 3 or 3], whole[len(whole) % 3 or 3:]
        whole = head + "".join(r"\s?" + tail[i:i + 3] for i in range(0, len(tail), 3))
    body = whole + (r"[.,]" + frac + r"0*" if frac else r"(?:[.,]0+)?")
    return re.compile(r"(?<![\d.,])" + body + r"(?![\d]|[.,]\d)")


@lru_cache(maxsize=100_000)
def _word_pattern(v: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(v) + r"(?!\w)")


@lru_cache(maxsize=100_000)
def _number_unit_pattern(num: str, unit: str) -> re.Pattern | None:
    """Число + любое написание единицы: 256 ГБ ~ «256GB», «256 гб», «12/256gb»; 15.6" ~ «15.6 дюйма»."""
    syn = UNIT_SYNONYMS.get(unit.strip(". ").replace("ё", "е"))
    if not syn:
        return None
    alts = "|".join(re.escape(s) + ("" if len(s) > 2 else r"(?![a-zа-я])") for s in sorted(syn, key=len, reverse=True))
    return re.compile(_number_pattern(num).pattern + r"\s?(?:" + alts + ")")


def _numeric_hit(num: str, unit: str, text: str) -> str | None:
    """exact — число с единицей (или длинное число без известной единицы), loose — просто число где-то в тексте."""
    nums = [num]
    if unit in ("гб", "gb") and float(num.replace(",", ".")) % 1024 == 0:  # 1024 ГБ ~ «1 ТБ»
        nums.append(str(int(float(num) // 1024)))
    for i, n in enumerate(nums):
        p = _number_unit_pattern(n, unit if i == 0 else "тб")
        if p is not None and p.search(text):
            return "exact"
    if _number_pattern(num).search(text):
        known = _number_unit_pattern(num, unit) is not None
        return "loose" if known or len(num.replace(",", "").replace(".", "")) < 2 else "exact"
    return None


def value_in_text(value: str | None, text: str, compact: str | None = None, unit: str | None = None) -> str | None:
    """-> 'exact' | 'loose' | None; text — уже norm(). Для чисел учитывает единицу (из value или поля unit)."""
    v = norm(value)
    if not v:
        return None
    if _NUM.match(v):
        return _numeric_hit(v, norm(unit), text)
    if (m := _NUM_UNIT.match(v)) and m[2]:
        return _numeric_hit(m[1], m[2], text)
    if v in text and _word_pattern(v).search(text):
        return "exact"
    parts = [p for p in _SPLIT_PARTS.split(v) if len(p) > 1]
    if len(parts) > 1 and all(p in text and _word_pattern(p).search(text) for p in parts):
        return "exact"
    if compact is not None:
        cv = _NON_WORD.sub("", v)
        if len(cv) >= 3 and cv in compact:
            return "loose"
    return None


def card_text(title: str, desc: str, seller_attrs: str) -> str:
    attrs = json.loads(seller_attrs) if seller_attrs else {}
    return "\n".join(x for x in (title, desc, "; ".join(f"{k}: {v}" for k, v in attrs.items())) if x)


def _rate(hits: list[str | None]) -> dict:
    n = len(hits)
    if not n:
        return {"n_checked": 0, "extractable": None, "extractable_loose": None}
    exact = sum(h == "exact" for h in hits)
    return {"n_checked": n, "extractable": round(exact / n, 3), "extractable_loose": round((exact + sum(
        h == "loose" for h in hits)) / n, 3)}


def build(out: Path, top_k: int = 25, sample: int = 1500, max_params: int = 80, seed: int = 0, con=None) -> dict:
    from .clicks import connect

    con = con or connect()
    cats = pd.read_parquet(out / "categories.parquet")
    by_cat = pd.read_parquet(out / "clicks_by_category.parquet")
    total_clicks = int(by_cat["clicks"].sum())
    leaf = cats[cats["is_leaf"] & (cats["n_offers"] > 0)].set_index("id")
    by_cat = by_cat[by_cat["category_id"].isin(leaf.index)]
    chosen = list(by_cat.head(top_k)["category_id"])
    chosen += [c for c in MUST_CATEGORIES if c not in chosen and c in leaf.index]
    clicks = by_cat.set_index("category_id")

    con.execute(f"""CREATE OR REPLACE TEMP TABLE off_top AS
        SELECT sku_id, category_id, vendor, model, seller FROM read_parquet('{out / 'offers.parquet'}')
        WHERE category_id IN ({','.join(f"'{c}'" for c in chosen)})""")
    params = con.execute(f"""
        SELECT o.category_id, p.sku_id, p.name, p.is_search, p.unit, p.value
        FROM read_parquet('{out / 'params.parquet'}') p JOIN off_top o USING (sku_id)
        WHERE NOT p.is_service AND p.value IS NOT NULL""").df()
    offers = con.execute("SELECT * FROM off_top").df()
    cards = con.execute(f"""
        SELECT c.sku_id, c.title, c.description_clean, c.seller_attrs, c.n_seller_attrs
        FROM read_parquet('{out / 'cards.parquet'}') c JOIN off_top o USING (sku_id)""").df()
    top_q = con.execute(f"""SELECT category_id, query, clicks FROM read_parquet('{out / 'top_queries.parquet'}')
        WHERE category_id IN ({','.join(f"'{c}'" for c in chosen)})""").df()

    # предпочтительное значение параметра: точное (_search), если есть, иначе бакет/обычное
    pref_flag = params.groupby(["sku_id", "name"])["is_search"].transform("max")
    pref = params[params["is_search"] == pref_flag]
    texts = {r.sku_id: card_text(r.title, r.description_clean, r.seller_attrs) for r in cards.itertuples()}

    result = []
    for cid in chosen:
        c_off = offers[offers["category_id"] == cid]
        n_off = len(c_off)
        c_cards = cards[cards["sku_id"].isin(c_off["sku_id"])]
        with_text = [s for s in c_off["sku_id"] if s in texts]
        checked = set(pd.Series(with_text, dtype=object).sample(min(sample, len(with_text)), random_state=seed)) \
            if with_text else set()
        norm_text = {s: norm(texts[s]) for s in checked}
        compact = {s: _NON_WORD.sub("", t) for s, t in norm_text.items()}
        title_n = {s: norm(texts[s].split("\n", 1)[0]) for s in checked}

        def field_stats(col: str, c_off=c_off, n_off=n_off, checked=checked, norm_text=norm_text, compact=compact,
                        title_n=title_n) -> dict:
            filled = {s: v for s, v in zip(c_off["sku_id"], c_off[col], strict=True)
                      if isinstance(v, str) and v.strip()}
            hits = [value_in_text(filled[s], norm_text[s], compact[s]) for s in checked if s in filled]
            hits_t = [value_in_text(filled[s], title_n[s]) for s in checked if s in filled]
            return {"fill_rate": round(len(filled) / max(n_off, 1), 3), **_rate(hits),
                    "extractable_title": _rate(hits_t)["extractable"],
                    "top_values": Counter(filled.values()).most_common(10)}

        c_par = params[params["category_id"] == cid]
        pref_by_name = dict(iter(pref[pref["category_id"] == cid].groupby("name")))
        stats = []
        for name, g in c_par.groupby("name"):
            n_with = g["sku_id"].nunique()
            gp = pref_by_name[name]
            # при нескольких значениях — последнее; unit — у точного варианта _search
            first = {s: (v, u) for s, v, u in zip(gp["sku_id"], gp["value"], gp["unit"], strict=True)}
            hits = [value_in_text(first[s][0], norm_text[s], compact[s], first[s][1]) for s in checked if s in first]
            hits_t = [value_in_text(first[s][0], title_n[s], None, first[s][1]) for s in checked if s in first]
            search, bucket = g[g["is_search"]], g[~g["is_search"]]
            stats.append({
                "name": name, "fill_rate": round(n_with / max(n_off, 1), 3), "n_offers": int(n_with),
                "has_search": bool(len(search)), "has_bucket": bool(len(bucket)),
                "multi_valued_share": round(float((gp.groupby("sku_id").size() > 1).mean()), 3),
                "units": Counter(gp["unit"].dropna()).most_common(3),
                "top_values": Counter(gp["value"]).most_common(10),
                "top_bucket_values": Counter(bucket["value"]).most_common(5) if len(search) and len(bucket) else [],
                **_rate(hits), "extractable_title": _rate(hits_t)["extractable"],
            })
        stats.sort(key=lambda x: -x["fill_rate"])
        keys: Counter = Counter()
        for sa in c_cards["seller_attrs"]:
            keys.update(json.loads(sa).keys())
        row = leaf.loc[cid]
        cl = clicks.loc[cid] if cid in clicks.index else None
        sellers = c_off["seller"].fillna("")
        result.append({
            "id": cid, "title": row["title"], "path": list(row["path"]), "n_offers": int(n_off),
            "n_cards_with_offer": int(len(c_cards)),
            "clicks": int(cl["clicks"]) if cl is not None else 0,
            "clicks_share": round(int(cl["clicks"]) / total_clicks, 4) if cl is not None else 0.0,
            "mvideo_offer_share": round(float((sellers == "М.Видео").mean()), 3),
            "char_block_share": round(float((c_cards["n_seller_attrs"] > 0).mean()), 3) if len(c_cards) else 0.0,
            "vendor": field_stats("vendor"), "model": field_stats("model"),
            "params": stats[:max_params], "n_params_total": len(stats),
            "top_seller_attr_keys": keys.most_common(40),
            "top_queries": top_q[top_q["category_id"] == cid].head(20)[["query", "clicks"]].values.tolist(),
        })
    data = {"total_clicks": total_clicks, "sample_per_category": sample,
            "note": "fill_rate — доля офферов категории с параметром; extractable — доля проверенных офферов "
                    "(с текстом карточки), где значение есть в title+описании+seller_attrs (числа — вместе с единицей "
                    "в любом написании: 256 ГБ ~ 256GB; 1024 ГБ ~ 1 ТБ); extractable_loose — плюс голое число где "
                    "угодно и совпадение без пробелов/дефисов (верхняя оценка); extractable_title — только заголовок; "
                    "значения — точный вариант _search, если он есть",
            "categories": result}
    with atomic_path(out / "taxonomy_candidates.json") as tmp:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return {"categories": len(result), "chosen": chosen,
            "clicks_share_covered": round(sum(r["clicks_share"] for r in result), 4)}


def param_stats(out: Path, con=None) -> int:
    """Заполненность всех параметров во всех категориях -> param_stats.parquet (для хвоста и cold start)."""
    from .clicks import connect, copy_to

    con = con or connect()
    copy_to(con, f"""
        WITH o AS (SELECT sku_id, category_id FROM read_parquet('{out / 'offers.parquet'}')),
             n AS (SELECT category_id, count(*) AS n_offers FROM o GROUP BY 1)
        SELECT o.category_id, p.name, count(DISTINCT p.sku_id) AS n_offers_with,
               round(count(DISTINCT p.sku_id) / any_value(n.n_offers), 4) AS fill_rate,
               bool_or(p.is_search) AS has_search, bool_or(NOT p.is_search) AS has_bucket,
               mode(p.unit) AS unit, mode(p.value) AS top_value
        FROM read_parquet('{out / 'params.parquet'}') p JOIN o USING (sku_id) JOIN n USING (category_id)
        WHERE NOT p.is_service GROUP BY 1, 2 ORDER BY 1, 4 DESC""", out / "param_stats.parquet")
    return con.execute(f"SELECT count(*) FROM read_parquet('{out / 'param_stats.parquet'}')").fetchone()[0]
