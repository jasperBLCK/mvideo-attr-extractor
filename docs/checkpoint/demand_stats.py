"""Клики по запросам-примерам для слайда «Проблема» (docs/checkpoint/concept.html) и грубая нижняя оценка спроса.

Доли спроса на слайде (56,5% / 24,9%) — из docs/EDA.md §6: детектор `src/attrx/data/traffic.py` (`attrx traffic`)
размечает запросы словарём каталога. Этот скрипт — проще и независимо: только чтение data/raw (query_clicks.parquet,
sku_desc.parquet), результат — docs/checkpoint/demand_stats.json. Его доля `share_any_attribute` — нижняя оценка
по регулярным выражениям: бренд кликнутого товара, число с единицей («45см», «512gb»), «8/256» и цвет; тип, модель
и размер без единицы («телевизоры 55») не считаются. FOCUS — запросы-примеры: клики по точному совпадению запроса.
Запуск: .venv/bin/python docs/checkpoint/demand_stats.py (≈2 мин, ≤2 ГБ RAM).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = Path(__file__).with_name("demand_stats.json")

B = r"(^|[^0-9a-zа-яё])"
NUM_UNIT = (r"\d+([.,]\d+)?\s*(гб|gb|тб|tb|мб|mb|мач|mah|вт|w|гц|hz|мп|mp|л|кг|см|мм|квт|ач)([^a-zа-яё]|$)"
            r"|\d+([.,]\d+)?\s*(дюйм|литр|ядер|\"|'')")
PAIR = r"\d+\s*/\s*\d+"
COLOR = (B + r"(черн|бел(ый|ая|ое|ые|ого|ую)|красн|син(ий|яя|ее|ие|его|юю)|сер(ый|ая|ое|ые|ого|ую)|золот|серебр|розов|"
         r"зел[её]н|голуб|фиолет|бежев|коричн|оранжев|ж[её]лт|графит|бирюз)"
         r"|(^|[^a-z])(black|white|silver|gold|blue|pink|green|gr[ae]y)([^a-z]|$)")
BRAND = "length(br) >= 2 AND strpos(qt, br) > 0"
FOCUS = {  # запрос -> какая характеристика в нём названа
    "игровые ноутбуки": "тип", "встраиваемые посудомоечные машины 45см": "ширина", "смартфоны samsung": "бренд",
    "телевизоры 55": "диагональ", "накопительные водонагреватели 50л": "объём", "iphone 17 pro max 512gb": "память",
    "холодильник черный": "цвет",
}


def main() -> None:
    if not (RAW / "query_clicks.parquet").exists():
        sys.exit("нет data/raw/query_clicks.parquet — см. scripts/download_data.sh")
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'; SET threads=2; SET preserve_insertion_order=false;")
    con.execute(f"""
        CREATE TEMP VIEW q AS
        SELECT sku_id, lower(trim("toValidUTF8(query_text)")) AS qt, lower(trim("toValidUTF8(sku_brand_name)")) AS br
        FROM '{RAW / "query_clicks.parquet"}'""")
    num, pair, color = (f"regexp_matches(qt, '{p}')" for p in (NUM_UNIT, PAIR, COLOR))
    t = time.time()
    clicks = con.execute(f"""
        SELECT count(*) AS rows,
               avg(({num})::INT) AS share_num_unit,
               avg(({pair})::INT) AS share_memory_pair,
               avg(({color})::INT) AS share_color,
               avg((length(br) >= 2)::INT) AS share_brand_field_filled,
               avg(({BRAND})::INT) AS share_brand_in_query,
               avg(({num} OR {pair} OR {color} OR {BRAND})::INT) AS share_any_attribute
        FROM q""").df().iloc[0].to_dict()

    def top(where: str, n: int) -> list[list]:
        rows = con.execute(f"SELECT qt, count(*) AS c FROM q WHERE {where} GROUP BY qt ORDER BY c DESC LIMIT {n}")
        return [list(r) for r in rows.fetchall()]

    examples = {"num_unit": top(num, 10), "memory_pair": top(pair, 5), "color": top(color, 5), "all": top("true", 10)}
    in_list = ", ".join("'" + q + "'" for q in FOCUS)
    counts = dict(con.execute(f"SELECT qt, count(*) FROM q WHERE qt IN ({in_list}) GROUP BY qt").fetchall())
    focus = sorted(([q, FOCUS[q], counts.get(q, 0)] for q in FOCUS), key=lambda r: -r[2])
    desc = con.execute(f"""
        SELECT count(*) AS rows,
               median(length(coalesce(description, ''))) AS median_description_chars,
               avg((length(coalesce(description, '')) < 200)::INT) AS share_description_lt_200,
               avg((strpos(coalesce(description, ''), '<') > 0)::INT) AS share_description_with_html,
               avg((strpos(coalesce(description, ''), 'Характеристики') > 0)::INT) AS share_with_word_harakteristiki
        FROM '{RAW / "sku_desc.parquet"}'""").df().iloc[0].to_dict()
    res = {"source": "data/raw/query_clicks.parquet, data/raw/sku_desc.parquet", "method": __doc__.strip(),
           "clicks": clicks, "focus_queries": focus, "examples": examples, "descriptions": desc, "seconds": round(time.time() - t, 1)}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"clicks": clicks, "focus": focus, "descriptions": desc}, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
