"""Агрегаты по поисковым кликам (query_clicks, 31 млн строк) через DuckDB — без загрузки в pandas целиком.

Найдено при сверке с YML по sku_id: `sku_seo_id` = id категории каталога (совпадает с categoryId оффера в 97,5%
случаев, расхождения — кросс-листинг в соседние категории), `sku_subject_id` — отдельный справочник «предметов»
(тип товара), лишь частично совпадающий с id каталога. Поэтому категорию по кликам берём из seo_id.
"""
from __future__ import annotations

from pathlib import Path

from . import atomic_path

# нормализация запроса: нижний регистр, ё -> е, схлопнутые пробелы
QUERY_NORM = "trim(regexp_replace(replace(lower(\"toValidUTF8(query_text)\"), 'ё', 'е'), '\\s+', ' ', 'g'))"


def connect(threads: int = 3, memory: str = "4GB", tmp: Path | None = None):
    import duckdb

    con = duckdb.connect()
    con.execute(f"SET threads={threads}; SET memory_limit='{memory}'; SET preserve_insertion_order=false")
    try:  # в Jupyter без ipywidgets сама настройка прогресс-бара падает — она не обязательна
        con.execute("SET enable_progress_bar=false")
    except Exception:  # noqa: BLE001
        pass
    if tmp is not None:
        con.execute(f"SET temp_directory='{tmp}'")
    return con


def copy_to(con, select_sql: str, path: Path) -> None:
    """COPY (select) -> parquet zstd через временный файл (атомарная замена)."""
    with atomic_path(path) as tmp:
        con.execute(f"COPY ({select_sql}) TO '{tmp}' (FORMAT parquet, COMPRESSION zstd)")


def aggregate(raw: Path, out: Path, top_n: int = 300, con=None) -> dict:
    """-> clicks_by_sku / clicks_by_category / top_queries.parquet; возвращает сводку для отчёта."""
    con = con or connect(tmp=out / ".duckdb_tmp")
    src, offers = raw / "query_clicks.parquet", out / "offers.parquet"
    q = lambda sql: con.execute(sql).fetchall()  # noqa: E731
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW qc AS
        SELECT sku_id::VARCHAR AS sku_id, {QUERY_NORM} AS query, sku_position::INT AS pos,
               sku_subject_id::VARCHAR AS subject_id, sku_seo_id::VARCHAR AS seo_id, sku_price AS price,
               "toValidUTF8(sku_name)" AS sku_name, "toValidUTF8(sku_brand_name)" AS sku_brand
        FROM read_parquet('{src}')""")
    con.execute(f"CREATE OR REPLACE TEMP TABLE off AS SELECT sku_id, category_id FROM read_parquet('{offers}')")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE by_sku AS
        SELECT sku_id, count(*) AS clicks, count(DISTINCT query) AS n_queries, avg(pos) AS mean_position,
               mode(seo_id) AS seo_id, mode(subject_id) AS subject_id, median(price) AS price,
               any_value(sku_name) AS sku_name, any_value(sku_brand) AS sku_brand
        FROM qc GROUP BY sku_id""")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE by_sku2 AS
        SELECT b.*, coalesce(o.category_id, b.seo_id) AS category_id,
               CASE WHEN o.category_id IS NOT NULL THEN 'offer' ELSE 'clicks_seo' END AS category_source
        FROM by_sku b LEFT JOIN off o USING (sku_id)""")
    copy_to(con, "SELECT * FROM by_sku2 ORDER BY clicks DESC", out / "clicks_by_sku.parquet")
    copy_to(con, """SELECT category_id, sum(clicks)::BIGINT AS clicks, count(*) AS n_skus,
                           (sum(clicks) FILTER (WHERE category_source = 'offer'))::BIGINT AS clicks_with_offer
                    FROM by_sku2 GROUP BY category_id ORDER BY clicks DESC""", out / "clicks_by_category.parquet")
    copy_to(con, f"""
          WITH cq AS (
            SELECT s.category_id, qc.query, count(*) AS clicks, count(DISTINCT qc.sku_id) AS n_skus
            FROM qc JOIN by_sku2 s USING (sku_id) WHERE qc.query <> ''
            GROUP BY s.category_id, qc.query)
          SELECT category_id, query, clicks, n_skus FROM cq
          QUALIFY row_number() OVER (PARTITION BY category_id ORDER BY clicks DESC, query) <= {int(top_n)}
          ORDER BY category_id, clicks DESC""", out / "top_queries.parquet")
    total, n_sku, n_q = q("SELECT count(*), count(DISTINCT sku_id), count(DISTINCT query) FROM qc")[0]
    with_offer = q("SELECT count(*), sum(clicks) FROM by_sku2 WHERE category_source = 'offer'")[0]
    seo_eq, subj_eq, n_join = q("""
        SELECT sum((b.seo_id = o.category_id)::INT), sum((b.subject_id = o.category_id)::INT), count(*)
        FROM by_sku b JOIN off o USING (sku_id)""")[0]
    multi = q("""SELECT sum((n_seo > 1)::INT), sum((n_subj > 1)::INT) FROM (
                   SELECT count(DISTINCT seo_id) n_seo, count(DISTINCT subject_id) n_subj FROM qc GROUP BY sku_id)""")[0]
    con.execute("DROP VIEW qc")
    return {
        "rows": int(total), "skus": int(n_sku), "distinct_queries": int(n_q),
        "skus_with_offer": int(with_offer[0]), "clicks_with_offer_share": round(with_offer[1] / total, 4),
        "seo_id_eq_category_share": round(seo_eq / n_join, 4),
        "subject_id_eq_category_share": round(subj_eq / n_join, 4),
        "skus_with_several_seo_ids": int(multi[0]), "skus_with_several_subject_ids": int(multi[1]),
    }
