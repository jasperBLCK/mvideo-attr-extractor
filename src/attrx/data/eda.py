"""EDA реальных данных кейса: поля, пропуски, распределения, длинный хвост категорий, структура атрибутов,
извлекаемость эталона из текста, поисковый спрос, справедливость (М.Видео vs маркетплейс) и бизнес-цифры.

Один код для двух входов: notebooks/01_eda.ipynb вызывает функции по разделам и показывает таблицы и графики,
`attrx eda-real` делает всё разом без Jupyter. Выход: графики docs/img/eda_*.png, цифры
data/processed/eda_stats.json и data/processed/business_stats.json.
Только агрегаты DuckDB поверх data/processed/*.parquet (≤ 4 ГБ RAM, ~2 мин); data/raw/skus.pkl не читается.
Сырые описания (data/raw/sku_desc.parquet) — только для доли HTML-разметки, на детерминированной выборке 20%
(sku_id % 5 = 0). Трафик-веса берутся из submission/traffic.json (`attrx traffic`).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import atomic_path

# --------------------------------------------------------------------------- стиль графиков

INK, ACCENT, GREY, LIGHT, GRID, MUTED = "#1A1A1A", "#E31E24", "#8C8C8C", "#C8C8C8", "#E6E6E6", "#555555"
SEGMENTS = ("М.Видео", "маркетплейс", "без оффера")
SEG_COLOR = {"М.Видео": ACCENT, "маркетплейс": INK, "без оффера": GREY}
SEG_SQL = ("CASE WHEN seller_type = 'mvideo' THEN 'М.Видео' WHEN has_offer THEN 'маркетплейс' "
           "ELSE 'без оффера' END")
MAIN = {"761": "Смартфоны", "914": "iPhone", "987": "Ноутбуки", "1682": "Телевизоры", "2447": "Стиральные машины",
        "666": "Холодильники", "31607": "Наушники"}
# обязательные поля кейса и ключевые свойства главных категорий (имена параметров YML)
KEY_PARAMS = {
    "761": ["Встроенная память (ROM)", "Диагональ", "Емкость аккумулятора", "Цвет", "Оперативная память (RAM)"],
    "914": ["Встроенная память (ROM)", "Диагональ", "Емкость аккумулятора", "Цвет"],
    "987": ["Модель процессора", "Оперативная память (RAM)", "Объём SSD", "Диагональ экрана", "Цвет"],
    "1682": ["Диагональ", "Разрешение", "Технология экрана", "Цвет"],
    "2447": ["Тип стиральной машины", "Максимальная загрузка", "Ширина", "Цвет"],
    "666": ["Тип холодильника", "Размораживание морозильной камеры", "Высота", "Цвет"],
    "31607": ["Тип подключения", "Тип наушников", "Цвет"],
}
# канонические колонки тепловой карты заполненности: регулярка по имени параметра YML
FILL_ATTRS = {
    "Бренд": "vendor", "Модель": "model", "Цвет": r"^цвет$", "Диагональ": r"^диагональ(?: экрана)?$",
    "Встроенная память": r"^встроенная память", "Оперативная память": r"^оперативная память",
    "Аккумулятор": r"^емкость аккумулятора", "Процессор": r"^(?:модель процессора|процессор)$",
    "Накопитель": r"^(?:объём ssd|объем ssd|общий объем накопителя)",
    "Мощность": r"^(?:потребляемая |максимальная )?мощность$", "Ширина": r"^ширина$", "Высота": r"^высота$",
    "Объём": r"^(?:общий объем|объем|общий полезный объем)$",
}


def setup_style() -> None:
    """Спокойный стиль: почти чёрный + красный акцент М.Видео + серые; тонкие оси, без лишних рамок."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 170, "figure.facecolor": "white", "axes.facecolor": "white",
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 10, "axes.labelsize": 10, "axes.labelcolor": MUTED, "axes.edgecolor": LIGHT,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "grid.color": GRID, "grid.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False, "legend.fontsize": 9,
        "text.color": INK, "axes.prop_cycle": mpl.cycler(color=[INK, ACCENT, GREY, LIGHT]),
    })


def pct(x: float, digits: int = 0) -> str:
    return f"{x * 100:.{digits}f}%".replace(".", ",")


def num(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def _barh(ax, labels: list[str], values: list[float], colors=INK, fmt=pct, max_x: float | None = None) -> None:
    """Горизонтальные столбики с подписью значения у конца; самый большой — сверху."""
    y = list(range(len(labels)))[::-1]
    ax.barh(y, values, color=colors, height=0.62)
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0)
    hi = max_x or (max(values) if values else 1) or 1
    ax.set_xlim(0, hi * 1.18)
    for yi, v in zip(y, values, strict=True):
        ax.text(v + hi * 0.015, yi, fmt(v), va="center", fontsize=8.5, color=INK)
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_visible(False)


# --------------------------------------------------------------------------- контекст

@dataclass
class EDA:
    """Пути и подключение DuckDB; кэш промежуточных результатов между разделами."""

    processed: Path
    raw: Path | None = None
    img: Path | None = None
    traffic_path: Path | None = None
    stats: dict = field(default_factory=dict)
    cache: dict = field(default_factory=dict)
    _con: object = None

    @property
    def con(self):
        """Своё подключение DuckDB: без SET enable_progress_bar — в Jupyter без ipywidgets он падает
        (по умолчанию прогресс-бар в ядре и так выключен)."""
        if self._con is None:
            import duckdb

            self._con = duckdb.connect()
            self._con.execute("SET threads=3; SET memory_limit='4GB'; SET preserve_insertion_order=false")
        return self._con

    def p(self, name: str) -> str:
        return f"read_parquet('{self.processed / (name + '.parquet')}')"

    def df(self, sql: str) -> pd.DataFrame:
        return self.con.execute(sql).df()

    def one(self, sql: str):
        return self.con.execute(sql).fetchone()

    def save(self, fig, name: str) -> None:
        if self.img is not None:
            self.img.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.img / f"eda_{name}.png", bbox_inches="tight", facecolor="white")

    def traffic(self) -> dict:
        if "traffic" not in self.cache:
            path = self.traffic_path or self.processed.parent.parent / "submission" / "traffic.json"
            self.cache["traffic"] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return self.cache["traffic"]

    def taxonomy(self) -> dict:
        path = self.processed / "taxonomy_candidates.json"
        return {c["id"]: c for c in json.loads(path.read_text(encoding="utf-8"))["categories"]}


# --------------------------------------------------------------------------- (a) источники и связки

def sources(e: EDA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Таблица источников и доли связок по sku_id."""
    n_cards, n_offer, n_seo = e.one(f"""SELECT count(*), count(*) FILTER (WHERE has_offer),
        count(*) FILTER (WHERE category_source = 'clicks_seo') FROM {e.p('cards')}""")
    n_offers, n_vendor_null = e.one(f"SELECT count(*), count(*) FILTER (WHERE vendor IS NULL) FROM {e.p('offers')}")
    n_cats, n_leaf, n_leaf_off = e.one(f"""SELECT count(*), count(*) FILTER (WHERE is_leaf),
        count(*) FILTER (WHERE is_leaf AND n_offers > 0) FROM {e.p('categories')}""")
    n_params, n_names, n_service = e.one(f"""SELECT count(*), count(DISTINCT name), count(*) FILTER (WHERE is_service)
        FROM {e.p('params')}""")
    clicks, skus, sku_off, clicks_off = e.one(f"""SELECT sum(clicks), count(*), count(*) FILTER (WHERE category_source='offer'),
        sum(clicks) FILTER (WHERE category_source = 'offer') FROM {e.p('clicks_by_sku')}""")
    sku_card, clicks_card = e.one(f"""SELECT count(*), sum(s.clicks) FROM {e.p('clicks_by_sku')} s
        JOIN {e.p('cards')} c USING (sku_id)""")
    n_queries = e.one(f"SELECT count(DISTINCT query) FROM {e.p('queries_by_category')}")[0] \
        if (e.processed / "queries_by_category.parquet").exists() else None
    src = pd.DataFrame([
        ("sku_desc.parquet → cards", n_cards, "sku_id", "заголовок + описание (HTML → текст, блок «Характеристики» → seller_attrs)"),
        ("skus.pkl (YML) → offers", n_offers, "sku_id = @id", "vendor, model, name, категория, продавец — эталон"),
        ("skus.pkl (YML) → params", n_params, "sku_id", f"{num(n_names)} имён параметров, {pct(n_service / n_params)} служебных"),
        ("skus.pkl (YML) → categories", n_cats, "id", f"дерево 3 уровня, {num(n_leaf)} листьев, {num(n_leaf_off)} с офферами"),
        ("query_clicks.parquet", int(clicks), "sku_id", f"клики поиска: {num(skus)} SKU"
         + (f", {num(n_queries)} уникальных запросов" if n_queries else "")),
    ], columns=["источник", "строк", "ключ", "что внутри"])
    joins = pd.DataFrame([
        ("карточки ∩ офферы (= есть эталон)", n_offer, n_offer / n_cards, "доля карточек"),
        ("офферы ∩ карточки", n_offer, n_offer / n_offers, "доля офферов"),
        ("карточки без оффера, категория по seo_id кликов", n_seo, n_seo / n_cards, "доля карточек"),
        ("карточки без оффера и без кликов (только текст)", n_cards - n_offer - n_seo,
         (n_cards - n_offer - n_seo) / n_cards, "доля карточек"),
        ("кликнутые SKU ∩ офферы", sku_off, clicks_off / clicks, "доля кликов"),
        ("кликнутые SKU ∩ карточки", sku_card, clicks_card / clicks, "доля кликов"),
    ], columns=["связка", "SKU", "доля", "от чего доля"])
    e.stats["sources"] = {
        "cards": int(n_cards), "cards_with_offer": int(n_offer), "cards_category_by_clicks": int(n_seo),
        "cards_text_only": int(n_cards - n_offer - n_seo), "offers": int(n_offers), "offers_vendor_null": int(n_vendor_null),
        "categories": int(n_cats), "leaf_categories": int(n_leaf), "leaf_with_offers": int(n_leaf_off),
        "param_rows": int(n_params), "param_names": int(n_names), "param_service_share": round(n_service / n_params, 4),
        "clicks": int(clicks), "clicked_skus": int(skus), "clicked_skus_with_offer": int(sku_off),
        "clicks_with_offer_share": round(clicks_off / clicks, 4), "clicked_skus_with_card": int(sku_card),
        "clicks_with_card_share": round(clicks_card / clicks, 4), "distinct_queries": n_queries,
    }
    return src, joins


def fig_coverage(e: EDA):
    """Карточки и клики: где есть эталон (оффер YML), где только категория по кликам, где только текст."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    s = e.stats["sources"]
    rows = [
        ("Карточки товаров\n1,18 млн", [(s["cards_with_offer"] / s["cards"], ACCENT),
                                        (s["cards_category_by_clicks"] / s["cards"], INK),
                                        (s["cards_text_only"] / s["cards"], LIGHT)]),
        ("Клики поиска\n31 млн", [(s["clicks_with_offer_share"], ACCENT), (1 - s["clicks_with_offer_share"], LIGHT)]),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 2.6))
    for i, (label, parts) in enumerate(rows):
        y, left = len(rows) - 1 - i, 0.0
        for w, color in parts:
            ax.barh(y, w, left=left, color=color, height=0.55, edgecolor="white", linewidth=2)
            if w >= 0.045:
                ax.text(left + w / 2, y, pct(w, 1), ha="center", va="center", fontsize=9,
                        color="white" if color in (ACCENT, INK) else INK)
            left += w
        ax.text(-0.012, y, label, ha="right", va="center", fontsize=9.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(rows) - 0.45)
    ax.axis("off")
    ax.legend(handles=[Patch(color=ACCENT, label="есть оффер в YML-каталоге (эталонные атрибуты)"),
                       Patch(color=INK, label="нет оффера, категория по seo_id кликов"),
                       Patch(color=LIGHT, label="нет оффера (только текст)")],
              loc="upper center", bbox_to_anchor=(0.5, 0.02), ncols=3, fontsize=8.3, handlelength=1.2)
    ax.set_title("Эталон есть у каждой пятой карточки, но на эти товары приходится 93% кликов поиска", pad=8)
    e.save(fig, "coverage")
    return fig


# --------------------------------------------------------------------------- (b) поля, пропуски, качество текста

def fields(e: EDA) -> pd.DataFrame:
    """Поля трёх основных таблиц: тип, заполненность, пустые значения."""
    out = []
    for table, cols in (("cards", ["sku_id", "title", "description_clean", "seller_attrs", "n_seller_attrs",
                                   "category_id", "category_source", "seller", "seller_type", "has_offer", "split",
                                   "description_len"]),
                        ("offers", ["sku_id", "vendor", "model", "name", "category_id", "seller", "n_params"]),
                        ("params", ["sku_id", "name", "is_search", "unit", "value", "is_service"])):
        types = {r[0]: r[1] for r in e.con.execute(f"DESCRIBE SELECT * FROM {e.p(table)}").fetchall()}
        sel = ", ".join(
            f"avg(({c} IS NOT NULL)::INT) AS \"{c}__nn\", "
            + (f"avg((length({c}) = 0 OR {c} = '{{}}')::INT) AS \"{c}__empty\""
               if types[c] == "VARCHAR" and table != "params" else f"0.0 AS \"{c}__empty\"")
            for c in cols)
        row = e.con.execute(f"SELECT {sel} FROM {e.p(table)}").fetchone()
        for i, c in enumerate(cols):
            out.append((table, c, types[c], row[2 * i], row[2 * i + 1] or 0.0))
    df = pd.DataFrame(out, columns=["таблица", "поле", "тип", "заполнено", "пусто ('' или {})"])
    e.stats["fields"] = {f"{t}.{c}": round(float(nn), 4) for t, c, _, nn, _ in out}
    return df


def quality(e: EDA) -> pd.DataFrame:
    """Качество текста по сегментам продавца: короткие/пустые поля, заглушки, HTML, блок «Характеристики»."""
    df = e.df(f"""
        SELECT {SEG_SQL} AS segment, count(*) AS cards,
          avg((length(title) < 15)::INT) AS short_title,
          avg(regexp_matches(title, ' (UNIMALL|GSS) -$')::INT) AS placeholder_title,
          avg((description_len = 0)::INT) AS empty_desc,
          avg((description_len < 50)::INT) AS desc_lt_50,
          avg((description_len < 200)::INT) AS desc_lt_200,
          avg((n_seller_attrs > 0)::INT) AS char_block,
          avg(n_seller_attrs) FILTER (WHERE n_seller_attrs > 0) AS seller_attrs_if_block,
          avg((category_id IS NULL)::INT) AS no_category
        FROM {e.p('cards')} GROUP BY 1""").set_index("segment").reindex(list(SEGMENTS))
    if e.raw is not None and (e.raw / "sku_desc.parquet").exists():
        html = e.df(f"""
            WITH s AS (SELECT sku_id::BIGINT AS sku_id, {SEG_SQL} AS segment FROM {e.p('cards')})
            SELECT segment, avg(regexp_matches(r.description, '<\\s*/?\\s*[a-zA-Z][a-zA-Z0-9]*[^<>]*>')::INT) AS html,
                   avg(regexp_matches(r.description, '&[a-zA-Z]+;|&#[0-9]+;')::INT) AS entities
            FROM read_parquet('{e.raw / 'sku_desc.parquet'}') r JOIN s USING (sku_id)
            WHERE r.sku_id % 5 = 0 GROUP BY 1""").set_index("segment")
        df = df.join(html)
    total = e.df(f"""SELECT count(*) AS cards, avg((length(title) < 15)::INT) AS short_title,
          avg(regexp_matches(title, ' (UNIMALL|GSS) -$')::INT) AS placeholder_title,
          avg((description_len < 50)::INT) AS desc_lt_50, avg((description_len < 200)::INT) AS desc_lt_200,
          avg((n_seller_attrs > 0)::INT) AS char_block FROM {e.p('cards')}""").iloc[0].to_dict()
    e.stats["quality"] = {"by_segment": json.loads(df.round(4).to_json(orient="index", force_ascii=False)),
                          "total": {k: round(float(v), 4) for k, v in total.items()}}
    return df


def fig_quality(e: EDA, df: pd.DataFrame):
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = [("desc_lt_200", "описание короче 200 символов"), ("html", "HTML-разметка в исходном описании"),
               ("char_block", "есть блок «Характеристики» продавца"),
               ("placeholder_title", "заголовок-заглушка «Чехол … GSS -»"), ("no_category", "нет категории")]
    metrics = [(k, t) for k, t in metrics if k in df.columns]
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    h = 0.24
    y = np.arange(len(metrics))[::-1]
    for j, seg in enumerate(SEGMENTS):
        vals = [float(df.loc[seg, k]) for k, _ in metrics]
        ax.barh(y + (1 - j) * h, vals, height=h * 0.92, color=SEG_COLOR[seg], label=seg)
        for yi, v in zip(y + (1 - j) * h, vals, strict=True):
            ax.text(v + 0.008, yi, pct(v), va="center", fontsize=7.8, color=MUTED)
    ax.set_yticks(y, [t for _, t in metrics])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="lower right", ncols=3, handlelength=1, bbox_to_anchor=(1, -0.12))
    ax.set_title("Тексты карточек: эталонные (с оффером) чище, «хвост» без оффера — короткий и шаблонный")
    e.save(fig, "quality")
    return fig


# --------------------------------------------------------------------------- (c) длины текстов

def lengths(e: EDA) -> pd.DataFrame:
    df = e.df(f"""
        SELECT {SEG_SQL} AS segment,
          quantile_cont(length(title), [0.1, 0.5, 0.9]) AS title_q,
          quantile_cont(description_len, [0.1, 0.25, 0.5, 0.75, 0.9]) AS desc_q
        FROM {e.p('cards')} GROUP BY 1""").set_index("segment").reindex(list(SEGMENTS))
    out = pd.DataFrame({
        "заголовок p10": df["title_q"].map(lambda q: q[0]), "заголовок медиана": df["title_q"].map(lambda q: q[1]),
        "заголовок p90": df["title_q"].map(lambda q: q[2]), "описание p10": df["desc_q"].map(lambda q: q[0]),
        "описание p25": df["desc_q"].map(lambda q: q[1]), "описание медиана": df["desc_q"].map(lambda q: q[2]),
        "описание p75": df["desc_q"].map(lambda q: q[3]), "описание p90": df["desc_q"].map(lambda q: q[4]),
    })
    e.stats["lengths"] = json.loads(out.round(1).to_json(orient="index", force_ascii=False))
    return out


def fig_lengths(e: EDA):
    import matplotlib.pyplot as plt

    t = e.df(f"""SELECT {SEG_SQL} AS segment, least(length(title), 200) // 5 * 5 AS b, count(*) AS n
                 FROM {e.p('cards')} GROUP BY 1, 2""")
    d = e.df(f"""SELECT {SEG_SQL} AS segment, floor(log10(description_len + 1) * 12) / 12 AS b, count(*) AS n
                 FROM {e.p('cards')} GROUP BY 1, 2""")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, data, title in ((axes[0], t, "Длина заголовка, символов"),
                            (axes[1], d, "Длина описания, символов (лог. шкала)")):
        for seg in SEGMENTS:
            g = data[data.segment == seg].sort_values("b")
            share = g.n / g.n.sum()
            x = g.b if ax is axes[0] else 10 ** g.b
            ax.step(x, share, where="post", color=SEG_COLOR[seg], lw=1.8, label=seg)
        ax.set_title(title, fontsize=10.5)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: pct(v)))
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    axes[1].set_xscale("log")
    axes[1].set_xlim(10, 30000)
    axes[1].set_xticks([10, 100, 1000, 10000], ["10", "100", "1 000", "10 000"])
    axes[0].set_ylabel("доля карточек сегмента")
    axes[0].legend(loc="upper right")
    fig.suptitle("Без оффера описания вдвое короче; у эталонных карточек медиана 0,9–1,1 тыс. символов",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    e.save(fig, "lengths")
    return fig


# --------------------------------------------------------------------------- (d) категории: длинный хвост

def categories(e: EDA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Листовые категории: офферы, карточки с эталоном, клики; полосы head/torso/tail по доле офферов."""
    leaf = e.df(f"""
        SELECT k.id, k.title, k.depth, k.n_offers, coalesce(c.clicks, 0) AS clicks,
               coalesce(g.gold_cards, 0) AS gold_cards
        FROM {e.p('categories')} k
        LEFT JOIN {e.p('clicks_by_category')} c ON c.category_id = k.id
        LEFT JOIN (SELECT category_id, count(*) AS gold_cards FROM {e.p('cards')} WHERE has_offer GROUP BY 1) g
               ON g.category_id = k.id
        WHERE k.is_leaf AND k.n_offers > 0 ORDER BY k.n_offers DESC""")
    leaf["cum_offers"] = leaf.n_offers.cumsum() / leaf.n_offers.sum()
    leaf["band"] = pd.cut(leaf.cum_offers, [0, 0.5, 0.8, 1.0001], labels=["head", "torso", "tail"])
    bands = leaf.groupby("band", observed=True).agg(
        categories=("id", "size"), offers=("n_offers", "sum"), clicks=("clicks", "sum"), gold_cards=("gold_cards", "sum"),
        median_offers=("n_offers", "median"))
    for c in ("offers", "clicks", "gold_cards"):
        bands[c + "_share"] = bands[c] / bands[c].sum()
    depth = e.df(f"""SELECT depth, count(*) AS categories, count(*) FILTER (WHERE is_leaf) AS leaves,
                     sum(n_offers) AS offers FROM {e.p('categories')} GROUP BY 1 ORDER BY 1""")
    clicks_all = e.df(f"SELECT clicks FROM {e.p('clicks_by_category')} ORDER BY clicks DESC").clicks
    cum_clicks = clicks_all.cumsum() / clicks_all.sum()
    e.stats["categories"] = {
        "leaf_with_offers": len(leaf), "median_offers_per_leaf": float(leaf.n_offers.median()),
        "leaf_lt_10_offers": int((leaf.n_offers < 10).sum()), "leaf_lt_100_offers": int((leaf.n_offers < 100).sum()),
        "top_for_50_offers": int((leaf.cum_offers < 0.5).sum() + 1), "top_for_80_offers": int((leaf.cum_offers < 0.8).sum() + 1),
        "categories_with_clicks": len(clicks_all), "top_for_50_clicks": int((cum_clicks < 0.5).sum() + 1),
        "top_for_80_clicks": int((cum_clicks < 0.8).sum() + 1),
        "bands": json.loads(bands.round(4).to_json(orient="index")),
        "depth": json.loads(depth.to_json(orient="records")),
    }
    e.cache["leaf"] = leaf
    return bands.reset_index(), depth


def fig_long_tail(e: EDA):
    import matplotlib.pyplot as plt
    import numpy as np

    leaf = e.cache["leaf"]
    clicks = e.df(f"SELECT clicks FROM {e.p('clicks_by_category')} ORDER BY clicks DESC").clicks
    fig, ax = plt.subplots(figsize=(9.2, 4))
    for series, color, label in ((leaf.n_offers, INK, "офферы каталога"), (clicks, ACCENT, "клики поиска")):
        y = series.cumsum().to_numpy() / series.sum()
        ax.plot(np.arange(1, len(y) + 1), y, color=color, lw=2, label=label)
    s = e.stats["categories"]
    for k, frac in ((s["top_for_50_offers"], 0.5), (s["top_for_80_offers"], 0.8)):
        ax.plot([k], [frac], "o", ms=6, color=INK, mec="white", mew=1.5)
        ax.annotate(f"{k} категорий — {pct(frac)} офферов", (k, frac), xytext=(8, -14), textcoords="offset points",
                    fontsize=8.5, color=INK)
    k = s["top_for_80_clicks"]
    ax.plot([k], [0.8], "o", ms=6, color=ACCENT, mec="white", mew=1.5)
    ax.annotate(f"{k} категорий — 80% кликов", (k, 0.8), xytext=(-12, 6), textcoords="offset points",
                ha="right", fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1 000"])
    ax.set_xlabel("категории по убыванию (лог. шкала)")
    ax.set_ylabel("накопленная доля")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: pct(v)))
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    ax.set_title(f"Длинный хвост: {num(s['leaf_with_offers'])} листовых категорий, у {num(s['leaf_lt_10_offers'])} "
                 f"меньше 10 офферов")
    e.save(fig, "long_tail")
    return fig


# --------------------------------------------------------------------------- (e) структурированные атрибуты

def params_overview(e: EDA) -> pd.DataFrame:
    by_seller = e.df(f"""
        SELECT CASE WHEN seller = 'М.Видео' THEN 'М.Видео' ELSE 'маркетплейс' END AS segment, count(*) AS offers,
               quantile_cont(n_params, [0.1, 0.25, 0.5, 0.75, 0.9]) AS q, avg((n_params < 5)::INT) AS lt5,
               avg((n_params >= 20)::INT) AS ge20
        FROM {e.p('offers')} GROUP BY 1""").set_index("segment")
    pairs = e.one(f"""
        WITH x AS (SELECT sku_id, name, bool_or(is_search) s, bool_or(NOT is_search) b FROM {e.p('params')}
                   WHERE NOT is_service GROUP BY 1, 2)
        SELECT count(*), avg((s AND b)::INT), avg((s AND NOT b)::INT), avg((b AND NOT s)::INT) FROM x""")
    names = e.one(f"""SELECT median(n) FROM (SELECT category_id, count(*) AS n FROM {e.p('param_stats')} GROUP BY 1)""")[0]
    out = pd.DataFrame({
        "офферов": by_seller.offers, "медиана характеристик": by_seller.q.map(lambda q: q[2]),
        "p10": by_seller.q.map(lambda q: q[0]), "p90": by_seller.q.map(lambda q: q[4]),
        "< 5 характеристик": by_seller.lt5, "≥ 20 характеристик": by_seller.ge20})
    e.stats["params"] = {
        "by_seller": json.loads(out.round(4).to_json(orient="index", force_ascii=False)),
        "sku_param_pairs": int(pairs[0]), "pairs_bucket_and_exact": round(pairs[1], 4),
        "pairs_exact_only": round(pairs[2], 4), "pairs_bucket_only": round(pairs[3], 4),
        "median_param_names_per_category": float(names),
    }
    return out


def fill_matrix(e: EDA, top: int = 14) -> pd.DataFrame:
    """Заполненность ключевых свойств в топ-категориях по кликам (из param_stats + vendor/model офферов)."""
    tax = e.taxonomy()
    ids = list(dict.fromkeys(list(MAIN) + list(tax)))[:top]
    ps = e.df(f"""SELECT category_id, lower(name) AS name, fill_rate FROM {e.p('param_stats')}
                  WHERE category_id IN ({','.join(f"'{c}'" for c in ids)})""")
    vm = e.df(f"""SELECT category_id, avg((vendor IS NOT NULL)::INT) AS vendor, avg((model IS NOT NULL)::INT) AS model
                  FROM {e.p('offers')} WHERE category_id IN ({','.join(f"'{c}'" for c in ids)}) GROUP BY 1""") \
        .set_index("category_id")
    rows = {}
    for cid in ids:
        g = ps[ps.category_id == cid]
        row = {}
        for col, rx in FILL_ATTRS.items():
            if rx in ("vendor", "model"):
                row[col] = float(vm.loc[cid, rx]) if cid in vm.index else None
            else:
                m = g[g.name.str.contains(rx, regex=True)]
                row[col] = float(m.fill_rate.max()) if len(m) and m.fill_rate.max() >= 0.02 else None
        rows[f"{tax[cid]['title'] if cid in tax else MAIN.get(cid, cid)}"] = row
    df = pd.DataFrame(rows).T
    df = df.loc[:, df.notna().sum() >= 2]
    e.stats["fill_matrix"] = json.loads(df.round(3).to_json(orient="index", force_ascii=False))
    return df


def fig_fill(e: EDA, df: pd.DataFrame):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("ink", ["#FFFFFF", "#E9E9E9", "#B5B5B5"])
    fig, ax = plt.subplots(figsize=(10.5, 0.36 * len(df) + 1.6))
    vals = df.to_numpy(dtype=float)
    ax.imshow(vals, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if v != v:  # NaN: свойства нет в категории
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color=LIGHT)
                continue
            ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=8.2,
                    color=ACCENT if v < 0.5 else INK, fontweight="bold" if v < 0.5 else "normal")
    ax.set_xticks(range(df.shape[1]), df.columns, rotation=30, ha="right")
    ax.set_yticks(range(df.shape[0]), df.index)
    ax.tick_params(length=0)
    ax.set_xticks([x - 0.5 for x in range(1, df.shape[1])], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, df.shape[0])], minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Заполненность ключевых свойств в каталоге, % офферов (красным — меньше половины)")
    e.save(fig, "fill")
    return fig


def _param_values(e: EDA, cids: list[str], name: str, exact: bool | None = True, top: int = 8) -> pd.DataFrame:
    ids = ",".join(f"'{c}'" for c in cids)
    cond = "" if exact is None else f"AND p.is_search = {str(exact).upper()}"
    return e.df(f"""
        WITH o AS (SELECT sku_id FROM {e.p('offers')} WHERE category_id IN ({ids}))
        SELECT p.value, any_value(p.unit) AS unit, count(DISTINCT p.sku_id) AS n
        FROM {e.p('params')} p JOIN o USING (sku_id)
        WHERE p.name = '{name}' AND p.value IS NOT NULL {cond}
        GROUP BY 1 ORDER BY n DESC LIMIT {top}""")


def values(e: EDA) -> dict[str, pd.DataFrame]:
    """Распределения значений ключевых атрибутов главных категорий (доли офферов)."""
    out = {
        "Смартфоны: встроенная память": _param_values(e, ["761", "914"], "Встроенная память", exact=False, top=6),
        "Смартфоны: цвет (фильтр)": _param_values(e, ["761", "914"], "Цвет", exact=False, top=8),
        "Смартфоны: ёмкость аккумулятора, мАч": _param_values(e, ["761"], "Емкость аккумулятора", top=7),
        "Ноутбуки: оперативная память, ГБ": _param_values(e, ["987"], "Оперативная память (RAM)", top=6),
        "Ноутбуки: диагональ": _param_values(e, ["987"], "Диагональ экрана", top=7),
        "Телевизоры: диагональ, дюймы": _param_values(e, ["1682"], "Диагональ", top=8),
    }
    brands = e.df(f"""SELECT vendor AS value, NULL AS unit, count(*) AS n FROM {e.p('offers')}
                      WHERE category_id IN ('761', '914') GROUP BY 1 ORDER BY n DESC LIMIT 8""")
    out["Смартфоны: бренд"] = brands
    totals = dict(e.con.execute(f"""SELECT category_id, count(*) FROM {e.p('offers')}
                                    WHERE category_id IN ('761','914','987','1682') GROUP BY 1""").fetchall())
    for k, df in out.items():
        base = totals["761"] + totals["914"] if k.startswith("Смартфоны") and "аккум" not in k else \
            totals["761"] if "аккум" in k else totals["987"] if k.startswith("Ноутбуки") else totals["1682"]
        df["share"] = df.n / base
    e.stats["values"] = {k: [[str(v), round(float(s), 4)] for v, s in zip(df.value, df.share, strict=True)]
                         for k, df in out.items()}
    return out


def fig_values(e: EDA, vals: dict[str, pd.DataFrame]):
    import matplotlib.pyplot as plt

    keys = ["Смартфоны: встроенная память", "Смартфоны: бренд", "Смартфоны: цвет (фильтр)",
            "Ноутбуки: оперативная память, ГБ", "Ноутбуки: диагональ", "Телевизоры: диагональ, дюймы"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    for ax, k in zip(axes.flat, keys, strict=True):
        df = vals[k]
        labels = [re.sub(r"\(.*\)", "", str(v)).replace("\"", "″") + ("″" if k.endswith("дюймы") else "")
                  for v in df.value]
        _barh(ax, labels, df.share.tolist(), colors=INK)
        ax.set_title(k, fontsize=9.5)
    fig.suptitle("Значения ключевых атрибутов: несколько частых значений покрывают большую часть офферов",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    e.save(fig, "values")
    return fig


# --------------------------------------------------------------------------- нормализация: реальные примеры

_MEM_UNIT = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?\s?(ГБ|Гб|гб|GB|Gb|gb|G|Г|Тб|ТБ|TB|Tb|тб)(?![A-Za-zА-Яа-яЁё])")
_PAIR_SLASH = re.compile(r"(?<![\w.+])\d{1,2}\s?/\s?\d{2,4}(?!\d)")
_PAIR_PLUS = re.compile(r"(?<![\w.+])\d{1,2}\s?\+\s?\d{2,4}(?!\d)")
_PAIR_UNITS = re.compile(r"\d{1,2}\s?(?:GB|ГБ|Гб|gb)\s?[/+]\s?\d{2,4}")
_DIAG = {"«55\"»": re.compile(r'\d+(?:[.,]\d+)?\s?(?:"|”|″|\'\')'),
         "«55 дюймов»": re.compile(r"\d+(?:[.,]\d+)?\s?(?:дюйм|inch)", re.I),
         "«139 см»": re.compile(r"\d+(?:[.,]\d+)?\s?см\b", re.I)}
_TV_CODE = re.compile(r"\b[A-Za-z]{0,4}(24|32|40|43|50|55|58|65|70|75|85|98|100)[A-Za-z]{1,4}\d", re.I)
_COLOR_EN = re.compile(r"\b(black|white|blue|green|gr[ae]y|silver|gold|pink|purple|titanium|midnight|starlight|"
                       r"orange|cosmic|deep|mist|sage|lavender|mint)\b", re.I)
_COLOR_RU = re.compile(r"(черн|бел[ыао]|син[иея]|зелен|сер[ыао]|серебр|золот|розов|фиолет|титан|голуб|оранж)", re.I)


def normalization(e: EDA) -> dict[str, pd.DataFrame]:
    """Почему без нормализации нельзя: написания единиц, форматы памяти и диагонали, цвета, бакеты фильтров."""
    titles = e.df(f"SELECT title, category_id FROM {e.p('cards')}")
    units = Counter(m for t in titles.title for m in _MEM_UNIT.findall(t))
    unit_df = pd.DataFrame(units.most_common(), columns=["написание", "упоминаний в заголовках"])
    ph = titles[titles.category_id.isin(["761", "914", "930"])].title
    pair_df = pd.DataFrame([
        ("«8/256GB»", ph.map(lambda t: bool(_PAIR_SLASH.search(t))).mean()),
        ("«12+256»", ph.map(lambda t: bool(_PAIR_PLUS.search(t))).mean()),
        ("«8GB/256GB», «8GB+256GB»", ph.map(lambda t: bool(_PAIR_UNITS.search(t))).mean()),
    ], columns=["формат RAM/ROM в заголовке (смартфоны, планшеты)", "доля заголовков"])
    rows = []
    for cid, title in (("987", "Ноутбуки"), ("1682", "Телевизоры"), ("761", "Смартфоны")):
        t = titles[titles.category_id == cid].title
        row = {"категория": title, "заголовков": len(t)}
        row.update({k: t.map(lambda s, rx=rx: bool(rx.search(s))).mean() for k, rx in _DIAG.items()})
        row["код модели с диагональю («55C6K», «UE55DU7100»)"] = t.map(lambda s: bool(_TV_CODE.search(s))).mean() \
            if cid == "1682" else None
        rows.append(row)
    diag_df = pd.DataFrame(rows)
    phones = titles[titles.category_id.isin(["761", "914"])].title
    color_df = pd.DataFrame([
        ("английское название цвета (Black, Titanium Gray, Starlight)", phones.map(lambda t: bool(_COLOR_EN.search(t))).mean()),
        ("русское название цвета", phones.map(lambda t: bool(_COLOR_RU.search(t))).mean()),
    ], columns=["цвет в заголовке смартфона", "доля"])
    syn = e.df(f"""
        WITH o AS (SELECT sku_id FROM {e.p('offers')} WHERE category_id IN ('761', '914')),
             c AS (SELECT p.sku_id, max(p.value) FILTER (WHERE p.name = 'Цвет' AND p.is_search) AS ru,
                          max(p.value) FILTER (WHERE p.name = 'Цвет производителя') AS vendor_color
                   FROM {e.p('params')} p JOIN o USING (sku_id) WHERE p.name IN ('Цвет', 'Цвет производителя')
                   GROUP BY 1)
        SELECT vendor_color AS "цвет производителя", ru AS "цвет в фильтре", count(*) AS "офферов" FROM c
        WHERE ru IS NOT NULL AND vendor_color IS NOT NULL AND lower(vendor_color) <> lower(ru)
        GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 12""")
    buckets = []
    for cid, bucket, exact in (("761", "Встроенная память", "Встроенная память (ROM)"),
                               ("761", "Диагональ экрана", "Диагональ"), ("1682", "Диагональ", "Диагональ экрана"),
                               ("666", "Высота холодильника", "Высота"), ("2352", "Ширина", "Ширина"),
                               ("2447", "Макс загрузка (хлопок)", "Максимальная загрузка")):
        same = bucket == exact
        r = e.df(f"""
            WITH o AS (SELECT sku_id FROM {e.p('offers')} WHERE category_id = '{cid}'),
                 b AS (SELECT sku_id, max(value) AS bv FROM {e.p('params')} JOIN o USING (sku_id)
                       WHERE name = '{bucket}' AND {'NOT is_search' if same else 'TRUE'} GROUP BY 1),
                 x AS (SELECT sku_id, max(value) AS xv, max(unit) AS u FROM {e.p('params')} JOIN o USING (sku_id)
                       WHERE name = '{exact}' AND {'is_search' if same else 'TRUE'} GROUP BY 1)
            SELECT bv, xv, u, count(*) AS n FROM b JOIN x USING (sku_id) WHERE bv <> xv
            GROUP BY 1, 2, 3 ORDER BY n DESC LIMIT 1""")
        if len(r):
            buckets.append((MAIN.get(cid, cid) if cid in MAIN else "Посудомоечные машины", bucket, r.bv[0], exact,
                            f"{r.xv[0]} {r.u[0] or ''}".strip()))
    bucket_df = pd.DataFrame(buckets, columns=["категория", "параметр-бакет (фильтр)", "значение",
                                               "точный параметр", "значение "])
    e.stats["normalization"] = {
        "memory_unit_spellings": units.most_common(), "memory_pairs": pair_df.values.tolist(),
        "diagonal_formats": json.loads(diag_df.round(4).to_json(orient="records", force_ascii=False)),
        "phone_title_colors": color_df.values.tolist(), "color_synonyms": syn.values.tolist(),
        "bucket_vs_exact": bucket_df.values.tolist(),
    }
    return {"units": unit_df, "pairs": pair_df, "diagonal": diag_df, "colors": color_df, "color_synonyms": syn,
            "buckets": bucket_df}


def fig_units(e: EDA, norm_tables: dict[str, pd.DataFrame]):
    import matplotlib.pyplot as plt

    df = norm_tables["units"]
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    x = range(len(df))
    ax.bar(x, df.iloc[:, 1], color=INK, width=0.62)
    ax.set_xticks(list(x), [f"«{u}»" for u in df.iloc[:, 0]])
    ax.tick_params(axis="x", length=0)
    for xi, v in zip(x, df.iloc[:, 1], strict=True):
        ax.text(xi, v, num(v), ha="center", va="bottom", fontsize=8, color=INK)
    ax.spines["left"].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_title(f"Объём памяти в заголовках: одна единица — {len(df)} написаний")
    e.save(fig, "units")
    return fig


# --------------------------------------------------------------------------- (f) извлекаемость эталона из текста

EXTRACT_ROWS = [("761", "vendor", "бренд"), ("761", "model", "модель"), ("761", "Встроенная память (ROM)", "память"),
                ("761", "Оперативная память (RAM)", "RAM"), ("761", "Цвет", "цвет"),
                ("761", "Емкость аккумулятора", "аккумулятор"), ("761", "Диагональ", "диагональ"),
                ("914", "Цвет", "цвет"), ("987", "Модель процессора", "процессор"),
                ("987", "Оперативная память (RAM)", "RAM"), ("987", "Объём SSD", "SSD"),
                ("987", "Диагональ экрана", "диагональ"), ("987", "Цвет", "цвет"),
                ("1682", "Диагональ", "диагональ"), ("2447", "Максимальная загрузка", "загрузка"),
                ("666", "Размораживание морозильной камеры", "No Frost"), ("31607", "Тип подключения", "подключение")]


def extractability(e: EDA) -> pd.DataFrame:
    """Доля эталонных значений, которые вообще есть в тексте карточки — потолок recall для любого метода
    (посчитано в `attrx prepare` → taxonomy_candidates.json, выборка до 1 500 офферов на категорию)."""
    tax = e.taxonomy()
    rows = []
    for cid, attr, label in EXTRACT_ROWS:
        c = tax.get(cid)
        if not c:
            continue
        s = c[attr] if attr in ("vendor", "model") else next((p for p in c["params"] if p["name"] == attr), None)
        if not s:
            continue
        rows.append({"категория": c["title"], "атрибут": label, "параметр": attr, "fill": s["fill_rate"],
                     "в тексте": s["extractable"], "в тексте (с голым числом)": s.get("extractable_loose"),
                     "в заголовке": s.get("extractable_title"), "проверено офферов": s["n_checked"]})
    df = pd.DataFrame(rows)
    # все заметные свойства топ-25 категорий (до 80 самых заполненных в каждой, fill ≥ 50%)
    allp = pd.DataFrame([(p["extractable"], {str(v).lower() for v, _ in p["top_values"]} <= {"да", "нет"})
                         for c in tax.values() for p in c["params"]
                         if p["fill_rate"] >= 0.5 and p["extractable"] is not None], columns=["ext", "flag"])
    e.stats["extractability"] = {
        "rows": json.loads(df.to_json(orient="records", force_ascii=False)),
        "top25_params_fill_ge_50": len(allp),
        "top25_params_median_extractable": float(allp.ext.median()),
        "top25_params_extractable_ge_50": int((allp.ext >= 0.5).sum()),
        "top25_params_extractable_ge_50_share": round(float((allp.ext >= 0.5).mean()), 4),
        "top25_params_yes_no_flag_share": round(float(allp.flag.mean()), 4),
        "top25_params_non_flag_median_extractable": float(allp[~allp.flag].ext.median()),
    }
    return df


def fig_extractability(e: EDA, df: pd.DataFrame):
    import matplotlib.pyplot as plt

    d = df.copy()
    d["label"] = d["категория"] + ": " + d["атрибут"]
    d = d.sort_values("в тексте")
    fig, ax = plt.subplots(figsize=(9.2, 0.33 * len(d) + 1.4))
    y = range(len(d))
    ax.barh(y, d["в тексте (с голым числом)"].fillna(d["в тексте"]), color=LIGHT, height=0.62,
            label="в тексте, включая голое число без единицы")
    ax.barh(y, d["в тексте"], color=INK, height=0.62, label="в тексте (значение с единицей / точно)")
    ax.plot(d["в заголовке"], list(y), "o", color=ACCENT, ms=6, mec="white", mew=1.2, label="уже в заголовке")
    for yi, v in zip(y, d["в тексте"], strict=True):
        ax.text(1.01, yi, pct(v), va="center", fontsize=8.5, color=INK)
    ax.set_yticks(list(y), d.label)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: pct(v)))
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.06), ncols=3, fontsize=8.2, handlelength=1.2)
    ax.set_title("Потолок recall: есть ли эталонное значение в тексте карточки (доля офферов)")
    e.save(fig, "extractability")
    return fig


# --------------------------------------------------------------------------- (g) поисковый спрос

def search(e: EDA) -> dict[str, pd.DataFrame]:
    """Клики по категориям, длина запросов, что называют в запросах (по submission/traffic.json)."""
    cache = e.processed / "queries_by_category.parquet"
    qlen = e.df(f"""SELECT least(len(string_split(query, ' ')), 6) AS words, sum(clicks) AS clicks
                    FROM read_parquet('{cache}') GROUP BY 1 ORDER BY 1""") if cache.exists() else pd.DataFrame()
    if len(qlen):
        qlen["share"] = qlen.clicks / qlen.clicks.sum()
    tr = e.traffic()
    meta = tr.get("_meta", {})
    kinds = []
    for key, title in (("smartphone", "Смартфоны (+ iPhone)"), ("987", "Ноутбуки"), ("1682", "Телевизоры"),
                       ("2447", "Стиральные машины"), ("666", "Холодильники"), ("31607", "Наушники")):
        c = tr.get(key)
        if not c or "kinds" not in c:
            continue
        k = c["kinds"]
        kinds.append({"категория": title, "клики": c["clicks"], "любой атрибут": c["any_attribute"],
                      "бренд": k["brand"], "модель": k["model"],
                      "характеристика": c.get("any_param", max(k["number"], k["color"], k["text"])),
                      "число с единицей / значение": k["number"], "цвет": k["color"], "текстовое значение": k["text"]})
    kinds_df = pd.DataFrame(kinds)
    tops = {}
    for key in ("761", "914", "987", "1682", "2447", "666", "31607"):
        c = tr.get(key)
        if c:
            tops[key] = pd.DataFrame([(it["attribute"], it["share"], ", ".join(it["params"][1:]),
                                       "; ".join(f"{x['query']} ({num(x['clicks'])})" for x in it["examples"][:2]))
                                      for it in c["top"]], columns=["атрибут", "доля кликов", "близнецы", "примеры"])
    e.stats["search"] = {
        "query_words": json.loads(qlen.round(4).to_json(orient="records")) if len(qlen) else [],
        "categories_scored": meta.get("n_categories"), "clicks_covered_share": meta.get("clicks_covered_share"),
        "any_attribute_share": meta.get("any_attribute_share"), "any_param_share": meta.get("any_param_share"),
        "kinds_share": meta.get("kinds_share"),
        "main": json.loads(kinds_df.round(4).to_json(orient="records", force_ascii=False)) if len(kinds_df) else [],
    }
    return {"query_words": qlen, "kinds": kinds_df, **{f"top_{k}": v for k, v in tops.items()}}


def fig_search_kinds(e: EDA, kinds_df: pd.DataFrame):
    import matplotlib.pyplot as plt
    import numpy as np

    series = [("бренд", INK), ("модель", GREY), ("характеристика", ACCENT)]
    d = kinds_df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    y = np.arange(len(d))
    h = 0.26
    for j, (col, color) in enumerate(series):
        yy = y + (1 - j) * h
        ax.barh(yy, d[col], height=h * 0.92, color=color, label=f"назван {col}" if col != "характеристика" else
                "названа характеристика (память, диагональ, тип, цвет…)")
        for yi, v in zip(yy, d[col], strict=True):
            ax.text(v + 0.008, yi, pct(v), va="center", fontsize=7.8, color=MUTED)
    ax.set_yticks(y, d["категория"])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.xaxis.set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncols=3, fontsize=8.2, handlelength=1.2)
    share = e.stats["search"].get("any_param_share")
    ax.set_title("Что покупатели называют в поисковом запросе, доля кликов категории"
                 + (f" (по всему поиску: характеристика — {pct(share)})" if share else ""))
    e.save(fig, "search_kinds")
    return fig


CASE_REQUIRED = {"761": {"vendor", "model", "Встроенная память (ROM)", "Встроенная память", "Диагональ",
                         "Диагональ экрана", "Емкость аккумулятора", "Цвет", "Цвет производителя"},
                 "914": {"vendor", "model", "Встроенная память (ROM)", "Встроенная память", "Диагональ",
                         "Емкость аккумулятора", "Цвет", "Цвет производителя"},
                 "987": {"vendor", "Модель процессора", "Процессор", "Тип процессора", "Оперативная память",
                         "Оперативная память (RAM)", "Объём SSD", "Общий объем накопителя", "Диагональ экрана", "Цвет"}}
_PRETTY = {"vendor": "бренд", "model": "модель"}


def fig_traffic_top(e: EDA, n: int = 7):
    import matplotlib.pyplot as plt

    tr = e.traffic()
    panels = [("761", "Смартфоны"), ("914", "iPhone"), ("987", "Ноутбуки"), ("1682", "Телевизоры"),
              ("2447", "Стиральные машины"), ("666", "Холодильники")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6))
    for ax, (cid, title) in zip(axes.flat, panels, strict=True):
        c = tr.get(cid)
        if not c:
            ax.axis("off")
            continue
        items = [it for it in c["top"] if it["attribute"] not in ("Серия", "Модель") and it["share"] >= 0.005][:n]
        labels = [_PRETTY.get(it["attribute"], it["attribute"])[:32] for it in items]
        req = CASE_REQUIRED.get(cid, set())
        colors = [ACCENT if (it["attribute"] in req or set(it["params"]) & req) else INK for it in items]
        _barh(ax, labels, [it["share"] for it in items], colors=colors)
        ax.set_title(f"{title}: {pct(c['any_attribute'])} кликов с атрибутом", fontsize=9.5)
    from matplotlib.patches import Patch

    fig.legend(handles=[Patch(color=ACCENT, label="обязательное поле кейса (смартфон, ноутбук)"),
                        Patch(color=INK, label="другие атрибуты")], loc="lower center", ncols=2, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Трафик-веса: доля поисковых кликов, где в запросе назван атрибут", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    e.save(fig, "traffic_top")
    return fig


# --------------------------------------------------------------------------- (h) справедливость: М.Видео vs маркетплейс

def _ids(xs) -> str:
    return ",".join("'" + str(x).replace("'", "''") + "'" for x in xs)


def key_presence(e: EDA) -> pd.DataFrame:
    """Офферы главных категорий: сегмент продавца, заполнены ли все ключевые свойства, клики поиска (один проход)."""
    if "key_presence" not in e.cache:
        names = sorted({p for ps in KEY_PARAMS.values() for p in ps})
        df = e.df(f"""
            WITH o AS (SELECT sku_id, category_id,
                              CASE WHEN seller = 'М.Видео' THEN 'М.Видео' ELSE 'маркетплейс' END AS segment
                       FROM {e.p('offers')} WHERE category_id IN ({_ids(KEY_PARAMS)})),
                 p AS (SELECT sku_id, list(DISTINCT name) AS names FROM {e.p('params')} JOIN o USING (sku_id)
                       WHERE name IN ({_ids(names)}) AND value IS NOT NULL GROUP BY 1)
            SELECT o.sku_id, o.category_id, o.segment, p.names, coalesce(c.clicks, 0) AS clicks
            FROM o LEFT JOIN p USING (sku_id) LEFT JOIN {e.p('clicks_by_sku')} c USING (sku_id)""")
        df["all_key"] = [hasattr(n, "tolist") and set(KEY_PARAMS[c]) <= set(n.tolist())  # NA — нет ни одного
                         for c, n in zip(df.category_id, df.names, strict=True)]
        e.cache["key_presence"] = df
    return e.cache["key_presence"]


def _extract_by_seller(e: EDA, per_group: int = 300) -> pd.DataFrame:
    """Есть ли эталонное значение в тексте — отдельно для карточек М.Видео и маркетплейса (главные категории,
    детерминированная выборка до per_group карточек на категорию и сегмент)."""
    from .taxo import _NON_WORD, card_text, norm, value_in_text

    cards = e.df(f"""
        SELECT c.sku_id, c.category_id, c.title, c.description_clean, c.seller_attrs, c.seller_type, o.vendor, o.model
        FROM {e.p('cards')} c JOIN {e.p('offers')} o USING (sku_id)
        WHERE c.category_id IN ({_ids(KEY_PARAMS)}) AND c.seller_type IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY c.category_id, c.seller_type ORDER BY hash(c.sku_id)) <= {int(per_group)}""")
    e.con.register("sampled", cards[["sku_id"]])
    names = sorted({p for ps in KEY_PARAMS.values() for p in ps})
    vals = e.df(f"""SELECT p.sku_id, p.name, p.unit, p.value, p.is_search FROM {e.p('params')} p JOIN sampled USING (sku_id)
                    WHERE p.name IN ({_ids(names)}) AND p.value IS NOT NULL""")
    e.con.unregister("sampled")
    vals = vals.sort_values("is_search").groupby(["sku_id", "name"]).last()  # точное значение, если есть
    rows = []
    for r in cards.itertuples():
        text = norm(card_text(*(x if isinstance(x, str) else "" for x in (r.title, r.description_clean,
                                                                          r.seller_attrs))))
        compact = _NON_WORD.sub("", text)
        checks = [("vendor", r.vendor, None), ("model", r.model, None)]
        checks += [(p, vals.loc[(r.sku_id, p), "value"], vals.loc[(r.sku_id, p), "unit"])
                   for p in KEY_PARAMS[r.category_id] if (r.sku_id, p) in vals.index]
        for attr, value, unit in checks:
            if isinstance(value, str) and value.strip():
                hit = value_in_text(value, text, compact, unit if isinstance(unit, str) else None)
                rows.append((r.category_id, r.seller_type, attr, hit == "exact", hit is not None))
    return pd.DataFrame(rows, columns=["category_id", "seller_type", "attribute", "exact", "loose"])


def fairness(e: EDA, per_group: int = 300) -> pd.DataFrame:
    """Сравнение сегментов продавца на карточках с эталоном: текст, заполненность фида, извлекаемость."""
    cards = e.df(f"""
        SELECT CASE WHEN seller_type = 'mvideo' THEN 'М.Видео' ELSE 'маркетплейс' END AS segment,
          median(description_len) AS desc_median, avg((description_len < 200)::INT) AS desc_lt_200,
          median(length(title)) AS title_median, avg((n_seller_attrs > 0)::INT) AS char_block
        FROM {e.p('cards')} WHERE has_offer GROUP BY 1""").set_index("segment")
    offers = e.df(f"""
        SELECT CASE WHEN seller = 'М.Видео' THEN 'М.Видео' ELSE 'маркетплейс' END AS segment,
          median(n_params) AS params_median, avg((n_params >= 20)::INT) AS params_ge_20
        FROM {e.p('offers')} GROUP BY 1""").set_index("segment")
    kp = key_presence(e)
    key = kp.groupby(["category_id", "segment"]).agg(offers=("sku_id", "size"), all_key=("all_key", "mean")).reset_index()
    # iPhone не в счёт: Apple не публикует ёмкость аккумулятора, поле пусто у 92% офферов при любом продавце
    key_seg = kp[kp.category_id != "914"].groupby("segment").all_key.mean()
    ext = _extract_by_seller(e, per_group)
    ext["segment"] = ext.seller_type.map({"mvideo": "М.Видео", "marketplace": "маркетплейс"})
    ext_seg = ext.groupby("segment")[["exact", "loose"]].mean()
    ext_attr = ext.assign(kind=ext.attribute.map(lambda a: a if a in ("vendor", "model") else "свойства")) \
        .groupby(["kind", "segment"]).exact.mean().unstack()
    out = pd.DataFrame({
        "описание короче 200 символов": cards.desc_lt_200, "есть блок «Характеристики»": cards.char_block,
        "в фиде ≥ 20 характеристик": offers.params_ge_20,
        "заполнены все ключевые свойства (6 главных категорий)": key_seg,
        "бренд есть в тексте": ext_attr.loc["vendor"], "модель есть в тексте": ext_attr.loc["model"],
        "значение ключевого свойства есть в тексте": ext_attr.loc["свойства"],
    }).T[["М.Видео", "маркетплейс"]]
    e.stats["fairness"] = {"table": json.loads(out.round(4).to_json(orient="index", force_ascii=False)),
                           "desc_median": cards.desc_median.to_dict(), "title_median": cards.title_median.to_dict(),
                           "params_median": offers.params_median.to_dict(),
                           "key_all_by_category": json.loads(key.round(4).to_json(orient="records", force_ascii=False)),
                           "extract_overall": json.loads(ext_seg.round(4).to_json(orient="index", force_ascii=False)),
                           "extract_sample_per_group": per_group, "extract_checks": len(ext)}
    e.cache["extract_by_seller"] = ext
    return out


def fig_fairness(e: EDA, df: pd.DataFrame):
    import matplotlib.pyplot as plt

    d = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.2, 0.5 * len(d) + 1.4))
    y = range(len(d))
    for yi, (a, b) in zip(y, zip(d["М.Видео"], d["маркетплейс"], strict=True), strict=True):
        ax.plot([a, b], [yi, yi], color=LIGHT, lw=2.2, zorder=1)
    ax.scatter(d["маркетплейс"], list(y), s=58, color=INK, zorder=2, label="маркетплейс", edgecolor="white", lw=1.5)
    ax.scatter(d["М.Видео"], list(y), s=58, color=ACCENT, zorder=3, label="М.Видео", edgecolor="white", lw=1.5)
    for yi, (a, b) in zip(y, zip(d["М.Видео"], d["маркетплейс"], strict=True), strict=True):
        lo, hi = sorted((a, b))
        ax.text(lo - 0.015, yi, pct(lo), ha="right", va="center", fontsize=8, color=MUTED)
        ax.text(hi + 0.015, yi, pct(hi), ha="left", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(list(y), d.index)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.08, 1.1)
    ax.xaxis.set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncols=2)
    ax.set_title("Справедливость: карточки М.Видео и маркетплейса (только с эталоном)", pad=26)
    e.save(fig, "fairness")
    return fig


# --------------------------------------------------------------------------- бизнес-цифры

def business(e: EDA) -> dict:
    """Цифры для питча (data/processed/business_stats.json): что именно показывают данные, без экстраполяций."""
    s = e.stats
    src = s["sources"]
    kp = key_presence(e)
    req = {}
    for cid, g in kp.groupby("category_id"):
        miss = ~g.all_key
        req[cid] = {"title": MAIN[cid], "key_params": KEY_PARAMS[cid], "offers": len(g),
                    "offers_missing_any_key_param": round(float(miss.mean()), 4),
                    "search_clicks_on_offers_missing_key_param": round(float(g.clicks[miss].sum() / max(g.clicks.sum(), 1)), 4)}
    no_offer = e.one(f"""SELECT count(*), avg((n_seller_attrs > 0)::INT), avg((category_id IS NULL)::INT),
                         avg(regexp_matches(title, ' (UNIMALL|GSS) -$')::INT) FROM {e.p('cards')} WHERE NOT has_offer""")
    tr = e.traffic().get("_meta", {})
    seg = s["quality"]["by_segment"]
    out = {
        "source": "data/processed/*.parquet (выгрузка организаторов), submission/traffic.json; код: src/attrx/data/eda.py",
        "catalog": {
            "cards_total": src["cards"],
            "cards_with_structured_attributes": src["cards_with_offer"],
            "cards_with_structured_attributes_share": round(src["cards_with_offer"] / src["cards"], 4),
            "cards_without_structured_attributes": src["cards"] - src["cards_with_offer"],
            "cards_without_structured_attributes_share": round(1 - src["cards_with_offer"] / src["cards"], 4),
            "note": "«без структурированных атрибутов» = SKU нет в выданном YML-снимке каталога (546 069 офферов); "
                    "это не значит, что у товара нет атрибутов в системах М.Видео — только что в данных кейса их нет, "
                    "и для таких карточек атрибуты можно получить только из текста",
            "cards_without_category": src["cards"] - src["cards_with_offer"] - src["cards_category_by_clicks"],
            "no_offer_cards_with_char_block_share": round(float(no_offer[1]), 4),
            "no_offer_cards_placeholder_title_share": round(float(no_offer[3]), 4),
            "leaf_categories_with_offers": s["categories"]["leaf_with_offers"],
            "leaf_categories_lt_10_offers": s["categories"]["leaf_lt_10_offers"],
        },
        "required_attributes_gaps": req,
        "search": {
            "clicks_total": src["clicks"],
            "clicks_on_skus_with_catalog_attributes_share": src["clicks_with_offer_share"],
            "clicks_with_any_attribute_in_query_share": tr.get("any_attribute_share"),
            "clicks_with_characteristic_beyond_brand_model_share": tr.get("any_param_share"),
            "clicks_by_mention_kind": tr.get("kinds_share"),
            "categories_scored": tr.get("n_categories"), "clicks_covered_share": tr.get("clicks_covered_share"),
            "note": "доли кликов поиска, в запросах которых детектор нашёл упоминание атрибута (оценка снизу)",
        },
        "by_seller_type": {
            k: {"cards": seg[k]["cards"], "desc_lt_200": seg[k]["desc_lt_200"], "char_block": seg[k]["char_block"]}
            for k in seg},
        "fairness": s.get("fairness", {}).get("table"),
        "extractability_ceiling": {f"{r['категория']}: {r['атрибут']}": r["в тексте"]
                                   for r in s.get("extractability", {}).get("rows", [])},
    }
    return out


# --------------------------------------------------------------------------- всё разом

def run(processed: Path, img: Path | None, raw: Path | None = None, traffic: Path | None = None) -> dict:
    """`attrx eda-real`: все разделы -> графики docs/img/eda_*.png + eda_stats.json + business_stats.json."""
    import time

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = time.perf_counter()
    setup_style()
    e = EDA(processed=processed, raw=raw, img=img, traffic_path=traffic)
    sources(e)
    plt.close(fig_coverage(e))
    fields(e)
    plt.close(fig_quality(e, quality(e)))
    lengths(e)
    plt.close(fig_lengths(e))
    categories(e)
    plt.close(fig_long_tail(e))
    params_overview(e)
    plt.close(fig_fill(e, fill_matrix(e)))
    plt.close(fig_values(e, values(e)))
    plt.close(fig_units(e, normalization(e)))
    plt.close(fig_extractability(e, extractability(e)))
    tables = search(e)
    if len(tables["kinds"]):
        plt.close(fig_search_kinds(e, tables["kinds"]))
        plt.close(fig_traffic_top(e))
    plt.close(fig_fairness(e, fairness(e)))
    biz = business(e)
    write_json(processed / "eda_stats.json", e.stats)
    write_json(processed / "business_stats.json", biz)
    return {"seconds": round(time.perf_counter() - t0, 1), "figures": sorted(p.name for p in (img or Path()).glob("eda_*.png")),
            "stats": str(processed / "eda_stats.json"), "business": str(processed / "business_stats.json")}


def write_json(path: Path, data: dict) -> None:
    with atomic_path(path) as tmp:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
