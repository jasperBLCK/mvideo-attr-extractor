"""Подготовка данных: data/raw -> data/processed (parquet, zstd). Запуск: `attrx prepare`.

Шаги (по порядку; можно запускать по отдельности через --steps):
  yml       skus.pkl -> categories / offers / params            (~12 ГБ RAM на время загрузки pickle)
  clicks    query_clicks -> clicks_by_sku / clicks_by_category / top_queries + сверка subject_id/seo_id
  cards     sku_desc + offers (+ seo_id кликов) -> cards (очищенный текст, seller_attrs, категория, продавец, split)
  taxonomy  офферы × параметры × тексты карточек -> taxonomy_candidates.json + param_stats.parquet
  sample    ~400 карточек HOLDOUT -> data/sample/real_cards.jsonl + real_params.jsonl (для демо и тестов)
"""
from __future__ import annotations

import json
import re
import resource
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa

STEPS = ("yml", "clicks", "cards", "taxonomy", "sample")


def _log(msg: str) -> None:
    print(msg, flush=True)


def peak_rss_gb(who: int = resource.RUSAGE_SELF) -> float:
    """Пиковое RSS (Linux: ru_maxrss в КБ); RUSAGE_CHILDREN — самый «тяжёлый» из завершённых воркеров."""
    return resource.getrusage(who).ru_maxrss / 1024**2


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    from . import atomic_path

    with atomic_path(path) as tmp:
        df.to_parquet(tmp, index=False, compression="zstd")


def step_yml(raw: Path, out: Path) -> dict:
    from .yml import convert

    cats, offers = convert(raw / "skus.pkl", out, log=_log)
    # число офферов: прямо в категории и во всём поддереве
    direct = offers["category_id"].value_counts()
    cats["n_offers"] = cats["id"].map(direct).fillna(0).astype("int64")
    sub = {}
    for ids, n in zip(cats["path_ids"], cats["n_offers"], strict=True):
        for cid in ids:
            sub[cid] = sub.get(cid, 0) + n
    cats["n_offers_total"] = cats["id"].map(sub).fillna(0).astype("int64")
    title = dict(zip(cats["id"], cats["title"], strict=True))
    offers["category_title"] = offers["category_id"].map(title)
    write_parquet(cats, out / "categories.parquet")
    write_parquet(offers, out / "offers.parquet")
    return {"categories": len(cats), "offers": len(offers),
            "offers_without_category": int(offers["category_id"].isna().sum()),
            "offers_unknown_category": int((offers["category_id"].notna() & offers["category_title"].isna()).sum()),
            "offers_multi_category": int((offers["n_category_ids"] > 1).sum())}


MVIDEO_SELLER = "М.Видео"

CARDS_SCHEMA = pa.schema([
    ("sku_id", pa.string()),
    ("title", pa.string()),
    ("description_clean", pa.string()),  # без HTML и без блока «Характеристики:»
    ("seller_attrs", pa.string()),  # JSON {ключ: значение} из блока «Характеристики:» (порядок сохранён)
    ("n_seller_attrs", pa.int32()),
    ("category_id", pa.string()),
    ("category_title", pa.string()),
    ("category_path", pa.string()),  # JSON-список названий от корня
    ("category_source", pa.string()),  # offer | clicks_seo | null
    ("seller", pa.string()),
    ("seller_type", pa.string()),  # mvideo | marketplace | null (нет оффера)
    ("has_offer", pa.bool_()),
    ("split", pa.string()),  # train / dev / holdout = 80/10/10 по md5(sku_id)
    ("description_len", pa.int32()),
])


def seller_type(seller: str | None) -> str | None:
    if not seller:
        return None
    return "mvideo" if seller == MVIDEO_SELLER else "marketplace"


def _clean_batch(batch: tuple[list, list, list]) -> tuple[list, list, list, list, list]:
    """Воркер пула: очистка заголовков/описаний и разбор блока характеристик."""
    from .text import clean_title, prepare_description

    ids, titles, descs = batch
    t_out, d_out, a_out, n_out = [], [], [], []
    for t, d in zip(titles, descs, strict=True):
        desc, attrs = prepare_description(d)
        t_out.append(clean_title(t))
        d_out.append(desc)
        a_out.append(json.dumps(attrs, ensure_ascii=False))
        n_out.append(len(attrs))
    return ids, t_out, d_out, a_out, n_out


def step_cards(raw: Path, out: Path, workers: int = 3, batch_size: int = 20_000) -> dict:
    import multiprocessing as mp

    import pyarrow.parquet as pq

    from . import atomic_path, split_of

    def columns(path: Path, *cols: str) -> list[list]:  # через pyarrow: пропуски -> None (а не NaN, как в pandas 3)
        t = pq.read_table(path, columns=list(cols))
        return [t.column(c).to_pylist() for c in cols]

    sku, cat, sel = columns(out / "offers.parquet", "sku_id", "category_id", "seller")
    off_cat, off_seller = dict(zip(sku, cat, strict=True)), dict(zip(sku, sel, strict=True))
    seo: dict[str, str] = {}
    if (out / "clicks_by_sku.parquet").exists():  # категория для карточек без оффера — по seo_id кликов
        seo = dict(zip(*columns(out / "clicks_by_sku.parquet", "sku_id", "seo_id"), strict=True))
    ids, titles, paths = columns(out / "categories.parquet", "id", "title", "path")
    cat_title = dict(zip(ids, titles, strict=True))
    cat_path = {i: json.dumps(p, ensure_ascii=False) for i, p in zip(ids, paths, strict=True)}
    del sku, cat, sel, ids, titles, paths

    pf = pq.ParquetFile(raw / "sku_desc.parquet")

    def batches():
        for b in pf.iter_batches(batch_size=batch_size, columns=["sku_id", "title", "description"]):
            yield ([str(x) for x in b.column("sku_id").to_pylist()], b.column("title").to_pylist(),
                   b.column("description").to_pylist())

    n = 0
    with atomic_path(out / "cards.parquet") as tmp, pq.ParquetWriter(tmp, CARDS_SCHEMA, compression="zstd") as writer:
        with mp.get_context("spawn").Pool(workers) as pool:
            for ids, titles, descs, attrs, n_attrs in pool.imap(_clean_batch, batches()):
                cat_ids, src = [], []
                for sku in ids:
                    if (c := off_cat.get(sku)) is not None:
                        cat_ids.append(c)
                        src.append("offer")
                    elif (c := seo.get(sku)) is not None:
                        cat_ids.append(c)
                        src.append("clicks_seo")
                    else:
                        cat_ids.append(None)
                        src.append(None)
                sellers = [off_seller.get(s) for s in ids]
                writer.write_table(pa.Table.from_pydict({
                    "sku_id": ids, "title": titles, "description_clean": descs, "seller_attrs": attrs,
                    "n_seller_attrs": n_attrs, "category_id": cat_ids,
                    "category_title": [cat_title.get(c) if c else None for c in cat_ids],
                    "category_path": [cat_path.get(c) if c else None for c in cat_ids],
                    "category_source": src, "seller": sellers, "seller_type": [seller_type(s) for s in sellers],
                    "has_offer": [s in off_cat for s in ids], "split": [split_of(s) for s in ids],
                    "description_len": [len(d) for d in descs],
                }, schema=CARDS_SCHEMA))
                n += len(ids)
                if n % 200_000 < batch_size:
                    _log(f"  карточек: {n:,}")
    return cards_summary(out / "cards.parquet")


def cards_summary(path: Path) -> dict:
    df = pd.read_parquet(path, columns=["has_offer", "category_source", "seller_type", "split", "n_seller_attrs",
                                        "description_len"])
    df["has_block"] = df["n_seller_attrs"] > 0
    by_seller = df.assign(seller_type=df["seller_type"].fillna("no_offer")).groupby("seller_type").agg(
        cards=("has_block", "size"), with_char_block=("has_block", "mean"),
        median_desc_len=("description_len", "median"), empty_desc=("description_len", lambda s: float((s == 0).mean())))
    return {
        "cards": len(df), "with_offer": int(df["has_offer"].sum()),
        "category_source": df["category_source"].fillna("none").value_counts().to_dict(),
        "split": df["split"].value_counts().to_dict(),
        "split_with_offer": df.loc[df["has_offer"], "split"].value_counts().to_dict(),
        "with_char_block": int(df["has_block"].sum()),
        "by_seller_type": json.loads(by_seller.round(4).to_json(orient="index")),
    }


def step_clicks(raw: Path, out: Path) -> dict:
    import shutil

    from .clicks import aggregate

    try:
        return aggregate(raw, out)
    finally:
        shutil.rmtree(out / ".duckdb_tmp", ignore_errors=True)


def step_taxonomy(raw: Path, out: Path) -> dict:
    from .taxo import build, param_stats

    info = build(out)
    info["param_stats_rows"] = param_stats(out)
    return info


SAMPLE_QUOTAS = {"761": 40, "914": 20, "987": 60}  # смартфоны (+iPhone) и ноутбуки — обязательные категории кейса
# параметры под обязательные атрибуты кейса — в сэмпл попадают всегда
REQUIRED_PARAM = re.compile(r"(?i)цвет|памят|диагонал|аккумулятор|процессор|ssd|накопител|модель|серия|бренд")


def sample_params(category: dict, top: int) -> set[str]:
    """Параметры категории для сэмпла: top по fill_rate × extractable_loose (заполнен И находится в тексте)
    плюс параметры под обязательные атрибуты кейса."""
    score = sorted(category["params"], key=lambda p: -(p["fill_rate"] * (p["extractable_loose"] or 0)))
    return {p["name"] for p in score[:top]} | {p["name"] for p in category["params"]
                                               if REQUIRED_PARAM.search(p["name"]) and p["fill_rate"] >= 0.3}


def step_sample(raw: Path, out: Path, total: int = 400, max_desc: int = 2000, top_params: int = 20,
                seed: int = 7) -> dict:
    """Реальный сэмпл для демо и тестов из HOLDOUT: data/sample/real_cards.jsonl + real_params.jsonl (< 2 МБ).

    В real_params — параметры из sample_params() (заполненные и находимые в тексте + обязательные), бакет
    фильтра — только если у параметра нет точного варианта _search. Полный набор — data/processed/params.parquet.
    """
    from ..config import ROOT
    from . import atomic_path
    from .clicks import connect
    from .taxo import MUST_CATEGORIES

    con = connect()
    taxo = json.loads((out / "taxonomy_candidates.json").read_text(encoding="utf-8"))["categories"]
    chosen = [c["id"] for c in taxo]
    keep_params = {c["id"]: sample_params(c, top_params) for c in taxo}
    rest = [c for c in chosen if c not in MUST_CATEGORIES]
    quotas = dict(SAMPLE_QUOTAS)
    left = total - sum(quotas.values())
    for i, c in enumerate(rest):
        quotas[c] = left // len(rest) + (i < left % len(rest))
    ids = ",".join(f"'{c}'" for c in quotas)
    cards = con.execute(f"""SELECT sku_id, title, description_clean, seller_attrs, category_id, category_title, seller,
                                   seller_type, description_len
        FROM read_parquet('{out / 'cards.parquet'}')
        WHERE has_offer AND split = 'holdout' AND category_id IN ({ids})""").df()
    picked = []
    for cid, q in quotas.items():
        g = cards[(cards["category_id"] == cid) & (cards["description_len"] <= max_desc)]
        short = g[g["description_len"] < 80].sample(min(len(g[g["description_len"] < 80]), max(1, q // 10)),
                                                     random_state=seed)
        g = g.drop(short.index)
        n_mv = min((q - len(short)) // 2, int((g["seller_type"] == "mvideo").sum()))
        mv = g[g["seller_type"] == "mvideo"].sample(n_mv, random_state=seed)
        mp = g[g["seller_type"] != "mvideo"]
        mp = mp.sample(min(len(mp), q - len(short) - n_mv), random_state=seed)
        picked.append(pd.concat([short, mv, mp]))
    sample = pd.concat(picked)
    if (deficit := total - len(sample)) > 0:  # в мелких категориях holdout меньше квоты — добираем из остальных
        pool = cards[cards["category_id"].isin(rest) & (cards["description_len"] <= max_desc)
                     & ~cards.index.isin(sample.index)]
        sample = pd.concat([sample, pool.sample(min(deficit, len(pool)), random_state=seed)])
    sample = sample.sort_values(["category_id", "sku_id"])
    sku_list = ",".join(f"'{s}'" for s in sample["sku_id"])
    offers = con.execute(f"""SELECT sku_id, vendor, model FROM read_parquet('{out / 'offers.parquet'}')
        WHERE sku_id IN ({sku_list})""").df().set_index("sku_id")
    params = con.execute(f"""SELECT sku_id, name, unit, value, is_search FROM read_parquet('{out / 'params.parquet'}')
        WHERE sku_id IN ({sku_list}) AND NOT is_service""").df()
    by_sku = {s: g for s, g in params.groupby("sku_id")}
    cat_of = dict(zip(sample["sku_id"], sample["category_id"], strict=True))

    def none(x):
        return None if x is None or (isinstance(x, float) and x != x) else x

    def sku_params(sku: str) -> list[dict]:
        g = by_sku.get(sku)
        if g is None:
            return []
        g = g[g["name"].isin(keep_params.get(cat_of[sku], set())) & g["value"].notna()]
        exact = set(g.loc[g["is_search"], "name"])
        g = g[g["is_search"] | ~g["name"].isin(exact)]
        return [{"name": p.name, "unit": none(p.unit), "value": p.value, "is_search": bool(p.is_search)}
                for p in g.itertuples()]

    dst = ROOT / "data" / "sample"
    dst.mkdir(parents=True, exist_ok=True)
    with atomic_path(dst / "real_cards.jsonl") as tc, atomic_path(dst / "real_params.jsonl") as tp, \
            tc.open("w", encoding="utf-8") as fc, tp.open("w", encoding="utf-8") as fp:
        for r in sample.itertuples():
            fc.write(json.dumps({"id": r.sku_id, "title": r.title, "description": r.description_clean,
                                 "seller_attrs": json.loads(r.seller_attrs), "category": none(r.category_title),
                                 "category_id": r.category_id, "seller": none(r.seller)}, ensure_ascii=False) + "\n")
            fp.write(json.dumps({"id": r.sku_id, "vendor": none(offers.at[r.sku_id, "vendor"]),
                                 "model": none(offers.at[r.sku_id, "model"]), "params": sku_params(r.sku_id)},
                                ensure_ascii=False, separators=(",", ":")) + "\n")
    size = sum((dst / f).stat().st_size for f in ("real_cards.jsonl", "real_params.jsonl"))
    return {"cards": len(sample), "by_category": sample["category_id"].value_counts().to_dict(),
            "mvideo": int((sample["seller_type"] == "mvideo").sum()),
            "short_desc": int((sample["description_len"] < 80).sum()), "size_mb": round(size / 1024**2, 2)}


def run(raw: Path, out: Path, steps: tuple[str, ...] = STEPS) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    t_all = time.perf_counter()
    for name in steps:
        fn = globals()[f"step_{name}"]
        t = time.perf_counter()
        _log(f"[{name}] ...")
        info = fn(raw, out) or {}
        info["seconds"] = round(time.perf_counter() - t, 1)
        info["peak_rss_gb"] = round(peak_rss_gb(), 2)
        report[name] = info
        _log(f"[{name}] {json.dumps(info, ensure_ascii=False)}")
    report["total_seconds"] = round(time.perf_counter() - t_all, 1)
    report["peak_rss_gb"] = round(peak_rss_gb(), 2)
    report["peak_rss_worker_gb"] = round(peak_rss_gb(resource.RUSAGE_CHILDREN), 2)
    prev = out / "prepare_report.json"
    merged = json.loads(prev.read_text(encoding="utf-8")) if prev.exists() else {}
    merged.update({k: v for k, v in report.items() if k in STEPS})  # последние цифры по каждому шагу
    totals = {"steps": list(steps), **{k: report[k] for k in ("total_seconds", "peak_rss_gb", "peak_rss_worker_gb")}}
    merged["last_run"] = totals
    if set(steps) == set(STEPS):
        merged["full_run"] = totals
    for k in ("total_seconds", "peak_rss_gb", "peak_rss_worker_gb"):  # старый плоский формат
        merged.pop(k, None)
    from . import atomic_path

    with atomic_path(prev) as tmp:
        tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
