"""YML-фид М.Видео (`skus.pkl`, xmltodict) -> плоские таблицы categories / offers / params.

Структура: obj["yml_catalog"]["shop"]["categories"]["category"] = [{@id, @parentId?, #text}],
obj["yml_catalog"]["shop"]["offers"]["offer"] = [{@id, @available, vendor, model, name, categories: {categoryId},
param: [{@name, @unit?, #text}]}]. У атрибута бывает пара: бакет фильтра («Ширина» = «80 - 120 мм») и точное
значение «Ширина_search» (@unit «мм», «90»). `param` и `categoryId` при одном элементе — не список, а скаляр/словарь.
"""
from __future__ import annotations

import gc
import pickle
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import atomic_path

SEARCH_SUFFIX = "_search"
SELLER_PARAM = "Продавец"
# Служебные параметры фида: не характеристики товара (продавец, маркетинг, лояльность, условия гарантии).
# Признак: у них нет точного варианта «_search» (кроме гарантии), значения — акции/подборки витрины.
SERVICE_PARAMS = frozenset({
    "Продавец", "Потратить бонусы", "Подборки", "Топ коллекция", "Акции",
    "Гарантия", "Гарантия предоставляется", "Гарантия на АКБ", "Гарантия на аксессуары",
})

PARAMS_SCHEMA = pa.schema([
    ("sku_id", pa.string()),
    ("name", pa.string()),  # базовое имя без суффикса _search
    ("is_search", pa.bool_()),  # True — точное значение «<имя>_search», False — бакет фильтра/обычное значение
    ("unit", pa.string()),
    ("value", pa.string()),
    ("is_service", pa.bool_()),
])


class _BuiltinsOnlyUnpickler(pickle.Unpickler):
    """Pickle фида содержит только встроенные типы: любой импорт класса — повод остановиться."""

    def find_class(self, module: str, name: str):  # noqa: D401
        raise pickle.UnpicklingError(f"skus.pkl: запрещённый объект {module}.{name}")


def load_yml(path: Path) -> dict:
    """Загружает skus.pkl (~27 с, ~12 ГБ RAM). Держать в памяти не дольше, чем нужно."""
    with Path(path).open("rb") as f:
        return _BuiltinsOnlyUnpickler(f).load()


def as_list(x: Any) -> list:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def text_of(x: Any) -> str | None:
    """Скаляр xmltodict: строка, {'#text': ...} или список — берём первый текст."""
    if x is None:
        return None
    if isinstance(x, list):
        return text_of(x[0]) if x else None
    if isinstance(x, dict):
        return text_of(x.get("#text"))
    s = str(x).strip()
    return s or None


def split_param_name(name: str) -> tuple[str, bool]:
    if name.endswith(SEARCH_SUFFIX):
        return name[: -len(SEARCH_SUFFIX)], True
    return name, False


def is_service(name: str) -> bool:
    return name in SERVICE_PARAMS


# --------------------------------------------------------------------------- категории

def categories_frame(cats: Iterable[dict]) -> pd.DataFrame:
    """id, parent_id, title, path (названия от корня), path_ids, depth (корень = 0), is_leaf."""
    rows = {}
    for c in as_list(cats):
        cid = text_of(c.get("@id"))
        if cid:
            rows[cid] = (text_of(c.get("@parentId")), text_of(c.get("#text")) or "")
    children = Counter(p for p, _ in rows.values() if p)
    memo: dict[str, list[str]] = {}

    def path_ids(cid: str) -> list[str]:
        if cid in memo:
            return memo[cid]
        chain, seen, cur = [], set(), cid
        while cur and cur in rows and cur not in seen:  # защита от циклов и битых parentId
            seen.add(cur)
            chain.append(cur)
            cur = rows[cur][0]
        memo[cid] = chain[::-1]
        return memo[cid]

    out = []
    for cid, (parent, title) in rows.items():
        ids = path_ids(cid)
        out.append({"id": cid, "parent_id": parent, "title": title, "path": [rows[i][1] for i in ids],
                    "path_ids": ids, "depth": len(ids) - 1, "is_leaf": children[cid] == 0})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- офферы и параметры

_OFFER_RENAME = {"@id": "sku_id", "@available": "available"}
_OFFER_SKIP = {"param", "categories"}


def _offer_scalar_keys(offers: list[dict]) -> list[str]:
    """Скалярные поля оффера (vendor, model, name, price, url...) — берём все, что встречаются."""
    keys: Counter = Counter()
    for o in offers:
        keys.update(k for k in o if k not in _OFFER_SKIP)
    return [k for k, _ in keys.most_common()]


def convert_offers(offers: list[dict], out_dir: Path, chunk: int = 50_000, log=print) -> pd.DataFrame:
    """Пишет params.parquet потоково (чанками), возвращает таблицу офферов (её дополняет prepare)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = _offer_scalar_keys(offers)
    log(f"  поля оффера: {keys}")
    cols: dict[str, list] = {k: [] for k in keys}
    cat_ids, n_cat_ids, sellers, n_params, n_raw_params = [], [], [], [], []
    name_cache: dict[str, tuple[str, bool]] = {}
    with atomic_path(out_dir / "params.parquet") as tmp, \
            pq.ParquetWriter(tmp, PARAMS_SCHEMA, compression="zstd") as writer:
        buf: dict[str, list] = {f.name: [] for f in PARAMS_SCHEMA}
        for i, o in enumerate(offers):
            sku = text_of(o.get("@id")) or ""
            for k in keys:
                cols[k].append(text_of(o.get(k)))
            cats = [text_of(c) for c in as_list((o.get("categories") or {}).get("categoryId"))]
            cats = [c for c in cats if c]
            cat_ids.append(cats[0] if cats else None)
            n_cat_ids.append(len(cats))
            seller, names, n_raw = None, set(), 0
            for p in as_list(o.get("param")):
                if not isinstance(p, dict) or not (raw := text_of(p.get("@name"))):
                    continue
                if raw not in name_cache:
                    name_cache[raw] = split_param_name(raw)
                name, is_search = name_cache[raw]
                value = text_of(p.get("#text"))
                service = is_service(name)
                if name == SELLER_PARAM and seller is None:
                    seller = value
                if not service:
                    names.add(name)
                n_raw += 1
                buf["sku_id"].append(sku)
                buf["name"].append(name)
                buf["is_search"].append(is_search)
                buf["unit"].append(text_of(p.get("@unit")))
                buf["value"].append(value)
                buf["is_service"].append(service)
            sellers.append(seller)
            n_params.append(len(names))
            n_raw_params.append(n_raw)
            if (i + 1) % chunk == 0:
                writer.write_table(pa.Table.from_pydict(buf, schema=PARAMS_SCHEMA))
                buf = {f.name: [] for f in PARAMS_SCHEMA}
                log(f"  офферов обработано: {i + 1:,}")
        if buf["sku_id"]:
            writer.write_table(pa.Table.from_pydict(buf, schema=PARAMS_SCHEMA))
    df = pd.DataFrame({_OFFER_RENAME.get(k, k): v for k, v in cols.items()})
    df["sku_id"] = df["sku_id"].astype(str)
    if "available" in df:
        df["available"] = df["available"].eq("true")
    df["category_id"] = cat_ids
    df["n_category_ids"] = n_cat_ids
    df["seller"] = sellers
    df["n_params"] = n_params
    df["n_raw_params"] = n_raw_params
    return df


def convert(pkl_path: Path, out_dir: Path, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """skus.pkl -> (categories, offers) + params.parquet на диске. Pickle освобождается до выхода."""
    obj = load_yml(pkl_path)
    shop = obj["yml_catalog"]["shop"]
    cats = categories_frame((shop.get("categories") or {}).get("category"))
    offers_list = as_list((shop.get("offers") or {}).get("offer"))
    log(f"  категорий: {len(cats):,}, офферов: {len(offers_list):,}")
    offers = convert_offers(offers_list, out_dir, log=log)
    del obj, shop, offers_list
    gc.collect()
    return cats, offers
