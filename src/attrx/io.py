"""Чтение датасета организаторов (CSV/JSON/JSONL/Parquet; XLSX — с extra [xlsx]) через маппинг config/dataset.yaml
и экспорт предсказаний.

Реальные данные после `attrx prepare`: load_processed_cards() (карточки из data/processed/cards.parquet)
и load_offer_params() (эталон фида: vendor/model/param оффера) — см. .claude/context/DATA.md.
"""
from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import ROOT, get_settings
from .schema import Card, ExtractionResult


def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if suf == ".jsonl":
        return pd.read_json(path, lines=True, dtype=False)
    if suf == ".json":
        return pd.read_json(path, dtype=False)
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".xlsx":
        try:
            return pd.read_excel(path, dtype=str)
        except ImportError as e:  # openpyxl — необязательная зависимость
            raise ImportError('Для чтения .xlsx нужен openpyxl: uv pip install -e ".[xlsx]" (или pip install openpyxl)') from e
    raise ValueError(f"Неизвестный формат: {path} (поддерживаются .csv, .json, .jsonl, .parquet, .xlsx)")


def _as_dict(v: Any) -> dict:
    if isinstance(v, dict):
        return v
    if isinstance(v, list):  # [{"name": .., "value": ..}] — частый формат маркетплейсов
        return {str(x.get("name") or x.get("key")): x.get("value") for x in v if isinstance(x, dict)}
    if isinstance(v, str) and v.strip().startswith(("{", "[")):
        try:
            return _as_dict(json.loads(v))
        except json.JSONDecodeError:
            return {}
    return {}


def load_mapping(path: Path | None = None) -> dict:
    return yaml.safe_load((path or get_settings().dataset_config).read_text(encoding="utf-8"))


def row_to_card(row: dict, m: dict) -> Card:
    seller: dict = {}
    if m.get("seller_attrs") and m["seller_attrs"] in row:
        seller = _as_dict(row[m["seller_attrs"]])
    if m.get("seller_attrs_prefix"):
        pre = m["seller_attrs_prefix"]
        seller |= {k[len(pre):]: v for k, v in row.items() if k.startswith(pre) and v not in (None, "")}

    def get(name: str) -> str:
        col = m.get(name)
        v = row.get(col) if col else None
        return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)

    return Card(id=get("id"), title=get("title"), description=get("description"), seller_attrs=seller,
                category=get("category") or None, category_id=get("category_id") or None,
                seller=get("seller") or None)


def load_cards(path: Path, mapping: dict | None = None) -> list[Card]:
    m = mapping or load_mapping()
    df = read_table(path)
    return [row_to_card(r, m) for r in df.to_dict(orient="records")]


PROCESSED_DIR = ROOT / "data" / "processed"


def load_processed_cards(path: Path | None = None, *, split: str | None = None,
                         category_ids: Iterable[str] | None = None, has_offer: bool | None = None,
                         limit: int | None = None) -> list[Card]:
    """Карточки из data/processed/cards.parquet (после `attrx prepare`) с фильтрами по сплиту/категориям/офферу."""
    import pyarrow.parquet as pq

    filters = []
    if split:
        filters.append(("split", "=", split))
    if category_ids is not None:
        filters.append(("category_id", "in", [str(c) for c in category_ids]))
    if has_offer is not None:
        filters.append(("has_offer", "=", has_offer))
    cols = ["sku_id", "title", "description_clean", "seller_attrs", "category_id", "category_title", "seller"]
    t = pq.read_table(path or PROCESSED_DIR / "cards.parquet", columns=cols, filters=filters or None)
    if limit is not None:
        t = t.slice(0, limit)
    return [Card(id=r["sku_id"], title=r["title"] or "", description=r["description_clean"] or "",
                 seller_attrs=json.loads(r["seller_attrs"] or "{}"), category=r["category_title"],
                 category_id=r["category_id"], seller=r["seller"]) for r in t.to_pylist()]


def load_offer_params(sku_ids: Iterable[str], processed: Path | None = None,
                      include_service: bool = False) -> dict[str, dict[str, Any]]:
    """Эталон из YML-фида: {sku_id: {vendor, model, category_id, params: [{name, unit, value, is_search}]}}.

    name — базовое имя без «_search»; is_search=True — точное значение (с unit), False — бакет фильтра.
    """
    import pyarrow.parquet as pq

    base = processed or PROCESSED_DIR
    ids = [str(x) for x in sku_ids]
    offers = pq.read_table(base / "offers.parquet", columns=["sku_id", "vendor", "model", "category_id"],
                           filters=[("sku_id", "in", ids)]).to_pylist()
    out = {r["sku_id"]: {"vendor": r["vendor"], "model": r["model"], "category_id": r["category_id"], "params": []}
           for r in offers}
    flt = [("sku_id", "in", ids)] + ([] if include_service else [("is_service", "=", False)])
    for r in pq.read_table(base / "params.parquet", columns=["sku_id", "name", "unit", "value", "is_search"],
                           filters=flt).to_pylist():
        if r["sku_id"] in out:
            out[r["sku_id"]]["params"].append({k: r[k] for k in ("name", "unit", "value", "is_search")})
    return out


def load_gold(path: Path, mapping: dict | None = None) -> dict[str, dict[str, Any]]:
    """Эталон: JSONL {"id":..., "attributes": {...}} или колонка gold в таблице датасета."""
    m = mapping or load_mapping()
    if path.suffix == ".jsonl":
        out = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[str(r["id"])] = r.get("attributes") or {}
        return out
    df = read_table(path)
    return {str(r[m["id"]]): _as_dict(r.get(m.get("gold", "gold"))) for r in df.to_dict(orient="records")}


def write_predictions(results: Iterable[ExtractionResult], path: Path) -> list[ExtractionResult]:
    path.parent.mkdir(parents=True, exist_ok=True)
    results = list(results)
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")
    # плоский CSV — удобно открыть в Excel и сдать как «файл предсказаний»
    with path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "attribute", "value", "unit", "confidence", "status", "sources", "evidence"])
        for r in results:
            for a in r.attributes:
                w.writerow([r.card_id, r.category, a.attribute, "" if a.value is None else a.value, a.unit or "",
                            a.confidence, a.status, "|".join(a.sources), a.evidence or ""])
    return results


def read_predictions(path: Path) -> Iterator[ExtractionResult]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield ExtractionResult.model_validate_json(line)
