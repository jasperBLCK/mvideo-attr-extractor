"""Подготовка реальных данных кейса: YML-фид, описания SKU, клики поиска -> data/processed/*.parquet."""
from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SPLITS = (("train", 80), ("dev", 90), ("holdout", 100))  # 80/10/10 по md5(sku_id)


def split_of(sku_id: object) -> str:
    """Детерминированный сплит по md5 от строкового sku_id: train 80% / dev 10% / holdout 10%."""
    h = int(hashlib.md5(str(sku_id).encode()).hexdigest()[:8], 16) % 100
    return next(name for name, edge in SPLITS if h < edge)


@contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """Пишем во временный файл рядом и переименовываем: параллельные читатели не увидят недописанный файл."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
