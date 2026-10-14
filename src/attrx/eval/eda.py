"""EDA датасета: размеры, пропуски, длины текстов, категории, ключи продавцов. Результат — Markdown + графики."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from ..schema import Card


def eda(cards: list[Card], out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{
        "id": c.id, "category": c.category or "<нет>", "title_len": len(c.title), "desc_len": len(c.description),
        "n_seller_attrs": len(c.seller_attrs), "empty_title": not c.title, "empty_desc": not c.description,
    } for c in cards])
    keys = Counter(k.strip().lower() for c in cards for k in c.seller_attrs)
    lines = [
        "# EDA", "", f"Карточек: **{len(df)}**", "",
        "## Пропуски", "",
        f"- пустое название: {df.empty_title.mean():.1%}",
        f"- пустое описание: {df.empty_desc.mean():.1%}",
        f"- нет атрибутов продавца: {(df.n_seller_attrs == 0).mean():.1%}", "",
        "## Категории", "", df.category.value_counts().to_frame().to_string(), "",
        "## Длины текстов", "", df[["title_len", "desc_len", "n_seller_attrs"]].describe().round(1).to_string(), "",
        "## Топ-30 ключей продавцов (кандидаты в aliases таксономии)", "",
        "\n".join(f"- {k}: {n}" for k, n in keys.most_common(30)), "",
    ]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        df.desc_len.hist(bins=40, ax=ax[0])
        ax[0].set_title("Длина описания")
        df.category.value_counts().head(15).plot.barh(ax=ax[1])
        ax[1].set_title("Категории")
        fig.tight_layout()
        fig.savefig(out_dir / "eda.png", dpi=120)
        lines += ["![eda](eda.png)", ""]
    except ImportError:
        pass
    report = "\n".join(lines)
    (out_dir / "eda.md").write_text(report, encoding="utf-8")
    return report
