"""attrx demo — на закоммиченном реальном сэмпле (400 карточек holdout + эталон каталога), те же метрики, что
eval-real; без LLM, torch и data/processed. Веса спроса без data/processed — из _meta config/traffic_weights.yaml."""
import json
import sys

import pytest
import yaml

from attrx.cli import main
from attrx.config import ROOT
from attrx.eval.metrics import load_traffic_weights
from attrx.taxonomy import load_taxonomy


def test_demo_runs_on_real_holdout_sample(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PREDICTIONS_PATH", str(tmp_path / "predictions.jsonl"))
    torch_before = "torch" in sys.modules
    main(["demo"])
    out = capsys.readouterr().out
    lines = (ROOT / "data" / "sample" / "real_cards.jsonl").read_text(encoding="utf-8").splitlines()
    ids = [json.loads(x)["id"] for x in lines if x.strip()]
    preds = [json.loads(x) for x in (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(ids) == 400 and [p["card_id"] for p in preds] == ids  # реальные карточки, а не синтетика
    m = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert m["n_cards"] == 400 and m["info"]["sources"] == ["rules", "seller"]
    assert {"micro", "recall_extractable", "coverage", "traffic_weighted_f1", "per_category", "segments"} <= set(m)
    assert m["micro"]["precision"] >= 0.9  # грубая страховка от регрессии правил
    for label in ("Precision", "Recall по извлекаемому эталону", "F1 с учётом поискового спроса", "Покрытие",
                  "F1 по сегментам", "smartphone"):
        assert label in out
    assert not any(p["llm_used"] for p in preds)
    if not torch_before:
        assert "torch" not in sys.modules  # демо не тянет torch


def test_traffic_weights_fall_back_to_committed_click_shares(tmp_path):
    tax = load_taxonomy()
    cid = tax.categories["smartphone"].category_ids[0]
    (tmp_path / "config").mkdir()
    data = {"_meta": {"total_clicks": 1000, "categories": {cid: {"clicks": 400}, "999999": {"clicks": 100}}},
            cid: {"vendor": 0.5}}
    (tmp_path / "config" / "traffic_weights.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    w = load_traffic_weights(tax, root=tmp_path)  # data/processed нет — клики категорий из _meta
    assert w[("smartphone", "brand")] == pytest.approx(0.4 * 0.5)  # доля кликов категории × вес атрибута
