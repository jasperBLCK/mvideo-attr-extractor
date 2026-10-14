from attrx.config import ROOT
from attrx.eval.report import MODES, compare, errors, segments, write_report
from attrx.io import load_cards, load_gold
from attrx.pipeline import Extractor

SAMPLE = ROOT / "data" / "sample"


def _data():
    ex = Extractor()
    return ex, load_cards(SAMPLE / "cards.jsonl"), load_gold(SAMPLE / "gold.jsonl")


def test_compare_modes_and_seller_helps():
    ex, cards, gold = _data()
    rows, by_mode = compare(ex, cards, gold, ex.tax, {k: MODES[k] for k in ("правила", "продавец + правила")})
    assert [r["mode"] for r in rows] == ["правила", "продавец + правила"]
    assert rows[1]["coverage"] >= rows[0]["coverage"]
    assert all(r["llm_calls"] == 0 for r in rows)


def test_segments_and_errors():
    ex, cards, gold = _data()
    results = list(ex.extract_many(cards))
    seg = segments(results, cards, gold, ex.tax)
    assert "нет атрибутов продавца" in seg and seg["есть атрибуты продавца"]["n_cards"] > 0
    for e in errors(results, gold, ex.tax):
        assert e["kind"] in {"wrong", "extra", "missed"}


def test_write_report(tmp_path):
    ex, cards, gold = _data()
    res = write_report(ex, cards, gold, ex.tax, tmp_path)
    for name in ("predictions.jsonl", "predictions.csv", "metrics.json", "compare.json", "segments.json", "errors.csv", "REPORT.md"):
        assert (tmp_path / name).exists(), name
    assert res["metrics"]["mode"] in {r["mode"] for r in res["compare"]}
