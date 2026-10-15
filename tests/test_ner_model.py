"""NER-модель (torch/transformers): без extra [ml] или без обученной модели в models/ner — тесты пропускаются."""
import os
from pathlib import Path

import pytest

from attrx.config import ROOT
from attrx.extractors import ner
from attrx.schema import Card
from attrx.taxonomy import load_taxonomy

MODEL = Path(os.getenv("NER_TEST_MODEL", str(ROOT / "models" / "ner")))
pytestmark = [
    pytest.mark.skipif(not ner.available(), reason="нужен extra [ml]: torch, transformers"),
    pytest.mark.skipif(not (MODEL / "config.json").exists(), reason="нет обученной модели (attrx train-ner --distant)"),
]


@pytest.fixture(scope="module")
def model():
    return ner.NerExtractor(MODEL, device=os.getenv("NER_DEVICE", "cpu"))


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


def test_model_card_and_calibration(model):
    info = ner.read_info(MODEL)
    assert info["supervision"].startswith("distant")
    assert {"brand", "model", "storage", "screen", "color"} <= set(model.attrs)
    cal = info["calibration"]
    assert cal["points"] > 1000 and cal["table"]
    vals = [ner.calibrate(cal["table"], x / 20) for x in range(21)]
    assert all(b >= a for a, b in zip(vals, vals[1:], strict=False))  # калибровка монотонна


def test_extracts_typical_smartphone_card(model, tax):
    card = Card(id="t", title="Смартфон Samsung Galaxy A55 8/256GB синий",
                description="Смартфон с экраном диагональю 6,6 дюйма и аккумулятором емкостью 5000 мАч.",
                category_id="761")
    cat = tax.categories["smartphone"]
    got = {c.attribute: c for c in model.extract(card, cat, tax)}
    assert got["brand"].value == "Samsung"
    assert got["storage"].value == 256.0
    assert got["screen"].value == 6.6
    assert got["battery"].value == 5000.0
    assert got["color"].value == "синий"
    text = card.full_text()
    for c in got.values():  # цитата — ровно фрагмент текста, уверенность — вероятность
        assert c.source == "ml" and text[c.span[0]:c.span[1]].strip() == c.raw and 0 < c.score < 1


def test_schema_of_category_limits_labels(model, tax):
    card = Card(id="t", title="Чайник электрический Polaris PWK 1755CAD 1,7 л 2200 Вт белый")
    cat = tax.categories["kettle"]
    ents = model.entities(card, cat, tax)
    assert ents and {a for a, *_ in ents} <= set(cat.attributes)
    got = {c.attribute: c.value for c in model.extract(card, cat, tax)}
    assert got.get("volume") == 1.7 and got.get("power") == 2200.0


def test_batch_equals_single(model, tax):
    cat = tax.categories["laptop"]
    cards = [Card(id=str(i), title=t) for i, t in enumerate([
        "Ноутбук ASUS Vivobook 15 X1504VA-BQ1069 15.6\"/Core i5 1335U/16/512/Win/серебристый",
        "Ноутбук Apple MacBook Air 13 M3 8 core/10 core/16/512/Midnight (MXCV3)"])]
    batch = model.extract_batch([(c, cat) for c in cards], tax)
    single = [model.extract(c, cat, tax) for c in cards]
    assert [[(x.attribute, x.value) for x in b] for b in batch] == [[(x.attribute, x.value) for x in s] for s in single]
