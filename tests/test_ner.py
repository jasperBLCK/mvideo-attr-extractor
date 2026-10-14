import os

import pytest

from attrx.extractors import ner, rules
from attrx.schema import Card
from attrx.taxonomy import load_taxonomy


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


def test_bio_labels_from_char_spans():
    # "Экран 6,1 128 ГБ": токены по символам
    offsets = [(0, 0), (0, 5), (6, 7), (7, 8), (8, 9), (10, 13), (14, 16), (0, 0)]
    spans = [(6, 9, "screen"), (10, 16, "storage")]
    assert ner.bio_labels(offsets, spans) == [None, "O", "B-screen", "I-screen", "I-screen", "B-storage", "I-storage", None]


def test_weak_spans_use_rules_and_respect_gold(tax):
    card = Card(id="x", title="Смартфон Samsung Galaxy A55 8/256 ГБ синий")
    cat = tax.categories["smartphone"]
    cands = rules.extract(card, cat, tax)
    spans = ner.weak_spans(card, cands, None, tax, cat)
    attrs = {a for _, _, a in spans}
    assert {"brand", "color"} <= attrs
    text = card.full_text()
    assert all(text[s:e] for s, e, _ in spans)
    # gold говорит, что цвет другой — спан цвета не используем как обучающую метку
    spans_g = ner.weak_spans(card, cands, {"brand": "Samsung", "color": "красный"}, tax, cat)
    assert "color" not in {a for _, _, a in spans_g}


@pytest.mark.skipif(not ner.available() or not os.getenv("ATTRX_SLOW"), reason="нужен extra [ml] и ATTRX_SLOW=1")
def test_train_and_extract_roundtrip(tmp_path, tax):
    cat = tax.categories["smartphone"]
    cards = [Card(id=str(i), title=f"Смартфон Samsung Galaxy A{i} 8/256 ГБ синий") for i in range(10, 26)]
    ex = [(c.full_text(), ner.weak_spans(c, rules.extract(c, cat, tax), None, tax, cat)) for c in cards]
    ner.train(ex, list(cat.attributes), tmp_path, epochs=15, lr=3e-4)
    m = ner.NerExtractor.load(tmp_path)
    got = {c.attribute for c in m.extract(Card(id="t", title="Смартфон Samsung Galaxy A99 8/256 ГБ синий"), cat, tax)}
    assert "brand" in got
