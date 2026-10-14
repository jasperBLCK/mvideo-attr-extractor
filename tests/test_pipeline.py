from attrx.config import Settings
from attrx.pipeline import Extractor
from attrx.schema import Card


def ex():
    return Extractor(settings=Settings(llm_enabled=False))


def get(result, attr):
    return next(a for a in result.attributes if a.attribute == attr)


def test_smartphone_full():
    r = ex().extract(Card(id="1", title="Смартфон Samsung Galaxy A55 8/256 ГБ голубой",
                          description="Дисплей 6.6 дюйма, батарея 5000 mAh"))
    assert r.category == "smartphone"
    assert get(r, "brand").value == "Samsung"
    assert get(r, "storage").value == 256
    assert get(r, "screen").value == 6.6
    assert get(r, "battery").value == 5000
    assert get(r, "color").value == "синий"
    assert all(a.status == "accepted" for a in r.attributes)


def test_ram_not_confused_with_storage():
    r = ex().extract(Card(id="2", title="Ноутбук ASUS Vivobook 15 Intel Core i5-1235U 16 ГБ SSD 512 ГБ", category="Ноутбуки"))
    assert get(r, "ram").value == 16
    assert get(r, "storage").value == 512


def test_no_hallucination_when_absent():
    r = ex().extract(Card(id="3", title="Смартфон vivo Y36 8/256 ГБ"))
    for attr in ("screen", "battery", "color"):
        a = get(r, attr)
        assert a.value is None and a.status == "missing"


def test_seller_conflict_goes_to_review():
    r = ex().extract(Card(id="4", title="Смартфон Infinix HOT 40i 4/128 ГБ", seller_attrs={"Объем памяти": "256 ГБ"}))
    a = get(r, "storage")
    assert a.status == "review"
    assert any("конфликт" in w for w in r.warnings)


def test_multiple_colors_in_description_not_accepted():
    r = ex().extract(Card(id="5", title="Смартфон Samsung Galaxy S24", description="Расцветки: черный, серый, фиолетовый."))
    assert get(r, "color").status != "accepted"


def test_unknown_category_falls_back_to_generic():
    r = ex().extract(Card(id="6", title="Умная колонка Яндекс Станция Мини", description="Цвет: розовый"))
    assert r.category == "generic"
    assert get(r, "color").value == "розовый"


def test_sample_quality_smoke():
    from attrx.config import ROOT
    from attrx.eval.metrics import evaluate
    from attrx.io import load_cards, load_gold

    e = ex()
    cards = load_cards(ROOT / "data/sample/cards.jsonl")
    m = evaluate(list(e.extract_many(cards)), load_gold(ROOT / "data/sample/gold.jsonl"), e.tax)
    assert m["micro"]["precision"] >= 0.95
    assert m["coverage"] >= 0.8
