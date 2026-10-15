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


def test_conflict_accept_is_one_constant_for_pipeline_and_calibration():
    from attrx import confidence, pipeline
    from attrx.confidence import Aggregate
    from attrx.eval import calibrate
    from attrx.schema import Candidate

    assert pipeline.CONFLICT_ACCEPT is confidence.CONFLICT_ACCEPT and calibrate.CONFLICT_ACCEPT is confidence.CONFLICT_ACCEPT
    e = Extractor(settings=Settings(llm_enabled=False, ml_enabled=False, confidence_threshold=0.7), calibrated=False)
    spec = e.tax.categories["smartphone"].attributes["storage"]
    cand = Candidate(attribute="storage", value=128.0, raw="128 ГБ", source="rules", score=0.9)
    alt = [{"value": 256.0, "confidence": 0.3, "sources": ["rules"]}]  # спор внутри одного источника
    for conf in (0.8, 0.9):
        _, status = e._finalize(Aggregate(128.0, conf, [cand], True, alt), spec, "smartphone", "storage")
        point = calibrate.Point(card="x", keys=[], raw=conf, ok=True, conflict=True, llm_only=False)
        assert (status == "accepted") == calibrate._accepted(point, conf, 0.7) == (conf >= confidence.CONFLICT_ACCEPT)


class StubNer:
    """Как NerExtractor: числа с единицей -> кандидаты source=ml; batches — размеры пачек одного прогона модели."""

    def __init__(self):
        self.batches = []

    def extract(self, card, category, tax):
        return self.extract_batch([(card, category)], tax)[0]

    def extract_batch(self, items, tax):
        import re

        from attrx.schema import Candidate

        self.batches.append(len(items))
        out = []
        for card, cat in items:
            found = []
            for m in re.finditer(r"(\d+(?:[.,]\d+)?)\s*(ГБ|дюйм\w*|мА·ч)", card.full_text()):
                attr = {"ГБ": "storage", "мА·ч": "battery"}.get(m.group(2), "screen")
                if attr in cat.attributes:
                    found.append(Candidate(attribute=attr, value=float(m.group(1).replace(",", ".")), raw=m.group(0),
                                           source="ml", score=0.8, span=m.span()))
            out.append(found)
        return out


def test_extract_many_runs_ner_once_per_pack_with_same_results():
    class Off:
        enabled = False
        calls = 0

    cards = [Card(id="a", title="Смартфон Infinix HOT 40i 4/128 ГБ", seller_attrs={"Объем памяти": "256 ГБ"}),
             Card(id="b", title="Смартфон Samsung Galaxy A55 8/256 ГБ голубой", description="Дисплей 6.6 дюйма, 5000 мА·ч"),
             Card(id="c", title="Смартфон vivo Y36", description="Экран — 6,64 дюйма, батарея 5000 мА·ч."),
             Card(id="d", title="Ноутбук ASUS Vivobook 15 Intel Core i5-1235U 16 ГБ SSD 512 ГБ", category="Ноутбуки")]
    e = Extractor(settings=Settings(llm_enabled=False, ml_enabled=False), client=Off())
    e.ml = StubNer()
    single = [e.extract(c) for c in cards]
    weak_cards = len(e.ml.batches)  # по одному прогону на карточку со слабыми атрибутами
    e.ml.batches = []
    many = list(e.extract_many(cards))
    assert e.ml.batches == [weak_cards] and weak_cards > 1  # один прогон NER на всю пачку
    assert [r.card_id for r in many] == [c.id for c in cards]
    for a, b in zip(single, many, strict=True):
        assert a.model_dump(exclude={"timings_ms"}) == b.model_dump(exclude={"timings_ms"})
    assert any("ml" in x.sources for r in many for x in r.attributes)  # NER действительно добавил значения
    assert all("ml" in r.timings_ms for r in many if any("ml" in x.sources for x in r.attributes))
