"""Заземление значений моделей (выравнивание цитаты с текстом) и инвариант доверия к LLM."""
import pytest

from attrx.config import Settings
from attrx.pipeline import Extractor
from attrx.schema import Candidate, Card
from attrx.taxonomy import load_taxonomy
from attrx.verify import align, ground, grounded


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


CARD = Card(id="g", title="Смартфон TECNO Spark 20 Pro 8/256GB",
            description="Оснащён 6,43-дюймовым AMOLED-экраном. Цвет корпуса — Moonlit Black.")


def cand(attr, value, quote, source="llm"):
    return Candidate(attribute=attr, value=value, raw=quote, source=source, score=0.8)


def test_align_tolerates_inflection_punctuation_and_spaces():
    text = CARD.full_text()
    s, e, score = align("6,43-дюймовый", text)  # модель процитировала в другой словоформе
    assert text[s:e] == "6,43-дюймовым" and score >= 0.85
    s, e, _ = align("Moonlit-black", text)
    assert text[s:e] == "Moonlit Black"
    s, e, _ = align("цвет корпуса Moonlit Black", text)  # «—» в тексте и его нет в цитате
    assert text[s:e] == "Цвет корпуса — Moonlit Black"


def test_align_rejects_fabricated_quote_and_other_numbers():
    text = CARD.full_text()
    assert align("экран 6.7 дюйма", text) is None  # числа нет в тексте — цитата выдумана
    assert align("лунный чёрный", text) is None
    assert align("6,45-дюймовым", text) is None  # число отличается — не заземлено


def test_grounded_inflected_quote_is_accepted_and_reanchored(tax):
    spec = tax.categories["smartphone"].attributes["screen"]
    g = ground(cand("screen", 6.43, "6,43-дюймовый"), CARD, spec, tax.dictionaries)
    assert g is not None and g.raw == "6,43-дюймовым"
    assert CARD.full_text()[g.span[0]:g.span[1]] == "6,43-дюймовым"


def test_value_must_follow_from_aligned_span(tax):
    attrs = tax.categories["smartphone"].attributes
    # цитата настоящая, но значение из неё не следует
    assert not grounded(cand("screen", 6.7, "6,43-дюймовый"), CARD, attrs["screen"], tax.dictionaries)
    assert not grounded(cand("color", "белый", "Moonlit Black"), CARD, attrs["color"], tax.dictionaries)
    assert grounded(cand("color", "чёрный", "Moonlit Black"), CARD, attrs["color"], tax.dictionaries)
    # продавец не проверяется (его поле — часть карточки)
    assert grounded(cand("color", "белый", "Цвет: белый", source="seller"), CARD, attrs["color"], tax.dictionaries)


class FakeLLM:
    enabled = True
    s = Settings(llm_enabled=True, llm_prefilter=False)

    def __init__(self, response):
        self.response, self.calls = response, 0

    def complete_json(self, system, user, schema, name="result", *, max_tokens=None):
        self.calls += 1
        return self.response


def test_llm_only_value_goes_to_review():
    """Инвариант доверия: значение, которое нашла ТОЛЬКО LLM, не принимается автоматически при любой уверенности."""
    card = Card(id="t", title="Смартфон TECNO Spark 20 Pro 8/256GB", description="Цвет корпуса — Moonlit Black.")
    llm = FakeLLM({"color": {"value": "чёрный", "evidence": "Moonlit Black"}})
    ex = Extractor(settings=Settings(llm_enabled=True, ml_enabled=False, confidence_threshold=0.0), client=llm)
    r = ex.extract(card, sources={"llm"})
    color = next(a for a in r.attributes if a.attribute == "color")
    assert color.value == "чёрный" and color.sources == ["llm"]
    assert color.status == "review"  # даже при пороге 0
    s, e = color.evidence_span
    assert card.full_text()[s:e] == "Moonlit Black"


def test_llm_value_confirmed_by_rules_can_be_accepted():
    card = Card(id="t2", title="Смартфон TECNO Spark 20 Pro 8/256GB чёрный")
    llm = FakeLLM({"color": {"value": "чёрный", "evidence": "чёрный"}})
    ex = Extractor(settings=Settings(llm_enabled=True, ml_enabled=False, confidence_threshold=0.5), client=llm)
    color = next(a for a in ex.extract(card, use_llm=True).attributes if a.attribute == "color")
    assert "rules" in color.sources and color.status == "accepted"


def test_cyrillic_lookalikes_and_normalizer_prefix_are_grounded(tax):
    card = Card(id="cpu", title="Ноутбук Apple MacBook Air 13 Чип М4 16/512",  # «М» — кириллица
                description="Процессор Pentium Gold 7505 для офиса")
    cpu = tax.categories["laptop"].attributes["cpu"]
    g = ground(cand("cpu", "Apple M4", "Чип M4"), card, cpu, tax.dictionaries)  # в цитате латиница
    assert g is not None and g.raw == "Чип М4"
    # нормализатор добавил производителя («Intel»), которого нет в тексте: значение следует из фрагмента
    assert grounded(cand("cpu", "Intel Pentium Gold 7505", "Pentium Gold 7505"), card, cpu, tax.dictionaries)
    assert not grounded(cand("cpu", "Intel Celeron N4020", "Celeron N4020"), card, cpu, tax.dictionaries)


def test_llm_corroborating_rules_can_be_accepted(tax):
    """LLM подтверждает слабое значение правил (noisy-OR) — так LLM помогает принятым значениям; одна LLM — нет."""
    from attrx.confidence import aggregate_full

    ex = Extractor(settings=Settings(llm_enabled=False, ml_enabled=False, confidence_threshold=0.7), calibrated=False)
    spec = tax.categories["smartphone"].attributes["color"]
    rules_c = Candidate(attribute="color", value="чёрный", raw="черный", source="rules", score=0.6)
    llm_c = Candidate(attribute="color", value="чёрный", raw="черный", source="llm", score=0.8)
    conf, status = ex._finalize(aggregate_full([rules_c, llm_c]), spec, "smartphone", "color")
    assert status == "accepted" and conf > 0.6
    _, status = ex._finalize(aggregate_full([llm_c.model_copy(update={"score": 0.99})]), spec, "smartphone", "color")
    assert status == "review"
    _, status = ex._finalize(aggregate_full([rules_c]), spec, "smartphone", "color")
    assert status == "review"  # одних правил с 0.6 мало — подтверждение LLM и дало принятие


def test_thousands_separator_numbers_are_grounded(tax):
    card = Card(id="bat", title="Смартфон TECNO Spark 20 Pro", description="Аккумулятор ёмкостью 5 000 мАч, экран 6,78\"")
    battery = tax.categories["smartphone"].attributes["battery"]
    for quote in ("5 000 мАч", "5000 мАч"):  # цитата с пробелом и без — текст с пробелом
        g = ground(cand("battery", 5000.0, quote), card, battery, tax.dictionaries)
        assert g is not None and card.full_text()[g.span[0]:g.span[1]] == "5 000 мАч"
    assert not grounded(cand("battery", 6000.0, "6000 мАч"), card, battery, tax.dictionaries)
    assert not grounded(cand("battery", 5000.0, "5000 мАч"), Card(id="n", title="Смартфон", description="5 500 мАч"),
                        battery, tax.dictionaries)


def test_brand_outside_dictionary_grounded_only_when_in_fragment(tax):
    brand = tax.categories["generic"].attributes["brand"]
    card = Card(id="br", title="Стул Праздничная Мебельщица Дубрава 2 шт")  # такого бренда нет в справочнике
    assert tax.dictionaries["brands"].get("Праздничная Мебельщица") is None
    g = ground(cand("brand", "Праздничная Мебельщица", "Праздничная Мебельщица", source="ml"), card, brand,
               tax.dictionaries)
    assert g is not None and g.raw == "Праздничная Мебельщица"
    # цитата настоящая, но значение в ней не стоит — отказ
    assert not grounded(cand("brand", "Мебельщица Плюс", "Праздничная Мебельщица", source="ml"), card, brand,
                        tax.dictionaries)


def test_corroboration_never_lowers_confidence(tax):
    """Согласие ещё одного источника не понижает уверенность: у сочетания «правила + LLM» нет своей кривой, но
    итог — максимум из общей кривой и калиброванной уверенности того же значения без LLM."""
    from attrx.confidence import Calibrator, aggregate_full

    spec = tax.categories["smartphone"].attributes["storage"]
    cal = Calibrator({"groups": {
        "smartphone/storage/rules": {"x": [0.0, 0.8], "y": [0.5, 0.97]},  # правила по памяти смартфона — надёжны
        "*|number": {"x": [0.0, 0.95], "y": [0.3, 0.82]},  # общая кривая чисел — заметно ниже
    }})
    rules_c = Candidate(attribute="storage", value=128.0, raw="128 ГБ", source="rules", score=0.94)
    llm_c = Candidate(attribute="storage", value=128.0, raw="128 ГБ", source="llm", score=0.9)
    alone = cal.monotone(aggregate_full([rules_c]), spec, "smartphone", "storage")
    both = cal.monotone(aggregate_full([rules_c, llm_c]), spec, "smartphone", "storage")
    assert alone == 0.97 and both >= alone
    # с конфликтом (в описании ещё «256 ГБ») — то же: согласие LLM не уводит слот ниже
    other = Candidate(attribute="storage", value=256.0, raw="256 ГБ", source="rules", score=0.45)
    a1 = cal.monotone(aggregate_full([rules_c, other]), spec, "smartphone", "storage")
    a2 = cal.monotone(aggregate_full([rules_c, other, llm_c]), spec, "smartphone", "storage")
    assert a2 >= a1
    # LLM, согласная с правилами, — не конфликт (одна группа значения), а LLM в одиночку — на проверку
    agg = aggregate_full([rules_c, llm_c])
    assert not agg.conflict and agg.sources == ["llm", "rules"]
    ex = Extractor(settings=Settings(llm_enabled=False, ml_enabled=False, confidence_threshold=0.5))
    ex.calibrator = cal
    assert ex._finalize(agg, spec, "smartphone", "storage")[1] == "accepted"
    assert ex._finalize(aggregate_full([llm_c]), spec, "smartphone", "storage")[1] == "review"


def test_agreeing_model_written_differently_is_not_a_conflict(tax):
    from attrx.confidence import aggregate_full
    from attrx.pipeline import value_key

    spec = tax.categories["smartphone"].attributes["model"]
    a = Candidate(attribute="model", value="Redmi Note 13", raw="Redmi Note 13", source="rules", score=0.9)
    b = Candidate(attribute="model", value="Redmi Note 13 8/256GB", raw="Redmi Note 13 8/256GB", source="llm", score=0.8)
    agg = aggregate_full([a, b], value_key(spec))
    assert not agg.conflict and agg.value == "Redmi Note 13" and agg.sources == ["llm", "rules"]


@pytest.mark.parametrize("phrase,color", [
    ("Корпус выполнен в серебристом цвете", "серебристый"),
    ("Модель в черном корпусе", "чёрный"),
    ("Панель с белым покрытием", "белый"),
    ("Вставки из синих материалов", "синий"),
])
def test_inflected_colors_are_matched_and_grounded(tax, phrase, color):
    from attrx.extractors import rules

    card = Card(id="col", title="Чайник Bosch TWK1234", description=phrase + ".")
    spec = tax.categories["kettle"].attributes["color"]
    # справочник находит падежную форму (дальше выученное по train соответствие может уточнить значение:
    # «синий» в описании чайника чаще про подсветку — это уже enum_map, а не сопоставление)
    eh = rules.enum_hits(rules.Ctx.of(card, "kettle"), spec, tax.dictionaries["colors"], [])
    assert eh is not None and eh.canons == [color]
    word = phrase.split()[-2] if color != "синий" else "синих"
    assert grounded(cand("color", color, word), card, spec, tax.dictionaries)  # LLM процитировала падежную форму
