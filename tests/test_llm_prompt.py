"""Промпт из схемы категории (тип, единица, справочник допустимых значений), разбор ответа и стоимость."""
import pytest

from attrx.extractors import llm as llm_extractor
from attrx.llm.cost import cost_table_md, hourly_cost, load_prices, per_1000_cards
from attrx.schema import Card
from attrx.taxonomy import load_taxonomy

TAX = load_taxonomy()
PHONE = TAX.categories["smartphone"]
CARD = Card(id="p1", title="Смартфон TECNO Spark 20 Pro 8/256GB",
            description="Экран 6,78 дюйма. Цвет корпуса — Moonlit Black. Аккумулятор 5000 мА·ч.",
            seller_attrs={"Бренд": "TECNO"})


class FakeLLM:
    enabled = True

    def __init__(self, response):
        self.response, self.calls = response, []

    def complete_json(self, system, user, schema, name="result", *, max_tokens=None):
        self.calls.append((system, user, schema))
        return self.response


def test_prompt_lists_enum_values_units_and_types():
    system, user, schema = llm_extractor.build_request(CARD, PHONE, TAX, ["color", "screen", "model"])
    colors = TAX.dictionaries["colors"].canon
    assert all(c in user for c in colors)  # справочник допустимых значений — в промпте
    assert "Диагональ экрана" in user and "дюйм" in user and "число" in user
    assert "Модель" in user and "строка" in user
    assert "Moonlit Black" in user and "Бренд: TECNO" in user  # текст карточки и атрибуты продавца дословно
    assert "null" in system and "ДОСЛОВНАЯ" in system
    assert schema["required"] == ["color", "screen", "model"] and schema["additionalProperties"] is False
    assert list(schema["properties"]["color"]["properties"]) == ["evidence", "value"]  # сначала цитата
    # по умолчанию справочник не зашит в схему: к канону значение приводит нормализация
    assert "enum" not in schema["properties"]["color"]["properties"]["value"]


def test_schema_enum_is_opt_in(monkeypatch):
    monkeypatch.setenv("LLM_SCHEMA_ENUM", "true")
    _, _, schema = llm_extractor.build_request(CARD, PHONE, TAX, ["color", "screen"])
    colors = TAX.dictionaries["colors"].canon
    assert schema["properties"]["color"]["properties"]["value"]["enum"] == [*colors, None]
    assert "enum" not in schema["properties"]["screen"]["properties"]["value"]


def test_large_dictionary_lists_only_values_found_in_text(monkeypatch):
    monkeypatch.setenv("LLM_ENUM_VALUES_MAX", "3")
    _, user, schema = llm_extractor.build_request(CARD, PHONE, TAX, ["color", "brand"])
    line = next(x for x in user.splitlines() if x.startswith("- color"))
    assert "найденных в тексте" in line and "чёрный" in line and "белый" not in line
    assert "enum" not in schema["properties"]["color"]["properties"]["value"]


def test_few_shot_is_optional():
    plain, *_ = llm_extractor.build_request(CARD, PHONE, TAX, ["color"], few_shot=False)
    shot, *_ = llm_extractor.build_request(CARD, PHONE, TAX, ["color"], few_shot=True)
    assert "Пример" not in plain and "Пример" in shot


def test_extract_normalizes_rejects_out_of_dictionary_and_locates_quote():
    llm = FakeLLM({"color": {"evidence": "Moonlit Black", "value": "чёрный"},
                   "screen": {"evidence": "Экран 6,78 дюйма", "value": "6,78 дюйма"},
                   "battery": {"evidence": "мА·ч", "value": "бирюзовый"},
                   "model": {"evidence": None, "value": None}})
    notes: list[str] = []
    cands, warnings = llm_extractor.extract(CARD, PHONE, TAX, ["color", "screen", "battery", "model"], llm, notes=notes)
    by = {c.attribute: c for c in cands}
    assert warnings == [] and set(by) == {"color", "screen"}
    assert by["color"].value == "чёрный" and by["screen"].value == 6.78 and by["color"].source == "llm"
    start, end = by["color"].span
    assert CARD.full_text()[start:end] == "Moonlit Black"
    assert any("battery" in n for n in notes)  # отброшенное значение видно для разбора/пополнения справочника
    # проверяемые типы (число, справочник) с цитатой — выше по уверенности, чем свободные строки
    assert by["screen"].score == llm_extractor.LLM_VERIFIED_SCORE > llm_extractor.LLM_BASE_SCORE


def test_llm_only_value_policy_in_pipeline():
    """Инвариант доверия (pipeline.py): значение ТОЛЬКО от LLM — всегда на проверку, даже заземлённое и уверенное."""
    from attrx.config import Settings
    from attrx.pipeline import Extractor

    card = Card(id="z", title="Смартфон TECNO Spark 20 Pro", description="Цвет корпуса — Moonlit Black. Серия Spark.")
    llm = FakeLLM({"color": {"evidence": "Moonlit Black", "value": "чёрный"},
                   "model": {"evidence": "Spark 20 Pro", "value": "Spark 20 Pro"}})
    # только LLM-источник: проверяем политику уверенности именно для значений, которые нашла одна LLM
    r = Extractor(settings=Settings(llm_enabled=True, ml_enabled=False), client=llm).extract(card, sources={"llm"})
    by = {a.attribute: a for a in r.attributes}
    assert by["color"].value == "чёрный" and by["color"].status == "review" and by["color"].sources == ["llm"]
    assert by["model"].sources == ["llm"] and by["model"].status == "review"


def test_extract_reports_unavailable_llm_as_warning():
    class Down(FakeLLM):
        def complete_json(self, *a, **k):
            raise llm_extractor.LLMUnavailable("нет соединения")

    cands, warnings = llm_extractor.extract(CARD, PHONE, TAX, ["color"], Down({}))
    assert cands == [] and warnings == ["llm: нет соединения"]


def test_extract_batch_keeps_order():
    llm = FakeLLM({"color": {"evidence": "Moonlit Black", "value": "чёрный"}})
    jobs = [(CARD, PHONE, ["color"]), (CARD, PHONE, []), (CARD, PHONE, ["color"])]
    res = llm_extractor.extract_batch(jobs, TAX, llm)
    assert [len(c) for c, _ in res] == [1, 0, 1] and len(llm.calls) == 2


def test_fit_text_keeps_relevant_sentences_verbatim():
    filler = "Стильный дизайн порадует владельца. " * 60
    desc = "Смартфон для всей семьи. " + filler + "Ёмкость аккумулятора 5000 мА·ч. " + filler + "Диагональ 6,7 дюйма."
    out = llm_extractor.fit_text(desc, [PHONE.attributes["battery"], PHONE.attributes["screen"]], 400)
    assert len(out) <= 450
    assert "Ёмкость аккумулятора 5000 мА·ч." in out and "Диагональ 6,7 дюйма." in out
    assert out.startswith("Смартфон для всей семьи.")
    assert all(p in desc for p in out.split(" … "))  # фрагменты — дословные куски описания


def test_locate_is_case_and_yo_insensitive():
    text = "Цвет: Чёрный оникс\nЭкран  6,7 дюйма"
    assert llm_extractor.locate("черный оникс", text) == (6, 18)
    assert llm_extractor.locate("экран 6,7", text) == (19, 29)
    assert llm_extractor.locate("нет такого", text) is None


def test_cost_per_1000_cards_every_card_vs_hybrid():
    prices = load_prices()
    st = {"calls": 10, "prompt_tokens": 15000, "completion_tokens": 1500, "latency_s": {"avg": 4.0}}
    rep = per_1000_cards(st, 10, llm_share=0.3, prices=prices)
    rows = {r["provider"]: r for r in rep["rows"]}
    assert rep["tokens_per_call"] == {"prompt": 1500.0, "completion": 150.0}
    # GigaChat Lite: (1500 + 150) токенов × 0,065 ₽ / 1000 = 0,10725 ₽ за карточку
    assert rows["gigachat-lite"]["every_card_rub_per_1000"] == pytest.approx(107.25, abs=0.01)
    assert rows["gigachat-lite"]["hybrid_rub_per_1000"] == pytest.approx(32.18, abs=0.01)
    assert rows["organizers"]["every_card_rub_per_1000"] is None  # тариф неизвестен — честно «неизвестно»
    h = prices["providers"]["onprem-gtx1060"]["hourly"]
    assert rows["onprem-gtx1060"]["every_card_rub_per_1000"] == pytest.approx(hourly_cost(h) * 4.0 / 3.6, abs=0.01)
    assert "неизвестно" in cost_table_md(rep)


def test_relevant_attrs_skips_what_text_never_mentions():
    card = Card(id="r", title="Смартфон Apple iPhone 15 128 ГБ", description="Батарея на весь день. Цвет корпуса — Pink.")
    got = llm_extractor.relevant_attrs(card, PHONE, TAX, ["battery", "screen", "color", "model"])
    assert got == ["color", "model"]  # батарея без числа и экран не упомянуты — LLM их не спрашиваем
    card2 = Card(id="r2", title="Смартфон", description="Аккумулятор, которого хватит надолго: 5 000 мА·ч.")
    assert llm_extractor.relevant_attrs(card2, PHONE, TAX, ["battery"]) == ["battery"]


def test_string_attrs_get_catalog_format_hints():
    _, user, _ = llm_extractor.build_request(CARD, PHONE, TAX, ["model"])
    line = next(x for x in user.splitlines() if x.startswith("- model"))
    assert "без бренда" in line  # модель в каталоге — без бренда, памяти и цвета
    laptop = TAX.categories["laptop"]
    _, user, _ = llm_extractor.build_request(CARD, laptop, TAX, ["cpu"])
    assert "Intel Core i5-1235U" in next(x for x in user.splitlines() if x.startswith("- cpu"))
