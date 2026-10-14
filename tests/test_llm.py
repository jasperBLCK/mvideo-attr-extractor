"""LLM мокается: проверяем маршрутизацию, заземление и защиту от выдуманных значений."""
from attrx.config import Settings
from attrx.genui.composer import compose_llm
from attrx.pipeline import Extractor
from attrx.schema import Card


class FakeLLM:
    enabled = True

    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.prompts = []

    def complete_json(self, system, user, schema, name="result"):
        self.calls += 1
        self.prompts.append((user, schema))
        return self.response


CARD = Card(id="x", title="Смартфон TECNO Spark 20 Pro 8/256GB", description="Большой экран на шесть и восемь десятых дюйма.")


def test_llm_only_for_unresolved_attrs():
    llm = FakeLLM({})
    Extractor(settings=Settings(llm_enabled=True), client=llm).extract(CARD)
    _, schema = llm.prompts[0]
    assert "storage" not in schema["properties"]  # закрыто правилами — в LLM не отправляем
    assert "screen" in schema["properties"]


def test_hallucinated_value_rejected():
    llm = FakeLLM({"battery": {"value": "5000 мАч", "evidence": "аккумулятор 5000 мАч"}, "screen": {"value": None, "evidence": None},
                   "color": {"value": None, "evidence": None}})
    r = Extractor(settings=Settings(llm_enabled=True), client=llm).extract(CARD)
    bat = next(a for a in r.attributes if a.attribute == "battery")
    assert bat.value is None and bat.status == "missing"
    assert any("незаземлённое" in w for w in r.warnings)


def test_grounded_llm_value_kept():
    card = Card(id="y", title="Смартфон TECNO Spark 20 Pro 8/256GB", description="Цвет корпуса — Moonlit Black")
    llm = FakeLLM({"color": {"value": "чёрный", "evidence": "Moonlit Black"}})
    r = Extractor(settings=Settings(llm_enabled=True), client=llm).extract(card)
    color = next(a for a in r.attributes if a.attribute == "color")
    assert color.value == "чёрный"


def test_genui_rejects_invalid_spec():
    r = Extractor(settings=Settings(llm_enabled=False)).extract(CARD)
    bad = FakeLLM({"title": "x", "layout": "column", "components": [{"type": "script", "text": "<script>alert(1)</script>"}]})
    spec, warn = compose_llm(r, "покажи всё", bad)
    assert spec.generated_by == "rules" and warn


def test_genui_valid_llm_spec_sanitized():
    r = Extractor(settings=Settings(llm_enabled=False)).extract(CARD)
    good = FakeLLM({"title": "Проверка", "layout": "grid", "components": [
        {"type": "evidence", "title": "Цитаты", "text": "<b>x</b>", "filter": None, "attributes": ["storage", "nonexistent"], "level": None}]})
    spec, warn = compose_llm(r, "покажи доказательства", good)
    assert spec.generated_by == "llm" and warn is None
    assert spec.components[0].attributes == ["storage"]
    assert "<" not in spec.components[0].text
