"""LLM мокается: проверяем маршрутизацию, заземление и защиту от выдуманных значений."""
from attrx.config import Settings
from attrx.genui.composer import compose_llm
from attrx.pipeline import Extractor
from attrx.schema import Card


class FakeLLM:
    enabled = True
    # маршрутизацию и заземление проверяем без префильтра «есть ли атрибут в тексте» (он проверен отдельно)
    s = Settings(llm_enabled=True, llm_prefilter=False)

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
    # в тексте диагональ словами, модель «вспоминает» 6.7 и цитирует то, чего в карточке нет
    llm = FakeLLM({"screen": {"value": "6.7 дюйма", "evidence": "экран 6.7 дюйма"}, "color": {"value": None, "evidence": None}})
    r = Extractor(settings=Settings(llm_enabled=True), client=llm).extract(CARD)
    screen = next(a for a in r.attributes if a.attribute == "screen")
    assert screen.value is None and screen.status == "missing"
    assert any("незаземлённое" in w for w in r.warnings)


def test_llm_not_called_when_text_has_nothing_to_extract():
    from attrx.extractors import llm as llm_extractor
    from attrx.taxonomy import load_taxonomy

    tax = load_taxonomy()
    llm = FakeLLM({"battery": {"value": "5000 мАч", "evidence": "аккумулятор 5000 мАч"}})
    llm.s = Settings(llm_enabled=True, llm_prefilter=True)
    # об аккумуляторе и цвете в карточке ни слова, диагональ — словами (заземлить число без цифр нельзя): не платим
    cands, warnings = llm_extractor.extract(CARD, tax.categories["smartphone"], tax, ["battery", "color", "screen"], llm)
    assert llm.calls == 0 and cands == [] and warnings == []


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


def test_llm_used_only_when_llm_was_actually_asked():
    """llm_used — был ли запрос к LLM (или ответ из кэша) по карточке, а не «не было ошибки»."""
    asked = FakeLLM({})
    r = Extractor(settings=Settings(llm_enabled=True), client=asked).extract(CARD)
    assert asked.calls == 1 and r.llm_used is True  # слабые атрибуты есть, префильтр выключен — вызов был

    skipped = FakeLLM({})
    skipped.s = Settings(llm_enabled=True, llm_prefilter=True)
    # экран — словами, о батарее и цвете ни слова: префильтр отсеял все слабые атрибуты, вызова не было
    r = Extractor(settings=Settings(llm_enabled=True), client=skipped).extract(CARD)
    assert skipped.calls == 0 and r.llm_used is False and r.warnings == []

    class Down(FakeLLM):
        def complete_json(self, *a, **k):
            from attrx.llm.client import LLMUnavailable

            raise LLMUnavailable("нет соединения")

    r = Extractor(settings=Settings(llm_enabled=True), client=Down({})).extract(CARD)
    assert r.llm_used is False and r.warnings == ["llm: нет соединения"]


def test_out_of_dictionary_llm_value_is_reported_as_dictionary_candidate():
    card = Card(id="h", title="Смартфон TECNO Spark 20 Pro 8/256GB",
                description="Корпус с эффектом хамелеон переливается на солнце.")
    llm = FakeLLM({"color": {"value": "хамелеон", "evidence": "эффектом хамелеон"}})
    r = Extractor(settings=Settings(llm_enabled=True), client=llm).extract(card)
    color = next(a for a in r.attributes if a.attribute == "color")
    assert color.value is None and color.status == "missing"  # вне справочника — в каталог не идёт
    note = next(w for w in r.warnings if w.startswith("кандидат в справочник"))
    assert note == ('кандидат в справочник colors: color = "хамелеон" (LLM, цитата «эффектом хамелеон») '
                    "— в справочнике нет, отброшено")
    # цитаты в тексте нет — это не кандидат, а выдумка: то же сообщение, что у незаземлённых значений
    llm = FakeLLM({"color": {"value": "хамелеон", "evidence": "цвет хамелеон"}})
    r = Extractor(settings=Settings(llm_enabled=True), client=llm).extract(CARD)
    assert "llm: отброшено незаземлённое значение color='хамелеон'" in r.warnings
    assert not any(w.startswith("кандидат в справочник") for w in r.warnings)


def test_extract_many_runs_llm_for_pack_in_parallel_with_same_results():
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    class MapLLM(FakeLLM):
        """Как LLMClient.map: задачи в нескольких потоках, порядок результатов сохраняется."""

        def __init__(self):
            super().__init__({})
            self.lock, self.threads, self.mapped = threading.Lock(), set(), 0

        def complete_json(self, system, user, schema, name="result"):
            with self.lock:
                self.calls += 1
                self.threads.add(threading.get_ident())
            time.sleep(0.02)  # запросы пересекаются по времени
            if "Moonlit Black" in user:
                return {"color": {"value": "чёрный", "evidence": "Moonlit Black"}}
            if "хамелеон" in user:
                return {"color": {"value": "хамелеон", "evidence": "эффектом хамелеон"}}
            return {"screen": {"value": "6.9 дюйма", "evidence": "экран 6.9 дюйма"}}  # выдумка

        def map(self, fn, items, concurrency=None):
            self.mapped += 1
            with ThreadPoolExecutor(max_workers=4) as pool:
                return list(pool.map(fn, items))

    cards = [Card(id=f"m{i}", title="Смартфон TECNO Spark 20 Pro 8/256GB", description=d) for i, d in enumerate([
        "Цвет корпуса — Moonlit Black", "Корпус с эффектом хамелеон.", "Большой экран.", "Цвет корпуса — Moonlit Black"])]
    llm = MapLLM()
    ex = Extractor(settings=Settings(llm_enabled=True), client=llm)
    single = [ex.extract(c) for c in cards]
    many = list(ex.extract_many(cards))
    assert llm.mapped == 1 and len(llm.threads) > 1  # одна пачка, запросы — в нескольких потоках
    assert [r.card_id for r in many] == [c.id for c in cards]  # порядок как у входа
    for a, b in zip(single, many, strict=True):
        assert a.model_dump(exclude={"timings_ms"}) == b.model_dump(exclude={"timings_ms"})
    assert next(x for x in many[0].attributes if x.attribute == "color").value == "чёрный"
    assert any(w.startswith("кандидат в справочник") for w in many[1].warnings)
    assert all(r.llm_used for r in many) and many[0].timings_ms["llm"] > 0
