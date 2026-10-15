"""Оценка ответа LLM в бенчмарке: валидность JSON, найденность цитаты, заземление, сверка с эталоном."""
from attrx.llm.bench import _new_counts, _rates, score_card
from attrx.schema import Card
from attrx.taxonomy import load_taxonomy

TAX = load_taxonomy()
PHONE = TAX.categories["smartphone"]
CARD = Card(id="b1", title="Смартфон Xiaomi Redmi 15 8/256GB", description="Диагональ 6,9 дюйма. Аккумулятор 7000 мА·ч.")


def test_score_card_counts_grounding_and_gold():
    attrs = ["screen", "battery", "storage"]
    data = {"screen": {"evidence": "Диагональ 6,9 дюйма", "value": "6.9 дюйма"},           # верно и заземлено
            "battery": {"evidence": "Аккумулятор 5000 мА·ч", "value": "5000 мА·ч"},     # цитаты нет в тексте
            "storage": {"evidence": None, "value": None}}                                # промолчала
    gold = {"b1": {"screen": "6.9", "battery": "7000 мАч", "storage": "256 ГБ"}}
    m = _new_counts()
    row = score_card(CARD, PHONE, attrs, data, gold, TAX, m)
    assert row["values"] == {"screen": 6.9} and row["rejected"] == ["battery"]
    assert (m["json_ok"], m["returned"], m["quote_found"], m["grounded"]) == (1, 2, 1, 1)
    assert (m["gold_slots"], m["gold_returned"], m["correct"]) == (3, 1, 1)
    r = _rates(m, 1)
    assert r["gold_precision"] == 1.0 and r["grounded_rate"] == 0.5 and r["quote_found_rate"] == 0.5
