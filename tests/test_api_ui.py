"""API интерфейса модератора: примеры карточек, разметка текста для подсветки цитат, очередь проверки
с приоритетом, решения и их отмена, сводка отчёта, справочники, статика без сборки."""
from fastapi.testclient import TestClient

from attrx.api import app as api
from attrx.genui.composer import compose_intent
from attrx.genui.schema import COMPONENT_JSON_SCHEMA
from attrx.schema import AttributeValue, Card, ExtractionResult

client = TestClient(api.app)


class _LLMOff:
    """LLM гарантированно выключена, даже если в окружении есть ключ."""

    enabled = False
    calls = 0


def _av(key, conf, status="review", weight=1.0, value="x"):
    return AttributeValue(attribute=key, title=key, value=value, confidence=conf, status=status,
                          sources=["rules"], evidence=None if value is None else str(value), traffic_weight=weight)


def _res(cid, attrs, warnings=()):
    return ExtractionResult(card_id=cid, category="smartphone", category_confidence=1.0, attributes=attrs,
                            warnings=list(warnings))


def test_examples_list_filter_and_lookup():
    data = client.get("/api/examples?limit=3").json()
    assert data["source"] and 0 < len(data["items"]) <= 3
    first = data["categories"][0]
    only = client.get("/api/examples", params={"category": first["name"], "limit": 1000}).json()
    assert only["total"] == first["count"]
    card = data["items"][0]
    assert client.get(f"/api/examples/{card['id']}").json()["id"] == card["id"]
    assert client.get("/api/examples/__nope__").status_code == 404


def test_card_text_blocks_point_into_full_text():
    card = Card(id="t", title="Смартфон Apple iPhone 15 128 ГБ", description='Экран 6,1".',
                seller_attrs={"Цвет": "черный", "Память": 128})
    t = api.card_text(card)
    assert t.full == card.full_text()
    assert [b.kind for b in t.blocks] == ["title", "description", "seller"]
    assert t.full[t.blocks[0].start:t.blocks[0].end] == card.title
    key, value = t.seller_pairs[1].key_span, t.seller_pairs[1].value_span
    assert t.full[key[0]:key[1]] == "Память" and t.full[value[0]:value[1]] == "128"


def test_extract_returns_text_and_fills_ui_fields():
    body = {"card": {"id": "x1", "title": "Смартфон Samsung Galaxy A55 8/256 ГБ голубой",
                     "description": "Дисплей 6.6 дюйма, батарея 5000 mAh."}}
    data = client.post("/api/extract", json=body).json()
    assert data["text"]["full"].startswith("Смартфон Samsung")
    assert data["result"]["category_title"]
    assert data["result"]["timings_ms"]["total"] >= 0
    assert {c["type"] for c in data["ui"]["components"]} >= {"summary", "attribute_table"}


def test_ui_intent_without_llm_is_composed_by_keywords(monkeypatch):
    monkeypatch.setattr(api.extractor(), "client", _LLMOff())
    r = _res("c", [_av("color", 0.4), _av("brand", 0.95, status="accepted")])
    data = client.post("/api/ui", json={"result": r.model_dump(), "intent": "Уверенность по всем характеристикам"}).json()
    assert data["ui"]["generated_by"] == "rules" and data["ui_warning"]
    assert data["ui"]["components"][0]["type"] == "confidence_chart"


def test_compose_intent_review_with_quotes_and_empty_case():
    r = _res("c", [_av("color", 0.4), _av("brand", 0.95, status="accepted")])
    spec = compose_intent(r, "Только то, что нужно проверить, с цитатами")
    types = [c.type for c in spec.components]
    assert types[:2] == ["review_queue", "evidence"] and spec.components[1].attributes == ["color"]
    clean = _res("c", [_av("brand", 0.95, status="accepted")])
    assert compose_intent(clean, "что проверить?").components[0].type == "alert"
    assert compose_intent(clean, "просто экран").components  # без ключевых слов — шаблон


def test_review_queue_priority_feedback_and_undo(tmp_path, monkeypatch):
    preds, fb = tmp_path / "pred.jsonl", tmp_path / "feedback.jsonl"
    rows = [_res("c1", [_av("color", 0.5, weight=0.2), _av("brand", 0.9, status="accepted")]),
            _res("c2", [_av("storage", 0.3, weight=0.9, value=128.0)], ["storage: конфликт значений, альтернативы: 256.0"]),
            _res("c3", [_av("model", 0.6, weight=0.6), _av("screen", 0.0, status="missing", value=None)])]
    preds.write_text("".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setenv("PREDICTIONS_PATH", str(preds))
    monkeypatch.setenv("FEEDBACK_PATH", str(fb))

    q = client.get("/api/review/queue").json()
    # приоритет = вес в поиске × (1 − уверенность): 0.63 > 0.24 > 0.10
    assert [i["id"] for i in q["items"]] == ["c2:storage", "c3:model", "c1:color"]
    assert q["items"][0]["alternatives"][0]["value"] == "256.0"  # альтернатива из предупреждения пайплайна
    assert q["stats"]["total"] == 3 and q["stats"]["remaining"] == 3
    with_missing = client.get("/api/review/queue?include_missing=true").json()
    assert "c3:screen" in [i["id"] for i in with_missing["items"]]

    rec = client.post("/api/feedback", json={"card_id": "c2", "attribute": "storage", "predicted": 128.0, "corrected": "256"}).json()
    assert rec["decision"]["action"] == "corrected"
    after = client.get("/api/review/queue").json()
    assert after["stats"]["done"] == 1 and after["items"][0]["decision"]["action"] == "corrected"

    assert client.post("/api/feedback/undo", json={"card_id": "c2", "attribute": "storage"}).status_code == 200
    assert client.get("/api/review/queue").json()["stats"]["done"] == 0
    assert client.post("/api/feedback/undo", json={"card_id": "c2", "attribute": "storage"}).status_code == 404


def test_report_bundle_names_every_artifact():
    rep = client.get("/api/report").json()
    meta = rep["_meta"]
    assert set(meta["files"]) | set(meta["missing"]) | set(meta["invalid"]) == set(api.REPORT_FILES)
    assert all(k in rep for k in meta["files"])


def test_categories_and_dictionary_values():
    cats = client.get("/api/categories").json()
    smartphone = next(c for c in cats if c["name"] == "smartphone")
    dict_name = next(a["dictionary"] for a in smartphone["attributes"] if a["dictionary"])
    assert client.get(f"/api/dictionary/{dict_name}").json()["values"]
    assert client.get("/api/dictionary/__nope__").status_code == 404


def test_genui_schema_offers_conflicts_component():
    assert "conflicts" in COMPONENT_JSON_SCHEMA["properties"]["type"]["enum"]


def test_static_ui_served_without_build():
    assert client.get("/").status_code == 200
    js = client.get("/static/app.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    assert js.headers["cache-control"] == "no-cache"
    assert client.get("/static/assets/icons.svg").headers["content-type"].startswith("image/svg")
    assert client.get("/static/assets/fonts/onest-cyrillic-wght-normal.woff2").headers["content-type"] == "font/woff2"


def test_default_screen_flags_category_by_pipeline_warning_not_own_threshold():
    from attrx.genui.composer import compose_default

    # классификатор уверен на 0.75 — ниже своего порога (0.9): пайплайн пишет предупреждение, экран его показывает
    flagged = _res("c", [_av("brand", 0.95, status="accepted")],
                   ["категория определена неуверенно (classifier: Смартфоны, 0.75) — карточка на проверку"])
    flagged.category_confidence = 0.75
    first = compose_default(flagged).components[0]
    assert first.type == "alert" and "Категория определена неуверенно" in first.text
    calm = _res("c", [_av("brand", 0.95, status="accepted")])
    calm.category_confidence = 0.5  # без предупреждения пайплайна своего порога у экрана нет
    assert compose_default(calm).components[0].type != "alert"
