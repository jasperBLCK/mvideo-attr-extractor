"""Удобство API: id карточки необязателен (генерируется по тексту, числовой приводится к строке); use_llm=true без
настроенной LLM — работает на правилах и NER с предупреждением в result.warnings; /api/metrics отдаёт итоговые
метрики submission/ раньше демо outputs/ и подсказывает, как их получить."""
import json

import pytest
from fastapi.testclient import TestClient

from attrx.api import app as api
from attrx.schema import Card, card_auto_id

client = TestClient(api.app)
PHONE = {"title": "Смартфон Samsung Galaxy A55 8/256 ГБ голубой", "description": "Дисплей 6.6 дюйма, батарея 5000 mAh."}


class _LLMOff:
    enabled = False
    calls = 0


def test_card_id_is_optional_and_stable():
    a, b = Card(**PHONE), Card.model_validate({**PHONE, "id": None})
    assert a.id.startswith("auto-") and a.id == b.id == card_auto_id(PHONE["title"], PHONE["description"], {})
    assert Card(title="другая карточка").id != a.id
    assert Card(id="", **PHONE).id == a.id  # пустой id из таблицы без колонки id — тоже генерируется
    assert Card(id="sku-1", **PHONE).id == "sku-1"  # явный id не трогаем (совместимость)
    assert Card.model_validate({"id": 30055003, **PHONE}).id == "30055003"  # числовой id из PIM


def test_extract_without_id(monkeypatch):
    monkeypatch.setattr(api.extractor(), "client", _LLMOff())
    data = client.post("/api/extract", json={"card": PHONE}).json()
    assert data["result"]["card_id"].startswith("auto-") and data["result"]["category"] == "smartphone"
    batch = client.post("/api/extract/batch", json={"cards": [PHONE, {**PHONE, "id": 7}]}).json()
    assert batch[0]["card_id"].startswith("auto-") and batch[1]["card_id"] == "7"


def test_use_llm_without_llm_warns_but_works(monkeypatch):
    monkeypatch.setattr(api.extractor(), "client", _LLMOff())
    r = client.post("/api/extract", json={"card": PHONE, "use_llm": True})
    assert r.status_code == 200
    res = r.json()["result"]
    assert any(w.startswith("LLM не настроена — использованы правила") for w in res["warnings"])
    assert not res["llm_used"] and any(a["value"] is not None for a in res["attributes"])
    plain = client.post("/api/extract", json={"card": PHONE}).json()["result"]  # use_llm не задан — не шумим
    assert not any(w.startswith("LLM не настроена") for w in plain["warnings"])
    batch = client.post("/api/extract/batch", json={"cards": [PHONE], "use_llm": True}).json()
    assert any(w.startswith("LLM не настроена") for w in batch[0]["warnings"])


def test_llm_off_warning_mentions_ner_only_when_loaded(monkeypatch):
    ex = api.extractor()
    monkeypatch.setattr(ex, "client", _LLMOff())
    monkeypatch.setattr(ex, "ml", object())
    assert api._llm_off_warning(ex, True) == api.LLM_OFF_WARNING == "LLM не настроена — использованы правила и NER"
    monkeypatch.setattr(ex, "ml", None)
    assert "NER не загружен" in api._llm_off_warning(ex, True)
    assert api._llm_off_warning(ex, None) is None and api._llm_off_warning(ex, False) is None


@pytest.fixture
def metric_files(tmp_path, monkeypatch):
    sub, out = tmp_path / "submission", tmp_path / "outputs"
    sub.mkdir()
    out.mkdir()
    monkeypatch.setattr(api, "SUBMISSION", sub)
    monkeypatch.setenv("PREDICTIONS_PATH", str(out / "predictions.jsonl"))
    (sub / "metrics.json").write_text(json.dumps({"who": "submission"}), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({"who": "outputs"}), encoding="utf-8")
    return sub, out


def test_metrics_prefer_final_submission_over_demo(metric_files):
    sub, out = metric_files
    r = client.get("/api/metrics")
    assert r.json() == {"who": "submission"} and r.headers["X-Attrx-Source"] == "submission"
    r = client.get("/api/metrics", params={"source": "outputs"})
    assert r.json() == {"who": "outputs"} and r.headers["X-Attrx-Source"] == "outputs"
    (sub / "metrics.json").unlink()
    assert client.get("/api/metrics").json() == {"who": "outputs"}  # итоговых нет — последний локальный прогон
    (out / "metrics.json").unlink()
    r = client.get("/api/metrics")
    assert r.status_code == 404 and "attrx demo" in r.json()["detail"] and "make report" in r.json()["detail"]
