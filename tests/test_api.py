from fastapi.testclient import TestClient

from attrx.api.app import app

client = TestClient(app)


def test_health():
    assert client.get("/api/health").json()["status"] == "ok"


def test_extract_returns_result_and_ui():
    body = {"card": {"id": "1", "title": "Ноутбук Lenovo IdeaPad i7-13620H/16/1TB SSD/15.6\" серый"}}
    data = client.post("/api/extract", json=body).json()
    assert data["result"]["category"] == "laptop"
    assert data["ui"]["components"]


def test_health_ml_model_means_ner_is_loaded_and_reports_category_threshold(monkeypatch, tmp_path):
    from attrx.api import app as api

    ex = api.extractor()
    monkeypatch.setenv("ML_ENABLED", "true")
    monkeypatch.setenv("ML_MODEL_PATH", str(tmp_path))  # папка есть, а модели (весов, torch) нет
    monkeypatch.setattr(ex, "ml", None)
    h = client.get("/api/health").json()
    assert h["ml_model"] is False  # раньше: True по одному факту существования папки
    monkeypatch.setattr(ex, "ml", object())
    assert client.get("/api/health").json()["ml_model"] is True
    clf = ex.tax.classifier()
    assert h["category_threshold"] == (clf.threshold if clf is not None else None)


def test_old_endpoints_kept_but_marked_deprecated():
    paths = client.get("/openapi.json").json()["paths"]
    assert paths["/api/samples"]["get"]["deprecated"] is True
    assert paths["/api/review"]["get"]["deprecated"] is True
    assert client.get("/api/samples").status_code == 200 and client.get("/api/review").status_code == 200
