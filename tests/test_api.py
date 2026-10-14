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
