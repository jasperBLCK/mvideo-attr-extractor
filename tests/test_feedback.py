"""Петля обратной связи: категория в записи решает, в какой справочник идёт синоним; синоним — формулировка
значения без «Ключ:» продавца; справочник учат только исправления; learned.yaml — валидный YAML, таксономия грузится."""
import json
import shutil

import pytest
import yaml
from fastapi.testclient import TestClient

from attrx.config import ROOT
from attrx.feedback import decision_action, dictionary_for, learn_synonyms, log_correction, value_text
from attrx.taxonomy import load_taxonomy


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """Копия таксономии и справочников: attrx learn пишет learned.yaml рядом с ними, config/ не трогаем."""
    conf = tmp_path / "config"
    shutil.copytree(ROOT / "config" / "dictionaries", conf / "dictionaries")
    shutil.copy(ROOT / "config" / "taxonomy.yaml", conf / "taxonomy.yaml")
    (conf / "dictionaries" / "learned.yaml").unlink(missing_ok=True)
    monkeypatch.setenv("TAXONOMY_PATH", str(conf / "taxonomy.yaml"))
    monkeypatch.setenv("FEEDBACK_PATH", str(tmp_path / "feedback.jsonl"))
    yield conf
    load_taxonomy.cache_clear()


def test_dictionary_is_resolved_by_card_category(cfg):
    tax = load_taxonomy(str(cfg / "taxonomy.yaml"))
    assert dictionary_for(tax, "type", "headphones") == "headphone_types"
    assert dictionary_for(tax, "type", "vacuum") == "vacuum_types"
    assert dictionary_for(tax, "type", None) is None  # старая запись без категории: у type три справочника — не учим
    assert dictionary_for(tax, "color", None) == "colors"  # у цвета справочник один — старую запись можно учесть
    assert dictionary_for(tax, "storage", "smartphone") is None  # число — не справочник


def test_type_correction_lands_in_dictionary_of_its_category_and_taxonomy_loads(cfg):
    canon = load_taxonomy(str(cfg / "taxonomy.yaml")).dictionaries["headphone_types"].canon[0]
    log_correction("h1", "type", "накладные", canon, raw="Тип: капельки", category="headphones")
    learned = learn_synonyms()
    assert learned == {"headphone_types": {"капельки": canon}}  # не washer_types, как раньше
    out = cfg / "dictionaries" / "learned.yaml"
    assert yaml.safe_load(out.read_text(encoding="utf-8")) == learned  # attrx learn пишет валидный YAML
    tax = load_taxonomy(str(cfg / "taxonomy.yaml"))  # и таксономия с ним загружается
    assert tax.dictionaries["headphone_types"].get("капельки") == canon
    assert tax.dictionaries["washer_types"].get("капельки") is None


def test_synonym_is_the_value_span_without_seller_key():
    assert value_text("Цвет: графитовый") == "графитовый"
    assert value_text("Цвет корпуса:  Graphite Grey ") == "Graphite Grey"
    assert value_text("графитовый") == "графитовый"
    assert value_text("16:9") == "16:9"  # двоеточие без пробела — часть значения, а не ключ
    assert value_text(None) == ""


def test_only_corrections_teach_the_dictionary(cfg):
    log_correction("c1", "color", "чёрный", "чёрный", raw="угольный", category="smartphone")  # принято
    log_correction("c2", "color", "чёрный", None, raw="угольный", category="smartphone")  # отклонено
    st: dict = {}
    assert learn_synonyms(stats=st) == {}
    assert (st["records"], st["accepted"], st["rejected"], st["learned"]) == (2, 1, 1, 0)
    # исправление по полю продавца: учится «графитовый», а не «цвет: графитовый»; «серый» — канон справочника
    log_correction("c3", "color", "чёрный", "Серый", raw="Цвет: графитовый", category="smartphone")
    assert learn_synonyms() == {"colors": {"графитовый": "серый"}}
    assert [decision_action(r) for r in ({"predicted": "a", "corrected": "a"}, {"predicted": "a", "corrected": ""},
                                         {"predicted": "a", "corrected": "b"})] == ["accepted", "rejected", "corrected"]
    # перепечатанное то же значение (6000 вместо 6000.0, другой регистр) — подтверждение, а не исправление
    assert decision_action({"predicted": 6000.0, "corrected": "6000"}) == "accepted"
    assert decision_action({"predicted": "Panasonic", "corrected": "panasonic"}) == "accepted"
    assert decision_action({"predicted": None, "corrected": "чёрный"}) == "corrected"  # заполнен пропуск


def test_api_feedback_stores_category_and_accepts_old_payload(cfg, tmp_path):
    from attrx.api.app import app

    client = TestClient(app)
    new = client.post("/api/feedback", json={"card_id": "h1", "category": "headphones", "attribute": "type",
                                             "predicted": "накладные", "corrected": "вкладыши", "raw": "капельки"}).json()
    assert new["category"] == "headphones" and new["decision"]["action"] == "corrected"
    old = client.post("/api/feedback", json={"card_id": "h2", "attribute": "color", "predicted": "чёрный",
                                             "corrected": "чёрный"}).json()  # клиент без category
    assert old["category"] is None and old["decision"]["action"] == "accepted"
    rows = [json.loads(x) for x in (tmp_path / "feedback.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["category"] for r in rows] == ["headphones", None]
