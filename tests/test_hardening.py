"""Безопасные умолчания: attrx serve слушает 127.0.0.1 (Docker — 0.0.0.0 явно); ATTRX_TOKEN закрывает запись решений
модератора заголовком X-Attrx-Token (без токена интерфейс работает как раньше); attrx learn не превращает опечатку
модератора в канон справочника — нужно значение из справочника или ≥ 2 одинаковых исправления на разных карточках."""
import shutil

import pytest
import yaml
from fastapi.testclient import TestClient

from attrx.cli import main
from attrx.config import ROOT
from attrx.feedback import CANDIDATES_FILE, learn, learn_synonyms, log_correction
from attrx.taxonomy import load_taxonomy

# ------------------------------------------------------------------ attrx serve


@pytest.fixture
def uvicorn_calls(monkeypatch):
    import uvicorn

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append(kw))
    monkeypatch.delenv("ATTRX_HOST", raising=False)
    return calls


def test_serve_listens_locally_by_default(uvicorn_calls, capsys):
    main(["serve"])
    assert uvicorn_calls[-1]["host"] == "127.0.0.1" and uvicorn_calls[-1]["port"] == 8000
    assert "--host 0.0.0.0" in capsys.readouterr().out  # подсказка, как открыть наружу


def test_serve_public_only_when_asked(uvicorn_calls, monkeypatch, capsys):
    monkeypatch.delenv("ATTRX_TOKEN", raising=False)
    main(["serve", "--host", "0.0.0.0"])
    assert uvicorn_calls[-1]["host"] == "0.0.0.0"
    assert "ATTRX_TOKEN" in capsys.readouterr().out  # наружу без токена — предупреждение
    monkeypatch.setenv("ATTRX_HOST", "0.0.0.0")
    main(["serve"])
    assert uvicorn_calls[-1]["host"] == "0.0.0.0"


def test_make_ner_explains_missing_extra_and_data(monkeypatch, tmp_path):
    """make ner = attrx train-ner --distant: без [ml] и без data/processed — понятная подсказка, а не трассировка."""
    from attrx.extractors import ner

    monkeypatch.setattr(ner, "available", lambda: False)
    with pytest.raises(SystemExit, match=r"extra \[ml\]"):
        main(["train-ner", "--distant", "--processed", str(tmp_path)])
    monkeypatch.setattr(ner, "available", lambda: True)
    with pytest.raises(SystemExit, match="attrx prepare -> attrx build-gold"):
        main(["train-ner", "--distant", "--processed", str(tmp_path)])  # пустая папка: обучение не стартует
    make = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "ner: ; $(PY)/attrx train-ner --distant" in make and "eda: ; $(PY)/attrx eda-real" in make


def test_docker_image_listens_on_all_interfaces():
    cmd = [x for x in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines() if x.startswith("CMD")]
    assert cmd and '"--host", "0.0.0.0"' in cmd[0] and '"8000"' in cmd[0]


# ------------------------------------------------------------------ ATTRX_TOKEN


@pytest.fixture
def api(tmp_path, monkeypatch):
    from attrx.api.app import app

    monkeypatch.setenv("FEEDBACK_PATH", str(tmp_path / "feedback.jsonl"))
    monkeypatch.delenv("ATTRX_TOKEN", raising=False)
    return TestClient(app)


FB = {"card_id": "c1", "category": "smartphone", "attribute": "color", "predicted": "чёрный", "corrected": "серый",
      "raw": "графитовый"}


def test_feedback_open_without_token_as_before(api, tmp_path):
    assert api.get("/api/health").json()["auth"] == {"write_token_required": False, "header": "X-Attrx-Token"}
    assert api.post("/api/feedback", json=FB).status_code == 200  # интерфейс токен не шлёт — и не должен
    assert api.post("/api/feedback/undo", json={"card_id": "c1", "attribute": "color"}).status_code == 200


def test_token_protects_writes_only(api, monkeypatch, tmp_path):
    monkeypatch.setenv("ATTRX_TOKEN", "s3cret")
    assert api.get("/api/health").json()["auth"]["write_token_required"] is True
    r = api.post("/api/feedback", json=FB)
    assert r.status_code == 401 and "X-Attrx-Token" in r.json()["detail"]
    assert api.post("/api/feedback", json=FB, headers={"X-Attrx-Token": "wrong"}).status_code == 401
    assert not (tmp_path / "feedback.jsonl").exists()  # отвергнутые запросы ничего не записали
    assert api.post("/api/feedback", json=FB, headers={"X-Attrx-Token": "s3cret"}).status_code == 200
    undo = {"card_id": "c1", "attribute": "color"}
    assert api.post("/api/feedback/undo", json=undo).status_code == 401
    assert api.post("/api/feedback/undo", json=undo, headers={"X-Attrx-Token": "s3cret"}).status_code == 200
    assert api.get("/api/review/queue").status_code == 200  # чтение не закрыто
    assert "s3cret" not in api.get("/api/health").text  # токен не утекает


# ------------------------------------------------------------------ attrx learn


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """Копия таксономии и справочников: learn пишет learned.yaml рядом с ними, config/ не трогаем."""
    conf = tmp_path / "config"
    shutil.copytree(ROOT / "config" / "dictionaries", conf / "dictionaries")
    shutil.copy(ROOT / "config" / "taxonomy.yaml", conf / "taxonomy.yaml")
    (conf / "dictionaries" / "learned.yaml").unlink(missing_ok=True)
    monkeypatch.setenv("TAXONOMY_PATH", str(conf / "taxonomy.yaml"))
    monkeypatch.setenv("FEEDBACK_PATH", str(tmp_path / "feedback.jsonl"))
    yield conf
    load_taxonomy.cache_clear()


def _colors(cfg):
    return load_taxonomy(str(cfg / "taxonomy.yaml")).dictionaries["colors"]


def test_typo_does_not_become_canonical_after_one_correction(cfg, tmp_path):
    log_correction("c1", "color", "чёрный", "чорный", raw="Цвет: лакричный", category="smartphone")
    res = learn()
    assert res["learned"] == {} and res["stats"]["candidates"] == 1
    cand = res["candidates"]["colors"]["лакричный"]
    assert cand["value"] == "чорный" and cand["votes"] == 1 and cand["need"] == 2 and cand["suggest"] == "чёрный"
    on_disk = yaml.safe_load((tmp_path / CANDIDATES_FILE).read_text(encoding="utf-8"))
    assert on_disk["colors"]["лакричный"]["value"] == "чорный"  # кандидаты — отдельный файл рядом с журналом
    colors = _colors(cfg)
    assert "чорный" not in colors.canon and colors.get("лакричный") is None and colors.get("чорный") is None
    assert "learned_candidates" not in load_taxonomy(str(cfg / "taxonomy.yaml")).dictionaries


def test_repeat_click_on_same_card_is_not_agreement(cfg):
    for _ in range(3):
        log_correction("c1", "color", "чёрный", "чорный", raw="лакричный", category="smartphone")
    assert learn_synonyms() == {}


def test_two_identical_corrections_on_different_cards_teach(cfg):
    log_correction("c1", "color", "чёрный", "чорный", raw="лакричный", category="smartphone")
    log_correction("c2", "color", "чёрный", " Чорный ", raw="Цвет: лакричный", category="laptop")  # то же значение
    st: dict = {}
    assert learn_synonyms(stats=st) == {"colors": {"лакричный": "чорный"}}
    assert st["learned"] == 1 and st["candidates"] == 0
    assert _colors(cfg).get("лакричный") == "чорный"  # согласились двое — значение стало каноном


def test_known_dictionary_value_teaches_at_once_and_conflicts_wait(cfg):
    log_correction("c1", "color", "чёрный", "серый", raw="лунная пыль", category="smartphone")  # канон справочника
    log_correction("c2", "color", "белый", "черный", raw="вороново крыло", category="smartphone")  # известный синоним
    log_correction("c3", "color", "белый", "синий", raw="северное сияние", category="smartphone")  # правки расходятся
    log_correction("c4", "color", "белый", "зелёный", raw="северное сияние", category="smartphone")
    res = learn()
    assert res["learned"] == {"colors": {"лунная пыль": "серый", "вороново крыло": "чёрный"}}
    cand = res["candidates"]["colors"]["северное сияние"]
    assert cand["options"] == {"зелёный": 1, "синий": 1} and "расходятся" in cand["reason"]
    colors = _colors(cfg)
    assert colors.get("вороново крыло") == "чёрный" and colors.get("северное сияние") is None


def test_learn_cli_prints_candidates(cfg, capsys):
    log_correction("c1", "color", "чёрный", "чорный", raw="лакричный", category="smartphone")
    main(["learn"])
    out = capsys.readouterr().out
    assert "Выучено синонимов: 0" in out and "Кандидатов: 1" in out and "похоже на «чёрный»" in out
