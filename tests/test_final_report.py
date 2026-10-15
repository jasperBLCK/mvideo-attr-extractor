"""Итоговый отчёт (attrx report-real): подвыборка для LLM, правило выбора режима, покрытие с учётом спроса, cap ошибок.
Без data/processed — на синтетической мете."""
import csv

from attrx.eval import final
from attrx.schema import AttributeValue, Card, ExtractionResult
from attrx.taxonomy import load_taxonomy


def _meta():
    meta, ids = {}, []
    for cat, n in (("smartphone", 300), ("laptop", 100), ("tv", 120), ("kettle", 80), ("fridge", 15)):
        for i in range(n):
            k = f"{cat}{i}"
            ids.append(k)
            meta[k] = {"category": cat, "size_band": "head"}
    for band, n in (("head", 500), ("torso", 300), ("tail", 200)):
        for i in range(n):
            k = f"g{band}{i}"
            ids.append(k)
            meta[k] = {"category": "generic", "size_band": band}
    return ids, meta


def test_llm_subsample_is_stratified_deterministic_and_ordered():
    ids, meta = _meta()
    a = final.llm_subsample(ids, meta, n=600, seed=0)
    assert a == final.llm_subsample(ids, meta, n=600, seed=0)  # детерминированно
    assert set(a) <= set(ids) and a == [i for i in ids if i in set(a)]  # подмножество, порядок как во входе
    cats = [meta[i]["category"] for i in a]
    assert cats.count("smartphone") + cats.count("laptop") >= 150
    assert cats.count("fridge") >= 12 and cats.count("tv") >= 12  # каждая категория представлена
    bands = [meta[i]["size_band"] for i in a if meta[i]["category"] == "generic"]
    assert min(bands.count(b) for b in ("head", "torso", "tail")) >= 40  # хвост каталога тоже
    assert 550 <= len(a) <= 650


def test_final_mode_rule_is_fixed_before_the_run():
    base = {"precision": 0.970, "f1": 0.845}
    assert final.choose_final({"продавец + правила": base, "+ NER": {"precision": 0.968, "f1": 0.849}}) == "+ NER"
    assert final.choose_final({"продавец + правила": base, "+ NER": {"precision": 0.960, "f1": 0.860}}) == \
        "продавец + правила"  # теряет точность больше 0,005
    assert final.choose_final({"продавец + правила": base, "+ NER": {"precision": 0.975, "f1": 0.840}}) == \
        "продавец + правила"  # не лучше по F1
    assert final.choose_final({"продавец + правила": base}) == "продавец + правила"


def test_traffic_weighted_coverage():
    m = {"per_category": {"smartphone": {"brand": {"n_cards": 10, "catalog_fill": 1.0, "union_fill": 1.0},
                                         "screen": {"n_cards": 10, "catalog_fill": 0.5, "union_fill": 0.9}},
                          "generic": {"color": {"n_cards": 5, "catalog_fill": 0.6, "union_fill": 0.8}}}}
    w = {("smartphone", "brand"): 0.5, ("smartphone", "screen"): 0.3, ("generic", "color"): 0.2}
    tc = final.traffic_coverage(m, w)
    assert tc["before"] == round(0.5 * 1 + 0.3 * 0.5 + 0.2 * 0.6, 4)
    assert tc["after"] == round(0.5 * 1 + 0.3 * 0.9 + 0.2 * 0.8, 4)


def _result(cid: str, value, conf: float, status: str) -> ExtractionResult:
    return ExtractionResult(card_id=cid, category="smartphone", category_confidence=1.0, attributes=[
        AttributeValue(attribute="storage", title="Память", value=value, confidence=conf, status=status,
                       sources=["rules"])])


def test_reliability_and_accepted_precision():
    tax = load_taxonomy()
    gold = {str(i): {"storage": 256.0} for i in range(10)}
    res = [_result(str(i), 256.0 if i < 8 else 128.0, 0.95 if i < 9 else 0.55, "accepted" if i < 9 else "review")
           for i in range(10)]
    rel = final.reliability(res, gold, tax)
    assert rel["n"] == 10 and sum(b["n"] for b in rel["bins"]) == 10
    top = next(b for b in rel["bins"] if b["lo"] == 0.9)
    assert top["n"] == 9 and abs(top["accuracy"] - 8 / 9) < 1e-3
    ap = final.accepted_precision(res, gold, tax)
    assert ap["accepted"] == 9 and abs(ap["precision"] - 8 / 9) < 1e-3 and ap["lower95"] < ap["precision"]


def test_errors_csv_cap_keeps_all_wrong(tmp_path):
    tax = load_taxonomy()
    gold = {str(i): {"storage": 256.0} for i in range(40)}
    res = [_result(str(i), 128.0 if i < 5 else None, 0.9 if i < 5 else 0.0, "accepted" if i < 5 else "missing")
           for i in range(40)]
    cards = [Card(id=str(i), title=f"Смартфон {i}") for i in range(40)]
    meta = {str(i): {"extractable": {"storage": i % 2 == 0}, "seller_type": "mvideo", "size_band": "head"} for i in range(40)}
    out = tmp_path / "errors.csv"
    st = final.write_errors(res, gold, meta, tax, cards, out, cap=12)
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert st["total"] == 40 and st["by_kind"] == {"wrong": 5, "missed": 35}
    assert sum(r["kind"] == "wrong" for r in rows) == 5 and len(rows) <= 12
    assert st["missed_not_in_text"] + st["missed_in_review"] + st["missed_not_found"] == 35


def test_llm_provider_choice(monkeypatch):
    for k in ("LLM_BASE_URL", "OPENAI_BASE_URL", "LLM_API_KEY", "OPENAI_API_KEY", "LLM_MODEL", "OPENAI_MODEL"):
        monkeypatch.delenv(k, raising=False)
    s, info = final.llm_settings("auto")
    assert info["choice"] == "local" and s.llm_base_url == final.LOCAL_LLM_URL and s.llm_model == final.LOCAL_LLM_MODEL
    assert s.llm_cache is False  # в отчёте — настоящие вызовы, а не ответы из кэша
    monkeypatch.setenv("LLM_BASE_URL", "http://10.0.0.5:8000/v1")
    s, info = final.llm_settings("auto")
    assert info["choice"] == "env" and s.llm_base_url == "http://10.0.0.5:8000/v1"  # явная настройка LLM_*
    # ВМ: шлюз организаторов побеждает, даже если LLM_* подставлены по умолчанию (Makefile, .env)
    monkeypatch.setenv("LLM_BASE_URL", final.LOCAL_LLM_URL)
    monkeypatch.setenv("LLM_MODEL", final.LOCAL_LLM_MODEL)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gw.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key")
    s, info = final.llm_settings("auto")
    assert info["choice"] == "openai" and s.llm_base_url == "https://gw.example/v1"
    assert s.llm_api_key == "sk-real-key" and s.llm_model == "auto" and s.llm_source == "OPENAI_*"
    s, info = final.llm_settings("env")
    assert info["choice"] == "env" and s.llm_base_url == final.LOCAL_LLM_URL  # env — обычный порядок: LLM_* первым
    s, info = final.llm_settings("local")
    assert info["choice"] == "local" and s.llm_base_url == final.LOCAL_LLM_URL  # явный выбор сильнее окружения


def test_report_refuses_silent_llm_change(tmp_path):
    import json

    import pytest

    final._check_llm_change(tmp_path, None, force=False)  # отчёта ещё нет — можно что угодно
    (tmp_path / "metrics.json").write_text(json.dumps({"llm": {"model": "qwen3:4b-instruct"}}), encoding="utf-8")
    final._check_llm_change(tmp_path, "qwen3:4b-instruct", force=False)  # та же модель — можно
    with pytest.raises(SystemExit):
        final._check_llm_change(tmp_path, "gpt-4.1-mini", force=False)
    with pytest.raises(SystemExit):  # был отчёт с LLM, а сейчас LLM недоступна — тоже не молча
        final._check_llm_change(tmp_path, None, force=False)
    final._check_llm_change(tmp_path, "gpt-4.1-mini", force=True)  # осознанная смена


def test_llm_probe_unreachable_server_is_reported_not_used():
    from attrx.config import Settings

    s = Settings(llm_enabled=True, llm_source="LLM_*", llm_base_url="http://127.0.0.1:9/v1", llm_model="m",
                 llm_api_key="not-needed", llm_cache=False, llm_timeout=1.0)
    client, info = final.llm_probe(s)
    assert client is None and info["model"] is None and info["unavailable"]


def test_operating_point_and_band_precision():
    tax = load_taxonomy()
    gold = {str(i): {"storage": 256.0} for i in range(6)}
    res = [_result("0", 256.0, 0.72, "accepted"), _result("1", 128.0, 0.75, "accepted"),
           _result("2", 256.0, 0.95, "accepted"), _result("3", 256.0, 0.65, "review"),
           _result("4", 256.0, 0.79, "accepted"), _result("5", 128.0, 0.85, "review")]
    band = final.band_precision(res, gold, tax, 0.7, 0.8)
    assert band["n"] == 3 and abs(band["precision"] - 2 / 3) < 1e-3  # только принятые с уверенностью в [0,7; 0,8)
    assert band["share_of_accepted"] == 0.75 and band["wrong"] == 1 and band["wrong_accepted_total"] == 1
    assert band["top_cells"][0]["n"] == 3 and band["top_cells"][0]["cell"].startswith("smartphone/storage")
    m = {"micro": {"precision": 0.9, "recall": 0.7, "f1": 0.79}, "coverage": 0.7, "new_fills": 5, "review_rate": 0.1,
         "accepted_precision": {"precision": 0.9}}
    op = final.operating_point(m, 0.69)
    assert op["threshold"] == 0.69 and op["precision"] == 0.9 and op["new_fills"] == 5


def test_export_eval_set_roundtrip(tmp_path):
    from attrx.io import load_cards, load_gold

    cards = [Card(id="1", title="Смартфон Samsung Galaxy A55 8/256 ГБ", description="Экран 6,6\"", category="Смартфоны",
                  category_id="761", seller="М.Видео")]
    gold = {"1": {"brand": "Samsung", "storage": 256.0}}
    meta = {"1": {"category": "smartphone", "category_id": "761", "split": "holdout", "seller_type": "mvideo",
                  "size_band": "head", "desc_band": "short", "extractable": {"brand": True, "storage": True}}}
    info = final.export_eval_set(cards, gold, meta, tmp_path)
    back = load_cards(tmp_path / info["cards"])
    assert back[0].title == cards[0].title and back[0].category_id == "761"
    assert load_gold(tmp_path / info["gold"]) == gold
