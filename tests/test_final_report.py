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
