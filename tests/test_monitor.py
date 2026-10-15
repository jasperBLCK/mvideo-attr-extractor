"""attrx monitor: доли review / missing, уверенность, неуверенная категория, generic, сдвиг категорий (PSI, JS),
доля LLM — против эталонного прогона; алерты по порогам; CLI пишет JSON и умеет падать на алертах (cron, CI)."""
import json

import pytest

from attrx import monitor
from attrx.cli import main
from attrx.feedback import log_correction
from attrx.schema import AttributeValue, ExtractionResult

UNSURE = "категория определена неуверенно (classifier: Прочее, 0.41) — карточка на проверку"


def _res(cid, category, statuses, llm=False, warnings=(), leaf=None, alts=False):
    attrs = [AttributeValue(attribute=f"a{i}", title=f"a{i}", value=None if st == "missing" else "x", confidence=conf,
                            status=st, alternatives=[{"value": "y"}] if alts and st != "missing" else [])
             for i, (st, conf) in enumerate(statuses)]
    return ExtractionResult(card_id=cid, category=category, category_confidence=0.5 if UNSURE in warnings else 1.0,
                            attributes=attrs, llm_used=llm, warnings=list(warnings), category_title=leaf or category,
                            category_path=[leaf or category])


GOOD = [("accepted", 0.95), ("accepted", 0.9), ("accepted", 0.92), ("accepted", 0.9), ("review", 0.5),
        ("missing", 0.0)]  # review и missing по 1/6 — ниже абсолютных порогов


def _reference():
    return ([_res(f"s{i}", "smartphone", GOOD) for i in range(60)]
            + [_res(f"g{i}", "generic", GOOD, leaf="Чайники") for i in range(40)])


def _drifted():
    """Новые категории каталога уходят в generic, категория неуверенная, больше review и пустых, чаще LLM."""
    bad = [("review", 0.4), ("review", 0.5), ("missing", 0.0), ("missing", 0.0)]
    return ([_res(f"s{i}", "smartphone", GOOD) for i in range(30)]
            + [_res(f"g{i}", "generic", bad, llm=True, warnings=[UNSURE, "llm: отброшено незаземлённое значение a1='z'"],
                    leaf="Электросамокаты") for i in range(70)])


def test_group_stats_shares():
    four = [("accepted", 0.95), ("accepted", 0.9), ("review", 0.5), ("missing", 0.0)]
    st = monitor.group_stats([_res("1", "tv", four, llm=True, warnings=[UNSURE], alts=True), _res("2", "tv", four)])
    assert st["n_cards"] == 2 and st["slots"] == 8
    assert (st["accepted_share"], st["review_share"], st["missing_share"]) == (0.5, 0.25, 0.25)
    assert st["mean_confidence"] == pytest.approx((0.95 + 0.9 + 0.5) / 3, abs=1e-4)
    assert st["category_unsure_share"] == 0.5 and st["llm_share"] == 0.5 and st["conflict_share"] == 0.5


def test_psi_and_js():
    a = monitor.Counter({"smartphone": 60, "generic": 40})
    assert monitor.psi(a, a) == 0 and monitor.js_divergence(a, a) == 0
    b = monitor.Counter({"smartphone": 30, "generic": 70})
    assert monitor.psi(b, a) > 0.25 and monitor.psi_level(monitor.psi(b, a)) == "сильный"
    assert 0 < monitor.js_divergence(b, a) < 1
    assert monitor.psi(monitor.Counter({"new": 10, **a}), a) > 0.25  # новая категория (9% карточек) — сильный сдвиг


def test_no_alerts_against_itself():
    ref = _reference()
    rep = monitor.monitor(ref, ref)
    assert rep["status"] == "ok" and rep["alerts"] == [] and rep["category_shift"]["psi"] == 0


def test_drift_raises_alerts_with_actions():
    rep = monitor.monitor(_drifted(), _reference())
    got = {(a["scope"], a["metric"], a["kind"]) for a in rep["alerts"]}
    assert ("все карточки", "psi", "max") in got
    assert ("все карточки", "generic_share", "rise") in got and ("все карточки", "generic_share", "max") in got
    assert ("generic", "review_share", "rise") in got and ("generic", "category_unsure_share", "max") in got
    assert ("generic", "llm_share", "rise") in got and ("generic", "mean_confidence", "drop") in got
    assert not [a for a in rep["alerts"] if a["scope"] == "smartphone"]  # смартфоны не поменялись
    assert rep["status"] == "alerts" and any("train-category" in s for s in rep["next_steps"])
    assert rep["category_shift"]["new_leaves"] == [{"title": "Электросамокаты", "n": 70}]
    assert rep["per_category"]["generic"]["llm_rejected_per_card"] == 1.0


def test_min_cards_guards_small_categories():
    cur = _reference() + [_res(f"k{i}", "kettle", [("review", 0.3)] * 4) for i in range(5)]
    rep = monitor.monitor(cur, _reference())
    assert rep["per_category"]["kettle"]["alerts"] == []  # 5 карточек < min_cards: шум, не алерт
    rep = monitor.monitor(cur, _reference(), {"min_cards": 5})
    assert {a["metric"] for a in rep["per_category"]["kettle"]["alerts"]} >= {"review_share", "mean_confidence"}


def test_thresholds_parse():
    t = monitor.parse_thresholds(["psi=0.1", "min_cards=10"])
    assert t["psi"] == 0.1 and t["min_cards"] == 10 and t["review_share"] == monitor.DEFAULT_THRESHOLDS["review_share"]
    with pytest.raises(ValueError, match="psi"):
        monitor.parse_thresholds(["nope=1"])


def _write(path, results):
    path.write_text("".join(r.model_dump_json() + "\n" for r in results), encoding="utf-8")
    return path


def test_cli_writes_report_and_fails_on_alert(tmp_path, capsys):
    cur, ref, out = _write(tmp_path / "pred.jsonl", _drifted()), _write(tmp_path / "ref.jsonl", _reference()), \
        tmp_path / "monitor.json"
    fb = tmp_path / "feedback.jsonl"
    log_correction("c1", "color", "чёрный", "серый", raw="графит", path=fb, category="smartphone")
    main(["monitor", "-i", str(cur), "--ref", str(ref), "--out", str(out), "--feedback", str(fb)])  # алерты не валят
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["status"] == "alerts" and rep["n_cards"] == 100 and rep["ref_n_cards"] == 100
    assert rep["feedback"]["corrected"] == 1
    printed = capsys.readouterr().out
    assert "PSI" in printed and "Что делать" in printed and "Решения модератора: 1" in printed
    with pytest.raises(SystemExit) as e:
        main(["monitor", "-i", str(cur), "--ref", str(ref), "--out", str(out), "--feedback", "", "--fail-on-alert"])
    assert e.value.code == 2
    main(["monitor", "-i", str(ref), "--ref", str(ref), "--out", str(out), "--feedback", "", "--fail-on-alert"])  # код 0
    with pytest.raises(SystemExit, match="имя=значение"):
        main(["monitor", "-i", str(cur), "--out", str(out), "-t", "psi"])


def test_cli_without_reference_uses_absolute_thresholds(tmp_path):
    cur, out = _write(tmp_path / "pred.jsonl", _drifted()), tmp_path / "m.json"
    main(["monitor", "-i", str(cur), "--ref", str(tmp_path / "нет.jsonl"), "--out", str(out), "--feedback", ""])
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["reference"] is None and rep["category_shift"] is None and "note" in rep
    assert {a["kind"] for a in rep["alerts"]} <= {"max", "min"} and rep["alerts"]


def test_feedback_summary_closes_the_loop(tmp_path):
    fb = tmp_path / "feedback.jsonl"
    log_correction("c1", "color", "чёрный", "чёрный", path=fb, category="smartphone")  # подтверждено
    log_correction("c2", "color", "чёрный", "серый", raw="графит", path=fb, category="smartphone")  # исправлено
    log_correction("c3", "brand", "X", None, path=fb, category="tv")  # отклонено
    (tmp_path / "learned_candidates.yaml").write_text("colors:\n  графит: {value: серый}\n", encoding="utf-8")
    st = monitor.feedback_stats(fb)
    assert (st["decisions"], st["accepted"], st["corrected"], st["rejected"]) == (3, 1, 1, 1)
    assert st["by_category"]["smartphone"] == {"accepted": 1, "corrected": 1} and st["pending_synonym_candidates"] == 1
    assert monitor.feedback_stats(tmp_path / "нет.jsonl") is None
