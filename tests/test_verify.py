"""attrx verify (make verify): строки «правила» и «продавец + правила» из compare.json пересчитываются на выгрузке
holdout тем же кодом, что итоговый отчёт; расхождение больше допуска — код 1; нет выгрузки — понятная ошибка.
Фикстура — 6 реальных карточек из data/sample (без data/processed, torch и LLM). Пересчёт идёт дважды: в фикстуре и
в сквозном прогоне CLI; остальные проверки CLI подставляют готовые строки (логика сверки та же)."""
import copy
import json
import sys

import pytest

from attrx import reproduce
from attrx.cli import main
from attrx.config import ROOT

SAMPLE = ROOT / "data" / "sample"


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """6 карточек + эталон в формате выгрузки (holdout_eval_cards.jsonl / holdout_eval_gold.jsonl) и их строки."""
    d = tmp_path_factory.mktemp("verify")
    gold = [json.loads(x) for x in (SAMPLE / "real_gold.jsonl").read_text(encoding="utf-8").splitlines()[:6]]
    ids = {g["id"] for g in gold}
    cards = [x for x in (SAMPLE / "real_cards.jsonl").read_text(encoding="utf-8").split("\n")
             if x.strip() and json.loads(x)["id"] in ids]
    c, g = d / "cards.jsonl", d / "gold.jsonl"
    c.write_text("\n".join(cards) + "\n", encoding="utf-8")
    g.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in gold) + "\n", encoding="utf-8")
    rows, info = reproduce.recompute(c, g)
    assert info["n_cards"] == 6 and [r["mode"] for r in rows] == ["правила", "продавец + правила"]
    return c, g, rows, info


@pytest.fixture
def precomputed(tiny, monkeypatch):
    """CLI без повторного прогона пайплайна: recompute отдаёт строки из фикстуры."""
    c, g, rows, info = tiny
    monkeypatch.setattr(reproduce, "recompute", lambda *a, **k: (copy.deepcopy(rows), dict(info)))
    return c, g, rows


def _compare_v2(rows, path, shift: float = 0.0):
    """compare.json как у отчёта v2: стратифицированный набор (с суффиксом) + приманки — весь holdout (другое число
    карточек) и подвыборка LLM с тем же режимом."""
    out = []
    for r in rows:
        out.append({**r, "mode": r["mode"] + " · стратифицированный набор", "set": "stratified",
                    "precision": round(r["precision"] + shift, 4)})
        out.append({**r, "set": "full_holdout", "n_cards": 9999, "precision": 0.1})
        out.append({**r, "mode": r["mode"] + " · подвыборка LLM", "set": "llm_subsample", "precision": 0.2})
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return path


def _args(c, g, cmp):
    return ["verify", "--cards", str(c), "--gold", str(g), "--compare", str(cmp)]


def test_verify_end_to_end_passes_and_picks_the_exported_set(tiny, tmp_path, capsys):
    c, g, rows, _ = tiny
    torch_before = "torch" in sys.modules
    main(_args(c, g, _compare_v2(rows, tmp_path / "compare.json")) + ["--out", str(tmp_path / "verify.json")])
    out = capsys.readouterr().out
    assert "OK: все" in out and "РАСХОЖДЕНИЕ" not in out
    rep = json.loads((tmp_path / "verify.json").read_text(encoding="utf-8"))
    assert rep["ok"] and {c["set"] for c in rep["checks"]} == {"stratified"}  # не весь holdout и не подвыборка LLM
    assert {c["metric"] for c in rep["checks"]} >= {"n_cards", "precision", "recall", "f1", "traffic_weighted_f1"}
    if not torch_before:
        assert "torch" not in sys.modules  # сверка офлайн: без NER


def test_verify_accepts_v1_compare_format_within_tolerance(precomputed, tmp_path, capsys):
    c, g, rows = precomputed
    v1 = [{**r, "set": "full", "recall": round(r["recall"] + 0.0015, 4)} for r in rows]  # в пределах допуска 0,002
    cmp = tmp_path / "compare_v1.json"
    cmp.write_text(json.dumps(v1, ensure_ascii=False), encoding="utf-8")
    main(_args(c, g, cmp))
    assert "OK: все" in capsys.readouterr().out


def test_verify_fails_on_mismatch_beyond_tolerance(precomputed, tmp_path, capsys):
    c, g, rows = precomputed
    with pytest.raises(SystemExit) as e:
        main(_args(c, g, _compare_v2(rows, tmp_path / "compare.json", shift=0.01)))
    assert e.value.code == 1
    out = capsys.readouterr().out
    assert "РАСХОЖДЕНИЕ" in out and "precision" in out


def test_verify_reports_missing_mode_row(precomputed, tmp_path):
    c, g, _ = precomputed
    cmp = tmp_path / "compare.json"
    cmp.write_text(json.dumps([{"mode": "+ NER", "set": "full", "n_cards": 6}]), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        main(_args(c, g, cmp))
    assert "нет строки «правила»" in str(e.value.code)


def test_verify_without_export_fails_with_clear_message(tmp_path):
    cmp = tmp_path / "compare.json"
    cmp.write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        main(_args(tmp_path / "holdout_eval_cards.jsonl", tmp_path / "holdout_eval_gold.jsonl", cmp))
    msg = str(e.value.code)
    assert "нет" in msg and "holdout_eval_cards.jsonl" in msg and "make report" in msg


def test_reference_row_matching():
    ref = [{"mode": "правила", "set": "full_holdout", "n_cards": 100},
           {"mode": "правила · стратифицированный набор", "set": "stratified", "n_cards": 10},
           {"mode": "правила · подвыборка LLM", "set": "llm_subsample", "n_cards": 10}]
    assert reproduce.reference_row(ref, "правила", 10)["set"] == "stratified"
    assert reproduce.reference_row(ref, "правила", 100)["set"] == "full_holdout"
    assert reproduce.reference_row(ref, "правила", 10, "full_holdout")["n_cards"] == 100  # явный набор
    assert reproduce.reference_row(ref, "+ NER", 10) is None
    assert reproduce.base_mode("продавец + правила · стратифицированный набор") == "продавец + правила"
