"""Порог автопринятия по умолчанию — из калибровки (.env.example его не перекрывает); XLSX — через extra [xlsx]."""
import json

import pandas as pd
import pytest

from attrx import io
from attrx.config import ROOT, Settings


def test_threshold_comes_from_calibration_and_env_example_does_not_override(monkeypatch):
    monkeypatch.delenv("CONFIDENCE_THRESHOLD", raising=False)
    cal = json.loads((ROOT / "config" / "calibration.json").read_text(encoding="utf-8")).get("threshold", 0.7)
    assert Settings().confidence_threshold == pytest.approx(cal)
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    assert not [x for x in lines if x.strip().startswith("CONFIDENCE_THRESHOLD")]  # только закомментирован
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.8")  # явная переменная по-прежнему перекрывает калибровку
    assert Settings().confidence_threshold == 0.8


def test_xlsx_needs_optional_extra_with_install_hint(tmp_path, monkeypatch):
    def no_openpyxl(*a, **k):
        raise ImportError("Missing optional dependency 'openpyxl'.")

    monkeypatch.setattr(pd, "read_excel", no_openpyxl)
    with pytest.raises(ImportError, match=r'\.\[xlsx\]'):
        io.read_table(tmp_path / "cards.xlsx")
    with pytest.raises(ValueError, match="xlsx"):
        io.read_table(tmp_path / "cards.xls")  # старый .xls не поддерживаем — сохранить как .xlsx или .csv


def test_xlsx_roundtrip_when_extra_installed(tmp_path):
    pytest.importorskip("openpyxl")
    p = tmp_path / "cards.xlsx"
    pd.DataFrame([{"id": "1", "title": "Смартфон Apple iPhone 15 128 ГБ", "description": ""}]).to_excel(p, index=False)
    cards = io.load_cards(p)
    assert cards[0].id == "1" and cards[0].title.startswith("Смартфон")
