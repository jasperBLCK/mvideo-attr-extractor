import pytest

from attrx.normalize import normalize_cpu, parse_number
from attrx.taxonomy import load_taxonomy


@pytest.mark.parametrize("raw,kind,expected", [
    ("256 ГБ", "memory", 256.0), ("1 ТБ", "memory", 1024.0), ("512gb", "memory", 512.0),
    ('6,7"', "diagonal", 6.7), ("15.6 дюйма", "diagonal", 15.6), ("6.67\"", "diagonal", 6.67),
    ("5000 мАч", "battery", 5000.0), ("4500 мА·ч", "battery", 4500.0), ("5000 mAh", "battery", 5000.0),
])
def test_parse_number(raw, kind, expected):
    assert parse_number(raw, kind)[0] == expected


@pytest.mark.parametrize("raw,expected", [
    ("intel core i5 1235u", "Intel Core i5-1235U"), ("i7-13620H", "Intel Core i7-13620H"),
    ("m2", "Apple M2"), ("apple m3 pro", "Apple M3 Pro"), ("amd ryzen 7 5700u", "AMD Ryzen 7 5700u"),
])
def test_cpu(raw, expected):
    assert normalize_cpu(raw) == expected


def test_color_dictionary_inflections_and_synonyms():
    colors = load_taxonomy().dictionaries["colors"]
    assert colors.get("Черная") == "чёрный"
    assert colors.get("Space Gray") == "серый"
    assert colors.get("голубые") == "синий"
    assert colors.get("Hazel") is None  # неизвестное значение не выдумываем
