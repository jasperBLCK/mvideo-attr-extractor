"""Трафик-веса: детектор упоминаний атрибутов в поисковых запросах (attrx.data.traffic) — на маленьких словарях."""
import pytest

from attrx.data.traffic import (
    CategoryVocab,
    Detector,
    _phonetic,
    parse_value,
    query_numbers,
    score_category,
    tokens,
    unit_canon,
)


def phones() -> Detector:
    offers = [("Samsung", "Galaxy S25 Ultra 12/256GB Titanium Gray", 30), ("Samsung", "Galaxy A56 8/256GB", 20),
              ("Apple", "iPhone 17 Pro Max", 40), ("Apple", "iPhone 16", 25), ("HONOR", "600 Pro", 10),
              ("Xiaomi", "Redmi Note 15 Pro", 15), ("Xiaomi", "17 Ultra", 5)]
    params = [("Встроенная память (ROM)", True, None, v, n) for v, n in
              (("256 ГБ", 60), ("512 ГБ", 40), ("128 ГБ", 30), ("1024 ГБ", 10))]
    params += [("Встроенная память", False, None, v, n) for v, n in
               (("256 ГБ", 60), ("512 ГБ", 40), ("128 ГБ", 30), ("1 ТБ", 10))]
    params += [("Оперативная память (RAM)", True, None, v, n) for v, n in (("8 ГБ", 50), ("12 ГБ", 60), ("6 ГБ", 20))]
    params += [("Максимальная емкость карты памяти", True, "ГБ", v, n) for v, n in (("1024", 30), ("512", 10))]
    params += [("Диагональ", True, '"', v, n) for v, n in (("6.7", 50), ("6.1", 40), ("6.9", 30))]
    params += [("Емкость аккумулятора", True, "мАч", v, n) for v, n in (("5000", 70), ("4500", 30))]
    params += [("Цвет", True, None, v, n) for v, n in (("черный", 50), ("белый", 30), ("серый", 20))]
    params += [("Состояние", True, None, v, n) for v, n in (("новый", 120), ("восстановленный", 20))]
    params += [("Поддержка NFC", True, None, v, n) for v, n in (("Да", 100), ("Нет", 20))]
    fill = {"Встроенная память (ROM)": 0.97, "Встроенная память": 0.97, "Оперативная память (RAM)": 0.94,
            "Максимальная емкость карты памяти": 0.4, "Диагональ": 0.95, "Емкость аккумулятора": 0.96, "Цвет": 0.94,
            "Состояние": 0.9, "Поддержка NFC": 0.9}
    return Detector(CategoryVocab.build("761", "Смартфоны", offers, params, fill, path=["Смартфоны и связь"]))


def tvs() -> Detector:
    offers = [("TCL", "55C6K", 20), ("Samsung", "UE55DU7100UXRU", 20), ("Haier", "55 Smart TV S5", 15),
              ("Hisense", "65A6N", 10), ("LG", "43UT80006LA", 10)]
    params = [("Диагональ", True, '"', v, n) for v, n in (("55", 40), ("43", 30), ("65", 20), ("32", 20))]
    params += [("Диагональ", False, None, v, n) for v, n in (("55", 40), ("43", 30), ("65", 20), ("32", 20))]
    params += [("Частота обновления", True, "Гц", v, n) for v, n in (("60", 60), ("120", 30), ("144", 10))]
    params += [("Алиса", True, None, v, n) for v, n in (("Да", 20), ("Нет", 80))]
    params += [("Разрешение", True, None, v, n) for v, n in (("4K Ultra HD", 60), ("Full HD", 30), ("HD", 20))]
    fill = {"Диагональ": 0.98, "Частота обновления": 0.98, "Алиса": 0.72, "Разрешение": 0.95}
    return Detector(CategoryVocab.build("1682", "Телевизоры", offers, params, fill))


def appliances() -> tuple[Detector, Detector, Detector]:
    fridge = CategoryVocab.build(
        "666", "Холодильники", [("Haier", "C2F637CWRG", 30), ("ATLANT", "ХМ 4021-000", 20)],
        [("Размораживание морозильной камеры", True, None, "No Frost", 50),
         ("Размораживание морозильной камеры", True, None, "ручное", 30),
         ("Количество камер", True, None, "2", 70), ("Количество камер", True, None, "1", 20),
         ("Цвет", True, None, "белый", 60), ("Цвет", True, None, "черный", 20),
         ("Высота", True, "см", "200", 30), ("Высота", True, "см", "190", 30),
         ("Высота холодильника", False, None, "186 - 199 см", 30), ("Высота холодильника", False, None, "200 см и более", 30),
         ("Ширина холодильника", False, None, "56 - 65 см", 55), ("Ширина холодильника", False, None, "более 90 см", 5),
         ("Количество дверей", True, None, "2", 40), ("Количество дверей", True, None, "1", 30)],
        {"Размораживание морозильной камеры": 0.93, "Количество камер": 0.91, "Цвет": 0.99, "Высота": 0.99,
         "Высота холодильника": 0.99, "Ширина холодильника": 0.99, "Количество дверей": 0.48},
        path=["Холодильники и морозильники", "Холодильники и морозильные камеры"])
    washer = CategoryVocab.build(
        "2447", "Стиральные машины", [("LG", "F2Y1NS6W", 20), ("Haier", "HW70-BP12959B", 20)],
        [("Максимальная загрузка", True, "кг", v, n) for v, n in (("7", 30), ("8", 25), ("9", 15), ("6", 20))]
        + [("Макс загрузка (хлопок)", False, None, v, n) for v, n in
           (("7 - 7.9 кг", 30), ("8 - 8.9 кг", 25), ("9 кг и более", 15), ("6 - 6.9 кг", 20))]
        + [("Тип стиральной машины", True, None, v, n) for v, n in (("узкая", 50), ("с вертикальной загрузкой", 10))]
        + [("Сушка", True, None, v, n) for v, n in (("Да", 10), ("Нет", 60))],
        {"Максимальная загрузка": 0.99, "Макс загрузка (хлопок)": 0.99, "Тип стиральной машины": 0.97, "Сушка": 0.66})
    headphones = CategoryVocab.build(
        "31607", "Наушники", [("Apple", "AirPods Pro 3", 20), ("Apple", "AirPods 4", 20), ("JBL", "Tune 520BT", 10)],
        [("Тип подключения", True, None, v, n) for v, n in (("True Wireless", 50), ("проводной/ беспроводной", 20))]
        + [("Тип подключения", True, None, "беспроводной", 20)]
        + [("Тип зарядки", True, None, v, n) for v, n in (("беспроводная", 10), ("быстрая", 10))]
        + [("Пар амбушюров в комплекте", True, None, v, n) for v, n in (("3", 30), ("2", 20), ("4", 10))],
        {"Тип подключения": 0.88, "Тип зарядки": 0.28, "Пар амбушюров в комплекте": 0.6})
    return Detector(fridge), Detector(washer), Detector(headphones)


def test_units_and_values():
    assert unit_canon("ГБ") == "гб" and unit_canon('"') == "дюйм" and unit_canon("МПикс") == "мп"
    assert unit_canon("mAh") == "мач" and unit_canon("непонятно") is None
    assert parse_value("256 ГБ", None) == ("num", (256.0, "гб"))
    assert parse_value("1 ТБ", None) == ("num", (1024.0, "гб"))  # 1 ТБ == 1024 ГБ
    assert parse_value("8 ГБ/256 ГБ", None) == ("pair", (8.0, 256.0))
    assert parse_value('15.6"(39.6 см)', None) == ("num", (15.6, "дюйм"))
    assert parse_value("3000-5000 мАч", None) == ("range", (3000.0, 5000.0, "мач"))
    assert parse_value("от 15 до 16.9", None) == ("range", (15.0, 16.9, None))
    assert parse_value("9 кг и более", None)[1][:2] == (9.0, float("inf"))
    assert parse_value("узкая", None) is None


def test_query_numbers_units_and_memory_pairs():
    nums, pairs, used = query_numbers("смартфон 8/256gb 6.7 дюйма 5000 мач")
    assert pairs == [(8.0, 256.0, "8/256gb")] and {"8", "256"} <= used
    assert {(x, u) for x, u, *_ in nums} >= {(6.7, "дюйм"), (5000.0, "мач")}
    nums, _, _ = query_numbers('телевизор 55" и 1тб')
    assert {(x, u) for x, u, *_ in nums} == {(55.0, "дюйм"), (1024.0, "гб")}


def test_tokens_aliases_and_glued_forms():
    assert tokens("айфон17про макс") == ["iphone", "17", "pro", "max"]
    assert tokens("самсунг с25") == ["самсунг", "s25"]  # кириллическая «с» перед цифрами — это латинская s
    assert tokens("холодильник нофрост") == ["холодильник", "no", "frost"]
    assert _phonetic("самсунг") == _phonetic("Samsung") and _phonetic("хайер") == _phonetic("Haier")


@pytest.mark.parametrize("query, expected, absent", [
    ("смартфон", set(), {"vendor", "model"}),  # только название категории — атрибутов нет
    ("смартфоны самсунг", {"vendor"}, {"model"}),
    ("samsung s25 ultra 12/256gb", {"vendor", "model", "Оперативная память (RAM)", "Встроенная память (ROM)"}, set()),
    ("iphone 17 pro max 256", {"model", "Встроенная память (ROM)", "Встроенная память"}, {"Диагональ"}),
    ("samsung s26 ultra 1tb", {"Встроенная память (ROM)"}, {"Максимальная емкость карты памяти"}),
    ("смартфон 6.7 дюйма 5000 мач черный", {"Диагональ", "Емкость аккумулятора", "Цвет"}, {"vendor"}),
    ("восстановленный смартфон с nfc", {"Состояние", "Поддержка NFC"}, set()),
    ("honor 600", {"vendor", "model"}, set()),
])
def test_phone_mentions(query, expected, absent):
    found = phones().mentions(query)
    assert expected <= set(found), found
    assert not (absent & set(found)), found


@pytest.mark.parametrize("query, expected, absent", [
    ("телевизоры 55", {"Диагональ"}, {"model"}),
    ("телевизор tcl 55", {"vendor", "Диагональ"}, set()),  # 55 после бренда — диагональ, а не номер модели
    ("tcl 55c6k", {"vendor", "model"}, set()),
    ("телевизор 4к с алисой 120гц", {"Разрешение", "Алиса", "Частота обновления"}, set()),
    ("телевизор 58 дюймов", {"Диагональ"}, set()),  # значения нет в каталоге, но единица однозначна
])
def test_tv_mentions(query, expected, absent):
    found = tvs().mentions(query)
    assert expected <= set(found), found
    assert not (absent & set(found)), found


def test_appliance_mentions():
    fridge, washer, headphones = appliances()
    found = fridge.mentions("холодильники двухкамерные nofrost белый")
    assert {"Количество камер", "Размораживание морозильной камеры", "Цвет"} <= set(found)
    assert "Количество камер" in fridge.mentions("холодильники двухкамерные")  # «камеры» есть и в пути категории
    assert "Количество дверей" in fridge.mentions("холодильники двухдверные")
    assert "Количество камер" in fridge.mentions("холодильник 2 камеры")
    found = fridge.mentions("холодильник 200 см")
    assert {"Высота", "Высота холодильника"} <= set(found) and "Ширина холодильника" not in found  # бакет-близнец
    assert {"Максимальная загрузка", "Макс загрузка (хлопок)"} <= set(washer.mentions("стиральная машинка 7 кг"))
    assert "Тип стиральной машины" in washer.mentions("стиральная машинка узкая lg")
    assert "Сушка" in washer.mentions("стиральная машина с сушкой")
    found = headphones.mentions("наушники беспроводные")
    assert "Тип подключения" in found and "Тип зарядки" not in found  # слово отдаём самому заполненному свойству
    found = headphones.mentions("airpods pro 3")
    assert "model" in found and "Пар амбушюров в комплекте" not in found  # 3 — номер модели


def test_score_category_weights_are_click_shares():
    det = tvs()
    queries = [("телевизоры 55", 60), ("телевизор", 30), ("телевизор samsung", 10)]
    r = score_category(det.v, queries)
    assert r.clicks == 100 and r.weights["Диагональ"] == 0.6 and r.weights["vendor"] == 0.1
    assert r.any_attribute == 0.7 and r.any_param == 0.6
    assert r.examples["Диагональ"][0] == ("телевизоры 55", 60, "55")


def test_group_summary_weights_by_clicks_and_maps_gold_names():
    from attrx.data.traffic import CategoryTraffic, group_summary

    kinds = {"brand": 0.5, "model": 0.5, "number": 0.1, "color": 0.0, "text": 0.0}
    phones = CategoryTraffic("761", "Смартфоны", 300, 10, 0.9, 0.1, kinds,
                             {"vendor": 0.8, "Встроенная память (ROM)": 0.1}, {}, [])
    iphone = CategoryTraffic("914", "iPhone", 100, 5, 0.8, 0.2, kinds,
                             {"model": 0.8, "Встроенная память": 0.2}, {}, [])
    g = group_summary([phones, iphone], {"brand": ["vendor"], "storage": ["Встроенная память (ROM)", "Встроенная память"],
                                         "battery": ["Емкость аккумулятора"]})
    assert g["clicks"] == 400 and g["category_ids"] == ["761", "914"]
    assert g["weights"]["vendor"] == 0.6 and g["weights"]["model"] == 0.2  # 0.8·300/400 и 0.8·100/400
    assert g["by_attribute"] == {"brand": 0.6, "storage": 0.075, "battery": 0.0}  # max(0.1·300/400, 0.2·100/400)
    assert g["any_attribute"] == 0.875
