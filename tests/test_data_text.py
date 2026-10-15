"""Очистка HTML и разбор блока «Характеристики:» (attrx.data.text) — на маленьких строках, без data/raw."""
from attrx.data import split_of
from attrx.data.text import clean_html, clean_title, parse_kv, prepare_description, split_characteristics
from attrx.data.yml import categories_frame, split_param_name


def test_clean_html_breaks_entities_and_spaces():
    raw = "<p>Мощность&nbsp;1400&nbsp;Вт</p><p>Тихий&amp;быстрый<br/>Объём 0,6 л</p>"
    assert clean_html(raw) == "Мощность 1400 Вт\n\nТихий&быстрый\nОбъём 0,6 л"  # абзацы <p> — через пустую строку


def test_clean_html_double_escaped_and_lists():
    raw = "<p>Смартфон&lt;br/&gt;&lt;br/&gt;Экран 6,3\"</p><ul><li>Цвет: черный</li><li>NFC: да</li></ul>"
    assert clean_html(raw) == 'Смартфон\n\nЭкран 6,3"\n\n• Цвет: черный\n• NFC: да'


def test_clean_html_keeps_comparisons_and_crlf():
    assert clean_html("Размер < 5 мм и >3 см\r\nстрока\u200b2") == "Размер < 5 мм и >3 см\nстрока2"
    assert clean_html(None) == "" and clean_html("") == ""
    assert clean_title("  Ключ&nbsp;гаечный <b>GSS</b>\n -  ") == "Ключ гаечный GSS -"


def test_characteristics_block_plain_text():
    text, attrs = prepare_description(
        "Двухголовый ключ 8x10мм.\n\nХарактеристики:\n• Тип ключа: накидной\n• Размер ключа: 8 мм x 10 мм\n"
        "• Материал: хромованадиевая сталь.\n\nЛучшие товары из Китая")
    assert attrs == {"Тип ключа": "накидной", "Размер ключа": "8 мм x 10 мм", "Материал": "хромованадиевая сталь"}
    assert list(attrs) == ["Тип ключа", "Размер ключа", "Материал"]  # порядок сохранён
    assert text == "Двухголовый ключ 8x10мм.\n\nЛучшие товары из Китая"


def test_characteristics_block_html_stops_at_next_section():
    text, attrs = prepare_description(
        "Офисный стул.<br><br>Технические характеристики:<br><br>• Материал каркаса: металл.<br>"
        "• Высота сиденья: 49 см.<br><br><br>Преимущества:<br>• практичное решение — легко мыть")
    assert attrs == {"Материал каркаса": "металл", "Высота сиденья": "49 см"}
    assert "Преимущества:\n• практичное решение — легко мыть" in text and "Материал" not in text


def test_characteristics_variants_dash_star_and_lone_bullet():
    _, a = prepare_description("Характеристики товара:\n- Цвет: красный\n- Вес: 2 кг")
    assert a == {"Цвет": "красный", "Вес": "2 кг"}
    _, a = prepare_description("ХАРАКТЕРИСТИКИ\n* Объем: 1,7 л\n* Мощность: 2200 Вт")
    assert a == {"Объем": "1,7 л", "Мощность": "2200 Вт"}
    _, a = prepare_description("Характеристики:\n\n•\nВместимость: до 27 бутылок.\n\n•\nШум: до 40 дБ.")
    assert a == {"Вместимость": "до 27 бутылок", "Шум": "до 40 дБ"}


def test_characteristics_inline_and_style_change():
    text, a = prepare_description("Кухонный нож. Характеристики: Материал: сталь. Длина лезвия: 14 см.")
    assert a == {"Материал": "сталь", "Длина лезвия": "14 см"} and text == "Кухонный нож."
    # маркированный блок не поглощает следующий абзац без маркера «Важно: ...»
    text, a = prepare_description("Характеристики:\n• Цвет: белый\n• Вес: 1 кг\nВажно: хранить в сухом месте")
    assert a == {"Цвет": "белый", "Вес": "1 кг"} and text == "Важно: хранить в сухом месте"


def test_no_block_no_change():
    s = "Характеристики модели:\nКАРАОКЕ\nподдержка 16:9"
    assert split_characteristics(s) == (s, {})
    # одна-две пары без заголовка — не блок (это может быть обычный текст)
    s2 = "• Важно: перед использованием прочитать инструкцию\n• Цвет: белый"
    assert split_characteristics(s2) == (s2, {})
    # три и больше маркированных пар подряд без заголовка — блок
    _, a = split_characteristics("Описание.\n• Цвет: белый\n• Вес: 1 кг\n• Тип: настенный")
    assert a == {"Цвет": "белый", "Вес": "1 кг", "Тип": "настенный"}


def test_parse_kv():
    assert parse_kv("• Габариты (ШхГ): 52 х 47 см.") == ("Габариты (ШхГ)", "52 х 47 см", True)
    assert parse_kv("Время работы: 10:30") == ("Время работы", "10:30", False)
    assert parse_kv("https://example.com: ссылка") is None
    assert parse_kv("просто строка без пары") is None
    assert parse_kv("• Ключ:") is None


def test_yml_helpers_and_split():
    assert split_param_name("Ширина_search") == ("Ширина", True)
    assert split_param_name("Ширина") == ("Ширина", False)
    cats = categories_frame([{"@id": "1", "#text": "Смартфоны и связь"},
                             {"@id": "2", "@parentId": "1", "#text": "Смартфоны"},
                             {"@id": "3", "@parentId": "2", "#text": "iPhone"}]).set_index("id")
    assert cats.loc["3", "path"] == ["Смартфоны и связь", "Смартфоны", "iPhone"]
    assert cats.loc["3", "depth"] == 2 and bool(cats.loc["3", "is_leaf"]) and not bool(cats.loc["1", "is_leaf"])
    assert split_of(761) == split_of("761") and split_of("761") in {"train", "dev", "holdout"}
    shares = [split_of(i) for i in range(20000)]
    assert 0.07 < shares.count("holdout") / len(shares) < 0.13


def test_value_in_text_units_and_numbers():
    from attrx.data.taxo import value_in_text

    t = "смартфон samsung galaxy s25 12/256gb navy, экран 6,7 дюйма, 5 000 мач, 1 тб"
    assert value_in_text("256 ГБ", t) == "exact"  # 256 ГБ ~ 256gb
    assert value_in_text("256", t, unit="ГБ") == "exact"
    assert value_in_text("12", t, unit="ГБ") == "loose"  # число есть, но без единицы (12/256gb)
    assert value_in_text("6.7", t, unit='"') == "exact"
    assert value_in_text("5000", t, unit="мАч") == "exact"
    assert value_in_text("1024 ГБ", t) == "exact"  # 1024 ГБ ~ 1 ТБ
    assert value_in_text("Navy", t) == "exact"
    assert value_in_text("6.6\" и более", t) is None  # бакет фильтра — не значение из текста
    assert value_in_text("67", t) is None  # не кусок другого числа (6,7)
