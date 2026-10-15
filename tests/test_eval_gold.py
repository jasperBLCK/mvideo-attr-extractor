"""Эталон из каталога и метрики на неполном эталоне: нормализация значений фида, правила сравнения, new_fills,
извлекаемый recall, категории по category_id. Без data/processed (кроме сэмпла data/sample/real_*.jsonl)."""
import pytest

from attrx.config import ROOT
from attrx.eval.gold import canon_values, model_key, number_in_text, resolve
from attrx.eval.metrics import evaluate, gold_matches, is_match, match_key, traffic_weights
from attrx.normalize import clean_cpu, cpu_key, is_range, model_core, normalize_colors, parse_number
from attrx.schema import AttributeValue, Card, ExtractionResult
from attrx.taxonomy import Taxonomy, load_taxonomy, norm


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


def spec(tax, cat, attr):
    return tax.categories[cat].attributes[attr]


@pytest.mark.parametrize("text,expected", [
    ('6.6" и более', True), ("3000-5000 мАч", True), ("от 15 до 16.9", True), ("до 2 л", True), ('39" - 48"', True),
    ('15.6"(39.6 см)', False), ("256 ГБ", False), ("2200", False), ("с нижней морозильной камерой", False),
])
def test_bucket_ranges_detected(text, expected):
    assert is_range(text) is expected


@pytest.mark.parametrize("raw,kind,expected", [
    ("SSD 1 ТБ", "storage_drive", 1024.0), ('15.6"(39.6 см)', "diagonal", 15.6), ("1024 ГБ", "memory", 1024.0),
    ("до 35 кв. м", "area_m2", 35.0), ("30 мин", "time_h", 0.5), ("2,2 кВт", "power_w", 2200.0),
    ("1200 об/мин", "rpm", 1200.0), ("45 см", "length_cm", 45.0), ("450 мм", "length_cm", 45.0), ("1.7 л", "volume_l", 1.7),
])
def test_parse_number_units(raw, kind, expected):
    assert parse_number(raw, kind)[0] == pytest.approx(expected)


def test_feed_value_with_separate_unit(tax):
    assert canon_values(spec(tax, "smartphone", "battery"), "5000", "мАч", tax) == ([5000.0], True)
    assert canon_values(spec(tax, "headphones", "battery_life"), "30", "мин", tax)[0] == [0.5]
    assert canon_values(spec(tax, "split_system", "btu"), "0", "BTU", tax) == ([], False)  # 0 — заглушка, не эталон


def test_exact_search_variant_preferred_over_bucket(tax):
    params = {"Диагональ экрана": [(False, None, "от 15 до 16.9"), (True, None, '15.6"(39.6 см)')]}
    g = resolve(spec(tax, "laptop", "screen"), "Lenovo", "V15", params, tax)
    assert g["values"] == [15.6] and g["source"] == "Диагональ экрана"
    # только бакет-диапазон — эталона нет (не угадываем точное значение)
    assert resolve(spec(tax, "laptop", "screen"), "Lenovo", "V15", {"Диагональ экрана": [(False, None, "от 15 до 16.9")]},
                   tax) is None
    # бакет-категория (не диапазон) годится, если точного варианта нет
    g = resolve(spec(tax, "headphones", "connection"), "JBL", "Wave", {"Тип подключения": [(False, None, "True Wireless")]}, tax)
    assert g["values"] == ["True Wireless"]


def test_colors_to_base_palette_and_multi_valued(tax):
    colors = tax.dictionaries["colors"]
    assert normalize_colors("черный/ серебристый", colors) == ["чёрный", "серебристый"]
    assert normalize_colors("темно-серый", colors) == ["серый"]
    assert normalize_colors("Midnight", colors) == ["чёрный"]
    assert normalize_colors("титановый/ белый", colors) == ["серый", "белый"]
    assert normalize_colors("другие цвета", colors) == []
    g = resolve(spec(tax, "smartphone", "color"), "Apple", "iPhone 16",
                {"Цвет": [(False, None, "другие цвета"), (True, None, "ультрамарин")]}, tax)
    assert g["values"] == ["синий"]
    # «другие цвета» в фильтре и нет точного — берём «Цвет производителя»
    g = resolve(spec(tax, "smartphone", "color"), "OnePlus", "15R",
                {"Цвет": [(False, None, "другие цвета")], "Цвет производителя": [(True, None, "Mint Breeze")]}, tax)
    assert g["values"] == ["зелёный"] and g["source"] == "Цвет производителя"


def test_model_core_strips_variant_parts(tax):
    colors = tax.dictionaries["colors"]
    assert model_core("Galaxy S25 5G 12/128GB Серый", "Samsung", colors) == "Galaxy S25"
    assert model_core("iPhone 17 256GB Black eSim (без RuStore)", "Apple", colors) == "iPhone 17"
    assert model_core('Katana 17/17.3"/Core i5-12450H/8/512/RTX3050/noOS/черный', "MSI", colors) == "Katana 17"
    assert model_core("Смартфон Samsung Galaxy A55 8/256 ГБ голубой", "Samsung", colors) == "Galaxy A55"
    assert model_key("Galaxy S24+") == model_key("galaxy s24 plus") != model_key("Galaxy S24")
    # мусорная «модель» без латиницы и цифр не становится эталоном
    assert canon_values(spec(tax, "kettle", "model"), "Чайник электрический стеклянный", None, tax, "X") == ([], False)


def test_model_alternatives_from_series(tax):
    g = resolve(spec(tax, "smartphone", "model"), "Samsung", "A36 8/256Gb 5G Dual Sim White",
                {"Серия": [(True, None, "Galaxy A36")], "Модель": [(True, None, "SM-A366")]}, tax)
    assert g["values"] == ["A36", "Galaxy A36", "SM-A366"]
    s = spec(tax, "smartphone", "model")
    assert gold_matches("Galaxy A36", g["values"], s, tax, "Samsung")
    assert gold_matches("Samsung A36", g["values"], s, tax, "Samsung")
    assert not gold_matches("Galaxy A35", g["values"], s, tax, "Samsung")


def test_cpu_canon_and_family(tax):
    assert clean_cpu("AMD Ryzen 3 Ryzen 3 7320U 2.4 ГГц") == "AMD Ryzen 3 7320U"
    assert clean_cpu("Apple Apple M5 Pro 4.61 ГГц") == "Apple M5 Pro"
    assert cpu_key("Intel Core i5 12450H 2 ГГц") == cpu_key("i5-12450H") == "i512450h"
    s = spec(tax, "laptop", "cpu")
    g = resolve(s, "MSI", "Katana", {"Процессор": [(True, None, "Intel Core i5 12450H 2.0 ГГц")],
                                     "Модель процессора": [(True, None, "Core i5")]}, tax)
    assert g["values"] == ["Intel Core i5-12450H", "Intel Core i5"]
    assert gold_matches("Intel Core i5-12450H", g["values"], s, tax)
    assert gold_matches("Core i5", g["values"], s, tax)
    assert not gold_matches("Intel Core i5-1235U", g["values"], s, tax)  # другая модель того же семейства
    assert gold_matches("Intel Core i5-1235U", ["Intel Core i5"], s, tax)  # эталон — только семейство
    assert not is_match("m4pro", ["m4"], s)  # M4 Pro — не M4


def test_numeric_tolerance_and_units(tax):
    s = spec(tax, "smartphone", "screen")
    assert gold_matches(6.7, [6.67], s, tax)  # 0.45% — в пределах 1%
    assert not gold_matches(6.5, [6.67], s, tax)
    st = spec(tax, "smartphone", "storage")
    assert gold_matches("1 ТБ", [1024.0], st, tax)
    assert match_key("256 ГБ", st, tax) == 256.0


def test_extractable_numbers_in_text():
    t = norm("Смартфон Xiaomi Redmi Note 13 Pro 12/512GB, экран 6,67\", 5 000 мАч")
    assert number_in_text(512.0, "memory", t) and number_in_text(12.0, "ram", t)
    assert number_in_text(6.67, "diagonal", t) and number_in_text(5000.0, "battery", t)
    assert number_in_text(1024.0, "memory", norm("Galaxy S24 Ultra 12/1TB"))
    assert not number_in_text(8.0, "ram", norm("Ноутбук 8 ядер"))  # голое «8» — не доказательство


def _res(card_id, cat, **vals):
    attrs = [AttributeValue(attribute=a, title=a, value=v, confidence=0.9 if v is not None else 0.0,
                            status="accepted" if v is not None else "missing") for a, v in vals.items()]
    return ExtractionResult(card_id=card_id, category=cat, category_confidence=1.0, attributes=attrs)


def test_incomplete_gold_new_fills_not_false_positives(tax):
    results = [
        _res("1", "smartphone", brand="Samsung", storage=256.0, color="чёрный", battery=5000.0),
        _res("2", "smartphone", brand="Apple", storage=128.0, color="синий", battery=None),
    ]
    gold = {"1": {"brand": "Samsung", "storage": 256.0, "color": ["серый", "чёрный"]},  # battery в каталоге не заполнен
            "2": {"brand": "Apple", "storage": 256.0, "battery": 3349.0}}
    meta = {"1": {"extractable": {"brand": True, "storage": True, "color": True}},
            "2": {"extractable": {"brand": True, "storage": True, "battery": False}}}
    m = evaluate(results, gold, tax, meta=meta)
    a = m["per_attribute"]
    assert a["battery"]["new_fills"] == 1 and a["battery"]["fp"] == 0 and a["battery"]["fn"] == 1
    assert a["color"]["tp"] == 1 and a["color"]["new_fills"] == 1  # многозначный эталон: «чёрный» допустим
    assert a["storage"]["tp"] == 1 and a["storage"]["fp"] == 1 and a["storage"]["fn"] == 1  # неверное значение
    assert a["battery"]["recall_extractable"] is None and a["battery"]["extractable_share"] == 0.0
    assert a["storage"]["recall_extractable"] == 0.5
    assert a["battery"]["catalog_fill"] == 0.5 and a["battery"]["union_fill"] == 1.0
    assert m["micro"]["precision"] == pytest.approx(4 / 5)  # 4 верных, 1 неверное; new_fills в знаменатель не входят


def test_traffic_weights_from_clicks_and_yaml(tax):
    w = traffic_weights(tax, {"761": {"Встроенная память (ROM)": 0.2}, "914": {"Встроенная память (ROM)": 0.6}},
                        clicks={"761": 3.0, "914": 1.0, "999": 4.0})
    assert w[("smartphone", "storage")] == pytest.approx(0.5 * (3 * 0.2 + 1 * 0.6) / 4)
    assert w[("generic", "brand")] == pytest.approx(0.5 * 1.0)  # клики вне таксономии -> generic


def test_category_from_catalog_id(tax):
    assert tax.detect_category(Card(id="1", title="Apple iPhone 15", category_id="914")) == ("smartphone", 1.0)
    # известная категория вне таксономии -> generic, а не «смартфон» по ключевому слову
    assert tax.detect_category(Card(id="2", title="Чехол для смартфона Samsung", category_id="1189")) == ("generic", 1.0)
    assert tax.detect_category(Card(id="3", title="Смартфон Samsung Galaxy A55"))[0] == "smartphone"
    assert tax.category_of_id(2447) == "washing_machine"


def test_taxonomy_validation(tmp_path):
    (tmp_path / "dictionaries").mkdir()
    (tmp_path / "dictionaries" / "colors.yaml").write_text("чёрный: [black]\n", encoding="utf-8")
    bad = tmp_path / "taxonomy.yaml"
    bad.write_text("categories:\n  generic:\n    attributes:\n      color: {title: Цвет, type: enum, dictionary: nope}\n",
                   encoding="utf-8")
    with pytest.raises(ValueError, match="справочника"):
        Taxonomy(bad)
    bad.write_text("categories:\n  a:\n    category_ids: [1]\n    attributes: {}\n  b:\n    category_ids: [1]\n"
                   "    attributes: {}\n  generic:\n    attributes: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="указана и в"):
        Taxonomy(bad)


@pytest.mark.skipif(not (ROOT / "data/sample/real_gold.jsonl").exists(), reason="нет data/sample/real_gold.jsonl")
def test_real_sample_gold_and_pipeline_smoke(tax):
    from attrx.config import Settings
    from attrx.eval.gold import load_jsonl
    from attrx.io import load_cards
    from attrx.pipeline import Extractor

    gold, meta = load_jsonl(ROOT / "data/sample/real_gold.jsonl")
    cards = load_cards(ROOT / "data/sample/real_cards.jsonl")
    assert set(gold) <= {c.id for c in cards} and len(gold) >= 0.9 * len(cards)
    ex = Extractor(settings=Settings(llm_enabled=False, ml_enabled=False))
    m = evaluate(list(ex.extract_many(cards[:120])), gold, tax, meta=meta)
    assert m["n_cards"] > 0 and 0 <= m["micro"]["precision"] <= 1 and m["per_category"]
