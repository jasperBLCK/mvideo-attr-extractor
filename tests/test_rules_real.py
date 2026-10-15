"""Правила и словари на реальных паттернах каталога: бренды каталога и защита от омонимов/совместимости/платформ,
модель после бренда, пара памяти, процессоры, диагональ, габариты, загрузка, выученное соответствие справочников,
калибровка, смещения цитат. Без data/processed — только config/ (справочники в git)."""
import pytest

from attrx.confidence import Calibrator, aggregate_full
from attrx.config import Settings
from attrx.dictgen import is_placeholder, translit_variants, vendor_key
from attrx.eval.calibrate import pav_curve, wilson_lower
from attrx.eval.gold import model_key
from attrx.eval.metrics import match_key
from attrx.extractors import rules, seller
from attrx.normalize import find_cpu, normalize_cpu
from attrx.pipeline import Extractor
from attrx.schema import Candidate, Card
from attrx.taxonomy import BrandDictionary, load_taxonomy, lower_keep, pick_mention


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


@pytest.fixture(scope="module")
def ex():
    return Extractor(settings=Settings(llm_enabled=False, ml_enabled=False))


def cands(tax, cat, title, description="", seller_attrs=None):
    card = Card(id="t", title=title, description=description, seller_attrs=seller_attrs or {})
    return card, rules.extract(card, tax.categories[cat], tax)


def best(cs, attr):
    xs = [c for c in cs if c.attribute == attr]
    return max(xs, key=lambda c: c.score) if xs else None


# ------------------------------------------------------------------ бренды -----------------------------------------
@pytest.mark.parametrize("title,brand", [
    ("Холодильник Bosch KGN39VI25R", "Bosch"),
    ("Фотошторы 365home Узор из листьев FP51187092", "365home"),
    ("Электрочайник Zigmund & Shtain KE-83 SD", "Zigmund & Shtain"),
    ("Стиральная машина Бирюса WM-SM610/08 S", "Бирюса"),
    ("Холодильник Атлант ХМ-4623-159-ND", "ATLANT"),  # кириллица по транслиту -> канон каталога
    ("Смартфон Самсунг Galaxy A55", "Samsung"),  # ручной синоним
])
def test_catalog_brands_found_in_title(tax, title, brand):
    _, cs = cands(tax, "generic", title)
    assert best(cs, "brand").value == brand


def test_compatibility_and_platform_contexts_are_not_brands(tax):
    _, cs = cands(tax, "generic", "Чехол Deppa для iPhone 15 Pro прозрачный")
    assert best(cs, "brand").value == "Deppa"  # «для iPhone» — бренд устройства
    _, cs = cands(tax, "tv", "Телевизор 55 дюймов Hyundai H-LED55BU7003 с Яндекс ТВ")
    assert best(cs, "brand").value == "Hyundai"
    _, cs = cands(tax, "tv", "Телевизор Haier 65 Smart TV S3 с Google TV")
    assert best(cs, "brand").value == "Haier"
    # только бренд устройства — бренд товара не назван: на проверку (низкая оценка), а не «Apple»
    _, cs = cands(tax, "generic", "Защитное стекло для iPhone 15")
    b = best(cs, "brand")
    assert b is None or b.score < 0.5


def test_homonym_brands_not_matched_in_text(tax):
    bd = tax.dictionaries["brands"]
    assert isinstance(bd, BrandDictionary)
    assert bd.is_word("СМ") and bd.is_word("Wireless")  # «45 см», «True Wireless» — не бренды
    _, cs = cands(tax, "headphones", "Наушники True Wireless JBL Wave Buds 2 белые")
    assert best(cs, "brand").value == "JBL"
    _, cs = cands(tax, "generic", "Встраиваемая посудомоечная машина 45 см Beko BDIS16020")
    assert best(cs, "brand").value == "Beko"
    # поле продавца «Бренд» принимает и омоним (это явное значение, а не слово в тексте)
    card = Card(id="s", title="Наушники", seller_attrs={"Бренд": "Wireless"})
    assert [c.value for c in seller.extract(card, tax.categories["headphones"], tax) if c.attribute == "brand"] == \
        ["Wireless"]


def test_sub_brand_mention_keeps_line_in_model(tax):
    _, cs = cands(tax, "smartphone", "Смартфон Xiaomi Redmi Note 13 8/256GB Ice Blue")
    assert best(cs, "brand").value == "Xiaomi"
    assert best(cs, "model").value == "Redmi Note 13"


def test_pick_mention_skips_compat_chain(tax):
    bd = tax.dictionaries["brands"]
    ms = bd.mentions(lower_keep("Чехол для Samsung Galaxy S24 Deppa"))
    assert pick_mention(ms).canon == "Deppa"  # Samsung — «для», Galaxy — вплотную за ним


def test_brand_dictionary_builder_helpers():
    assert is_placeholder("нет бренда") and is_placeholder("No Name") and is_placeholder("###")
    assert not is_placeholder("Bosch")
    assert vendor_key("Western Digital (WD)") == "westerndigital"
    assert "atlant" in translit_variants("атлант")


def test_placeholder_vendor_is_not_gold(tax):
    spec = tax.categories["generic"].attributes["brand"]
    assert match_key("нет бренда", spec, tax) is None
    assert match_key("Самсунг", spec, tax) == match_key("Samsung", spec, tax)


# ------------------------------------------------------------------ модель -----------------------------------------
@pytest.mark.parametrize("cat,title,model", [
    ("fridge", "Холодильник Hisense RB440N4AWE белый", "RB440N4AWE"),
    ("tv", "Телевизор LG 50QNED80A6A.ARUG", "50QNED80A6A.ARUG"),
    ("washing_machine", "Стиральная машина Hiberg i-DDQ6 - 612 G золотистая", "i-DDQ6"),
    ("vacuum", "Пылесос напольный Thomas Cycloon Hybrid Family & Pets 786552", "Cycloon Hybrid Family & Pets 786552"),
    ("generic", "Триммер садовый Huter GGT-2900T", "GGT-2900T"),
    ("generic", "Светильник потолочный Citilux Citilux MEGA CL752350G", "MEGA CL752350G"),  # повтор бренда
    ("kettle", "Электрочайник Oursson Электрический чайник Oursson KE1716P/RD (Красный)", "KE1716P"),
])
def test_model_after_brand(tax, cat, title, model):
    _, cs = cands(tax, cat, title)
    assert best(cs, "model").value == model


def test_model_display_keeps_suffix_but_matches_catalog_key(tax):
    _, cs = cands(tax, "fridge", "Холодильник Indesit ITS 4160 W")
    m = best(cs, "model").value
    assert m == "ITS 4160 W"
    assert model_key(m) == model_key("ITS 4160 W", "Indesit")  # эталон и предсказание — одна нормализация


def test_article_model_style_from_train(tax):
    card = Card(id="a", title="Ручка для письма Brauberg шариковая синяя, комплект 12 штук, 0,35 мм, 880015")
    ctx = rules.Ctx.of(card, "generic")
    _, pick = rules.extract_brand(ctx, "brand", tax)
    styles = {"Brauberg": {"n": 100, "after": 0.03, "article": 0.6}}
    out = rules.extract_model(ctx, "model", tax, pick, styles=styles)
    assert out[0].value == "880015" and out[0].score > 0.6
    out = rules.extract_model(ctx, "model", tax, pick, styles={})  # без статистики — описание с низкой оценкой
    assert out[0].score < 0.5


# ------------------------------------------------------------------ память и процессор -----------------------------
@pytest.mark.parametrize("cat,title,ram,storage", [
    ("smartphone", "Смартфон Samsung Galaxy S24 12/256GB", None, 256),
    ("smartphone", "Смартфон realme C71 8+256Gb", None, 256),
    ("tablet", "Планшет HUAWEI MatePad Pro 8+128GB WiFi", 8, 128),
    ("laptop", "Ноутбук ASUS Zenbook A14 14\" OLED/Snapdragon X1/16Гб/1Тб/Win 11", 16, 1024),
    ("laptop", "Ноутбук Maibenben M545/15.6\"/AMD Ryzen 5 4500U/16/512/Linux", 16, 512),
])
def test_memory_pairs(tax, cat, title, ram, storage):
    _, cs = cands(tax, cat, title)
    if ram is not None:
        assert best(cs, "ram").value == ram
    assert best(cs, "storage").value == storage


def test_memory_pair_needs_left_boundary(tax):
    # «RTX4060/16ГБ» — не пара (60 ГБ ОЗУ / 16 ГБ накопитель), ОЗУ — подписанные 32ГБ
    _, cs = cands(tax, "laptop", "Ноутбук ASUS ProArt H7606WV-ME106X/OLED/AMD Ryzen 9 AI/RTX4060/32ГБ/2ТБ/Windows 11 Pro")
    assert all(c.value != 60 for c in cs if c.attribute == "ram")
    assert best(cs, "storage") is None or best(cs, "storage").value != 32


@pytest.mark.parametrize("text,cpu", [
    ("apple macbook air 13 m5 16/512", "Apple M5"),
    ("core i3-1115g4/8/256", "Intel Core i3-1115G4"),
    ("core ultra 5-226v/16гб", "Intel Core Ultra 5 226v"),
    ("intel core 5 210h, 16 гб", "Intel Core 5 210h"),
    ("core i5 120u/16гб/1тб", "Intel Core 5 120u"),
    ("ноутбук e1504ga/15.6\"/n100/8/256", "Intel N100"),
    ("core i3-n305/16/512gb", "Intel Core i3-N305"),
    ("amd ryzen ai 9 hx 370, 32 гб", "AMD Ryzen AI 9 HX 370"),
])
def test_cpu_patterns(text, cpu):
    m, _w = find_cpu(text)
    assert normalize_cpu(text[m.start():m.end()]) == cpu


def test_cpu_bare_ultra_is_low_confidence():
    m, w = find_cpu("ноутбук игровой thunderobot zero ultra 7 gray")
    assert w < 0.7  # «Zero Ultra 7» — название модели, не процессор: на проверку


# ------------------------------------------------------------------ диагональ и габариты -----------------------------
def test_diagonal_hyphenated_and_rejects_aspect_and_hz(tax):
    _, cs = cands(tax, "smartphone", "Смартфон HONOR 400 Lite 8/256GB", "6,7-дюймовый AMOLED-экран")
    assert best(cs, "screen").value == 6.7
    _, cs = cands(tax, "tv", "Телевизор Hisense 75A6BG", "Экраном формата 16:9 и частотой 60 Гц, экраном 4K")
    assert all(c.value not in (16.0, 4.0, 60.0) for c in cs if c.attribute == "screen")


def test_tv_size_from_model_code_needs_corroboration(tax):
    _, cs = cands(tax, "tv", "Телевизор Hisense 75A6BG")
    c = best(cs, "screen")
    assert c.value == 75 and c.score < 0.7  # только из кода модели — на проверку
    _, cs = cands(tax, "tv", "Телевизор Hisense 75A6BG", "Диагональ 75 дюймов (190 см)")
    assert best(cs, "screen").value == 75 and best(cs, "screen").score >= 0.8


def test_tv_cm_diagonal_snaps_to_standard_size(tax):
    _, cs = cands(tax, "tv", "Телевизор LG OLED65G3RLA", "с экраном диагональю 164 см")
    assert {c.value for c in cs if c.attribute == "screen"} == {65.0}


def test_near_equal_numbers_aggregate_without_conflict():
    a = aggregate_full([Candidate(attribute="screen", value=43.0, raw="43 дюйма", source="rules", score=0.86),
                        Candidate(attribute="screen", value=42.9, raw="109 см", source="rules", score=0.86)])
    assert a.value == 43.0 and not a.conflict


@pytest.mark.parametrize("cat,attr,desc,value", [
    ("washing_machine", "depth", "полноразмерная модель габаритами 84,5х59,5х64 см, вмещает до 12 кг", 64.0),
    ("washing_machine", "depth", "Габариты: 84х44,6х60 см", 44.6),  # ширина стиральной машины ~60 см
    ("washing_machine", "depth", "Узкая стиральная машина глубиной 41 см рассчитана на загрузку 6 кг", 41.0),
    ("fridge", "height", "Размеры этой модели — 201х59,5х65 см", 201.0),
    ("fridge", "height", "Холодильник Candy CFN6200S высотой 200 см", 200.0),
])
def test_dimensions(tax, cat, attr, desc, value):
    _, cs = cands(tax, cat, "Товар", desc)
    assert best(cs, attr).value == value


def test_washer_load_not_weight(tax):
    _, cs = cands(tax, "washing_machine", "Стиральная машина Haier HW60",
                  "Вмещает до 6 кг белья. Вес — 56 кг, сушка до 4 кг")
    assert {c.value for c in cs if c.attribute == "load"} == {6.0}


def test_split_system_cooling_power_and_area(tax):
    _, cs = cands(tax, "split_system", "Сплит-система Rapid RAMI-07HJ/N1",
                  "эффективна на площади до 21 м². Производительность по теплу 2,7 кВт, по холоду — 2,65 кВт.")
    assert best(cs, "cooling_power").value == 2.65
    assert best(cs, "area").value == 21.0


# ------------------------------------------------------------------ справочники ------------------------------------
def test_enum_label_and_color_pattern_context(tax):
    _, cs = cands(tax, "generic", "Подушка декоративная PrintyFull Цветочные узоры: розовые FP19601074")
    assert best(cs, "color") is None  # узор, а не цвет товара
    _, cs = cands(tax, "generic", "Тумба под раковину Orange Line Li-100", "Цвет: белый")
    assert best(cs, "color").value == "белый"  # «Orange» в бренде — не цвет; подпись «Цвет:» надёжна


def test_learned_enum_map_applies_catalog_convention(tax):
    card = Card(id="v", title="Пылесос ручной (handstick) Dreame V11 серый")
    spec = tax.categories["vacuum"].attributes["type"]
    emap = {"title|ручной": ["вертикальный", 0.97, 231]}
    out = rules.extract_enum(rules.Ctx.of(card, "vacuum"), "type", spec, tax.dictionaries[spec.dictionary], [], emap)
    assert out[0].value == "вертикальный" and out[0].score > 0.9


def test_listed_colors_in_description_stay_ambiguous(tax):
    card = Card(id="c", title="Смартфон Samsung Galaxy S24", description="Расцветки: черный, серый, фиолетовый.")
    spec = tax.categories["smartphone"].attributes["color"]
    eh = rules.enum_hits(rules.Ctx.of(card, "smartphone"), spec, tax.dictionaries["colors"], [])
    assert eh.many and eh.keys() == [f"body|{'+'.join(sorted(set(eh.canons)))}"]


# ------------------------------------------------------------------ цитаты, статусы, калибровка ---------------------
def test_every_value_has_evidence_span_into_full_text(ex):
    card = Card(id="e", title="Смартфон Samsung Galaxy A55 8/256 ГБ голубой", description="Дисплей 6.6 дюйма, 5000 mAh",
                seller_attrs={"Цвет": "голубой"})
    r = ex.extract(card)
    text = card.full_text()
    for a in r.attributes:
        if a.value is None:
            continue
        s, e = a.evidence_span
        assert text[s:e] == a.evidence
    assert r.category_title and r.category_path and "total" in r.timings_ms
    assert all(a.traffic_weight > 0 for a in r.attributes)


def test_cross_source_conflict_goes_to_review(ex):
    r = ex.extract(Card(id="x", title="Смартфон Infinix HOT 40i 4/128 ГБ", seller_attrs={"Объем памяти": "256 ГБ"}))
    st = next(a for a in r.attributes if a.attribute == "storage")
    assert st.status == "review" and st.alternatives and st.alternatives[0]["sources"] == ["seller"]


def test_calibration_curve_is_monotonic_and_falls_back(tax):
    c = pav_curve([0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 0.9], [0, 1, 0, 1, 1, 1, 1], prior=0.7)
    assert all(a <= b for a, b in zip(c.y, c.y[1:], strict=False))
    spec = tax.categories["generic"].attributes["brand"]
    cal = Calibrator({"groups": {"rules|brand": {"x": [0.0, 0.9], "y": [0.3, 0.99]}}})
    assert cal(0.95, ["rules"], spec, False, "generic", "brand") == 0.99  # общая группа «источники|тип»
    assert cal(0.95, ["ml"], spec, False, "generic", "brand") == 0.95  # нет кривой — сырая оценка
    assert wilson_lower(950, 1000) < 0.95 < wilson_lower(9700, 10000)


def test_ram_with_glued_ssd_capacity(tax):
    _, cs = cands(tax, "laptop", "Ноутбук Chuwi HeroBook N100 8Gb SSD256Gb 14.1\" серый")
    assert best(cs, "ram").value == 8 and best(cs, "storage").value == 256


def test_label_at_description_start(tax):
    card = Card(id="l", title="Умная колонка Яндекс Станция Мини", description="Цвет: розовый")
    spec = tax.categories["generic"].attributes["color"]
    eh = rules.enum_hits(rules.Ctx.of(card, "generic"), spec, tax.dictionaries["colors"], [])
    assert eh.loc == "label" and eh.canons == ["розовый"]
