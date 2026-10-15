"""Distant supervision для NER: значения каталога -> спаны в тексте (без torch; своя мини-таксономия)."""
import pytest

from attrx.extractors import ner
from attrx.extractors.ner_distant import GoldAttr, bio_preview, label_card, restrict
from attrx.schema import Card
from attrx.taxonomy import Taxonomy

TAXONOMY = """
categories:
  smartphone:
    title: Смартфоны
    category_ids: ["761"]
    keywords: [смартфон]
    attributes:
      brand:   {title: Бренд, type: enum, dictionary: brands, gold: [vendor], aliases: [бренд]}
      model:   {title: Модель, type: string, kind: model, gold: [model], aliases: [модель]}
      storage: {title: Объём встроенной памяти, type: number, unit: ГБ, kind: memory, gold: [x], aliases: [встроенная память, объем памяти]}
      screen:  {title: Диагональ экрана, type: number, unit: дюйм, kind: diagonal, gold: [x], aliases: [диагональ, экран]}
      battery: {title: Ёмкость аккумулятора, type: number, unit: мА·ч, kind: battery, gold: [x], aliases: [аккумулятор, емкость аккумулятора]}
      color:   {title: Цвет, type: enum, dictionary: colors, gold: [Цвет], aliases: [цвет]}
  laptop:
    title: Ноутбуки
    category_ids: ["987"]
    keywords: [ноутбук]
    attributes:
      brand:   {title: Бренд, type: enum, dictionary: brands, gold: [vendor], aliases: [бренд]}
      cpu:     {title: Процессор, type: string, kind: cpu, gold: [x], aliases: [процессор]}
      ram:     {title: Объём оперативной памяти, type: number, unit: ГБ, kind: ram, gold: [x], aliases: [оперативная память, озу]}
      storage: {title: Накопитель, type: number, unit: ГБ, kind: storage_drive, gold: [x], aliases: [накопитель, ssd]}
      color:   {title: Цвет, type: enum, dictionary: colors, gold: [Цвет], aliases: [цвет]}
  kettle:
    title: Электрические чайники
    category_ids: ["482"]
    keywords: [чайник]
    attributes:
      volume:   {title: Объём, type: number, unit: л, kind: volume_l, gold: [x], aliases: [объем, объём]}
      material: {title: Материал корпуса, type: enum, dictionary: materials, gold: [x], aliases: [материал]}
      color:    {title: Цвет, type: enum, dictionary: colors, gold: [Цвет], aliases: [цвет]}
  generic:
    title: Прочее
    keywords: []
    attributes:
      brand: {title: Бренд, type: enum, dictionary: brands, gold: [vendor], aliases: [бренд]}
      model: {title: Модель, type: string, kind: model, gold: [model], aliases: [модель]}
      color: {title: Цвет, type: enum, dictionary: colors, gold: [Цвет], aliases: [цвет]}
"""
DICTS = {
    "brands": "Samsung: [samsung, самсунг]\nApple: [apple, iphone, macbook]\nXiaomi: [xiaomi]\n",
    "colors": ("чёрный: [черный, black, midnight]\nбелый: [white]\nсиний: [blue, голубой]\n"
               "серебристый: [silver, нержавеющая сталь]\nкрасный: [red]\nфиолетовый: [violet]\n"),
    "materials": "нержавеющая сталь: [сталь, stainless steel]\nстекло: [стеклянный, glass]\n",
}


@pytest.fixture(scope="module")
def tax(tmp_path_factory):
    root = tmp_path_factory.mktemp("tax")
    (root / "dictionaries").mkdir()
    (root / "taxonomy.yaml").write_text(TAXONOMY, encoding="utf-8")
    for name, body in DICTS.items():
        (root / "dictionaries" / f"{name}.yaml").write_text(body, encoding="utf-8")
    return Taxonomy(root / "taxonomy.yaml")


def lab(tax, title, desc="", gold=None, cat="smartphone", seller=None):
    card = Card(id="t", title=title, description=desc, seller_attrs=seller or {})
    text = card.full_text()
    g = {a: GoldAttr(v if isinstance(v, list) else [v], raw) for a, (v, raw) in (gold or {}).items()}
    res = label_card(text, len(card.title), g, tax.categories[cat], tax)
    return text, res, {a: [text[s:e] for s, e, b in res.spans if b == a] for a in tax.categories[cat].attributes}


def masked(text, res):
    return [text[s:e] for s, e in res.masks]


def test_memory_pair_brand_model_color(tax):
    text, res, got = lab(tax, "Смартфон Samsung Galaxy A55 8/256GB синий", gold={
        "brand": ("Samsung", "Samsung"), "model": ("Galaxy A55", "Galaxy A55 8/256GB синий"),
        "storage": (256.0, "256"), "color": ("синий", "синий")})
    assert got["brand"] == ["Samsung"]
    assert got["model"] == ["Galaxy A55"]  # модель обрезана по памяти и цвету
    assert got["storage"] == ["256GB"]  # из пары «8/256» — только объём накопителя
    assert got["color"] == ["синий"]
    assert res.status["storage"] == "labeled"


@pytest.mark.parametrize("title, value, expected", [
    ("Смартфон X 256 ГБ", 256.0, "256 ГБ"),
    ("Смартфон X 256Гб", 256.0, "256Гб"),
    ("Смартфон X 1 ТБ", 1024.0, "1 ТБ"),
    ("Смартфон X 12/1Tb", 1024.0, "1Tb"),
])
def test_number_with_any_unit_spelling(tax, title, value, expected):
    _, _, got = lab(tax, title, gold={"storage": (value, str(value))})
    assert got["storage"] == [expected]


def test_diagonal_battery_and_bare_number_after_attribute_name(tax):
    text, res, got = lab(tax, "Смартфон X", "Экран 6,7\" AMOLED. Большой 6.7-дюймовый дисплей.\n"
                         "Аккумулятор емкостью 5 000 мАч.", gold={"screen": (6.7, "6.7"), "battery": (5000.0, "5000")},
                         seller={"Диагональ": "6.7"})
    assert got["screen"] == ['6,7"', "6.7", "6.7"]  # с единицей, «-дюймовый» (только число), «Диагональ: 6.7»
    assert got["battery"] == ["5 000 мАч"]
    assert ner.span_value("5 000 мАч", tax.categories["smartphone"].attributes["battery"], tax)[0] == 5000.0


def test_colors_inflections_and_english(tax):
    _, _, got = lab(tax, "Смартфон Apple iPhone 15 Midnight", "Корпус черного цвета.",
                    gold={"color": ("чёрный", "черный")})
    assert got["color"] == ["Midnight", "черного"]


def test_conflicting_value_in_title_skips_attribute(tax):
    text, res, got = lab(tax, "Смартфон X черный/красный", gold={"color": ("чёрный", "черный")})
    assert got["color"] == []
    assert res.status["color"] == "ambiguous"
    assert {"черный", "красный"} <= set(masked(text, res))


def test_conflict_in_description_sentence_masks_only_that_sentence(tax):
    text, res, got = lab(tax, "Смартфон X черный", "Есть черная и белая версии. Черный корпус не царапается.",
                         gold={"color": ("чёрный", "черный")})
    assert got["color"] == ["черный", "Черный"]  # название и предложение без конфликта
    assert {"черная", "белая"} <= set(masked(text, res))  # предложение с двумя цветами — в маске


def test_no_gold_means_mask_not_negative(tax):
    text, res, got = lab(tax, "Смартфон X синий 128 ГБ", gold={"brand": ("Xiaomi", "Xiaomi")})
    assert got["color"] == [] and got["storage"] == []
    assert {"синий", "128 ГБ"} <= set(masked(text, res))  # каталог неполный: не учим «O»
    assert res.status["color"] == "nogold"


def test_memory_card_context_is_masked(tax):
    text, res, got = lab(tax, "Смартфон X 128 ГБ", "Поддержка карт памяти до 256 ГБ.", gold={"storage": (256.0, "256")})
    assert got["storage"] == []
    assert "256 ГБ" in masked(text, res)


def test_gpu_memory_is_not_ram(tax):
    _, res, got = lab(tax, "Ноутбук X 16GB/512GB SSD/RTX 4060 8GB", cat="laptop",
                      gold={"ram": (16.0, "16"), "storage": (512.0, "SSD 512 ГБ")})
    assert got["ram"] == ["16GB"] and got["storage"] == ["512GB"]
    assert res.status["ram"] == "labeled"


def test_compatibility_brand_is_not_the_brand(tax):
    _, _, got = lab(tax, "Чехол Borasco для Samsung Galaxy S24", cat="generic", gold={"brand": ("Borasco", "Borasco")})
    assert got["brand"] == ["Borasco"]


def test_brand_synonym_is_masked_not_labeled(tax):
    text, res, got = lab(tax, "Ноутбук Apple MacBook Air 13 M3 16/512", cat="laptop",
                         gold={"brand": ("Apple", "Apple"), "cpu": ("Apple M3", "Apple M3 4.05 ГГц")})
    assert got["brand"] == ["Apple"]
    assert "MacBook" in masked(text, res)
    assert got["cpu"] == ["M3"]


def test_cpu_any_spelling_and_family(tax):
    _, _, got = lab(tax, "Ноутбук X Core i5 12450H", "Процессор Intel Core i5-12450H.", cat="laptop",
                    gold={"cpu": (["Intel Core i5-12450H", "Intel Core i5"], "Intel Core i5 12450H 2.0 ГГц")})
    assert got["cpu"] == ["Core i5 12450H", "Intel Core i5-12450H"]
    _, _, got = lab(tax, "Ноутбук X i5-1235U", cat="laptop", gold={"cpu": ("Intel Core i5", "Intel Core i5")})
    assert got["cpu"] == ["i5-1235U"]  # в эталоне только семейство — спан с номером модели


def test_cpu_partial_match_is_masked(tax):
    text, res, got = lab(tax, "Ноутбук Apple MacBook Pro M4 Pro", cat="laptop", gold={"cpu": ("Apple M4", "Apple M4")})
    assert got["cpu"] == []  # «M4» в «M4 Pro» — другое значение
    assert any("M4" in m for m in masked(text, res))


def test_material_and_color_on_same_words_are_masked(tax):
    text, res, got = lab(tax, "Чайник X нержавеющая сталь 1,7 л", cat="kettle",
                         gold={"material": ("нержавеющая сталь", "нержавеющая сталь"), "color": ("серебристый", "серебристый"),
                               "volume": (1.7, "1.7")})
    assert got["material"] == [] and got["color"] == []
    assert "нержавеющая сталь" in masked(text, res)
    assert got["volume"] == ["1,7 л"]


def test_junk_model_value_is_not_labeled(tax):
    _, res, got = lab(tax, "Подушка декоративная PrintyFull 30x30см", cat="generic",
                      gold={"brand": ("PrintyFull", "PrintyFull"), "model": ("30x30см", "30x30см / узор")})
    assert got["brand"] == ["PrintyFull"] and got["model"] == []


def test_open_vocabulary_brand_value(tax):
    spec = tax.categories["generic"].attributes["brand"]
    assert ner.span_value("Borasco", spec, tax) == ("Borasco", 0.85)
    assert ner.span_value("самсунг", spec, tax)[0] == "Samsung"


def test_bio_labels_with_masks():
    offsets = [(0, 0), (0, 5), (6, 7), (7, 8), (8, 9), (10, 13), (14, 16), (17, 22), (0, 0)]
    labels = ner.bio_labels(offsets, [(6, 9, "screen")], masks=[(17, 22)])
    assert labels == [None, "O", "B-screen", "I-screen", "I-screen", "O", "O", None, None]
    assert ner.bio_labels([(0, 3), (4, 7)], [(4, 7, "brand")]) == ["O", "B-brand"]


def test_view_keeps_title_and_informative_fragments_with_offsets(tax):
    filler = " ".join(["Отличный выбор для дома и офиса."] * 40)
    card = Card(id="v", title="Смартфон Samsung Galaxy A55", description=filler + "\nАккумулятор емкостью 5000 мАч.")
    text = card.full_text()
    view = ner.build_view(text, len(card.title), ner.view_scorer(tax.categories["smartphone"], tax), max_chars=200)
    assert len(view.text) <= 200 and view.text.startswith(card.title)
    assert "5000 мАч" in view.text
    s = text.index("5000 мАч")
    v = view.from_orig(s, s + 8)
    assert view.text[v[0]:v[1]] == "5000 мАч" and view.to_orig(*v) == (s, s + 8)
    short = ner.build_view("коротко", 7)
    assert short.text == "коротко" and short.from_orig(0, 7) == (0, 7)


def test_segments_split_sentences_and_seller_pairs():
    text = "Название\nПервое предложение. Второе!\nЦвет: черный; Вес: 1 кг"
    segs = [text[s:e] for s, e in ner.segments(text, len("Название"))]
    assert segs == ["Название", "Первое предложение.", "Второе!", "Цвет: черный;", "Вес: 1 кг"]


def test_restrict_turns_rare_attribute_spans_into_masks():
    ex = ner.NerExample("Смартфоны", "Samsung 5000 мАч", [(0, 7, "brand"), (8, 16, "battery")])
    out = restrict([ex], ["brand"])[0]
    assert out.spans == [(0, 7, "brand")] and (8, 16) in out.masks


def test_isotonic_calibration_is_monotone():
    xs = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99] * 10
    ys = [0, 0, 1, 0, 1, 1, 1, 1] * 10
    table = ner.fit_isotonic(xs, ys)
    vals = [ner.calibrate(table, x / 100) for x in range(30, 100)]
    assert all(b >= a for a, b in zip(vals, vals[1:], strict=False))
    assert vals[0] < 0.2 and vals[-1] > 0.9
    assert ner.calibrate(None, 0.5) == pytest.approx(0.5 * ner.ML_BASE_SCORE)


def test_bio_preview_for_docs():
    ex = ner.NerExample("Смартфоны", "Samsung 256 ГБ", [(0, 7, "brand"), (8, 14, "storage")])
    assert bio_preview(ex) == [("Samsung", "B-brand"), ("256", "B-storage"), ("ГБ", "I-storage")]
