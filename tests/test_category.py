"""Классификатор категории: признаки, отображение на таксономию, инференс без scikit-learn, обучение на синтетике.

Тесты не зависят от data/processed: обучение — на крошечном синтетическом «каталоге» во временной папке.
Тест реального артефакта (models/category) выполняется, только если артефакт есть.
"""
import json
import random

import numpy as np
import pytest

from attrx import category as cat
from attrx.schema import Card

TAXONOMY_YAML = """
categories:
  smartphone: {title: Смартфоны, category_ids: ["761", "914"], keywords: [смартфон], attributes: {}}
  laptop:     {title: Ноутбуки, category_ids: ["987"], keywords: [ноутбук], attributes: {}}
  kettle:     {title: Электрические чайники, keywords: [чайник], attributes: {}}
  generic:    {title: Прочее, attributes: {}}
"""

ROOTS = {"r1": "Смартфоны и связь", "r2": "Компьютеры", "r3": "Техника для кухни"}
# лист -> (раздел, название, шаблон заголовка, число карточек)
LEAVES = {
    "761": ("r1", "Смартфоны", "Смартфон {b} {m} 8/256 ГБ {c}", 60),
    "914": ("r1", "iPhone", "Смартфон Apple iPhone {n} 128 ГБ {c}", 40),
    "555": ("r1", "Чехлы для смартфонов", "Чехол для смартфона {b} {m} силиконовый {c}", 30),
    "987": ("r2", "Ноутбуки", "Ноутбук {b} {m} 15.6 Intel Core i5 16 ГБ SSD 512 ГБ", 50),
    "321": ("r2", "Мыши", "Мышь беспроводная {b} {m} {c}", 25),
    "482": ("r3", "Электрические чайники", "Чайник электрический {b} {m} 1.7 л {c}", 35),
    "700": ("r3", "Блендеры", "Блендер погружной {b} {m} 800 Вт {c}", 20),
}
BRANDS = ["Samsung", "Xiaomi", "Honor", "Bosch", "Philips", "Lenovo", "Asus", "Tefal", "Redmond", "Logitech"]
COLORS = ["черный", "белый", "синий", "серый", "красный"]


@pytest.fixture()
def taxonomy_yaml(tmp_path, monkeypatch):
    p = tmp_path / "taxonomy.yaml"
    p.write_text(TAXONOMY_YAML, encoding="utf-8")
    monkeypatch.setenv("TAXONOMY_PATH", str(p))
    return p


def _title(leaf: str, rnd: random.Random) -> str:
    return LEAVES[leaf][2].format(b=rnd.choice(BRANDS), m=f"{rnd.choice('ABXZ')}{rnd.randint(10, 99)}",
                                  n=rnd.randint(11, 16), c=rnd.choice(COLORS))


def _write_processed(d):
    """Мини-«каталог» в формате data/processed: cards.parquet, categories.parquet, taxonomy_candidates.json."""
    import pandas as pd

    rnd = random.Random(0)
    cats = [{"id": r, "title": t, "path": [t], "path_ids": [r]} for r, t in ROOTS.items()]
    cats += [{"id": c, "title": t, "path": [ROOTS[r], t], "path_ids": [r, c]} for c, (r, t, _, _) in LEAVES.items()]
    rows = []
    for leaf, (_, title, _, n) in LEAVES.items():
        for i in range(n):
            split = "train" if i % 10 < 8 else ("dev" if i % 10 == 8 else "holdout")
            rows.append({"sku_id": f"{leaf}-{i}", "title": _title(leaf, rnd), "description_clean": f"{title}. Гарантия 1 год.",
                         "seller_attrs": "{}", "category_id": leaf, "split": split, "category_source": "offer",
                         "seller_type": "marketplace", "seller": "Продавец"})
    for i in range(300):  # карточки без категории (cold start)
        leaf = rnd.choice(list(LEAVES))
        rows.append({"sku_id": f"u{i}", "title": _title(leaf, rnd), "description_clean": "",
                     "seller_attrs": json.dumps({"Цвет": "черный"}, ensure_ascii=False), "category_id": None,
                     "split": "train", "category_source": None, "seller_type": None, "seller": "Партнёр"})
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(d / "cards.parquet")
    pd.DataFrame(cats).to_parquet(d / "categories.parquet")
    (d / "taxonomy_candidates.json").write_text(json.dumps(
        {"categories": [{"id": "761", "title": "Смартфоны"}, {"id": "482", "title": "Электрические чайники"}]},
        ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------------------------- признаки


def test_char_grams_like_char_wb():
    assert cat._char_grams("ab") == [" a", "ab", "b ", " ab", "ab ", " ab "]
    assert cat._word_grams(["смартфон", "256", "apple"]) == ["смартфон", "apple", "смартфон apple"]


def test_context_strips_html_and_limits_length():
    ctx = cat.context_text("<p>Чайник <b>электрический</b></p>" + "х" * 1000, {"Объём": "1.7 л"})
    assert "<" not in ctx and "чайник" in ctx and "объем 1.7 л" in ctx  # ё -> е, как в остальном пайплайне
    assert len(ctx) <= 2 * cat.CTX_CHARS + 1


def test_featurizer_tfidf_blocks_are_l2_normalized():
    docs = [("Смартфон Samsung Galaxy", "описание смартфона"), ("Ноутбук Lenovo IdeaPad", "описание ноутбука")] * 3
    fz = cat.Featurizer.fit(docs, min_df=(1, 1, 1))
    indptr, cols, vals = fz.transform(docs[:2])
    assert len(indptr) == 3 and len(cols) == len(vals) > 0
    block = np.searchsorted(np.asarray(fz.offsets[1:]), cols, side="right")
    row0 = np.arange(indptr[0], indptr[1])
    for b, w in enumerate(fz.weights):
        m = block[row0] == b
        assert np.isclose(np.sqrt((vals[row0][m] ** 2).sum()), w, atol=1e-5)
    # одинаковые документы -> одинаковые признаки (кэш n-грамм не меняет результат)
    again = fz.transform(docs[:2])
    assert all(np.array_equal(x, y) for x, y in zip(again, (indptr, cols, vals), strict=True))


# ---------------------------------------------------------------------------------------------- таксономия


def test_leaf_to_taxonomy_ids_and_provisional(taxonomy_yaml):
    specs = cat._taxonomy_specs()
    mapping, guessed = cat.leaf_to_taxonomy(specs, [{"id": "482", "title": "Электрические чайники"},
                                                   {"id": "999", "title": "Чайные пары"}])
    assert mapping["761"] == mapping["914"] == "smartphone" and mapping["987"] == "laptop"
    assert guessed == {"482": "kettle"}  # у kettle нет category_ids -> временное отображение по названию листа
    assert "999" not in mapping


def test_label_space_contiguous_roots_and_other_bucket():
    cats = {c: {"title": t, "path": [ROOTS[r], t], "root": r} for c, (r, t, _, _) in LEAVES.items()}
    cats |= {r: {"title": t, "path": [t], "root": r} for r, t in ROOTS.items()}
    counts = {c: v[3] for c, v in LEAVES.items()}
    ls = cat.build_label_space(counts, cats, max_classes=3, forced=["914", "700"], min_forced=10)
    ids = [c["id"] for c in ls.classes]
    assert {"761", "987", "914", "700"} <= set(ids) and "other:r1" in ids and "555" not in ids
    roots = [c["root"] for c in ls.classes]
    assert roots == sorted(roots)  # классы идут блоками по разделам
    assert ls.class_of("555", "r1") == ls.class_index["other:r1"]
    assert ls.class_of("123", "zz") == -1


# ---------------------------------------------------------------------------------------------- инференс


def _handmade_model(tmp_path):
    """Модель «вручную» (без scikit-learn): раздел r1 = {761, other:r1}, раздел r3 = {482}."""
    docs = [("смартфон samsung", ""), ("чехол для смартфона", ""), ("чайник электрический", "")]
    fz = cat.Featurizer.fit(docs, min_df=(1, 1, 1))
    roots = [{"id": "r1", "title": ROOTS["r1"], "n_train": 2}, {"id": "r3", "title": ROOTS["r3"], "n_train": 1}]
    classes = [{"id": "761", "title": "Смартфоны", "path": [ROOTS["r1"], "Смартфоны"], "root": 0, "n_train": 1},
               {"id": "other:r1", "title": "Прочее: " + ROOTS["r1"], "path": [ROOTS["r1"]], "root": 0, "n_train": 1},
               {"id": "482", "title": "Электрические чайники", "path": [ROOTS["r3"], "Электрические чайники"], "root": 1,
                "n_train": 1}]
    ls = cat.LabelSpace(roots, classes)
    k = len(roots) + len(classes)
    w = np.zeros((k, fz.n_features))
    for word, cols in {"смартфон": {0: 3, 2: 2}, "чехол": {0: 3, 2: -2, 3: 3}, "чайник": {1: 3}}.items():
        for c, v in cols.items():
            w[c, fz.vocab[0][word]] = v
    f, c, q, s = cat._prune(w, np.arange(k), keep=1000)
    order = np.lexsort((c, f))
    indptr = np.zeros(fz.n_features + 1, dtype=np.int64)
    np.cumsum(np.bincount(f, minlength=fz.n_features), out=indptr[1:])
    arrays = {"w_indptr": indptr.astype(np.int32), "w_col": c[order].astype(np.uint16), "w_q": q[order],
              "scale": s, "bias": np.zeros(k, dtype=np.float32)}
    meta, arr = cat._assemble(fz, arrays, ls, {"root": 0.5, "leaf": 0.5}, 0.6, {"provisional_leaves": []})
    sizes = cat.save_artifact(tmp_path / "model", meta, arr)
    assert set(sizes) == {cat.MODEL_FILE, cat.VOCAB_FILE, cat.META_FILE}
    return tmp_path / "model"


def test_inference_without_sklearn_roundtrip(tmp_path, taxonomy_yaml):
    clf = cat.CategoryClassifier.load(_handmade_model(tmp_path))
    assert clf is not None and clf.threshold == 0.6
    p = clf.predict(Card(id="1", title="Смартфон Samsung Galaxy A55"))
    assert (p.leaf_id, p.taxonomy_category) == ("761", "smartphone") and 0.5 < p.confidence <= 1.0
    assert [t[0] for t in p.top3][0] == "761" and p.top3[0][2] >= p.top3[1][2] >= p.top3[2][2]
    acc = clf.predict({"title": "Чехол для смартфона Samsung", "description": "<b>силикон</b>"})
    assert acc.taxonomy_category == "generic" and acc.leaf_id == "other:r1"  # аксессуар — не смартфон
    k = clf.predict(Card(id="3", title="Чайник электрический Tefal"))
    assert k.leaf_id == "482" and k.taxonomy_category == "generic"  # у kettle нет category_ids и provisional пуст
    assert clf.leaf_path("761") == [ROOTS["r1"], "Смартфоны"] and clf.leaf_path("nope") == []
    s = clf.scores_many([Card(id="x", title="смартфон"), Card(id="y", title="")])
    _, p_leaf = clf.proba_from_scores(s)
    assert np.allclose(p_leaf.sum(axis=1), 1.0)
    assert len(clf.predict_many([Card(id=str(i), title="смартфон") for i in range(5)], batch=2)) == 5
    assert clf.is_review(acc._replace(confidence=0.5)) and not clf.is_review(p._replace(confidence=0.9))
    with np.load(tmp_path / "model" / cat.MODEL_FILE, allow_pickle=False) as z:  # без pickle
        assert z["w_q"].dtype == np.int8


def test_load_returns_none_when_absent_disabled_or_broken(tmp_path, monkeypatch, taxonomy_yaml):
    assert cat.CategoryClassifier.load(tmp_path / "missing") is None
    d = _handmade_model(tmp_path)
    monkeypatch.setenv("CATEGORY_CLASSIFIER", "off")
    assert cat.CategoryClassifier.load(d) is None
    monkeypatch.delenv("CATEGORY_CLASSIFIER")
    (d / cat.META_FILE).write_text("{broken", encoding="utf-8")
    assert cat.CategoryClassifier.load(d) is None


def test_metrics_helpers():
    assert cat.choose_threshold(np.array([0.9, 0.8, 0.7, 0.6]), np.array([1, 1, 0, 1]), 0.99) == 0.8
    m = cat.prf(["a", "a", "b"], ["a", "b", "b"])
    assert m["accuracy"] == round(2 / 3, 4) and m["per_class"]["b"]["precision"] == 0.5
    r = cat.reliability(np.array([0.95, 0.95, 0.15]), np.array([1.0, 0.0, 0.0]))
    assert r["bins"][-1]["accuracy"] == 0.5 and r["ece"] > 0
    spec = [("smartphone", "Смартфоны", ["смартфон", "iphone"], []), ("laptop", "Ноутбуки", ["ноутбук"], [])]
    assert cat.keyword_category(spec, "Смартфон Apple iPhone 15", "Смартфон Apple iPhone 15") == "smartphone"
    assert cat.keyword_category(spec, "Мышь", "Мышь для ноутбука") == "laptop"  # слабость правил: аксессуар
    assert cat.keyword_category(spec, "Утюг", "Утюг") == "generic"


# ---------------------------------------------------------------------------------------------- обучение


def test_train_category_end_to_end_on_synthetic(tmp_path, taxonomy_yaml):
    pytest.importorskip("sklearn")
    processed, out = tmp_path / "processed", tmp_path / "model"
    _write_processed(processed)
    m = cat.train_category(processed, out, max_classes=4, n_jobs=1, epochs=10, alpha=1e-4, echo=lambda s: None)
    assert m["label_space"]["roots"] == 3 and m["label_space"]["classes"] == 7  # 4 листа + 3 «Прочее в разделе»
    assert m["holdout"]["taxonomy"]["model"]["accuracy"] >= 0.9
    assert m["holdout"]["leaf"]["top3"] >= m["holdout"]["leaf"]["top1"] >= 0.8
    assert {"reliability_taxonomy", "coverage_accuracy", "segments"} <= set(m["holdout"])
    assert m["unlabeled"]["n"] >= 0 and (out / cat.METRICS_FILE).exists()
    clf = cat.CategoryClassifier.load(out)
    assert clf is not None and clf.provisional_map == {"482": "kettle"}
    assert clf.predict(Card(id="a", title="Смартфон Honor X8 8/256 ГБ синий")).taxonomy_category == "smartphone"
    assert clf.predict(Card(id="b", title="Ноутбук Asus Z42 15.6 Intel Core i5")).taxonomy_category == "laptop"
    assert clf.predict(Card(id="c", title="Чайник электрический Tefal B17 1.7 л")).taxonomy_category == "kettle"
    acc = clf.predict(Card(id="d", title="Чехол для смартфона Xiaomi A20 силиконовый"))
    assert acc.taxonomy_category == "generic" and acc.leaf_id == "other:r1"


def test_default_artifact_smoke():
    clf = cat.CategoryClassifier.load()
    if clf is None:
        pytest.skip("нет обученного артефакта models/category (attrx train-category)")
    assert 0.0 < clf.threshold <= 1.0 and "generic" in clf.taxonomy_names
    p = clf.predict(Card(id="s", title="Смартфон Samsung Galaxy A55 8/256 ГБ синий"))
    assert p.taxonomy_category == "smartphone" and p.confidence >= clf.threshold
    assert clf.predict(Card(id="n", title="Ноутбук Lenovo IdeaPad Slim 3 15.6 Ryzen 5 16/512 ГБ")).taxonomy_category == "laptop"
