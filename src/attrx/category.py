"""Классификатор категории карточки по тексту: TF-IDF + иерархическая линейная модель (раздел → лист каталога).

Зачем. Набор атрибутов зависит от категории, а у 74,8% карточек (880 632 из 1 177 200) нет ни оффера в YML-фиде,
ни кликов — категорию можно определить только по тексту. Ключевые слова таксономии — базлайн на правилах;
этот модуль — дешёвая ML-модель поверх него (CPU, без torch и GPU, доли миллисекунды на карточку); LLM остаётся
для cold start. Подробно, с метриками на holdout — docs/CATEGORY.md.

Дизайн.
* Признаки — TF-IDF в трёх блоках: слова title (1–2-граммы), символьные n-граммы title (char_wb 2–5), слова первых
  300 символов описания + атрибутов продавца (1–2-граммы, вес блока 0.5). Значение = (1 + log tf) · idf, L2 по блоку.
* Метки — двухуровневые: раздел каталога (корень дерева YML) → лист. Листья — топ-N по числу карточек в train
  плюс все листья таксономии; остальные листья раздела объединены в класс «Прочее в разделе» (`other:<id раздела>`).
* Модель — линейные SVM (SGD, hinge), one-vs-rest: этап 1 — раздел, этап 2 — лист внутри раздела.
  P(лист) = P(раздел) · P(лист | раздел); вероятности — softmax по оценкам с температурами, подобранными на dev.
* Категория таксономии (config/taxonomy.yaml) = сумма вероятностей её листьев (`category_ids`). Отображение
  применяется при загрузке модели: правка таксономии не требует переобучения (если лист есть в пространстве меток).
* Артефакт — models/category/{model.npz, meta.json}: разреженные веса int8 + масштаб на класс, словарь и классы
  в JSON, без pickle. Для инференса нужен только numpy; scikit-learn — только для обучения (`attrx train-category`).

Точка интеграции:
    clf = CategoryClassifier.load()            # None, если артефакта нет (или CATEGORY_CLASSIFIER=off)
    p = clf.predict(card)                      # CategoryPrediction(leaf_id, leaf_title, taxonomy_category,
                                               #                    confidence, top3)
    if p.confidence < clf.threshold: ...       # карточку — на проверку с предупреждением
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import os
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from .config import ROOT

log = logging.getLogger(__name__)

FORMAT = "attrx-category/1"
MODEL_FILE, META_FILE, VOCAB_FILE, METRICS_FILE = "model.npz", "meta.json", "vocab.json.gz", "metrics.json"
DEFAULT_DIR = ROOT / "models" / "category"
GENERIC = "generic"
OTHER = "other:"  # префикс класса «Прочее в разделе»: other:<id раздела>
CTX_CHARS = 300  # сколько символов описания (и столько же — атрибутов продавца) идёт в признаки
CHAR_NGRAMS = (2, 5)
BLOCKS = ("title_word", "title_char", "context_word")
_OFF = {"0", "off", "false", "no"}
# строки весов, где признак есть у ≥ 1/8 классов (частые n-граммы), считаются плотным умножением матриц (BLAS):
# ~4,5 тыс. таких признаков дают ~90% работы разреженного скоринга
DENSE_ROW_SHARE = 0.125
SMALL_BATCH = 16  # до стольких карточек — только разреженный путь (одиночный вызов из пайплайна быстрее без BLAS)

# ============================================================================ признаки

_TAG_RE = re.compile(r"<[^>]{0,300}>")
_TOKEN_RE = re.compile(r"[^\W_]+")


def normalize(text: str | None) -> str:
    t = (text or "").lower().replace("ё", "е")
    return _TAG_RE.sub(" ", t) if "<" in t else t


def context_text(description: str | None, seller_attrs: Mapping[str, Any] | None = None) -> str:
    """Контекст карточки: первые CTX_CHARS символов описания (без HTML) + атрибуты продавца «ключ значение»."""
    desc = normalize(description)[:CTX_CHARS]
    if seller_attrs:
        sa = " ".join(f"{k} {v}" for k, v in seller_attrs.items())
        desc = f"{desc} {normalize(sa)[:CTX_CHARS]}"
    return desc


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize(text))


def _word_grams(tokens: list[str]) -> list[str]:
    """Слова (без чисел) + биграммы слов."""
    w = [t for t in tokens if not t.isdigit()]
    return w + [f"{a} {b}" for a, b in zip(w, w[1:], strict=False)]


def _char_grams(word: str) -> list[str]:
    """Символьные n-граммы внутри слова с пробелами по краям (как analyzer='char_wb' в scikit-learn)."""
    p = f" {word} "
    n_p = len(p)
    return [p[i:i + n] for n in range(CHAR_NGRAMS[0], min(CHAR_NGRAMS[1], n_p) + 1) for i in range(n_p - n + 1)]


class Featurizer:
    """TF-IDF по трём блокам (BLOCKS). Одна и та же функция строит признаки при обучении и на инференсе."""

    def __init__(self, vocab: Sequence[Sequence[str]], idf: np.ndarray, weights: Sequence[float]):
        self.terms = [list(v) for v in vocab]
        self.vocab = [{g: i for i, g in enumerate(v)} for v in self.terms]
        self.offsets = np.cumsum([0] + [len(v) for v in self.terms]).tolist()
        self.idf = np.asarray(idf, dtype=np.float32)
        self.weights = [float(x) for x in weights]
        if len(self.idf) != self.n_features or len(self.weights) != len(BLOCKS):
            raise ValueError("Featurizer: размер idf/весов не совпадает со словарём")
        self._char_cache: dict[str, list[int]] = {}

    @property
    def n_features(self) -> int:
        return int(self.offsets[-1])

    @classmethod
    def fit(cls, docs: Iterable[tuple[str, str]], min_df: Sequence[int] = (2, 3, 3),
            max_features: Sequence[int] = (150_000, 200_000, 60_000),
            weights: Sequence[float] = (1.0, 1.0, 0.5)) -> Featurizer:
        """Словарь по документной частоте: n-грамма нужна минимум в min_df документах, не больше max_features."""
        dfs: list[Counter] = [Counter(), Counter(), Counter()]
        char_cache: dict[str, list[str]] = {}
        n = 0
        for title, ctx in docs:
            n += 1
            tt = _tokens(title)
            dfs[0].update(set(_word_grams(tt)))
            chars: set[str] = set()
            for w in tt:
                g = char_cache.get(w)
                if g is None:
                    g = char_cache[w] = _char_grams(w)
                chars.update(g)
            dfs[1].update(chars)
            dfs[2].update(set(_word_grams(_tokens(ctx))))
        vocab, idf = [], []
        for c, md, mf in zip(dfs, min_df, max_features, strict=True):
            items = sorted(((g, d) for g, d in c.items() if d >= md), key=lambda x: (-x[1], x[0]))[:mf]
            items.sort(key=lambda x: x[0])
            vocab.append([g for g, _ in items])
            idf.extend(math.log((1 + n) / (1 + d)) + 1.0 for _, d in items)  # smooth idf, как в scikit-learn
        return cls(vocab, np.asarray(idf, dtype=np.float32), weights)

    def _char_ids(self, word: str) -> list[int]:
        ids = self._char_cache.get(word)  # char_wb-граммы зависят только от слова — кэш ускоряет в ~6 раз
        if ids is None:
            off = self.offsets[1]
            ids = [j + off for j in map(self.vocab[1].get, _char_grams(word)) if j is not None]
            if len(self._char_cache) > 500_000:
                self._char_cache.clear()
            self._char_cache[word] = ids
        return ids

    def ids(self, title: str, ctx: str) -> list[int]:
        """Глобальные индексы признаков документа (с повторами — из них считается tf)."""
        tt = _tokens(title)
        out = [j for j in map(self.vocab[0].get, _word_grams(tt)) if j is not None]
        for w in tt:
            out.extend(self._char_ids(w))
        o2 = self.offsets[2]
        out.extend(j + o2 for j in map(self.vocab[2].get, _word_grams(_tokens(ctx))) if j is not None)
        return out

    def transform(self, docs: Iterable[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Документы (title, контекст) -> CSR (indptr, indices, data)."""
        lens: list[int] = []
        cols: list[int] = []
        for title, ctx in docs:
            ids = self.ids(title, ctx)
            cols.extend(ids)
            lens.append(len(ids))
        return self._weigh(np.asarray(lens, dtype=np.int64), np.asarray(cols, dtype=np.int64))

    def _weigh(self, lens: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, n_feat, n_blocks = len(lens), max(self.n_features, 1), len(BLOCKS)
        rows = np.repeat(np.arange(n, dtype=np.int64), lens)
        keys, counts = np.unique(rows * n_feat + cols, return_counts=True)
        rows, cols = keys // n_feat, keys % n_feat
        block = np.searchsorted(np.asarray(self.offsets[1:]), cols, side="right")
        val = (1.0 + np.log(counts.astype(np.float64))) * self.idf[cols]
        grp = rows * n_blocks + block
        norm = np.sqrt(np.bincount(grp, weights=val * val, minlength=n * n_blocks))
        val = val * (np.asarray(self.weights)[block] / norm[grp])
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(rows, minlength=n), out=indptr[1:])
        return indptr, cols.astype(np.int32), val.astype(np.float32)


def _doc(card: Any) -> tuple[str, str]:
    """Card (или dict с теми же полями) -> (title, контекст)."""
    get = card.get if isinstance(card, Mapping) else (lambda k, d=None: getattr(card, k, d))
    sa = get("seller_attrs") or None
    return str(get("title") or ""), context_text(str(get("description") or ""), sa if isinstance(sa, Mapping) else None)


# ============================================================================ таксономия: лист -> категория


def _taxonomy_specs(taxonomy: Any = None) -> list[tuple[str, str, list[str], list[str]]]:
    """[(имя, title, keywords, category_ids)] из объекта Taxonomy или из YAML (по умолчанию config/taxonomy.yaml).

    YAML читается напрямую, без attrx.taxonomy, — чтобы taxonomy.py мог импортировать этот модуль без цикла.
    """
    if taxonomy is not None and hasattr(taxonomy, "categories"):
        return [(n, getattr(c, "title", n), list(getattr(c, "keywords", []) or []),
                 [str(x) for x in getattr(c, "category_ids", []) or []]) for n, c in taxonomy.categories.items()]
    import yaml

    path = Path(taxonomy) if taxonomy else Path(os.getenv("TAXONOMY_PATH") or ROOT / "config" / "taxonomy.yaml")
    if not path.is_absolute():
        path = ROOT / path
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [(n, c.get("title", n), list(c.get("keywords") or []), [str(x) for x in c.get("category_ids") or []])
            for n, c in (data.get("categories") or {}).items()]


def leaf_to_taxonomy(specs: list[tuple[str, str, list[str], list[str]]],
                     provisional: Sequence[Mapping[str, str]] = ()) -> tuple[dict[str, str], dict[str, str]]:
    """{id листа: категория таксономии} по `category_ids`.

    Для категорий без `category_ids` — временное отображение по топ-листьям из taxonomy_candidates.json
    (provisional: [{id, title}]): лист относится к категории, если его название совпадает с title категории или
    содержит её ключевое слово. Возвращает (отображение, какие листья взяты провизорно).
    """
    mapping: dict[str, str] = {}
    for name, _, _, ids in specs:
        for cid in ids:
            mapping.setdefault(cid, name)
    guessed: dict[str, str] = {}
    for name, title, keywords, ids in specs:
        if ids or name == GENERIC:
            continue
        keys = [normalize(k) for k in keywords if len(k) >= 4]
        for leaf in provisional:
            lt = normalize(leaf["title"])
            if leaf["id"] not in mapping and (lt == normalize(title) or any(k in lt for k in keys)):
                mapping[leaf["id"]] = guessed[leaf["id"]] = name
    return mapping, guessed


# ============================================================================ инференс


class CategoryPrediction(NamedTuple):
    leaf_id: str  # id листа YML-каталога или other:<id раздела> («Прочее в разделе» — лист вне пространства меток)
    leaf_title: str
    taxonomy_category: str  # имя категории в config/taxonomy.yaml (generic — нет в таксономии)
    confidence: float  # калиброванная P(taxonomy_category | текст)
    top3: list[tuple[str, str, float]]  # [(id, название, P(лист))] — три самых вероятных листа


def _softmax_rows(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(axis=1, keepdims=True)
    return z


class CategoryClassifier:
    """Загруженная модель: признаки -> оценки -> P(раздел), P(лист | раздел) -> категория таксономии."""

    def __init__(self, meta: dict, arrays: Mapping[str, np.ndarray], taxonomy: Any = None):
        if meta.get("format") != FORMAT:
            raise ValueError(f"неизвестный формат модели категории: {meta.get('format')!r}")
        self.meta = meta
        self.featurizer = Featurizer(meta["vocab"], arrays["idf"], meta["block_weights"])
        self.w_indptr = arrays["w_indptr"].astype(np.int64)
        self.w_col = arrays["w_col"].astype(np.int64)
        self.w_q = arrays["w_q"]
        self.scale = arrays["scale"].astype(np.float32)
        self.bias = arrays["bias"].astype(np.float32)
        self.roots: list[dict] = meta["roots"]
        self.classes: list[dict] = meta["classes"]
        self.n_roots, self.n_classes = len(self.roots), len(self.classes)
        self.class_root = np.array([c["root"] for c in self.classes], dtype=np.int64)
        starts = np.flatnonzero(np.r_[True, self.class_root[1:] != self.class_root[:-1]])
        if not np.array_equal(self.class_root[starts], np.arange(self.n_roots)):
            raise ValueError("классы должны идти блоками по разделам в порядке разделов")
        self.group_start = starts
        self.t_root = float(meta["temperature"]["root"])
        self.t_leaf = float(meta["temperature"]["leaf"])
        self.threshold = float(meta.get("threshold", 0.0))
        self.class_index = {c["id"]: i for i, c in enumerate(self.classes)}
        self._prepare_scoring()
        self.set_taxonomy(taxonomy)

    # ------------------------------------------------------------------ загрузка
    @classmethod
    def load(cls, path: str | Path | None = None, taxonomy: Any = None) -> CategoryClassifier | None:
        """Модель из models/category (или CATEGORY_MODEL_PATH); None — если артефакта нет или он выключен.

        taxonomy — объект attrx.taxonomy.Taxonomy или путь к YAML; по умолчанию config/taxonomy.yaml.
        Битый артефакт не роняет пайплайн: предупреждение в лог и None (остаётся определение по ключевым словам).
        """
        if os.getenv("CATEGORY_CLASSIFIER", "on").strip().lower() in _OFF:
            return None
        d = Path(path or os.getenv("CATEGORY_MODEL_PATH") or DEFAULT_DIR)
        if not d.is_absolute():
            d = ROOT / d
        if not all((d / f).exists() for f in (MODEL_FILE, META_FILE, VOCAB_FILE)):
            return None
        try:
            meta = json.loads((d / META_FILE).read_text(encoding="utf-8"))
            with gzip.open(d / VOCAB_FILE, "rt", encoding="utf-8") as f:
                meta["vocab"] = json.load(f)
            with np.load(d / MODEL_FILE, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            return cls(meta, arrays, taxonomy)
        except (OSError, ValueError, KeyError, TypeError, EOFError) as e:
            log.warning("модель категории %s не загружена: %s", d, e)
            return None

    def set_taxonomy(self, taxonomy: Any = None) -> None:
        """Отображение классов на категории таксономии (category_ids); вызывать после правки таксономии."""
        try:
            specs = _taxonomy_specs(taxonomy)
        except (OSError, ValueError) as e:
            log.warning("таксономия не прочитана (%s): все листья -> generic", e)
            specs = []
        mapping, self.provisional_map = leaf_to_taxonomy(specs, self.meta.get("provisional_leaves", []))
        names = [n for n, *_ in specs if n != GENERIC]
        self.taxonomy_names = [*names, GENERIC]
        tax_index = {n: i for i, n in enumerate(self.taxonomy_names)}
        self.class_tax = np.array([tax_index[mapping[c["id"]]] if c["id"] in mapping and mapping[c["id"]] in tax_index
                                   else tax_index[GENERIC] for c in self.classes], dtype=np.int64)
        self.unmapped_leaves = sorted(set(mapping) - set(self.class_index))  # листья таксономии вне пространства меток
        if self.unmapped_leaves:
            log.info("листья таксономии вне пространства меток модели (нужно переобучение): %s", self.unmapped_leaves)

    # ------------------------------------------------------------------ вычисления
    def _prepare_scoring(self, min_row: int | None = None) -> None:
        """Делит веса на плотную часть (частые признаки, float32-матрица для BLAS) и разреженную (остальные)."""
        k = self.n_roots + self.n_classes
        rowlen = np.diff(self.w_indptr)
        thr = min_row if min_row is not None else max(32, int(k * DENSE_ROW_SHARE))
        dense = np.flatnonzero(rowlen >= thr)
        self._dense_pos = np.full(len(rowlen), -1, dtype=np.int64)
        self._dense_pos[dense] = np.arange(len(dense))
        in_dense = np.repeat(self._dense_pos >= 0, rowlen)
        self._w_dense = np.zeros((len(dense), k), dtype=np.float32)
        self._w_dense[np.repeat(np.arange(len(dense)), rowlen[dense]), self.w_col[in_dense]] = self.w_q[in_dense]
        sp_len = np.where(self._dense_pos >= 0, 0, rowlen)
        self._sp_indptr = np.zeros(len(rowlen) + 1, dtype=np.int64)
        np.cumsum(sp_len, out=self._sp_indptr[1:])
        self._sp_col, self._sp_q = self.w_col[~in_dense], self.w_q[~in_dense]

    @staticmethod
    def _expand(doc: np.ndarray, cols: np.ndarray, vals: np.ndarray, w_indptr: np.ndarray, w_col: np.ndarray,
                w_q: np.ndarray, n: int, k: int) -> np.ndarray:
        """Разреженное X·W: каждый признак документа «разворачивается» в свою строку весов (CSR по признакам)."""
        start, cnt = w_indptr[cols], w_indptr[cols + 1] - w_indptr[cols]
        total = int(cnt.sum())
        if not total:
            return np.zeros((n, k))
        pos = np.repeat(start - np.cumsum(cnt) + cnt, cnt) + np.arange(total, dtype=np.int64)
        w = w_q[pos].astype(np.float32) * np.repeat(vals, cnt)
        return np.bincount(np.repeat(doc, cnt) * k + w_col[pos], weights=w, minlength=n * k).reshape(n, k)

    def _scores(self, indptr: np.ndarray, cols: np.ndarray, vals: np.ndarray) -> np.ndarray:
        """Оценки (n × (разделы + классы)) = X · W · scale + bias."""
        n, k = len(indptr) - 1, self.n_roots + self.n_classes
        doc = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
        if n <= SMALL_BATCH or not len(self._w_dense):
            s = self._expand(doc, cols, vals, self.w_indptr, self.w_col, self.w_q, n, k)
        else:  # пакет: частые признаки — плотным умножением (BLAS), редкие — разреженно
            dpos = self._dense_pos[cols]
            dense = dpos >= 0
            x = np.zeros((n, len(self._w_dense)), dtype=np.float32)
            x[doc[dense], dpos[dense]] = vals[dense]
            s = (x @ self._w_dense).astype(np.float64)
            s += self._expand(doc[~dense], cols[~dense], vals[~dense], self._sp_indptr, self._sp_col, self._sp_q, n, k)
        return s * self.scale + self.bias

    def proba_from_scores(self, s: np.ndarray, t_root: float | None = None,
                          t_leaf: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Оценки -> (P(раздел) n×R, P(лист) n×L)."""
        p_root = _softmax_rows(s[:, :self.n_roots] / (t_root or self.t_root))
        z = s[:, self.n_roots:] / (t_leaf or self.t_leaf)
        z -= np.maximum.reduceat(z, self.group_start, axis=1)[:, self.class_root]
        np.exp(z, out=z)
        sums = np.add.reduceat(z, self.group_start, axis=1)
        p_leaf = z / sums[:, self.class_root] * p_root[:, self.class_root]
        return p_root, p_leaf

    def scores_many(self, cards: Sequence[Any], batch: int = 512) -> np.ndarray:
        out = []
        for i in range(0, len(cards), batch):
            ip, ix, vx = self.featurizer.transform(_doc(c) for c in cards[i:i + batch])
            out.append(self._scores(ip, ix, vx))
        return np.vstack(out) if out else np.zeros((0, self.n_roots + self.n_classes))

    def taxonomy_proba(self, p_leaf: np.ndarray) -> np.ndarray:
        """P(категория таксономии) = сумма P(лист) по её листьям; столбцы — self.taxonomy_names."""
        m = np.zeros((self.n_classes, len(self.taxonomy_names)))
        m[np.arange(self.n_classes), self.class_tax] = 1.0
        return p_leaf @ m

    def decide(self, p_leaf: np.ndarray) -> list[CategoryPrediction]:
        p_tax = self.taxonomy_proba(p_leaf)
        tax = p_tax.argmax(axis=1)
        conf = p_tax[np.arange(len(tax)), tax]
        # лист — самый вероятный внутри выбранной категории таксономии (согласованность для UI)
        leaf = np.where(self.class_tax[None, :] == tax[:, None], p_leaf, -1.0).argmax(axis=1)
        k3 = min(3, self.n_classes)
        top = np.argpartition(-p_leaf, k3 - 1, axis=1)[:, :k3] if self.n_classes > k3 else \
            np.broadcast_to(np.arange(self.n_classes), (len(p_leaf), self.n_classes))
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(p_leaf, top, axis=1), axis=1), axis=1)
        cls = self.classes
        return [CategoryPrediction(
            leaf_id=cls[leaf[i]]["id"], leaf_title=cls[leaf[i]]["title"], taxonomy_category=self.taxonomy_names[tax[i]],
            confidence=round(float(conf[i]), 4),
            top3=[(cls[j]["id"], cls[j]["title"], round(float(p_leaf[i, j]), 4)) for j in top[i]])
            for i in range(len(p_leaf))]

    # ------------------------------------------------------------------ публичный API
    def predict_many(self, cards: Sequence[Any], batch: int = 512) -> list[CategoryPrediction]:
        """Пакетное предсказание: Card или dict с полями title, description, seller_attrs."""
        out: list[CategoryPrediction] = []
        for i in range(0, len(cards), batch):
            _, p_leaf = self.proba_from_scores(self.scores_many(cards[i:i + batch], batch))
            out.extend(self.decide(p_leaf))
        return out

    def predict(self, card: Any) -> CategoryPrediction:
        return self.predict_many([card])[0]

    def leaf_path(self, class_id: str) -> list[str]:
        """Путь в дереве каталога: ["Раздел", "Подраздел", "Лист"]; для «Прочее в разделе» — ["Раздел"]."""
        i = self.class_index.get(class_id)
        return list(self.classes[i]["path"]) if i is not None else []

    def is_review(self, prediction: CategoryPrediction) -> bool:
        """Уверенность ниже порога (meta.json: threshold, правило — в metrics.json) -> карточку на проверку."""
        return prediction.confidence < self.threshold

    def info(self) -> dict[str, Any]:
        return {"format": FORMAT, "n_roots": self.n_roots, "n_classes": self.n_classes,
                "n_features": self.featurizer.n_features, "threshold": self.threshold,
                "taxonomy": self.taxonomy_names, "provisional_map": self.provisional_map,
                "unmapped_taxonomy_leaves": self.unmapped_leaves, "trained": self.meta.get("trained")}


# ============================================================================ обучение (нужен scikit-learn, extra [ml])


@dataclass
class LabelSpace:
    """Пространство меток: разделы (этап 1) и классы-листья блоками по разделам (этап 2)."""

    roots: list[dict]  # [{id, title, n_train}]
    classes: list[dict]  # [{id, title, path, root, n_train}]; id листа или other:<id раздела>

    def __post_init__(self) -> None:
        self.root_index = {r["id"]: i for i, r in enumerate(self.roots)}
        self.class_index = {c["id"]: i for i, c in enumerate(self.classes)}

    def class_of(self, leaf_id: str, root_id: str) -> int:
        """Индекс класса карточки: лист, иначе «Прочее в разделе»; -1 — раздела/класса нет в train."""
        i = self.class_index.get(leaf_id)
        return i if i is not None else self.class_index.get(OTHER + root_id, -1)


def build_label_space(train_counts: Mapping[str, int], cats: Mapping[str, dict], max_classes: int,
                      forced: Iterable[str] = (), min_forced: int = 10) -> LabelSpace:
    """Топ-max_classes листьев по числу карточек train + листья таксономии (forced, если есть ≥ min_forced карточек);
    остальные листья каждого раздела -> один класс «Прочее в разделе»."""
    ranked = sorted(train_counts, key=lambda c: (-train_counts[c], c))
    keep = set(ranked[:max_classes]) | {c for c in forced if train_counts.get(c, 0) >= min_forced}
    root_n: Counter = Counter()
    for c, n in train_counts.items():
        root_n[cats[c]["root"]] += n
    roots = [{"id": r, "title": cats[r]["title"], "n_train": int(n)}
             for r, n in sorted(root_n.items(), key=lambda x: (-x[1], x[0]))]
    classes: list[dict] = []
    for ri, r in enumerate(roots):
        leaves = [c for c in ranked if cats[c]["root"] == r["id"]]
        kept = [c for c in leaves if c in keep]
        classes += [{"id": c, "title": cats[c]["title"], "path": cats[c]["path"], "root": ri,
                     "n_train": int(train_counts[c])} for c in kept]
        rest = sum(train_counts[c] for c in leaves if c not in keep)
        if rest:
            classes.append({"id": OTHER + r["id"], "title": f"Прочее: {r['title']}", "path": [r["title"]], "root": ri,
                            "n_train": int(rest), "n_leaves": len(leaves) - len(kept)})
    return LabelSpace(roots, classes)


def _read_sql(processed: Path, sql: str):
    import duckdb

    con = duckdb.connect()
    con.execute("SET threads=2")
    q = lambda p: "'" + str(processed / p).replace("'", "''") + "'"  # noqa: E731
    return con.execute(sql.replace("{cards}", q("cards.parquet")).replace("{cats}", q("categories.parquet"))).df()


def _ctx_column(df) -> list[str]:
    return [context_text(d if isinstance(d, str) else "", json.loads(sa) if isinstance(sa, str) and len(sa) > 2 else None)
            for d, sa in zip(df["d"], df["seller_attrs"], strict=True)]


def load_labeled(processed: Path) -> tuple[Any, dict[str, dict], int]:
    """Карточки с известной категорией (оффер или seo_id кликов), дерево категорий, число отброшенных карточек
    (id категории вне дерева каталога, например seo_id = 0)."""
    cats_df = _read_sql(processed, "SELECT id, title, path, path_ids FROM {cats}")
    cats = {r.id: {"title": r.title, "path": list(r.path), "root": r.path_ids[0]} for r in cats_df.itertuples()}
    df = _read_sql(processed, f"""
        SELECT c.sku_id, c.title, left(coalesce(c.description_clean, ''), {2 * CTX_CHARS}) AS d, c.seller_attrs,
               c.category_id, c.split, c.category_source, c.seller_type
        FROM {{cards}} c WHERE c.category_id IS NOT NULL ORDER BY c.sku_id""")
    known = df["category_id"].isin(cats.keys())
    dropped = int((~known).sum())
    df = df[known].reset_index(drop=True)
    df["title"] = [t if isinstance(t, str) else "" for t in df["title"]]
    df["ctx"] = _ctx_column(df)
    df["root"] = [cats[c]["root"] for c in df["category_id"]]
    return df.drop(columns=["d", "seller_attrs"]), cats, dropped


_WORKER_FZ: Featurizer | None = None


def _transform_chunk(chunk: list[tuple[str, str]]):
    assert _WORKER_FZ is not None
    return _WORKER_FZ.transform(chunk)


def featurize(fz: Featurizer, docs: Sequence[tuple[str, str]], n_jobs: int = 2, chunk: int = 20_000):
    """Матрица признаков scipy CSR; n_jobs > 1 — процессы (fork), словарь наследуется без копирования."""
    import multiprocessing as mp

    import scipy.sparse as sp

    global _WORKER_FZ
    parts_in = [list(docs[i:i + chunk]) for i in range(0, len(docs), chunk)]
    if n_jobs > 1 and len(parts_in) > 1 and "fork" in mp.get_all_start_methods():
        _WORKER_FZ = fz
        try:
            with mp.get_context("fork").Pool(n_jobs) as pool:
                parts = pool.map(_transform_chunk, parts_in)
        finally:
            _WORKER_FZ = None
    else:
        parts = [fz.transform(p) for p in parts_in]
    mats = [sp.csr_matrix((v, ix, ip), shape=(len(ip) - 1, fz.n_features)) for ip, ix, v in parts]
    return sp.vstack(mats, format="csr") if mats else sp.csr_matrix((0, fz.n_features), dtype=np.float32)


def _sgd(alpha: float, epochs: int, n_jobs: int, seed: int):
    from sklearn.linear_model import SGDClassifier

    return SGDClassifier(loss="hinge", alpha=alpha, max_iter=epochs, tol=None, n_jobs=n_jobs, random_state=seed)


def _prune(coef: np.ndarray, cols: np.ndarray, keep: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Плотные веса классов (c×V) -> разреженные int8: top-keep признаков по |w| у каждого класса, масштаб max|w|/127.

    Возвращает (признак, столбец, int8, масштаб столбцов)."""
    feats, colz, qs = [], [], []
    scale = np.zeros(len(cols), dtype=np.float32)
    for row, col in zip(coef, cols, strict=True):
        nz = np.flatnonzero(row)
        if len(nz) > keep:
            nz = nz[np.argpartition(-np.abs(row[nz]), keep - 1)[:keep]]
        if not len(nz):
            continue
        s = float(np.abs(row[nz]).max()) / 127.0
        q = np.rint(row[nz] / s).astype(np.int8)
        ok = q != 0
        feats.append(nz[ok])
        colz.append(np.full(int(ok.sum()), col, dtype=np.int64))
        qs.append(q[ok])
        scale[np.searchsorted(cols, col)] = s
    cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dtype=dt)  # noqa: E731
    return cat(feats, np.int64), cat(colz, np.int64), cat(qs, np.int8), scale


def fit_hierarchical(X, y_root: np.ndarray, y_class: np.ndarray, ls: LabelSpace, *, alpha: float, epochs: int,
                     n_jobs: int, seed: int, keep_root: int, keep_leaf: int) -> dict[str, np.ndarray]:
    """Этап 1: раздел (OvR по всем карточкам). Этап 2: лист внутри раздела (OvR только по карточкам раздела).
    Возвращает массивы артефакта: веса по признакам (CSR: w_indptr, w_col, w_q), scale, bias."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    n_roots, n_classes = len(ls.roots), len(ls.classes)
    k = n_roots + n_classes
    bias = np.zeros(k, dtype=np.float32)
    scale = np.zeros(k, dtype=np.float32)
    parts = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m = _sgd(alpha, epochs, n_jobs, seed).fit(X, y_root)
        cols = m.classes_.astype(np.int64)
        f, c, q, s = _prune(m.coef_, cols, keep_root)
        parts.append((f, c, q))
        scale[cols], bias[cols] = s, m.intercept_
        del m
        for ri in range(n_roots):
            idx = np.flatnonzero(y_root == ri)
            labels = np.unique(y_class[idx])
            if len(labels) < 2:  # единственный класс в разделе: P(лист | раздел) = 1
                continue
            m = _sgd(alpha, epochs, n_jobs, seed).fit(X[idx], y_class[idx])
            coef, b = m.coef_, m.intercept_
            if len(labels) == 2:  # бинарный случай: оценки ±d/2 — softmax по двум классам = сигмоида от d
                coef, b = np.vstack([-coef[0] / 2, coef[0] / 2]), np.array([-b[0] / 2, b[0] / 2])
            cols = n_roots + m.classes_.astype(np.int64)
            f, c, q, s = _prune(coef, cols, keep_leaf)
            parts.append((f, c, q))
            scale[cols], bias[cols] = s, b
            del m, coef
    f = np.concatenate([p[0] for p in parts])
    c = np.concatenate([p[1] for p in parts])
    q = np.concatenate([p[2] for p in parts])
    order = np.lexsort((c, f))
    indptr = np.zeros(X.shape[1] + 1, dtype=np.int64)
    np.cumsum(np.bincount(f, minlength=X.shape[1]), out=indptr[1:])
    return {"w_indptr": indptr.astype(np.int32), "w_col": c[order].astype(np.uint16 if k < 65_536 else np.int32),
            "w_q": q[order], "scale": scale, "bias": bias}


def _scores_X(clf: CategoryClassifier, X, batch: int = 4096) -> np.ndarray:
    """Оценки модели для готовой CSR-матрицы признаков (в словаре clf)."""
    out = []
    for i in range(0, X.shape[0], batch):
        part = X[i:i + batch]
        out.append(clf._scores(part.indptr.astype(np.int64), part.indices.astype(np.int64), part.data))
    return np.vstack(out) if out else np.zeros((0, clf.n_roots + clf.n_classes))


def fit_temperatures(clf: CategoryClassifier, s: np.ndarray, y_root: np.ndarray, y_class: np.ndarray) -> dict:
    """Температуры softmax (раздел; лист внутри раздела) — минимум NLL на dev (temperature scaling):
    сначала каждая по своему уровню, затем совместно по итоговой вероятности листа."""
    from scipy.optimize import minimize, minimize_scalar

    r = clf.n_roots
    rows = np.arange(len(s))

    def nll_root(log_t: float) -> float:
        z = s[:, :r] / math.exp(log_t)
        m = z.max(axis=1)
        return float(np.mean(m + np.log(np.exp(z - m[:, None]).sum(axis=1)) - z[rows, y_root]))

    sizes = np.bincount(clf.class_root, minlength=r)
    multi = sizes[y_root] > 1  # в разделе с одним классом P(лист | раздел) = 1 — температура не влияет
    sm, ym, ycm = s[multi][:, r:], y_root[multi], y_class[multi]  # y_class — индекс класса (без сдвига на разделы)
    rows_m = np.arange(len(sm))

    def nll_leaf(log_t: float) -> float:
        z = sm / math.exp(log_t)
        g = np.maximum.reduceat(z, clf.group_start, axis=1)
        lse = np.log(np.add.reduceat(np.exp(z - g[:, clf.class_root]), clf.group_start, axis=1)) + g
        return float(np.mean(lse[rows_m, ym] - z[rows_m, ycm]))

    out = {}
    for name, f in (("root", nll_root), ("leaf", nll_leaf)):
        res = minimize_scalar(f, bounds=(math.log(1e-3), math.log(20.0)), method="bounded", options={"xatol": 1e-3})
        out[name] = round(math.exp(res.x), 5)
        out[f"nll_{name}"] = round(float(res.fun), 5)

    # уточнение: обе температуры вместе по NLL итоговой P(лист) = P(раздел)·P(лист | раздел)
    # (на dev: ECE листа 0,014 -> 0,005 при той же точности)
    ok = y_class >= 0

    def nll_joint(x: np.ndarray) -> float:
        _, p = clf.proba_from_scores(s[ok], math.exp(x[0]), math.exp(x[1]))
        return float(-np.mean(np.log(np.clip(p[np.arange(int(ok.sum())), y_class[ok]], 1e-12, 1.0))))

    res = minimize(nll_joint, np.log([out["root"], out["leaf"]]), method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 200})
    out["nll_separate"] = round(nll_joint(np.log([out["root"], out["leaf"]])), 5)
    if res.fun < out["nll_separate"]:
        out["root"], out["leaf"] = round(math.exp(res.x[0]), 5), round(math.exp(res.x[1]), 5)
    out["nll_final_leaf"] = round(min(float(res.fun), out["nll_separate"]), 5)
    return out


def choose_threshold(conf: np.ndarray, correct: np.ndarray, target: float) -> float:
    """Минимальный порог уверенности, при котором точность среди карточек выше порога ≥ target (макс. покрытие)."""
    order = np.argsort(-conf, kind="stable")
    acc = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    ok = np.flatnonzero(acc >= target)
    if not len(ok):
        return 1.0
    return float(math.ceil(round(float(conf[order[ok[-1]]]) * 100, 6)) / 100)  # округление вверх до сотых


def prf(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, Any]:
    """Accuracy, macro-F1 и P/R/F1 по классам (классы — объединение истинных и предсказанных)."""
    labels = sorted(set(y_true) | set(y_pred))
    tp, fp, fn = Counter(), Counter(), Counter()
    for t, p in zip(y_true, y_pred, strict=True):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    per = {}
    for c in labels:
        p = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        r = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        per[c] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r) if p + r else 0.0, 4),
                  "support": tp[c] + fn[c], "predicted": tp[c] + fp[c]}
    n = len(y_true)
    return {"n": n, "accuracy": round(sum(tp.values()) / n, 4) if n else 0.0,
            "macro_f1": round(float(np.mean([v["f1"] for v in per.values()])), 4) if per else 0.0, "per_class": per}


def reliability(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> dict[str, Any]:
    """Бины уверенности -> точность (reliability diagram) и ECE."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            mc, acc = float(conf[m].mean()), float(correct[m].mean())
            ece += m.mean() * abs(mc - acc)
            rows.append({"bin": f"{edges[b]:.1f}–{edges[b + 1]:.1f}", "n": int(m.sum()), "mean_confidence": round(mc, 4),
                         "accuracy": round(acc, 4)})
    return {"bins": rows, "ece": round(float(ece), 4)}


def coverage_curve(conf: np.ndarray, correct: np.ndarray, thresholds: Iterable[float]) -> list[dict]:
    out = []
    for t in thresholds:
        m = conf >= t
        out.append({"threshold": round(float(t), 3), "coverage": round(float(m.mean()), 4),
                    "accuracy": round(float(correct[m].mean()), 4) if m.any() else None, "n": int(m.sum())})
    return out


def _kw_norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace("ё", "е")).strip()


def keyword_category(specs: list[tuple[str, str, list[str], list[str]]], title: str, full_text: str) -> str:
    """Базлайн на правилах: копия ветки ключевых слов Taxonomy.detect_category (cold start без category_id)."""
    t, text = _kw_norm(title), _kw_norm(full_text)
    best, score = GENERIC, 0.0
    for name, _, keywords, _ in specs:
        s = sum(2.0 if _kw_norm(k) in t else 1.0 for k in keywords if _kw_norm(k) in text)
        if s > score:
            best, score = name, s
    return best


def _full_text(title: str, description: str, seller_attrs: Mapping[str, Any] | None) -> str:
    seller = "; ".join(f"{k}: {v}" for k, v in (seller_attrs or {}).items())
    return "\n".join(x for x in (title, description, seller) if x)  # как Card.full_text()


def _provisional_leaves(processed: Path) -> list[dict[str, str]]:
    p = processed / "taxonomy_candidates.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return [{"id": str(c["id"]), "title": c["title"]} for c in data.get("categories", [])]


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else None


def _assemble(fz: Featurizer, arrays: dict[str, np.ndarray], ls: LabelSpace, temps: Mapping[str, float],
              threshold: float, extra: Mapping[str, Any]) -> tuple[dict, dict[str, np.ndarray]]:
    """meta.json + массивы model.npz; признаки без единого веса удаляются из словаря (меньше артефакт)."""
    counts = np.diff(arrays["w_indptr"].astype(np.int64))
    used = np.flatnonzero(counts > 0)
    blocks = np.searchsorted(np.asarray(fz.offsets[1:]), used, side="right")
    vocab = [[fz.terms[b][j - fz.offsets[b]] for j, bb in zip(used, blocks, strict=True) if bb == b]
             for b in range(len(BLOCKS))]
    indptr = np.zeros(len(used) + 1, dtype=np.int32)
    np.cumsum(counts[used], out=indptr[1:])
    out = {"w_indptr": indptr, "w_col": arrays["w_col"], "w_q": arrays["w_q"],
           "scale": arrays["scale"].astype(np.float32), "bias": arrays["bias"].astype(np.float32),
           "idf": fz.idf[used].astype(np.float32)}
    meta = {"format": FORMAT, **extra, "block_weights": fz.weights, "temperature": dict(temps), "threshold": threshold,
            "roots": ls.roots, "classes": ls.classes, "vocab": vocab}
    return meta, out


def save_artifact(out: Path, meta: dict, arrays: Mapping[str, np.ndarray]) -> dict[str, float]:
    """model.npz (веса), vocab.json.gz (словарь n-грамм), meta.json (классы, калибровка, порог) — без pickle.
    Запись атомарная (временный файл + rename): параллельные читатели не увидят половину модели."""
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(meta)
    vocab = meta.pop("vocab")
    tmp = out / (MODEL_FILE + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(out / MODEL_FILE)
    tmp = out / (VOCAB_FILE + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(vocab, f, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(out / VOCAB_FILE)
    tmp = out / (META_FILE + ".tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(out / META_FILE)
    return {f: round((out / f).stat().st_size / 1e6, 2) for f in (MODEL_FILE, VOCAB_FILE, META_FILE)}


SEARCH_ALPHAS = (2e-6, 5e-6, 1e-5)
COVERAGE_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


def _targets(part, ls: LabelSpace, cats: Mapping[str, dict]) -> tuple[np.ndarray, np.ndarray]:
    yr = np.array([ls.root_index.get(cats[c]["root"], -1) for c in part["category_id"]], dtype=np.int64)
    yc = np.array([ls.class_of(c, cats[c]["root"]) for c in part["category_id"]], dtype=np.int64)
    return yr, yc


def _model_from(fz: Featurizer, arrays: dict[str, np.ndarray], ls: LabelSpace,
                extra: Mapping[str, Any]) -> tuple[CategoryClassifier, dict, dict[str, np.ndarray]]:
    meta, arr = _assemble(fz, arrays, ls, {"root": 1.0, "leaf": 1.0}, 0.0, extra)
    return CategoryClassifier(meta, arr), meta, arr


def _calibrated_dev(clf: CategoryClassifier, X_dev, y_root: np.ndarray, y_class: np.ndarray) -> dict[str, Any]:
    """Температуры на dev + точность после калибровки (то, что будет на инференсе)."""
    s = _scores_X(clf, X_dev)
    ok = y_class >= 0
    temps = fit_temperatures(clf, s[ok], y_root[ok], y_class[ok])
    clf.t_root, clf.t_leaf = temps["root"], temps["leaf"]
    p_root, p_leaf = clf.proba_from_scores(s)
    return {"temps": temps, "p_leaf": p_leaf, "root_acc": round(float((p_root.argmax(1) == y_root).mean()), 4),
            "leaf_acc": round(float((p_leaf.argmax(1) == y_class).mean()), 4)}


def _tax_arrays(clf: CategoryClassifier, p_leaf: np.ndarray) -> tuple[list[str], np.ndarray]:
    p_tax = clf.taxonomy_proba(p_leaf)
    best = p_tax.argmax(axis=1)
    return [clf.taxonomy_names[i] for i in best], p_tax[np.arange(len(best)), best]


def _sample_unlabeled(processed: Path) -> list[dict[str, Any]]:
    """Детерминированная выборка ~2,3% карточек без категории (по md5 sku_id) — для проверки на cold start."""
    df = _read_sql(processed, """SELECT sku_id, title, description_clean AS d, seller_attrs, seller FROM {cards}
        WHERE category_id IS NULL AND substr(md5(sku_id), 1, 3) < '060' ORDER BY sku_id""")
    return [{"id": r.sku_id, "title": r.title if isinstance(r.title, str) else "",
             "description": r.d if isinstance(r.d, str) else "",
             "seller_attrs": json.loads(r.seller_attrs) if isinstance(r.seller_attrs, str) and len(r.seller_attrs) > 2
             else {}, "seller": r.seller if isinstance(r.seller, str) else None} for r in df.itertuples()]


def _short(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[:n - 1] + "…"


def train_category(processed: Path | None = None, out: Path | None = None, *, max_classes: int = 1000,
                   alpha: float = 2e-6, epochs: int = 5, keep_root: int = 20_000, keep_leaf: int = 4_000,
                   n_jobs: int = 2, seed: int = 0, search: bool = False, target_precision: float = 0.99,
                   min_confidence: float = 0.9, vocab_docs: int = 120_000, n_examples: int = 15,
                   echo: Callable[[str], None] = print) -> dict[str, Any]:
    """Обучение на train, подбор alpha и калибровка на dev, один отчёт на holdout; артефакт + metrics.json в out.

    Порог уверенности = max(минимальный порог с точностью категории ≥ target_precision на dev, min_confidence):
    на карточках с категорией модель почти безошибочна и первый порог вырождается (~0,5), а уверенность
    калибрована — min_confidence = 0,9 означает «каждая автоматически принятая категория верна с вероятностью ≥ 90%».
    """
    t_all = time.perf_counter()
    processed = Path(processed or ROOT / "data" / "processed")
    out = Path(out or DEFAULT_DIR)
    if not (processed / "cards.parquet").exists():
        raise FileNotFoundError(f"нет {processed / 'cards.parquet'} — сначала `attrx prepare`")
    timings: dict[str, float] = {}

    def lap(name: str, t0: float) -> None:
        timings[name] = round(time.perf_counter() - t0, 1)

    # 1. данные и пространство меток ------------------------------------------------------------------
    t0 = time.perf_counter()
    df, cats, dropped = load_labeled(processed)
    specs = _taxonomy_specs()
    provisional = _provisional_leaves(processed)
    tax_map, _ = leaf_to_taxonomy(specs, provisional)  # то же отображение, что применит модель при загрузке
    forced = set(tax_map) | {p["id"] for p in provisional}
    parts = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "dev", "holdout")}
    tr, dv, ho = parts["train"], parts["dev"], parts["holdout"]
    counts = tr["category_id"].value_counts().to_dict()
    ls = build_label_space(counts, cats, max_classes, forced)
    ys = {s: _targets(p, ls, cats) for s, p in parts.items()}
    n_leaf_classes = sum(not c["id"].startswith(OTHER) for c in ls.classes)
    lap("load", t0)
    echo(f"карточек с категорией: {len(df)} (train {len(tr)} / dev {len(dv)} / holdout {len(ho)}), "
         f"отброшено с id вне дерева: {dropped}; разделов {len(ls.roots)}, классов {len(ls.classes)} "
         f"(листьев {n_leaf_classes} + «Прочее в разделе» {len(ls.classes) - n_leaf_classes}) — {timings['load']} с")

    # 2. признаки -------------------------------------------------------------------------------------
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(len(tr), size=min(vocab_docs, len(tr)), replace=False))
    titles, ctxs = tr["title"].tolist(), tr["ctx"].tolist()
    fz = Featurizer.fit((titles[i], ctxs[i]) for i in sub)
    X_tr = featurize(fz, list(zip(titles, ctxs, strict=True)), n_jobs)
    docs_dv = list(zip(dv["title"], dv["ctx"], strict=True))
    lap("features", t0)
    echo(f"признаки: {fz.n_features} (словарь по {len(sub)} карточкам train), "
         f"{X_tr.nnz / max(X_tr.shape[0], 1):.0f} ненулевых на карточку — {timings['features']} с")

    # 3. обучение и подбор alpha на dev ----------------------------------------------------------------
    extra = {"trained": {"date": time.strftime("%Y-%m-%d"), "max_classes": max_classes, "epochs": epochs,
                         "keep_root": keep_root, "keep_leaf": keep_leaf, "seed": seed},
             "provisional_leaves": provisional}
    selection, best = [], None
    for a in (SEARCH_ALPHAS if search else (alpha,)):
        t0 = time.perf_counter()
        arrays = fit_hierarchical(X_tr, *ys["train"], ls, alpha=a, epochs=epochs, n_jobs=n_jobs, seed=seed,
                                  keep_root=keep_root, keep_leaf=keep_leaf)
        fit_s = time.perf_counter() - t0
        clf, meta, arr = _model_from(fz, arrays, ls, {**extra, "trained": {**extra["trained"], "alpha": a}})
        X_dv = featurize(clf.featurizer, docs_dv, n_jobs)
        dev = _calibrated_dev(clf, X_dv, *ys["dev"])
        row = {"alpha": a, "dev_leaf_top1": dev["leaf_acc"], "dev_root_acc": dev["root_acc"],
               "dev_nll_root": dev["temps"]["nll_root"], "dev_nll_leaf": dev["temps"]["nll_leaf"],
               "nnz": int(len(arr["w_q"])), "fit_s": round(fit_s, 1)}
        selection.append(row)
        echo(f"alpha={a:g}: dev лист top-1 {dev['leaf_acc']:.4f}, раздел {dev['root_acc']:.4f}, "
             f"весов {row['nnz']}, обучение {fit_s:.0f} с")
        if best is None or dev["leaf_acc"] > best[0]["dev_leaf_top1"]:
            best = (row, clf, meta, arr, dev)
        timings[f"fit_alpha_{a:g}"] = round(time.perf_counter() - t0, 1)
    assert best is not None
    row, clf, meta, arr, dev = best

    # 4. порог по dev: максимальное покрытие при точности категории таксономии ≥ target ----------------
    tax_true_dv = [tax_map.get(c, GENERIC) for c in dv["category_id"]]
    tax_pred_dv, conf_dv = _tax_arrays(clf, dev["p_leaf"])
    correct_dv = np.array([p == t for p, t in zip(tax_pred_dv, tax_true_dv, strict=True)])
    precision_threshold = choose_threshold(conf_dv, correct_dv, target_precision)
    threshold = max(precision_threshold, min_confidence)
    cov_dv = conf_dv >= threshold
    meta["temperature"] = {"root": dev["temps"]["root"], "leaf": dev["temps"]["leaf"]}
    meta["threshold"] = threshold
    meta["trained"]["alpha"] = row["alpha"]
    meta["trained"]["threshold_rule"] = {"target_precision": target_precision, "precision_threshold": precision_threshold,
                                         "min_confidence": min_confidence}
    sizes = save_artifact(out, meta, arr)
    clf = CategoryClassifier.load(out)
    if clf is None:
        raise RuntimeError(f"артефакт в {out} не загрузился")
    echo(f"калибровка: T(раздел)={meta['temperature']['root']}, T(лист)={meta['temperature']['leaf']}; порог {threshold} "
         f"(dev: покрытие {cov_dv.mean():.4f}, точность {correct_dv[cov_dv].mean():.4f}); артефакт {sizes} МБ")

    # 5. holdout — один раз ------------------------------------------------------------------------------
    t0 = time.perf_counter()
    yr_ho, yc_ho = ys["holdout"]
    s_ho = _scores_X(clf, featurize(clf.featurizer, list(zip(ho["title"], ho["ctx"], strict=True)), n_jobs))
    p_root, p_leaf = clf.proba_from_scores(s_ho)
    tax_pred, conf = _tax_arrays(clf, p_leaf)
    tax_true = [tax_map.get(c, GENERIC) for c in ho["category_id"]]
    correct = np.array([p == t for p, t in zip(tax_pred, tax_true, strict=True)])
    top3 = np.argsort(-p_leaf, axis=1)[:, :3]
    leaf_top1 = top3[:, 0] == yc_ho
    leaf_top3 = (top3 == yc_ho[:, None]).any(axis=1)
    leaf_conf = p_leaf[np.arange(len(p_leaf)), top3[:, 0]]
    explicit = np.array([c in ls.class_index for c in ho["category_id"]])
    full_df = _read_sql(processed, """SELECT sku_id, description_clean AS d, seller_attrs FROM {cards}
        WHERE split = 'holdout' AND category_id IS NOT NULL""")
    full = dict(zip(full_df["sku_id"], zip(full_df["d"], full_df["seller_attrs"], strict=True), strict=True))
    kw_pred = []
    for sku, title in zip(ho["sku_id"], ho["title"], strict=True):  # базлайн видит полный текст карточки
        d, sa = full[sku]
        sa = json.loads(sa) if isinstance(sa, str) and len(sa) > 2 else None
        kw_pred.append(keyword_category(specs, title, _full_text(title, d if isinstance(d, str) else "", sa)))
    kw_correct = np.array([p == t for p, t in zip(kw_pred, tax_true, strict=True)])
    train_titles = {normalize(t).strip() for t in tr["title"]}
    novel = np.array([normalize(t).strip() not in train_titles for t in ho["title"]])
    n_train_class = np.array([ls.classes[c]["n_train"] if c >= 0 else 0 for c in yc_ho])

    def seg(mask: np.ndarray) -> dict[str, Any]:
        return {"n": int(mask.sum()), "taxonomy_accuracy": round(float(correct[mask].mean()), 4) if mask.any() else None,
                "keyword_taxonomy_accuracy": round(float(kw_correct[mask].mean()), 4) if mask.any() else None,
                "leaf_top1": round(float(leaf_top1[mask].mean()), 4) if mask.any() else None}

    model_prf = prf(tax_true, tax_pred)
    kw_prf = prf(tax_true, kw_pred)
    # разбор ошибок: все ошибки категории таксономии, частые путаницы листьев, худшие листья по полноте
    tax_idx = np.array([clf.taxonomy_names.index(t) for t in tax_pred])
    leaf_cons = np.where(clf.class_tax[None, :] == tax_idx[:, None], p_leaf, -1.0).argmax(axis=1)
    cls_title = lambda c: ls.classes[c]["title"] if c >= 0 else "лист вне train"  # noqa: E731
    tax_errors = [{"sku_id": ho["sku_id"].iloc[i], "title": _short(ho["title"].iloc[i], 110),
                   "true_leaf": cats[ho["category_id"].iloc[i]]["title"], "true_category": tax_true[i],
                   "predicted_leaf": cls_title(int(leaf_cons[i])), "predicted_category": tax_pred[i],
                   "confidence": round(float(conf[i]), 4), "keyword_baseline": kw_pred[i],
                   "source": ho["category_source"].iloc[i]} for i in np.flatnonzero(~correct)[:50]]
    pairs = Counter((cls_title(int(t)), cls_title(int(q))) for t, q in zip(yc_ho, top3[:, 0], strict=True) if t != q)
    support, hits = Counter(yc_ho[yc_ho >= 0].tolist()), Counter(yc_ho[leaf_top1].tolist())
    worst = sorted((hits[c] / n, c, n) for c, n in support.items() if n >= 20)[:10]
    holdout = {
        "n": len(ho),
        "taxonomy": {"model": model_prf, "keyword_baseline": kw_prf},
        "leaf": {"top1": round(float(leaf_top1.mean()), 4), "top3": round(float(leaf_top3.mean()), 4),
                 "root_accuracy": round(float((p_root.argmax(1) == yr_ho).mean()), 4),
                 "in_label_space_share": round(float(explicit.mean()), 4),
                 "exact_leaf_top1": round(float(leaf_top1[explicit].mean()), 4),
                 "unseen_in_train": int((yc_ho < 0).sum())},
        "segments": {
            "source_offer": seg(ho["category_source"].to_numpy() == "offer"),
            "source_clicks_seo": seg(ho["category_source"].to_numpy() == "clicks_seo"),
            "seller_mvideo": seg(ho["seller_type"].to_numpy() == "mvideo"),
            "seller_marketplace": seg(ho["seller_type"].to_numpy() == "marketplace"),
            "title_seen_in_train": seg(~novel), "title_novel": seg(novel),
            "class_head_1000plus": seg(n_train_class >= 1000),
            "class_torso_100_999": seg((n_train_class >= 100) & (n_train_class < 1000)),
            "class_tail_lt100": seg(n_train_class < 100)},
        "reliability_taxonomy": reliability(conf, correct.astype(float)),
        "reliability_leaf": reliability(leaf_conf, leaf_top1.astype(float)),
        "coverage_accuracy": coverage_curve(conf, correct.astype(float), sorted({*COVERAGE_THRESHOLDS, threshold})),
        "at_threshold": {"threshold": threshold, "coverage": round(float((conf >= threshold).mean()), 4),
                         "accuracy": round(float(correct[conf >= threshold].mean()), 4),
                         "review_share": round(float((conf < threshold).mean()), 4)},
        "errors": {"n_taxonomy_errors": int((~correct).sum()), "taxonomy_errors": tax_errors,
                   "top_leaf_confusions": [{"true": a, "predicted": b, "n": n} for (a, b), n in pairs.most_common(15)],
                   "worst_leaves_by_recall": [{"leaf": ls.classes[c]["title"], "id": ls.classes[c]["id"], "support": n,
                                               "recall": round(r, 4)} for r, c, n in worst]},
    }
    lap("holdout", t0)
    echo(f"holdout ({len(ho)}): категория таксономии — accuracy {model_prf['accuracy']}, macro-F1 {model_prf['macro_f1']} "
         f"(ключевые слова: {kw_prf['accuracy']} / {kw_prf['macro_f1']}); лист top-1 {holdout['leaf']['top1']}, "
         f"top-3 {holdout['leaf']['top3']}; ECE {holdout['reliability_taxonomy']['ece']}")

    # 6. карточки без категории (cold start): распределение и примеры для просмотра глазами --------------
    t0 = time.perf_counter()
    unl = _sample_unlabeled(processed)
    clf_fresh = CategoryClassifier.load(out)  # холодный кэш — честная скорость
    assert clf_fresh is not None
    t1 = time.perf_counter()
    preds = clf_fresh.predict_many(unl)
    batch_s = time.perf_counter() - t1
    single = []
    for c in unl[:300]:
        t1 = time.perf_counter()
        clf_fresh.predict(c)
        single.append((time.perf_counter() - t1) * 1000)
    t1 = time.perf_counter()
    kw_unl = [keyword_category(specs, c["title"], _full_text(c["title"], c["description"], c["seller_attrs"])) for c in unl]
    kw_s = time.perf_counter() - t1
    conf_u = np.array([p.confidence for p in preds])
    leaf_u = np.array([max(t[2] for t in p.top3) for p in preds])
    dist = Counter(p.taxonomy_category for p in preds)
    kw_dist = Counter(kw_unl)
    agree = np.array([p.taxonomy_category == k for p, k in zip(preds, kw_unl, strict=True)])
    ex_rng = np.random.default_rng(seed)

    def example(i: int) -> dict[str, Any]:
        p, c = preds[i], unl[i]
        return {"sku_id": c["id"], "title": _short(c["title"], 110), "description": _short(c["description"], 110),
                "predicted_leaf": p.leaf_title, "path": " › ".join(clf_fresh.leaf_path(p.leaf_id)),
                "taxonomy_category": p.taxonomy_category, "confidence": p.confidence, "keyword_baseline": kw_unl[i],
                "top3": [f"{t} ({q:.2f})" for _, t, q in p.top3]}

    dis = np.flatnonzero(~agree)
    low = np.flatnonzero(conf_u < clf_fresh.threshold)
    unlabeled = {
        "n": len(unl), "sampling": "category_id IS NULL AND md5(sku_id) < '060…' (~2,3% карточек без категории)",
        "taxonomy_distribution": {k: round(v / len(unl), 4) for k, v in dist.most_common()},
        "keyword_taxonomy_distribution": {k: round(v / len(unl), 4) for k, v in kw_dist.most_common()},
        "agreement_with_keywords": round(float(agree.mean()), 4),
        "confident_share": round(float((conf_u >= clf_fresh.threshold).mean()), 4),
        "review_share_by_threshold": {str(t): round(float((conf_u < t).mean()), 4) for t in (0.5, 0.7, 0.9, 0.95, 0.99)},
        "leaf_confidence_quantiles": {q: round(float(np.quantile(leaf_u, float(q))), 4) for q in ("0.1", "0.25", "0.5")},
        "confidence_quantiles": {q: round(float(np.quantile(conf_u, float(q))), 4) for q in ("0.1", "0.25", "0.5", "0.75")},
        "top_leaves": [{"leaf": k, "share": round(v / len(unl), 4)}
                       for k, v in Counter(f"{p.leaf_title} [{p.leaf_id}]" for p in preds).most_common(15)],
        "examples": [example(int(i)) for i in ex_rng.choice(len(unl), size=min(n_examples, len(unl)), replace=False)],
        "disagreement_examples": [example(int(i)) for i in ex_rng.choice(dis, size=min(8, len(dis)), replace=False)]
        if len(dis) else [],
        "review_examples": [example(int(i)) for i in ex_rng.choice(low, size=min(8, len(low)), replace=False)]
        if len(low) else [],
    }
    throughput = {"cards": len(unl), "batch_cards_per_s": round(len(unl) / batch_s), "single_ms_p50":
                  round(float(np.median(single)), 3), "single_ms_p95": round(float(np.quantile(single, 0.95)), 3),
                  "keyword_baseline_cards_per_s": round(len(unl) / kw_s), "process": "1 процесс, CPU, холодный кэш"}
    lap("unlabeled", t0)
    echo(f"без категории ({len(unl)}): уверенных (≥ {clf_fresh.threshold}) {unlabeled['confident_share']:.1%}, "
         f"согласие с ключевыми словами {unlabeled['agreement_with_keywords']:.1%}; скорость {throughput['batch_cards_per_s']} "
         f"карт/с пакетом, {throughput['single_ms_p50']} мс на карточку")

    # 7. metrics.json ------------------------------------------------------------------------------------
    tax_path = Path(os.getenv("TAXONOMY_PATH") or ROOT / "config" / "taxonomy.yaml")
    timings["total"] = round(time.perf_counter() - t_all, 1)
    try:  # пик памяти (Linux: КБ); на Windows модуля resource нет
        import resource

        timings["peak_rss_gb"] = round(max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                           resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 2**20, 2)
    except ImportError:
        pass
    metrics = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "format": FORMAT,
        "config": {"max_classes": max_classes, "alpha": row["alpha"], "search": search, "epochs": epochs,
                   "keep_root": keep_root, "keep_leaf": keep_leaf, "seed": seed, "n_jobs": n_jobs,
                   "target_precision": target_precision, "vocab_docs": vocab_docs, "ctx_chars": CTX_CHARS,
                   "char_ngrams": list(CHAR_NGRAMS), "block_weights": fz.weights, "model": "SGDClassifier(hinge) OvR × 2 этапа"},
        "data": {"labeled": len(df), "dropped_unknown_category_id": dropped, "train": len(tr), "dev": len(dv),
                 "holdout": len(ho), "by_source_holdout": ho["category_source"].value_counts().to_dict()},
        "taxonomy": {"path": str(tax_path.relative_to(ROOT)) if tax_path.is_relative_to(ROOT) else str(tax_path),
                     "sha256_12": _sha(tax_path), "categories": clf.taxonomy_names, "mapped_leaves": len(tax_map),
                     "unmapped_leaves": clf.unmapped_leaves},
        "label_space": {"roots": len(ls.roots), "classes": len(ls.classes), "leaf_classes": n_leaf_classes,
                        "other_classes": len(ls.classes) - n_leaf_classes,
                        "train_share_in_leaf_classes": round(sum(c["n_train"] for c in ls.classes
                                                                 if not c["id"].startswith(OTHER)) / len(tr), 4),
                        "leaves_in_train": len(counts)},
        "selection_dev": selection,
        "calibration": dev["temps"],
        "threshold": {"value": threshold, "rule": "max(порог точности ≥ target_precision на dev, min_confidence)",
                      "target_precision": target_precision, "precision_threshold": precision_threshold,
                      "min_confidence": min_confidence, "dev_coverage": round(float(cov_dv.mean()), 4),
                      "dev_accuracy": round(float(correct_dv[cov_dv].mean()), 4)},
        "holdout": holdout, "unlabeled": unlabeled, "throughput": throughput,
        "artifact": {"dir": str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out), "files_mb": sizes,
                     "total_mb": round(sum(sizes.values()), 2), "weights_nnz": int(len(arr["w_q"])),
                     "features": clf.featurizer.n_features, "format": "npz (int8 веса, CSR) + JSON, без pickle"},
        "timings_s": timings,
    }
    (out / METRICS_FILE).write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    echo(f"готово за {timings['total']} с -> {out}/{{{MODEL_FILE}, {VOCAB_FILE}, {META_FILE}, {METRICS_FILE}}}")
    return metrics
