from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from .config import get_settings
from .schema import Card

YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)  # C-загрузчик в ~10 раз быстрее (300 КБ трафик-весов)


def yaml_load(text: str):
    """yaml.safe_load через C-загрузчик, если libyaml есть, иначе чистый Python."""
    return yaml.load(text, Loader=YAML_LOADER)  # noqa: S506 — SafeLoader/CSafeLoader


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace("ё", "е")).strip()


def lower_keep(s: str) -> str:
    """Нижний регистр и ё->е БЕЗ сжатия пробелов: смещения совпадают с исходной строкой (для evidence_span).

    str.lower() у редких символов меняет длину («İ» -> «i̇») — тогда понижаем посимвольно, сохраняя длину."""
    low = s.lower().replace("ё", "е")
    if len(low) == len(s):
        return low
    return "".join(ch.lower() if len(ch.lower()) == 1 else ch for ch in s).replace("ё", "е")


def compact_key(s: str) -> str:
    """Ключ без регистра, пробелов и пунктуации: «De'Longhi» ~ «delonghi», «Zigmund & Shtain» ~ «zigmundshtain»."""
    return re.sub(r"[\W_]+", "", norm(str(s)))


TYPES = frozenset({"enum", "number", "string", "bool"})


@dataclass
class AttrSpec:
    name: str
    title: str
    type: str
    unit: str | None = None
    kind: str | None = None
    dictionary: str | None = None
    aliases: list[str] = field(default_factory=list)
    traffic_weight: float = 1.0
    # откуда эталон в YML-фиде: "vendor" | "model" | имена param (без «_search») по приоритету — см. eval/gold.py
    gold: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)  # для type=bool: признаки значения «да» в тексте
    required: bool = True


@dataclass
class CategorySpec:
    name: str
    title: str
    keywords: list[str]
    attributes: dict[str, AttrSpec]
    category_ids: list[str] = field(default_factory=list)  # листовые категории YML-фида


ADJ_END = re.compile(r"(ый|ий|ой)$")
# все падежи прилагательного: «черный/черная/черное/черные», «черного/черному/черным/черном», «черных/черными»,
# «черную/черною» — длинные окончания первыми; граница слова — в шаблоне синонима
ADJ_FORMS = "(?:ыми|ими|ого|его|ому|ему|ый|ий|ой|ая|яя|ое|ее|ые|ие|ую|юю|ом|ем|ым|им|ых|их|ою|ею)"


def _inflect(syn: str) -> str:
    """«чёрный» матчит «черная/черные/черного»: продавцы пишут прилагательные в любом роде и числе.
    Пробел синонима матчит любой пробельный промежуток (в сыром тексте бывают двойные пробелы и переводы строк)."""
    return r"\s+".join(ADJ_END.sub("", re.escape(w)) + ADJ_FORMS if ADJ_END.search(w) and len(w) > 4 else re.escape(w)
                       for w in syn.split(" "))


class Dictionary:
    """Справочник допустимых значений: канон + синонимы. Защищает от выдуманных значений.

    Ключи YAML, начинающиеся с «_», — метаданные справочника (self.meta), а не значения."""

    def __init__(self, name: str, mapping: dict[str, list[str]], meta: dict | None = None):
        self.name = name
        self.meta = dict(meta or {})
        self.canon = list(mapping)
        self.lookup: dict[str, str] = {}
        for canon, syns in mapping.items():
            for s in [canon, *(syns or [])]:
                self.lookup[norm(str(s))] = canon

    def add(self, synonym: str, canon: str) -> None:
        if canon not in self.canon:
            self.canon.append(canon)
        self.lookup[norm(synonym)] = canon

    def get(self, text: str) -> str | None:
        t = norm(text)
        if t in self.lookup:
            return self.lookup[t]
        hits = self.find_all(t)
        return hits[0][0] if hits and hits[0][1] == t else None

    def _patterns(self) -> list[tuple[str, str, re.Pattern]]:
        """Скомпилированные шаблоны синонимов (длинные первыми) + префильтр-подстрока: без него каждый вызов
        перекомпилировал сотни регулярок (кэш re на 512 шаблонов не вмещает все справочники)."""
        if getattr(self, "_compiled_n", -1) != len(self.lookup):
            items = []
            for syn in sorted(self.lookup, key=len, reverse=True):
                if re.fullmatch(r"[\d.,]+", syn):  # «2» (камеры) — только точное значение поля, в тексте это шум
                    continue
                first = syn.split(" ")[0]
                pre = ADJ_END.sub("", first) if ADJ_END.search(first) and len(first) > 4 else first
                items.append((syn, pre, re.compile(r"(?<![\w-])" + _inflect(syn) + r"(?![\w-])")))
            self._compiled, self._compiled_n = items, len(self.lookup)
        return self._compiled

    def find_all(self, text: str) -> list[tuple[str, str, int, int]]:
        """Все вхождения синонимов в тексте: (canon, raw, start, end) — смещения в norm(text). Длинные приоритетнее."""
        return self.find_all_raw(norm(text))

    def find_all_raw(self, low: str, start: int = 0, end: int | None = None) -> list[tuple[str, str, int, int]]:
        """То же по уже пониженному тексту (lower_keep) без нормализации пробелов: смещения — в исходной строке,
        поиск только в low[start:end]."""
        end = len(low) if end is None else end
        t = low[start:end]
        found: list[tuple[str, str, int, int]] = []
        taken: list[tuple[int, int]] = []
        for syn, pre, pattern in self._patterns():
            if pre not in t:
                continue
            for m in pattern.finditer(low, start, end):
                if any(m.start() < e and s < m.end() for s, e in taken):
                    continue
                taken.append((m.start(), m.end()))
                found.append((self.lookup[syn], m.group(0), m.start(), m.end()))
        return sorted(found, key=lambda x: x[2])


# ------------------------------------------------------------------ бренды -----------------------------------------
TOKEN_RE = re.compile(r"[^\W_]+")
# слово перед брендом, после которого это бренд УСТРОЙСТВА, а не товара: «чехол для iPhone», «картридж для HP»
COMPAT_PREV = frozenset({"для", "for", "к", "ко", "под", "совместим", "совместимый", "совместима", "совместимо",
                         "совместимые", "совместимых", "совместимость", "compatible", "fits", "вместо", "аналог",
                         "replacement"})
COMPAT_WITH = frozenset({"с", "c", "со", "with"})  # «совместим с Samsung»


@dataclass(frozen=True)
class Mention:
    canon: str
    start: int  # смещения в тексте, переданном в mentions()
    end: int
    index: int  # номер первого токена упоминания
    last: int  # номер последнего токена
    compat: bool  # бренд в контексте совместимости («для iPhone»)


class BrandDictionary(Dictionary):
    """Справочник брендов каталога (тысячи значений): поиск по окнам токенов и хэш-таблице компактных ключей
    вместо тысяч регулярок. Бренды-омонимы обычных слов и единиц (метаданные `_word`, отобраны по train:
    «СМ» из «45 см», «Wireless» из «True Wireless», «Galaxy» из «Samsung Galaxy») в тексте не ищутся —
    только в поле «Бренд» продавца. `_prior` — точность бренда как первого упоминания в названии (train)."""

    def __init__(self, name: str, mapping: dict[str, list[str]], meta: dict | None = None):
        super().__init__(name, mapping, meta)
        self.words = {compact_key(x) for x in (self.meta.get("word") or [])}
        self.prior: dict[str, float] = {str(k): float(v) for k, v in (self.meta.get("prior") or {}).items()}
        self._keys: dict[str, str] | None = None
        self.max_tokens = 1

    def add(self, synonym: str, canon: str) -> None:
        super().add(synonym, canon)
        self._keys = None

    def keys(self) -> dict[str, str]:
        """compact(синоним) -> канон; без омонимов (_word), однобуквенных и чисто цифровых ключей."""
        if self._keys is None:
            keys: dict[str, str] = {}
            longest = 1
            for syn, canon in sorted(self.lookup.items()):
                k = compact_key(syn)
                if len(k) < 2 or k.isdigit() or compact_key(canon) in self.words:
                    continue
                keys.setdefault(k, canon)
                longest = max(longest, len(TOKEN_RE.findall(syn)))
            self._keys, self.max_tokens = keys, min(longest, 6)
        return self._keys

    def is_word(self, canon: str) -> bool:
        return compact_key(canon) in self.words

    def mentions(self, low: str, start: int = 0, end: int | None = None) -> list[Mention]:
        """Упоминания брендов в low[start:end] (low — нижний регистр, см. lower_keep): самое длинное совпадение
        окна из 1..max_tokens токенов (разделители между токенами — не длиннее 3 символов, без перевода строки)."""
        keys = self.keys()
        end = len(low) if end is None else end
        toks = [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(low, start, end)]
        out: list[Mention] = []
        i = 0
        while i < len(toks):
            hit = None
            for k in range(min(self.max_tokens, len(toks) - i), 0, -1):
                if k > 1 and any(toks[j + 1][1] - toks[j][2] > 3 or "\n" in low[toks[j][2]:toks[j + 1][1]]
                                 for j in range(i, i + k - 1)):
                    continue
                canon = keys.get("".join(t[0] for t in toks[i:i + k]))
                if canon is not None:
                    # «Xiaomi Redmi» (синоним Xiaomi): если короче — тот же бренд, берём короткое упоминание,
                    # хвост («Redmi Note 13») — линейка, то есть часть модели
                    k = next((j for j in range(1, k) if keys.get("".join(t[0] for t in toks[i:i + j])) == canon), k)
                    hit = (canon, k)
                    break
            if hit is None:
                i += 1
                continue
            canon, k = hit
            p1 = toks[i - 1][0] if i > 0 and "\n" not in low[toks[i - 1][2]:toks[i][1]] else ""
            p2 = toks[i - 2][0] if i > 1 and p1 else ""
            compat = p1 in COMPAT_PREV or (p1 in COMPAT_WITH and p2.startswith(("совмест", "compat")))
            out.append(Mention(canon, toks[i][1], toks[i + k - 1][2], i, i + k - 1, compat))
            i += k
        return out

    def find_all(self, text: str) -> list[tuple[str, str, int, int]]:
        t = norm(text)
        return [(m.canon, t[m.start:m.end], m.start, m.end) for m in self.mentions(t)]

    def find_all_raw(self, low: str, start: int = 0, end: int | None = None) -> list[tuple[str, str, int, int]]:
        return [(m.canon, low[m.start:m.end], m.start, m.end) for m in self.mentions(low, start, end)]


def pick_mention(mentions: list[Mention]) -> Mention | None:
    """Бренд товара в названии: первое упоминание вне контекста совместимости («чехол для iPhone») и не вплотную
    за таким упоминанием («для Samsung Galaxy» — Galaxy тоже про устройство). На train (187 тыс. названий)
    это vendor каталога в 99,8% случаев (attrx build-dicts)."""
    chain = -10
    for m in mentions:
        if m.compat or m.index == chain + 1:
            chain = m.last
            continue
        return m
    return None


DICTIONARY_CLASSES: dict[str, type[Dictionary]] = {"brands": BrandDictionary}


def load_dictionary(path: Path) -> Dictionary:
    data = yaml_load(path.read_text(encoding="utf-8")) or {}
    meta = {str(k)[1:]: data.pop(k) for k in list(data) if str(k).startswith("_")}
    return DICTIONARY_CLASSES.get(path.stem, Dictionary)(path.stem, data, meta)


# хук классификатора категории (cold start: у карточки нет category_id): card -> (категория таксономии, уверенность)
CategoryHook = Callable[[Card], "tuple[str, float] | None"]
_UNSET = object()


@dataclass
class CategoryDecision:
    """Категория карточки и откуда она: catalog (category_id каталога), title (название категории),
    classifier (ML по тексту, attrx.category), hook, keywords (запасной путь)."""

    name: str
    confidence: float
    source: str
    title: str | None = None  # лист каталога (классификатор) — для UI
    path: list[str] = field(default_factory=list)
    review: bool = False  # уверенность ниже порога: карточка на проверку


class Taxonomy:
    def __init__(self, path: Path):
        data = yaml_load(path.read_text(encoding="utf-8"))
        self.categories: dict[str, CategorySpec] = {}
        self.by_category_id: dict[str, str] = {}  # id категории YML -> имя категории таксономии
        for cname, c in data["categories"].items():
            attrs = {a: AttrSpec(name=a, **spec) for a, spec in c["attributes"].items()}
            ids = [str(x) for x in c.get("category_ids") or []]
            self.categories[cname] = CategorySpec(cname, c.get("title", cname), c.get("keywords", []), attrs, ids)
            for cid in ids:
                if cid in self.by_category_id:
                    raise ValueError(f"taxonomy: категория {cid} указана и в {self.by_category_id[cid]}, и в {cname}")
                self.by_category_id[cid] = cname
        if "generic" not in self.categories:
            raise ValueError("taxonomy: нужна категория generic (фолбэк для длинного хвоста)")
        self.dictionaries: dict[str, Dictionary] = {}
        ddir = path.parent / "dictionaries"
        for f in sorted(ddir.glob("*.yaml")):
            if f.stem == "learned":
                continue
            self.dictionaries[f.stem] = load_dictionary(f)
        learned = ddir / "learned.yaml"
        if learned.exists():
            for dname, pairs in (yaml_load(learned.read_text(encoding="utf-8")) or {}).items():
                for syn, canon in (pairs or {}).items():
                    self.dictionaries.setdefault(dname, Dictionary(dname, {})).add(syn, canon)
        self.category_hook: CategoryHook | None = None
        self._classifier = _UNSET  # attrx.category.CategoryClassifier, грузится при первой карточке без категории
        # выученное на train соответствие «найденные значения справочника -> значение каталога» (attrx build-dicts)
        import json

        emap = path.parent / "enum_map.json"
        self.enum_maps: dict[str, dict[str, dict[str, list]]] = {}
        if emap.exists():
            self.enum_maps = {k: v for k, v in json.loads(emap.read_text(encoding="utf-8")).items()
                              if not k.startswith("_")}
        # как бренд записывает модель в каталоге (после бренда / артикулом) и точность способа — по train
        mstyle = path.parent / "model_styles.json"
        self.model_styles: dict[str, dict] = {}
        if mstyle.exists():
            self.model_styles = {k: v for k, v in json.loads(mstyle.read_text(encoding="utf-8")).items()
                                 if not k.startswith("_")}
        self._validate()

    def _validate(self) -> None:
        """Ошибки конфигурации — сразу при загрузке, а не тихие пропуски атрибутов на проде."""
        for cname, cat in self.categories.items():
            for a, spec in cat.attributes.items():
                where = f"taxonomy: {cname}.{a}"
                if spec.type not in TYPES:
                    raise ValueError(f"{where}: неизвестный type {spec.type!r} (допустимо: {sorted(TYPES)})")
                if spec.type == "enum" and spec.dictionary not in self.dictionaries:
                    raise ValueError(f"{where}: нет справочника {spec.dictionary!r} в config/dictionaries/")
                if spec.type == "bool" and not spec.keywords:
                    raise ValueError(f"{where}: для bool нужны keywords (признаки «да» в тексте)")

    def category_of_id(self, category_id: str | int | None) -> str | None:
        return self.by_category_id.get(str(category_id)) if category_id not in (None, "") else None

    def set_category_hook(self, hook: CategoryHook | None) -> None:
        """Подключить свой классификатор категории для карточек без category_id (вместо attrx.category)."""
        self.category_hook = hook

    def classifier(self):
        """ML-классификатор категории (models/category, attrx train-category) или None: нет артефакта,
        CATEGORY_CLASSIFIER=off, нет numpy. Загружается один раз на таксономию."""
        if self._classifier is _UNSET:
            self._classifier = None
            try:
                from .category import CategoryClassifier

                self._classifier = CategoryClassifier.load(taxonomy=self)
            except Exception as e:  # noqa: BLE001 — без классификатора остаются ключевые слова
                import logging

                logging.getLogger(__name__).warning("классификатор категории не загружен: %s", e)
        return self._classifier

    def classify(self, card: Card) -> CategoryDecision:
        """1) category_id из каталога: категория таксономии или generic (лист каталога вне таксономии), 1.0;
        2) точное название категории; 3) хук / ML-классификатор по тексту (cold start: 75% карточек без категории),
        ниже его порога — лучшая гипотеза + проверка; 4) ключевые слова, если классификатора нет."""
        cid = str(card.category_id) if card.category_id not in (None, "") else ""
        if cid and self.by_category_id:
            name = self.category_of_id(cid)
            if name:
                return CategoryDecision(name, 1.0, "catalog")
            clf = self.classifier() if cid not in _catalog_categories(_processed_dir()) else None
            if clf is None or cid in getattr(clf, "class_index", {}):
                return CategoryDecision("generic", 1.0, "catalog")  # известный лист каталога вне таксономии
        if card.category:
            c = norm(card.category)
            for name, spec in self.categories.items():
                if c == name or c == norm(spec.title):
                    return CategoryDecision(name, 1.0, "title")
        if self.category_hook is not None:
            res = self.category_hook(card)
            if res is not None and res[0] in self.categories:
                return CategoryDecision(res[0], float(res[1]), "hook", review=float(res[1]) < 0.6)
        clf = self.classifier()
        if clf is not None:
            p = clf.predict(card)
            name = p.taxonomy_category if p.taxonomy_category in self.categories else "generic"
            return CategoryDecision(name, round(float(p.confidence), 3), "classifier", p.leaf_title,
                                    clf.leaf_path(p.leaf_id), bool(clf.is_review(p)))
        if card.category:  # название категории без классификатора: ключевые слова в нём
            c = norm(card.category)
            for name, spec in self.categories.items():
                if any(norm(k) in c for k in spec.keywords):
                    return CategoryDecision(name, 0.9, "title")
        title, text = norm(card.title), norm(card.full_text())
        best, score = "generic", 0.0
        for name, spec in self.categories.items():
            sc = sum(2.0 if norm(k) in title else 1.0 for k in spec.keywords if norm(k) in text)
            if sc > score:
                best, score = name, sc
        conf = min(0.95, 0.6 + 0.15 * score) if score else 0.3
        return CategoryDecision(best, conf, "keywords", review=conf < 0.6)

    def detect_category(self, card: Card) -> tuple[str, float]:
        """Совместимость: (категория таксономии, уверенность) — см. classify."""
        d = self.classify(card)
        return d.name, d.confidence


@lru_cache(maxsize=2)
def _catalog_categories(processed: str) -> dict[str, tuple[str, list[str]]]:
    """id категории YML -> (название, путь в дереве) из data/processed/categories.parquet (если данные подготовлены)."""
    p = Path(processed) / "categories.parquet"
    if not p.exists():
        return {}
    try:
        import pyarrow.parquet as pq

        rows = pq.read_table(p, columns=["id", "title", "path"]).to_pylist()
    except Exception:  # noqa: BLE001 — нет pyarrow/битый файл: путь категории просто не показываем
        return {}
    return {str(r["id"]): (r["title"], list(r["path"] or [])) for r in rows}


@lru_cache(maxsize=2)
def _traffic_weights(path: str) -> dict[str, dict[str, float]]:
    """config/traffic_weights.yaml (attrx traffic): {category_id: {vendor | model | имя param: доля кликов}}."""
    p = Path(path)
    if not p.exists():
        return {}
    raw = yaml_load(p.read_text(encoding="utf-8")) or {}
    raw = raw.get("weights", raw) if isinstance(raw, dict) else {}
    return {str(k): v for k, v in raw.items() if not str(k).startswith("_") and isinstance(v, dict)}


@lru_cache(maxsize=1)
def _root() -> Path:
    from .config import ROOT

    return ROOT


@lru_cache(maxsize=1)
def _traffic_path() -> str:
    return str(_root() / "config" / "traffic_weights.yaml")


@lru_cache(maxsize=1)
def _processed_dir() -> str:
    return str(_root() / "data" / "processed")


def category_info(card: Card, cat: CategorySpec) -> tuple[str, list[str]]:
    """Название и путь категории для UI: из дерева каталога по category_id, иначе — категория таксономии."""
    hit = _catalog_categories(_processed_dir()).get(str(card.category_id or "")) \
        if card.category_id else None
    if hit:
        return hit
    return (card.category or cat.title), [card.category or cat.title]


def attr_traffic_weight(card: Card, spec: AttrSpec) -> float:
    """Вес атрибута по поисковому спросу в категории карточки (доля кликов с этим атрибутом в запросе);
    без данных о категории — traffic_weight из таксономии."""
    if not card.category_id:
        return spec.traffic_weight
    w = _traffic_weights(_traffic_path()).get(str(card.category_id))
    vals = [float(w[g]) for g in spec.gold if w and g in w]
    return round(max(vals), 4) if vals else spec.traffic_weight


@lru_cache(maxsize=4)
def load_taxonomy(path: str | None = None) -> Taxonomy:
    return Taxonomy(Path(path) if path else get_settings().taxonomy_path)
