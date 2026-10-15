"""Правила и словари: дешёвый, быстрый и объяснимый первый слой.

Каждый кандидат несёт `span` — смещения цитаты в Card.full_text() (подсветка в UI) — и `raw` = сама цитата.
Поиск идёт по одной строке full_text() в нижнем регистре той же длины (taxonomy.lower_keep): название [0, te),
описание [ds, de). `score` — «сырая» уверенность правила; в вероятность верности её переводит калибровка
(confidence.py, config/calibration.json), поэтому важно не абсолютное значение, а порядок: явная подпись
(«Цвет: …», «общий объём …») > название > описание; модель с цифрами > описательные слова.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from ..normalize import (
    MODEL_NOISE,
    NUM,
    UNIT_PATTERNS,
    find_cpu,
    fmt_number,
    model_cut_pos,
    normalize_cpu,
    to_float,
)
from ..schema import Candidate, Card
from ..taxonomy import (
    AttrSpec,
    BrandDictionary,
    CategorySpec,
    Dictionary,
    Mention,
    Taxonomy,
    compact_key,
    lower_keep,
    pick_mention,
)

# Правдоподобные диапазоны величины (по виду) и уточнения для пары (категория, атрибут).
RANGES: dict[str, tuple[float, float]] = {
    "memory": (2, 2048),
    "ram": (1, 256),
    "storage_drive": (16, 16384),
    "diagonal": (1.0, 120.0),
    "battery": (100, 40000),
    "load_kg": (1, 30),
    "rpm": (300, 2500),
    "length_cm": (10, 300),
    "size_mm": (20, 60),
    "volume_l": (0.1, 1500),
    "power_w": (50, 10000),
    "power_kw": (0.5, 30),
    "refresh_hz": (24, 600),
    "time_h": (0.5, 500),
    "time_min": (5, 600),
    "btu": (4000, 80000),
    "area_m2": (5, 300),
}
ATTR_RANGES: dict[tuple[str, str], tuple[float, float]] = {
    ("smartphone", "screen"): (3.5, 8.5),
    ("laptop", "screen"): (10.0, 18.5),
    ("tablet", "screen"): (6.5, 15.0),
    ("tv", "screen"): (15.0, 120.0),
    ("washing_machine", "depth"): (25.0, 80.0),
    ("washing_machine", "load"): (1.0, 20.0),
    ("fridge", "height"): (40.0, 230.0),
    ("fridge", "total_volume"): (20.0, 1000.0),
    ("kettle", "volume"): (0.3, 5.0),
    ("vacuum", "dust_volume"): (0.1, 10.0),
    ("air_fryer", "bowl_volume"): (0.5, 30.0),
    ("smartwatch", "case_size"): (20.0, 60.0),
}

GB = r"(?:гб|gb|гбайт|gbyte)"
TB = r"(?:тб|tb)"
RAM_HINT = re.compile(r"ram|озу|оперативн\w*|lpddr\d\w*|ddr\d\w*")
SSD_HINT = r"(?:ssd|hdd|emmc|накопител\w*|диск\w*|ufs|встроенн\w*|пзу|rom)"
RAM_VALUES = frozenset({1, 2, 3, 4, 6, 8, 10, 12, 16, 18, 20, 24, 32, 36, 48, 64, 96, 128})
STORAGE_VALUES = frozenset({8, 16, 32, 64, 128, 256, 512, 1000, 1024, 2000, 2048, 4096})
# «8/256 ГБ», «12+512GB», «16Гб/1Тб», «16G/1TB», «8/256» (без единицы — только из множеств типичных объёмов);
# слева граница: «RTX4060/16ГБ» -> не пара (60, 16); в списке через «/» («…/16/512/Linux») — пара
MEM_PAIR = re.compile(r"(?<![\w.,+-])(\d{1,3})\s*(gb|гб|g)?\s*[/+]\s*(\d{1,4})\s*(gb|гб|tb|тб|g|t)?(?![\w.,])")
INCH = r"(?:\"|''|”|″|дюйм\w*|inch(?:es)?|in\b)"
NOT_DIAGONAL_AFTER = re.compile(r"\s*(?::\s*\d|[kк](?![a-zа-я])|гц|hz|%|мм|mm|бит|bit|x\s*\d|х\s*\d|-?\s*ядер|core)")
LABEL_SEP = r"\s*(?::|—|–|-|=)?\s*"
GLUED_UNIT_PREFIX = re.compile(r"(?:ssd|hdd|emmc|ufs)$")  # «SSD256Gb», «eMMC64GB» — подпись слитно с числом
COLOR_PATTERN_CTX = re.compile(r"(?<![а-я])(?:узор|принт|рисун|изображен|орнамент|фон(?![а-я])|фоне|цвет(?:ы|ами|ов|очн)|"
                               r"картин|раскрас)|набор\s+(?:из\s+)?\d|\d+\s*цвет")


@dataclass
class Ctx:
    """Карточка как одна строка full_text(): low — нижний регистр той же длины; [0, te) — название, [ds, de) — описание."""

    card: Card
    category: str
    text: str
    low: str
    te: int
    ds: int
    de: int

    @classmethod
    def of(cls, card: Card, category: str) -> Ctx:
        text = card.full_text()
        te = len(card.title) if card.title else 0
        ds = te + 1 if card.title else 0
        de = ds + len(card.description) if card.description else te
        return cls(card, category, text, lower_keep(text), te, ds, de)


def _cand(ctx: Ctx, attr: str, value, start: int, end: int, score: float, raw: str | None = None) -> Candidate:
    while start < end and ctx.text[start].isspace():
        start += 1
    while end > start and ctx.text[end - 1].isspace():
        end -= 1
    return Candidate(attribute=attr, value=value, raw=(raw or ctx.text[start:end]).strip(), source="rules",
                     score=round(min(0.99, score), 3), span=(start, end))


def _range(kind: str, category: str, attr: str) -> tuple[float, float]:
    return ATTR_RANGES.get((category, attr), RANGES.get(kind, (float("-inf"), float("inf"))))


def _in_range(x: float, kind: str, category: str, attr: str = "") -> bool:
    lo, hi = _range(kind, category, attr)
    return lo <= x <= hi


@lru_cache(maxsize=512)
def _stem_re_cached(words: tuple[str, ...]) -> str:
    return _stem_re(list(words))


def _stem_re(words: list[str]) -> str:
    """Подсказки-названия атрибута как основы слов: «площадь помещения» -> «площад\\w*\\s+помещени\\w*»
    (ловит «на площади помещения», «производительностью по холоду»)."""
    out = []
    for w in sorted({w.lower().replace("ё", "е") for w in words if len(w) > 3}, key=len, reverse=True):
        parts = []
        for p in w.split():
            if len(p) > 5 and re.search(r"[а-я]$", p):
                p = re.escape(re.sub(r"(?:ость|ости|ый|ий|ой|ая|яя|ое|ее|ые|ие|[аяоеыиуюйь])$", "", p)) + r"\w*"
            else:
                p = re.escape(p)
            parts.append(p)
        out.append(r"\s+".join(parts))
    return "|".join(out)


# ------------------------------------------------------------------ бренд и модель ---------------------------------
def brand_mentions(ctx: Ctx, brands: BrandDictionary) -> tuple[Mention | None, list[Mention]]:
    ms = brands.mentions(ctx.low, 0, ctx.te)
    return pick_mention(ms), ms


PLATFORM_NEXT = frozenset({"tv", "тв", "алиса", "алисой", "алисы", "os", "ос", "play", "assistant", "ассистент"})
PLATFORM_PREV = frozenset({"с", "c", "со", "на", "with", "on", "под", "ос", "os", "платформе", "платформой",
                           "платформа", "системой", "система", "управлением", "поддержкой", "голосовым"})


def _is_platform(ctx: Ctx, m: Mention) -> bool:
    """«Телевизор Haier … с Google TV», «на платформе Яндекс ТВ», «с Алисой»: упоминание бренда платформы/сервиса,
    а не производителя товара."""
    after = re.match(r"[\s\-]*([^\W_]+)", ctx.low[m.end:m.end + 20])
    before = re.search(r"([^\W_]+)[\s\-]*$", ctx.low[max(0, m.start - 25):m.start])
    nxt = after.group(1) if after else ""
    prv = before.group(1) if before else ""
    return nxt in PLATFORM_NEXT and (prv in PLATFORM_PREV or m.index > 0 and prv not in ("телевизор", "тв"))


def extract_brand(ctx: Ctx, attr: str, tax: Taxonomy) -> tuple[list[Candidate], Mention | None]:
    """Бренд = первое упоминание бренда каталога в названии вне контекста совместимости («для iPhone») и платформы
    («с Google TV»). Уверенность — точность бренда на train (`_prior` справочника)."""
    brands = tax.dictionaries.get("brands")
    if not isinstance(brands, BrandDictionary):
        return [], None
    ms = [m for m in brands.mentions(ctx.low, 0, ctx.te) if not _is_platform(ctx, m)]
    pick = pick_mention(ms)
    if pick is not None:
        prior = brands.prior.get(pick.canon, 0.99)
        score = 0.5 + 0.47 * prior - (0.05 if ms and ms[0] is not pick else 0.0)
        return [_cand(ctx, attr, pick.canon, pick.start, pick.end, score)], pick
    if ms:  # только бренды устройства («Чехол для iPhone 15»): бренд товара не назван — на проверку
        m = ms[0]
        return [_cand(ctx, attr, m.canon, m.start, m.end, 0.35)], None
    # в описании: подпись «Бренд: X» / «Производитель — X» надёжна, просто первое упоминание — нет
    body = [m for m in brands.mentions(ctx.low, ctx.ds, ctx.de) if not m.compat and not _is_platform(ctx, m)]
    for m in body:
        if re.search(r"(?:бренд|торгов\w*\s+марк\w*|марк[аи]|производител\w*|brand|manufacturer)" + LABEL_SEP + r"$",
                     ctx.low[max(ctx.ds, m.start - 30):m.start]):
            return [_cand(ctx, attr, m.canon, m.start, m.end, 0.85)], None
    if body:
        m = body[0]
        prior = brands.prior.get(m.canon, 0.99)
        return [_cand(ctx, attr, m.canon, m.start, m.end, 0.45 + 0.2 * prior)], None
    return [], None


# слова категории в начале остатка названия — как normalize.CATEGORY_WORDS (эталон), с падежными окончаниями
CATEGORY_LEAD = re.compile(
    r"(?:смартфон\w*|телефон\w*|ноутбук\w*|планшет\w*|телевизор\w*|наушники|гарнитур\w*|смарт-часы|умные\s+часы|часы|"
    r"пылесос\w*|робот-пылесос\w*|чайник\w*|электрочайник\w*|электрическ\w*|холодильник\w*|стиральн\w*\s+машин\w*|"
    r"сплит-систем\w*|кондиционер\w*|аэрогрил\w*|игров\w*|беспроводн\w*|проводн\w*|вертикальн\w*|мобильн\w*|"
    r"true\s+wireless)\s+")


# «число + единица» — характеристика, а не код модели: «20W», «12 В», «2 Ач», «5000 мАч», «1,7 л»
SPEC_TOKEN = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)?\s*(?:в|v|ач|ah|а·ч|мач|mah|вт|w|квт|kw|а|a|л|мл|кг|г|мм|см|м|гб|gb|"
                        r"тб|tb|мб|mb|гц|hz|\"|''|”|″|дюйм\w*|шт)(?![a-zа-яё0-9])", re.I)
# слово совместимости перед брендом устройства: после него — модель чужого устройства («Baseus 20W для iPhone 15»);
# «для» без бренда после него — часть названия модели в каталоге («365home Штора для ванной 180х180см»)
COMPAT_CUT = re.compile(r"(?<![\w-])(?:для|for|под|к|ко|совместим\w*|compatible|with|с|c|со)(?![\w-])")


def compat_cut_pos(ctx: Ctx, tax: Taxonomy, start: int) -> int:
    """Где в названии (от start) начинается «для/for/совместим с + бренд устройства»: модель товара — до этого места."""
    brands = tax.dictionaries.get("brands")
    if not isinstance(brands, BrandDictionary):
        return ctx.te
    for m in brands.mentions(ctx.low, start, ctx.te):
        if m.compat:
            words = list(COMPAT_CUT.finditer(ctx.low, start, m.start))
            if words:
                w = words[-1]
                if w.group(0) in ("с", "c", "со", "with") and len(words) > 1:
                    w = words[-2]  # «совместим с Samsung» — режем перед «совместим»
                return w.start()
    return ctx.te


def _model_shape_score(value: str) -> float:
    """Похоже ли на модель: буквы+цифры > латиница > кириллица с цифрами > описательные слова. Характеристики
    вместо модели («20W», «12 В 2 Ач») — низкая оценка: такое значение уходит на проверку."""
    specs = SPEC_TOKEN.findall(value)
    if specs and not re.search(r"[^\W_]", SPEC_TOKEN.sub(" ", value)):
        return 0.3  # только характеристики, кода модели нет
    has_d = bool(re.search(r"\d", value))
    has_lat = bool(re.search(r"[A-Za-z]", value))
    has_cyr = bool(re.search(r"[А-Яа-яЁё]", value))
    words = value.split()
    if has_d and has_lat:
        s = 0.9
    elif has_lat and not has_cyr:
        s = 0.78
    elif has_d and not has_cyr:
        s = 0.72
    elif has_d:
        s = 0.68
    else:
        s = 0.4
    if re.match(r"[а-яё]", value):  # «для книг и учебников», «школьный …» — описание, не модель
        s = min(s, 0.3)
    if len(value) > 40 or len(words) > 5:
        s -= 0.15
    if len(specs) >= 2:  # «GSR 120-LI 12 В 2 Ач» — в «модель» попал хвост характеристик
        s = min(s, 0.5)
    return s


def _model_from(ctx: Ctx, start: int, end: int, colors: Dictionary | None) -> tuple[int, int, str] | None:
    """Модель в ctx.text[start:end]: без слова категории в начале, до хвоста характеристик (память, диагональ,
    цвет, запятая, скобка, « - »). Та же логика, что normalize.model_core (эталон), но со смещениями."""
    pos = start
    while True:  # пробелы/пунктуация и слова категории в начале («Смартфон Samsung Смартфон Galaxy …»)
        m = re.match(r"[\s\-–—:,.|/]+", ctx.low[pos:end])
        if m:
            pos += m.end()
            continue
        m = CATEGORY_LEAD.match(ctx.low, pos, end)
        if m:
            pos = m.end()
            continue
        break
    if pos >= end:
        return None
    cut = model_cut_pos(ctx.low, pos, end, keep_power=True)
    if colors is not None:
        for _, _, s, _ in colors.find_all_raw(ctx.low, pos, cut):
            if s > pos:  # цвет в самом начале — часть названия («Black Shark»)
                cut = min(cut, s)
                break
    e = cut
    while e > pos and ctx.text[e - 1] in " \t\n-–—,.;:/(":
        e -= 1
    if e <= pos:
        return None
    value = re.sub(r"\s+", " ", MODEL_NOISE.sub(" ", ctx.text[pos:e])).strip(" -–—,.;:/")
    return (pos, e, value) if value else None


ARTICLE = re.compile(r"(?<![\w.,/-])\d{5,7}(?![\w.,/-])")


def find_article(ctx: Ctx) -> tuple[int, int] | None:
    """Артикул в названии — отдельное 5–7-значное число, последнее (Brauberg «… синяя, 880015»)."""
    hits = [m for m in ARTICLE.finditer(ctx.low, 0, ctx.te)
            if not re.match(r"\s*(?:мм|см|м|мл|л|г|кг|шт|вт|мач|руб|₽|%)", ctx.low[m.end():m.end() + 4])]
    return (hits[-1].start(), hits[-1].end()) if hits else None


def extract_model(ctx: Ctx, attr: str, tax: Taxonomy, pick: Mention | None,
                  styles: dict | None = None) -> list[Candidate]:
    """Модель = кусок названия после бренда до хвоста характеристик (на train совпадает с моделью каталога
    в 93% карточек хвоста и 98% категорий таксономии). Если бренд на train пишет модель артикулом
    (config/model_styles.json) — артикул; уверенность — точность способа для бренда. Без бренда — первый «код»."""
    colors = tax.dictionaries.get("colors")
    styles = getattr(tax, "model_styles", {}) if styles is None else styles
    style = styles.get(pick.canon) if pick is not None else None
    if style and style["article"] > style["after"]:
        art = find_article(ctx)
        if art is not None:
            s, e = art
            return [_cand(ctx, attr, ctx.text[s:e], s, e, 0.3 + 0.67 * style["article"])]
    if pick is not None:
        start = pick.end
        brands = tax.dictionaries.get("brands")
        if isinstance(brands, BrandDictionary):  # «Citilux Citilux Старлайт …», «Tesler Чайник TESLER KT-1760»
            same = compact_key(ctx.text[pick.start:pick.end])
            for m in brands.mentions(ctx.low, pick.end, ctx.te):
                if compact_key(ctx.text[m.start:m.end]) == same:
                    start = m.end
        res = _model_from(ctx, start, compat_cut_pos(ctx, tax, start), colors)
        if res is None:
            return []
        s, e, value = res
        score = _model_shape_score(value)
        if style:  # точность «после бренда» именно у этого бренда (train) важнее общей формы строки
            score = min(score, 0.3 + 0.67 * style["after"]) if style["after"] < 0.5 else 0.3 + 0.67 * style["after"]
        return [_cand(ctx, attr, value, s, e, score, raw=ctx.text[s:e])]
    # бренда нет: первый токен «буквы+цифры» (TWK8611P, KT-1809) — код модели, но только ДО «для + бренд устройства»:
    # в «Чехол для Samsung Galaxy A55» A55 — модель телефона, а не чехла
    limit = compat_cut_pos(ctx, tax, 0)
    for m in re.finditer(r"(?<![\w-])(?=[\w-]*\d)(?=[\w-]*[a-zа-я])[a-zа-я0-9][\w-]{2,}", ctx.low[:limit]):
        tok = m.group(0)
        if re.fullmatch(r"\d+(?:[.,]\d+)?(?:гб|gb|тб|tb|мм|см|л|кг|вт|w|мач|mah|гц|hz|шт|g|k|к|x\d+|х\d+)?", tok):
            continue
        res = _model_from(ctx, m.start(), limit, colors)
        if res is None:
            return []
        s, e, value = res
        return [_cand(ctx, attr, value, s, e, _model_shape_score(value) - 0.25, raw=ctx.text[s:e])]
    return []


# ------------------------------------------------------------------ справочники (enum) -----------------------------
@lru_cache(maxsize=512)
def _label_re_for(names: tuple[str, ...]) -> re.Pattern:
    # подпись — с начала строки/фрагмента: перед ней начало текста, перевод строки, «;», «•», «|», «.» или пробел
    # (lookbehind, а не захват: поиск начинается с начала описания, и «Цвет: …» в самом начале тоже подпись)
    return re.compile(r"(?<![^\n;•|.\s])(?:" + _stem_re(list(names)) + r")" + LABEL_SEP)


def _label_re(spec: AttrSpec) -> re.Pattern:
    return _label_re_for((spec.title, *spec.aliases))


@dataclass
class EnumHits:
    """Что нашёл справочник для атрибута: где (label — подпись «Цвет: …», title, body), какие значения по порядку
    и первое упоминание каждого. Ключи выученного соответствия (config/enum_map.json) строятся отсюда."""

    loc: str
    hits: list[tuple[str, str, int, int]]  # (canon, raw, start, end) — по одному на значение, по порядку
    many: bool = False

    @property
    def canons(self) -> list[str]:
        return [h[0] for h in self.hits]

    def keys(self) -> list[str]:
        """Ключи от частного к общему: набор значений в этой позиции, первый синоним, первое значение.
        Если найдено несколько разных значений (перечисление расцветок), ключ — только сам набор: статистика
        одиночных упоминаний к перечислению не относится."""
        combo = "+".join(sorted(set(self.canons)))
        if self.many:
            return [f"{self.loc}|{combo}"]
        return [f"{self.loc}|{combo}", f"{self.loc}|~{compact_key(self.hits[0][1])}", f"{self.loc}|={self.hits[0][0]}"]


def enum_hits(ctx: Ctx, spec: AttrSpec, d: Dictionary, skip: list[tuple[int, int]]) -> EnumHits | None:
    """Совпадения справочника в порядке приоритета: подпись в описании/полях продавца («Цвет: чёрный») > название >
    описание. Совпадения внутри бренда («Orange Line», «Bronze Gym») и узоры («Цветочные узоры: розовые») — не значения."""

    def ok(h) -> bool:
        return not any(h[2] < b and a < h[3] for a, b in skip)

    def distinct(hits) -> list[tuple[str, str, int, int]]:
        seen: dict[str, tuple[str, str, int, int]] = {}
        for h in hits:
            seen.setdefault(h[0], h)
        return list(seen.values())

    is_color = spec.dictionary == "colors"
    for lm in _label_re(spec).finditer(ctx.low, ctx.ds, len(ctx.low)):
        seg_end = min(len(ctx.low), lm.end() + 40)
        nl = ctx.low.find("\n", lm.end(), seg_end)
        seg_end = nl if nl >= 0 else seg_end
        hits = [h for h in d.find_all_raw(ctx.low, lm.end(), seg_end) if ok(h)]
        if hits and hits[0][2] - lm.end() <= 3:
            return EnumHits("label", distinct(hits)[:3])
    for loc, start, end in (("title", 0, ctx.te), ("body", ctx.ds, ctx.de)):
        if end <= start:
            continue
        hits = [h for h in d.find_all_raw(ctx.low, start, end) if ok(h)]
        if is_color:
            hits = [h for h in hits if not COLOR_PATTERN_CTX.search(ctx.low[max(start, h[2] - 40):h[2]])]
        if hits:
            ds = distinct(hits)
            return EnumHits(loc, ds[:4], many=len(ds) > (2 if is_color and loc == "title" else 1))
    return None


def extract_enum(ctx: Ctx, attr: str, spec: AttrSpec, d: Dictionary, skip: list[tuple[int, int]],
                 emap: dict | None = None) -> list[Candidate]:
    """Значение справочника. Без выученного соответствия: подпись 0.9, название 0.9, одно значение в описании 0.75,
    перечисление разных значений — каждое 0.45 (конфликт -> проверка). С соответствием emap (выучено на train,
    attrx build-dicts): набор найденных значений в этой позиции -> значение каталога и его точность
    («ручной (handstick)» -> «вертикальный», «No Frost + капельная в описании» -> «No Frost»)."""
    eh = enum_hits(ctx, spec, d, skip)
    if eh is None:
        return []
    if emap:
        for key in eh.keys():
            if key in emap:
                value, p, _n = emap[key]
                if not value:
                    return []  # на train такой набор чаще ошибочен, чем верен — не предлагаем
                h = next((h for h in eh.hits if h[0] == value), eh.hits[0])
                return [_cand(ctx, attr, value, h[2], h[3], min(0.97, 0.3 + 0.67 * float(p)))]
    if eh.loc in ("label", "title"):
        canon, _, s, e = eh.hits[0]
        return [_cand(ctx, attr, canon, s, e, 0.6 if eh.many else 0.9)]
    if not eh.many:
        canon, _, s, e = eh.hits[0]
        return [_cand(ctx, attr, canon, s, e, 0.75)]
    return [_cand(ctx, attr, canon, s, e, 0.45) for canon, _, s, e in eh.hits[:3]]


# ------------------------------------------------------------------ числа ------------------------------------------
def _num_unit(ctx: Ctx, kind: str, start: int, end: int) -> list[tuple[float, int, int]]:
    """(значение в канонической единице, начало, конец) для «число + единица» вида kind в low[start:end]."""
    out = []
    for unit_re, mult in UNIT_PATTERNS.get(kind, []):  # «1,3 л», «1,3-литровым»
        for m in re.finditer(NUM + r"\s*-?\s*" + unit_re + r"(?![a-zа-я])", ctx.low[:end]):
            if m.start() < start:
                continue
            if m.start() > 0 and (ctx.low[m.start() - 1].isalnum() or ctx.low[m.start() - 1] in ".,") \
                    and not GLUED_UNIT_PREFIX.search(ctx.low, max(0, m.start() - 4), m.start()):
                continue  # «RTX4060», «1.5» — часть другого числа или кода; «SSD256Gb» — накопитель, берём
            out.append((fmt_number(to_float(m.group(1)) * mult), m.start(), m.end()))
    return sorted(out, key=lambda x: x[1])


def _memory_pairs(ctx: Ctx, start: int, end: int, category: str) -> list[tuple[float, float, int, int, bool]]:
    """(RAM, накопитель, начало, конец, с_единицей) для пар «8/256 ГБ», «16Гб/1Тб», «12+512»."""
    out = []
    for m in MEM_PAIR.finditer(ctx.low, start, end):
        a, u1, b, u2 = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
        if u2 in ("tb", "тб", "t"):
            b *= 1024
        elif u2 is None and u1 is None:  # без единиц — только типичные объёмы и только у гаджетов
            if category not in ("smartphone", "tablet", "laptop") or b not in STORAGE_VALUES:
                continue
        elif u2 == "g" and u1 not in (None, "g"):
            continue
        if a not in RAM_VALUES or b < 8 or b <= a:
            continue
        out.append((float(a), float(b), m.start(), m.end(), bool(u1 or u2)))
    return out


def extract_memory(ctx: Ctx, attr: str, spec: AttrSpec, category: str) -> list[Candidate]:
    """ОЗУ / встроенная память / накопитель: пара «8/256» (название приоритетнее), подписи RAM/SSD, одиночный объём."""
    kind = spec.kind or ""
    out: list[Candidate] = []
    for start, end, bonus in ((0, ctx.te, 0.08), (ctx.ds, ctx.de, 0.0)):
        if end <= start:
            continue
        pairs = _memory_pairs(ctx, start, end, category)
        if pairs:
            ram, storage, s, e, unit = pairs[0]
            val = ram if kind == "ram" else storage
            if _in_range(val, kind, category, attr):
                out.append(_cand(ctx, attr, fmt_number(val), s, e, (0.86 if unit else 0.8) + bonus))
            if bonus:  # пара в названии — главный источник, описание не смотрим
                return out
            continue
        hint = RAM_HINT.pattern if kind == "ram" else SSD_HINT
        for unit_re, mult in ((GB, 1), (TB, 1024)):
            # «16 ГБ RAM» / «RAM 16 ГБ» / «SSD 512 ГБ»; «16 ГБ SSD 512 ГБ»: подсказка относится к следующему числу
            for pat in (NUM + r"\s*" + unit_re + r"\s*(?:[\w-]+\s){0,1}(?:" + hint + r")(?!\s*:?\s*\d)",
                        r"(?:" + hint + r")[\s:]*(?:[\w-]+\s){0,1}" + NUM + r"\s*" + unit_re + r"(?![a-zа-я])"):
                for m in re.finditer(pat, ctx.low[:end]):
                    if m.start() < start:
                        continue
                    val = to_float(m.group(1)) * mult
                    if kind == "ram" and val not in RAM_VALUES:
                        continue
                    if _in_range(val, kind, category, attr):
                        out.append(_cand(ctx, attr, fmt_number(val), m.start(), m.end(), 0.85 + bonus))
        if out:
            continue
        vals = [(v, s, e) for v, s, e in _num_unit(ctx, "memory", start, end)]
        if kind in {"ram", "storage_drive"}:
            # ноутбук: «8 ГБ 256 ГБ» без подписей — меньшее (<=64) это ОЗУ, большее (>=128) накопитель
            small = [x for x in vals if x[0] <= 64 and not re.search(r"видео|gddr|rtx|gtx|radeon|graphics|arc", ctx.low[max(0, x[1] - 25):x[1]])]
            big = [x for x in vals if x[0] >= 128]
            pick = small if kind == "ram" else big
            if small and big and len({v for v, _, _ in pick}) == 1:
                v, s, e = pick[0]
                out.append(_cand(ctx, attr, fmt_number(v), s, e, 0.7 + bonus))
        elif kind == "memory":
            # смартфон/планшет: одиночный объём «256 ГБ» — скорее ПЗУ, если рядом нет подписи RAM
            for v, s, e in vals:
                ctx_txt = ctx.low[max(0, s - 25): e + 25]
                if RAM_HINT.search(ctx_txt) or not _in_range(v, kind, category, attr) or v < 16:
                    continue
                out.append(_cand(ctx, attr, fmt_number(v), s, e, 0.7 + bonus))
    return out


TV_SIZES = frozenset({19, 20, 22, 24, 28, 32, 39, 40, 42, 43, 45, 48, 49, 50, 55, 58, 60, 65, 70, 75, 77, 83, 85, 86,
                      98, 100, 115})


def extract_diagonal(ctx: Ctx, attr: str, spec: AttrSpec, category: str) -> list[Candidate]:
    """Диагональ: число с дюймами/см; число после «диагональ/экран» (кроме «16:9», «4K», «120 Гц», «2.5K»);
    у телевизоров — размер в начале кода модели («65A6BG», «QE55…», «Haier 65 Smart TV») с подтверждением."""
    kind = "diagonal"
    out: list[Candidate] = []
    for start, end, bonus in ((0, ctx.te, 0.08), (ctx.ds, ctx.de, 0.0)):
        if end <= start:
            continue
        found = False
        for m in re.finditer(NUM + r"\s*-?\s*(" + INCH + r"|см|cm)", ctx.low[:end]):  # «6,7-дюймовый»
            if m.start() < start or (m.start() > 0 and (ctx.low[m.start() - 1].isalnum() or ctx.low[m.start() - 1] in ".,")):
                continue
            unit = m.group(2)
            val = to_float(m.group(1)) / (2.54 if unit in ("см", "cm") else 1.0)
            if unit in ("см", "cm") and not re.search(r"диагонал|экран|дисплей", ctx.low[max(0, m.start() - 40):m.start()]):
                continue  # «45 см» — габарит, а не диагональ
            if unit in ("см", "cm"):  # «164 см» -> 64.6 -> стандартный размер 65", если он в пределах 1"
                val = round(val, 1)
                if category == "tv":
                    near = min(TV_SIZES, key=lambda x: abs(x - val))
                    val = float(near) if abs(near - val) <= 1.0 else val
            val = fmt_number(val)
            if _in_range(val, kind, category, attr):
                integer_nominal = category == "laptop" and val in (13, 15, 17) and unit not in ("см", "cm")
                found = True
                out.append(_cand(ctx, attr, val, m.start(), m.end(), (0.62 if integer_nominal else 0.86) + bonus))
        if not found:
            for m in re.finditer(r"(?:диагонал\w*|экран\w*|дисплей\w*)[^\d\n]{0,15}?" + NUM, ctx.low[:end]):
                if m.start() < start or NOT_DIAGONAL_AFTER.match(ctx.low, m.end()):
                    continue
                val = to_float(m.group(1))
                if _in_range(val, kind, category, attr):
                    out.append(_cand(ctx, attr, fmt_number(val), m.start(1), m.end(1), 0.72 + bonus))
    if category == "tv" and not any(c.span and c.span[1] <= ctx.te for c in out):
        out += _tv_size_from_model(ctx, attr, out)
    return out


def _tv_size_from_model(ctx: Ctx, attr: str, found: list[Candidate]) -> list[Candidate]:
    """«Телевизор Hisense 75A6BG» -> 75: стандартный размер в начале кода модели. Уверенно — только если размер
    подтверждён описанием (то же число в дюймах/см); иначе — низкая уверенность (на проверку)."""
    title = ctx.low[:ctx.te]
    for m in re.finditer(r"(?<![\w.,])(?:[a-z]{1,4}-?(?:led|oled|qled)?-?)?(\d{2,3})(?=[a-z]{1,6}\d|\s+(?:smart|led|oled|"
                         r"qled|mini|uhd|fhd|hd|4k|ultra|android|google)|[a-z]{2,}(?![\w]*\d)|[a-z]\d)", title):
        size = int(m.group(1))
        if size not in TV_SIZES:
            continue
        corroborated = any(abs(float(c.value) - size) <= 0.6 for c in found)
        s, e = m.start(1), m.end(1)
        return [_cand(ctx, attr, float(size), s, e, 0.8 if corroborated else 0.6)]
    return []


def extract_battery(ctx: Ctx, attr: str, spec: AttrSpec, category: str) -> list[Candidate]:
    out = []
    for start, end, bonus in ((0, ctx.te, 0.08), (ctx.ds, ctx.de, 0.0)):
        for v, s, e in _num_unit(ctx, "battery", start, end):
            if _in_range(v, "battery", category, attr):
                out.append(_cand(ctx, attr, v, s, e, 0.86 + bonus))
    return out


DIMS = re.compile(r"(?<![\w.,])(\d{2,4}(?:[.,]\d{1,2})?)\s*(?:см|мм|cm|mm)?\s*[xх×*]\s*(\d{2,4}(?:[.,]\d{1,2})?)\s*"
                  r"(?:см|мм|cm|mm)?\s*[xх×*]\s*(\d{2,4}(?:[.,]\d{1,2})?)\s*(см|мм|cm|mm|м\b)?")
DIM_ORDER = re.compile(r"\(?\s*([вшгhwd])\s*[xх×*/]\s*([вшгhwd])\s*[xх×*/]\s*([вшгhwd])\s*\)?")
DIM_LETTER = {"в": "h", "h": "h", "ш": "w", "w": "w", "г": "d", "d": "d"}


def extract_dimension(ctx: Ctx, attr: str, spec: AttrSpec, category: str) -> list[Candidate]:
    """Высота/глубина из габаритов «85х59,5х64 см» (порядок ВхШхГ, если подписан иначе — по подписи)
    и из «высотой 201 см» / «глубина: 45 см»."""
    want = {"height": "h", "depth": "d", "width": "w"}.get(attr) or ("h" if "высот" in spec.title.lower() else "d")
    out: list[Candidate] = []
    lo, hi = _range("length_cm", category, attr)
    for m in DIMS.finditer(ctx.low, 0, ctx.de):
        nums = [to_float(m.group(i)) for i in (1, 2, 3)]
        unit = m.group(4) or ""
        if unit in ("мм", "mm") or (not unit and max(nums) > 300):
            nums = [x / 10 for x in nums]
        elif unit == "м":
            nums = [x * 100 for x in nums]
        lab = DIM_ORDER.search(ctx.low[max(0, m.start() - 40):m.start()])
        order = [DIM_LETTER[x] for x in lab.groups()] if lab else ["h", "w", "d"]
        if len(set(order)) != 3:
            continue
        dims = dict(zip(order, nums, strict=True))
        # без подписи: высота — наибольший габарит у холодильников и стиральных машин, иначе порядок сомнителен
        plausible = lab is not None or dims["h"] == max(nums)
        if lab is None and category == "washing_machine" and plausible:
            # ширина стиральной машины — 59–60.5 см: если ровно один из двух других габаритов такой, второй — глубина
            rest = [x for x in nums if x != dims["h"]] if len(set(nums)) == 3 else []
            wide = [x for x in rest if 59 <= x <= 60.5]
            if len(rest) == 2 and len(wide) == 1:
                dims["w"], dims["d"] = wide[0], next(x for x in rest if x != wide[0])
        val = fmt_number(dims[want])
        if lo <= val <= hi and plausible:
            out.append(_cand(ctx, attr, val, m.start(), m.end(), 0.86 if lab else 0.8))
    stem = {"h": r"высот\w*", "d": r"глубин\w*", "w": r"ширин\w*"}[want]
    for m in re.finditer(r"(?<![\w-])" + stem + r"(?:\s+(?:всего|корпуса|прибора|устройства|модели))?" + LABEL_SEP
                         + r"(?:до\s+|около\s+|всего\s+)?" + NUM + r"\s*(см|мм|cm|mm)(?![a-zа-я])", ctx.low[:ctx.de]):
        if re.search(r"(?:при\s+открыт|с\s+открыт|ножк|подставк|упаковк|дверц)", ctx.low[m.end():m.end() + 30]):
            continue
        val = to_float(m.group(1)) / (10 if m.group(2) in ("мм", "mm") else 1)
        val = fmt_number(val)
        if lo <= val <= hi:
            out.append(_cand(ctx, attr, val, m.start(), m.end(), 0.88))
    return out


EXTRA_HINTS: dict[tuple[str, str], str] = {
    ("washing_machine", "load"): r"загрузк\w*|вмещ\w*|вместим\w*|рассчитан\w*\s+на|белья",
    ("split_system", "cooling_power"): r"холод\w*|охлажд\w*",
    ("split_system", "area"): r"площад\w*|помещени\w*|до\s*$",
    ("kettle", "volume"): r"вмещ\w*|вместим\w*|залить|кипяч\w*|объ[её]м\w*|[её]мкост\w*",
    ("vacuum", "dust_volume"): r"контейнер\w*|пылесборник\w*|мусор\w*|пыл[иь]\w*|грязн\w*\s+вод\w*",
    ("fridge", "total_volume"): r"общ\w*",
    ("vacuum", "battery_time"): r"работ\w*|автономн\w*|заряд\w*",
    ("headphones", "battery_life"): r"работ\w*|автономн\w*|заряд\w*|прослушиван\w*",
}
NEG_BEFORE: dict[tuple[str, str], re.Pattern] = {
    ("vacuum", "dust_volume"): re.compile(r"(?:чист\w*\s+вод\w*|для\s+воды)[^\d]{0,12}$"),
    ("split_system", "cooling_power"): re.compile(r"(?:тепл\w*|обогрев\w*|потребл\w*)[^\d]{0,14}$"),
    ("washing_machine", "load"): re.compile(r"(?:вес\w*|масс\w*|сушк\w*)\s*(?:—|-|:)?\s*(?:до\s+)?$"),
    ("fridge", "total_volume"): re.compile(r"(?:холодильн\w*|морозильн\w*)(?:\s+\w+){0,2}\s*(?:—|-|:)?\s*(?:объем\w*)?\s*$"),
}
NEG_AFTER: dict[tuple[str, str], re.Pattern] = {
    ("washing_machine", "load"): re.compile(r"\s*(?:\w+\s+){0,2}(?:для\s+)?(?:сушк\w*|сухого)"),
}


def extract_quantity(ctx: Ctx, attr: str, spec: AttrSpec, category: str) -> list[Candidate]:
    """Прочие величины (кг, об/мин, л, Вт, Гц, ч, BTU, м²): число + единица. В названии — любое такое число;
    в описании — только рядом с названием атрибута («вмещает до 8 кг белья», «общий объём 361 л»), иначе
    «вес 56 кг» стиральной машины станет её загрузкой."""
    out: list[Candidate] = []
    key = (category, attr)
    hint = _stem_re_cached((*spec.aliases, spec.title))
    if key in EXTRA_HINTS:
        hint = EXTRA_HINTS[key] + ("|" + hint if hint else "")
    hint_re = re.compile(hint) if hint else None
    strong_re = re.compile(EXTRA_HINTS[key]) if key in EXTRA_HINTS else None
    for v, s, e in _num_unit(ctx, spec.kind or "", 0, ctx.de):
        if not _in_range(v, spec.kind or "", category, attr):
            continue
        before = ctx.low[max(0, s - 60):s]
        if key in NEG_BEFORE and NEG_BEFORE[key].search(before[-30:]):
            continue
        if key in NEG_AFTER and NEG_AFTER[key].match(ctx.low, e):
            continue
        if s < ctx.te:
            out.append(_cand(ctx, attr, v, s, e, 0.8))
        elif hint_re is not None and hint_re.search(before):
            strong = strong_re is not None and strong_re.search(before[-30:])
            out.append(_cand(ctx, attr, v, s, e, 0.85 if strong else 0.75))
    # несколько кандидатов: подписанные сильнее («общий объём») вытесняют прочие
    top = max((c.score for c in out), default=0)
    return [c for c in out if c.score >= top - 0.06]


def extract_bool(ctx: Ctx, attr: str, spec: AttrSpec) -> list[Candidate]:
    """Признак «да» по ключевым словам (инверторный, inverter); «не инверторный»/«без …» не считаются."""
    for start, end, score in ((0, ctx.te, 0.85), (ctx.ds, ctx.de, 0.75)):
        for k in spec.keywords:
            for m in re.finditer(re.escape(k.lower().replace("ё", "е")), ctx.low[:end]):
                if m.start() < start or re.search(r"(?:\bне\s*|\bбез\s+)$", ctx.low[max(0, m.start() - 5):m.start()]):
                    continue
                e = min(end, m.end() + 12)
                nl = ctx.low.find("\n", m.end(), e)
                return [_cand(ctx, attr, "да", m.start(), nl if nl >= 0 else e, score)]
    return []


def extract_cpu(ctx: Ctx, attr: str, brand: str | None) -> list[Candidate]:
    out = []
    for start, end, bonus in ((0, ctx.te, 0.05), (ctx.ds, ctx.de, 0.0)):
        hit = find_cpu(ctx.low, start, end)
        if hit is None:
            continue
        m, w = hit
        raw = ctx.text[m.start():m.end()]
        value = normalize_cpu(raw)
        if value.startswith("Apple M") and brand not in (None, "Apple") and "macbook" not in ctx.low[:ctx.te]:
            w -= 0.25  # «M2» у ноутбука не Apple — скорее не процессор
        out.append(_cand(ctx, attr, value, m.start(), m.end(), 0.78 + 0.12 * w + bonus))
    return out


# ------------------------------------------------------------------ название против описания -----------------------
# Название против описания в числе: значение из названия побеждает, слот — на проверку («review»). На dev варианты
# off / boost / review / drop не отличаются больше чем на 0,0002 точности (расхождение редкое); «review» выполняет
# обещание «конфликт -> проверка» и не принимает число, которому противоречит другая часть карточки.
TITLE_DESC_POLICY = "review"


def _same_num(a, b) -> bool:
    return abs(float(a) - float(b)) <= 0.01 * max(abs(float(b)), 1e-9)


def title_desc_conflicts(card: Card, cands: list[Candidate], category: CategorySpec) -> set[str]:
    """Числовые атрибуты, где правила нашли в НАЗВАНИИ одно значение, а в ОПИСАНИИ — другое («256 ГБ» в названии,
    «128 ГБ» в описании): название приоритетнее, но расхождение решает модератор."""
    te = len(card.title or "")
    out = set()
    for attr, spec in category.attributes.items():
        if spec.type != "number":
            continue
        cs = [c for c in cands if c.attribute == attr and c.source == "rules" and c.span]
        title = [c for c in cs if c.span[1] <= te]
        desc = [c for c in cs if c.span[0] > te]
        if title and any(not any(_same_num(d.value, t.value) for t in title) for d in desc):
            out.add(attr)
    return out


def _prefer_title(card: Card, out: list[Candidate], category: CategorySpec) -> list[Candidate]:
    if TITLE_DESC_POLICY == "off":
        return out
    te = len(card.title or "")
    conflicts = title_desc_conflicts(card, out, category)
    if not conflicts:
        return out
    keep = []
    for attr in conflicts:
        cs = [c for c in out if c.attribute == attr and c.span]
        title = [c for c in cs if c.span[1] <= te]
        top = max(c.score for c in cs)
        for c in title:  # значение из названия побеждает значение из описания
            c.score = round(min(0.99, max(c.score, top + 0.02)), 3)
    for c in out:
        if (TITLE_DESC_POLICY == "drop" and c.attribute in conflicts and c.span and c.span[0] > te
                and not any(_same_num(c.value, t.value) for t in out
                            if t.attribute == c.attribute and t.span and t.span[1] <= te)):
            continue
        keep.append(c)
    return keep


# ------------------------------------------------------------------ точка входа ------------------------------------
def extract(card: Card, category: CategorySpec, tax: Taxonomy, emaps: dict | None = None) -> list[Candidate]:
    """Кандидаты правил для всех атрибутов категории. emaps — {attr: выученное соответствие справочника} (см. extract_enum)."""
    ctx = Ctx.of(card, category.name)
    out: list[Candidate] = []
    pick: Mention | None = None
    brand_span: list[tuple[int, int]] = []
    if "brand" in category.attributes:
        cands, pick = extract_brand(ctx, "brand", tax)
        out += cands
        brand_span = [c.span for c in cands if c.span]
    if emaps is None:
        emaps = getattr(tax, "enum_maps", {}).get(category.name, {})
    for attr, spec in category.attributes.items():
        if attr == "brand":
            continue
        if spec.kind == "model":
            out += extract_model(ctx, attr, tax, pick)
        elif spec.type == "enum" and spec.dictionary in tax.dictionaries:
            out += extract_enum(ctx, attr, spec, tax.dictionaries[spec.dictionary], brand_span, emaps.get(attr))
        elif spec.type == "number":
            kind = spec.kind or ""
            if kind in {"memory", "ram", "storage_drive"}:
                out += extract_memory(ctx, attr, spec, category.name)
            elif kind == "diagonal":
                out += extract_diagonal(ctx, attr, spec, category.name)
            elif kind == "battery":
                out += extract_battery(ctx, attr, spec, category.name)
            elif kind == "length_cm":
                out += extract_dimension(ctx, attr, spec, category.name) or extract_quantity(ctx, attr, spec, category.name)
            elif kind in UNIT_PATTERNS:
                out += extract_quantity(ctx, attr, spec, category.name)
        elif spec.type == "bool":
            out += extract_bool(ctx, attr, spec)
        elif spec.kind == "cpu":
            out += extract_cpu(ctx, attr, pick.canon if pick else None)
    return _prefer_title(card, out, category)
