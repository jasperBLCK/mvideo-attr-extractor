"""Трафик-веса атрибутов из реального поиска М.Видео (query_clicks, 31 млн кликов): `attrx traffic`.

Вес атрибута в категории = доля поисковых кликов категории, в запросах которых покупатель назвал этот атрибут
(значение или само свойство): «iphone 17 pro max 256» -> модель + встроенная память; «телевизоры 55» -> диагональ;
«холодильник no frost черный» -> размораживание + цвет. Это прямой ответ на метрику кейса «качество с учётом
трафика» и приоритет очереди модерации: ошибка в атрибуте, по которому ищут, стоит дороже.

Как считаем (без ML, воспроизводимо):
1. Клики -> (категория кликнутого SKU, нормализованный запрос, бренд кликнутого SKU) -> кэш
   data/processed/queries_by_category.parquet (DuckDB, ~1 мин, ≤ 4 ГБ RAM).
2. Словарь категории из каталога (офферы + param): бренды (+ кириллические написания: ручной список и найденные
   в логах — токен запроса, ≥ 60% кликов которого приходится на один бренд и который транслитерацией похож на него:
   «самсунг» -> Samsung), токены моделей («s26», «iphone 17», «55c6k»), числа с единицами и диапазоны бакетов
   («256 ГБ», «6.7"», «3000-5000 мАч»), текстовые значения («узкая», «No Frost», «с нижней морозильной камерой»),
   слова из имён свойств-флагов («Игровой», «Алиса», «Сушка»), «двухкамерный» -> «Количество камер», цвета.
3. `Detector.mentions(query)` размечает запрос; вес = Σ кликов запросов с упоминанием / Σ кликов категории.
   «Близнецы» (бакет фильтра и точное `_search`-значение одного свойства) получают одинаковый вес.
Оценка нижняя: не ловим описательные запросы без значения («маленький холодильник») и сильные опечатки.
"""
from __future__ import annotations

import bisect
import json
import re
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# --------------------------------------------------------------------------- нормализация запроса

_CYR = re.compile(r"[а-я]")
_TOKEN = re.compile(r"[a-zа-я0-9]+(?:[.,]\d+)?")
_CYR_DIGIT = re.compile(r"(?<=[а-я])(?=\d)|(?<=\d)(?=[а-я]{3,})")  # «айфон17про» -> «айфон 17 про»; «256гб» не трогаем
_NUM = re.compile(r"^\d+(?:[.,]\d+)?$")
_LOOKALIKE = re.compile(r"(?<![а-яa-z0-9])([асмхреокт])(\d{1,3}[a-z]?)(?![\d.,])")  # «самсунг с25» -> s25
_LOOKALIKE_MAP = dict(zip("асмхреокт", "asmxpeokt", strict=True))

STOPWORDS = frozenset("с со и или для в во на по от до при из к у а не о об под над за между".split())
# слова, которые не несут значения атрибута (в именах свойств и значениях)
GENERIC = frozenset("""
поддержка наличие функция функции тип типа вид класс уровень особенности параметр параметры общий общая основной
основная дополнительный встроенный встроенная система технология количество число режим режимы показатель
другие другой другая прочие прочее есть нет да отсутствует стандартный обычный модель серия цвет цвета
""".split())
# токены моделей, которые не отличают одну модель от другой
MODEL_GENERIC = frozenset("""
pro max plus ultra mini lite air neo fe se new edition global version series smart dual sim nfc lte 5g 4g 3g 2g
gb tb ram rom wifi wi fi led uhd hd fhd qled oled tv black white gray grey silver gold blue red green pink
purple violet titanium graphite midnight starlight orange yellow beige cream mint navy rose the and for with
kit set pack combo gen generation plus+ edge prime power play star style sport active classic standard
gaming laptop notebook business home office entry mainstream baseline ddr ssd hdd win windows dos noos phone
mobile tablet wireless bluetooth true inch
""".split())
# процессоры и видеокарты в поле model ноутбуков: «ryzen 7», «i5», «rtx 4060» — это свойства, а не модель
CPU_WORDS = frozenset("""
intel core amd ryzen i3 i5 i7 i9 celeron pentium xeon athlon rtx gtx geforce radeon nvidia arc iris snapdragon
dimensity helio exynos tensor kirin unisoc mediatek qualcomm
""".split())
MODEL_STOP = MODEL_GENERIC | CPU_WORDS
# после этих слов число — версия, а не значение свойства: «windows 11», «android 14», «usb 3», «hdmi 2»
VERSION_WORDS = frozenset("windows win android ios macos usb hdmi bluetooth wifi type dolby ddr ps pci".split())
MODEL_SUFFIX = frozenset("pro max plus ultra mini lite air neo fe se e s t".split())  # «17 pro max», «15 plus»
JUNK_VENDORS = frozenset({"нет бренда", "no name", "noname", "###", "без бренда", "нет", "-", "n/a", "unknown"})

# написание в запросе -> токен(ы) каталога (транслит частых слов; «нофрост» -> «no frost»)
TOKEN_ALIASES: dict[str, tuple[str, ...]] = {
    "айфон": ("iphone",), "айфоны": ("iphone",), "айфона": ("iphone",), "ифон": ("iphone",),
    "макбук": ("macbook",), "макбуки": ("macbook",), "аймак": ("imac",), "айпад": ("ipad",), "айпады": ("ipad",),
    "аирподс": ("airpods",), "аирподсы": ("airpods",), "эйрподс": ("airpods",), "эйрподсы": ("airpods",),
    "аирпод": ("airpods",), "эирподс": ("airpods",), "галакси": ("galaxy",), "редми": ("redmi",),
    "про": ("pro",), "макс": ("max",), "ультра": ("ultra",), "плюс": ("plus",), "лайт": ("lite",),
    "эйр": ("air",), "аир": ("air",), "нео": ("neo",), "смарт": ("smart",), "тв": ("tv",), "ноу": ("no",),
    "фрост": ("frost",), "нофрост": ("no", "frost"), "nofrost": ("no", "frost"), "ноуфрост": ("no", "frost"),
    "блютуз": ("bluetooth",), "блютус": ("bluetooth",), "вайфай": ("wifi",), "4к": ("4k",), "8к": ("8k",),
    "олед": ("oled",), "тайпси": ("type", "c"), "тайп": ("type",), "сайд": ("side",), "бай": ("by",),
    "нот": ("note",), "ноут": ("note",),
}

# кириллические написания брендов (ядро; остальное находится в логах кликов — mine_brand_aliases)
BRAND_ALIASES: dict[str, tuple[str, ...]] = {
    "samsung": ("самсунг", "самсун"), "apple": ("эпл", "эппл", "апл", "аппл"),
    "xiaomi": ("сяоми", "ксиоми", "ксяоми", "ксиаоми", "шаоми"), "huawei": ("хуавей", "хуавэй", "хуавеи"),
    "honor": ("хонор",), "lg": ("лж", "элджи"), "haier": ("хайер", "хаер", "хаир"), "atlant": ("атлант",),
    "indesit": ("индезит",), "bosch": ("бош",), "beko": ("беко",), "candy": ("канди",), "gorenje": ("горенье",),
    "hisense": ("хайсенс", "хисенс"), "sony": ("сони",), "philips": ("филипс",), "tefal": ("тефаль",),
    "redmond": ("редмонд",), "dyson": ("дайсон",), "marshall": ("маршал", "маршалл"), "realme": ("реалми",),
    "poco": ("поко",), "tecno": ("техно", "текно"), "infinix": ("инфиникс",), "asus": ("асус",),
    "lenovo": ("леново",), "acer": ("асер",), "midea": ("мидея", "мидеа"), "electrolux": ("электролюкс",),
    "polaris": ("поларис",), "kitfort": ("китфорт",), "vivo": ("виво",), "oppo": ("оппо",), "google": ("гугл",),
    "braun": ("браун",), "liebherr": ("либхер",), "toshiba": ("тошиба",), "grundig": ("грюндиг", "грундиг"),
    "weissgauff": ("вайсгауф",), "hotpoint": ("хотпоинт",), "dreame": ("дриме",), "jbl": ("джибиэль",),
    "msi": ("мси",), "dell": ("делл",), "яндекс": ("yandex",), "бирюса": ("biryusa",),
    "oneplus": ("one plus", "ванплас", "ван плас"),
}

# цвета: русские корни + английские названия (в т.ч. маркетинговые: midnight, starlight, titanium)
COLOR_RE = re.compile(
    r"(?<![a-zа-я])(?:черн\w*|бел(?:ый|ая|ое|ые|ого|ой|ую|ых|ым|ыми)|сер(?:ый|ая|ое|ые|ого|ой|ую|ых|ым|ыми)"
    r"|серебр\w*|золот\w*|красн\w*|син(?:ий|яя|ее|ие|его|ей|юю|их|им)|голуб\w*|зелен\w*|желт\w*|оранж\w*"
    r"|розов\w*|фиолет\w*|сирен\w*|лилов\w*|бежев\w*|коричн\w*|графит\w*|бирюз\w*|мятн\w*|лаванд\w*|бордов\w*"
    r"|кремов\w*|титанов\w*|бронз\w*|медн(?:ый|ая|ое|ые)|перламутр\w*|black|white|silver|gold|blue|red|green"
    r"|gr[ae]y|pink|purple|violet|beige|yellow|orange|graphite|midnight|starlight|titanium|navy|mint"
    r"|lavender|cream|ivory)(?![a-zа-я])")
COLOR_PARAMS = ("цвет", "цвет корпуса", "основной цвет", "цвет производителя")

# --------------------------------------------------------------------------- числа и единицы

# каноническая единица -> написания (регулярные выражения) в запросах и значениях каталога
_UNITS: tuple[tuple[str, str], ...] = (
    ("гб", r"гигабайт\w*|гбайт|гиг\w*|гб|gb"), ("тб", r"терабайт\w*|тб|tb"), ("мб", r"мб|mb"),
    ("мач", r"мач|mah|ма\s?ч"), ("дюйм", r"дюйм\w*|inch\w*|\"|”|″|''"),
    ("см", r"сантиметр\w*|см|cm"), ("мм", r"миллиметр\w*|мм|mm"), ("кг", r"килограмм\w*|кг|kg"),
    ("мл", r"мл|ml"), ("л", r"литр\w*|л|l"), ("квт", r"киловатт\w*|квт|kw"), ("вт", r"ватт\w*|вт|w"),
    ("ггц", r"ггц|ghz"), ("гц", r"герц\w*|гц|hz"), ("мп", r"мегапиксел\w*|мпикс|мп|mp"), ("дб", r"дб|db"),
    ("об/мин", r"об/мин|оборот\w*|rpm"), ("ядер", r"ядер\w*|ядра|core\w*"), ("ач", r"ач|ah"),
    ("°с", r"°с|°c|градус\w*"), ("бар", r"бар|bar"), ("шт", r"шт\w*"), ("мин", r"минут\w*|мин"),
    ("ч", r"час\w*|ч"), ("м", r"метр\w*|м|m"),
)
_UNIT_ALT = "|".join(f"(?P<u{i}>{rx})" for i, (_, rx) in enumerate(_UNITS))
_NUM_RX = r"\d{1,3}(?:[  ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?"
NUM_UNIT_RE = re.compile(rf"(?<![\w.,])(?P<num>{_NUM_RX})\s?(?:{_UNIT_ALT})(?![a-zа-я0-9])")
PAIR_RE = re.compile(r"(?<![\w.,])(?P<a>\d{1,2})\s?[/+]\s?(?P<b>\d{1,4})\s?(?P<u>гб|gb|тб|tb)?(?![\d.,])")
_MEMORY = frozenset({32, 64, 128, 256, 512, 1024, 2048})
# единицы из поля @unit фида и из текста значений
_UNIT_CANON = {'"': "дюйм", "дюйм": "дюйм", "мпикс": "мп", "мп": "мп", "*с": "°с", "°с": "°с", "°c": "°с",
               "об/мин": "об/мин", "мач": "мач", "ма·ч": "мач", "гб": "гб", "тб": "тб", "мб": "мб", "см": "см",
               "мм": "мм", "м": "м", "кг": "кг", "г": "г", "л": "л", "мл": "мл", "вт": "вт", "квт": "квт",
               "гц": "гц", "ггц": "ггц", "мгц": "мгц", "дб": "дб", "в": "в", "ч": "ч", "мин": "мин", "шт": "шт",
               "ач": "ач", "бар": "бар", "пикс": "пикс"}
_RANGE_RE = re.compile(rf"^(?:от\s*)?(?P<lo>{_NUM_RX})\s*(?P<u1>[^\d\s\-–]*)\s*(?:[-–]|до)\s*(?P<hi>{_NUM_RX})\s*(?P<u2>.*)$")
_UPTO_RE = re.compile(rf"^(?:до|менее|не более)\s*(?P<hi>{_NUM_RX})\s*(?P<u>.*)$")
_FROM_RE = re.compile(rf"^(?:более|свыше|от|больше)\s*(?P<lo>{_NUM_RX})\s*(?P<u>.*)$")
_AND_MORE_RE = re.compile(rf"^(?P<lo>{_NUM_RX})\s*(?P<u>.*?)\s*(?:и более|и выше|и больше)$")
_VAL_NUM_UNIT_RE = re.compile(rf"^(?P<num>{_NUM_RX})\s*(?P<u>[^\d\s(/]{{0,8}})\.?\s*(?:\(.*\))?$")
_VAL_PAIR_RE = re.compile(r"^(?P<a>\d+)\s*гб\s*/\s*(?P<b>\d+)\s*гб$")
_COUNT_PREFIX = {"одно": 1, "двух": 2, "дву": 2, "трех": 3, "четырех": 4, "пяти": 5, "шести": 6, "семи": 7,
                 "восьми": 8}
_COUNT_RE = re.compile(r"^(одно|двух|дву|трех|четырех|пяти|шести|семи|восьми)([а-я]{3,})$")


def norm(s: object) -> str:
    """Нижний регистр, ё -> е, схлопнутые пробелы — так же, как запросы в clicks.QUERY_NORM."""
    return " ".join(str(s or "").lower().replace("ё", "е").split())


def to_number(s: str) -> float:
    return float(re.sub(r"[  ]", "", s).replace(",", "."))


def unit_canon(u: str | None) -> str | None:
    """«ГБ» -> гб, '"' -> дюйм, «МПикс» -> мп, «gb» -> гб; неизвестная единица -> None."""
    if not u:
        return None
    u = norm(u).strip(" .")
    if u in _UNIT_CANON:
        return _UNIT_CANON[u]
    for canon, rx in _UNITS:
        if re.fullmatch(rx, u):
            return canon
    return None


def num_key(x: float) -> float:
    return round(x, 3)


# приведение к базовой единице: «598 мм» и «45 см» сравнимы (ширина посудомоечной машины), «1 ТБ» = 1024 ГБ
_TO_BASE = {"мм": ("см", 0.1), "м": ("см", 100.0), "г": ("кг", 0.001), "мл": ("л", 0.001), "квт": ("вт", 1000.0),
            "тб": ("гб", 1024.0), "мб": ("гб", 1 / 1024)}


def to_base(x: float, unit: str | None) -> tuple[float, str | None]:
    if unit in _TO_BASE:
        base, k = _TO_BASE[unit]
        return x * k, base
    return x, unit


_ENDINGS = tuple(sorted({
    "иями", "ями", "ами", "ием", "ией", "ого", "его", "ому", "ему", "ыми", "ими", "ых", "их", "ая", "яя", "ое",
    "ее", "ые", "ие", "ый", "ий", "ой", "ую", "юю", "ым", "им", "ом", "ем", "ах", "ях", "ам", "ям", "ов", "ев",
    "ей", "ия", "ья", "ью", "ии", "ию", "а", "я", "ы", "и", "у", "ю", "о", "е", "ь", "й"}, key=len, reverse=True))


def stem(w: str) -> str:
    """Грубый стеммер для русских слов (окончания прилагательных и существительных); латиница — как есть."""
    if len(w) <= 3 or not _CYR.search(w):
        return w
    for e in _ENDINGS:
        if w.endswith(e) and len(w) - len(e) >= 3:
            return w[: -len(e)]
    return w


_DIMS = re.compile(r"(?<=\d)\s?[xх×*]\s?(?=\d)")  # «160х200», «60x60» — два размера


def prep(text: str) -> str:
    """norm + «с25» -> s25 (кириллическая буква-двойник) + «160х200» -> «160 200» + «айфон17про» -> «айфон 17 про»."""
    text = _LOOKALIKE.sub(lambda m: _LOOKALIKE_MAP[m[1]] + m[2], norm(text))
    return _CYR_DIGIT.sub(" ", _DIMS.sub(" ", text))


def tokens(text: str) -> list[str]:
    """Токены строки с раскрытием алиасов («айфон17про» -> iphone, 17, pro; «айфонов» -> iphone)."""
    out: list[str] = []
    for t in _TOKEN.findall(prep(text)):
        out.extend(TOKEN_ALIASES.get(t) or (TOKEN_ALIASES.get(stem(t)) if _CYR.search(t) else None) or (t,))
    return out


# основы общих слов: «экраном», «системой», «корпуса» не говорят, о каком свойстве речь
GENERIC_STEMS = frozenset({stem(w) for w in GENERIC} | {"экран", "корпус", "размер", "вес", "материал", "сист",
                                                        "систем", "работ", "устройств", "товар", "издели", "комплект"})
_INTERNAL_PARAM = re.compile(r"для отчета|bi/bo|служебн")


def content_stems(text: str, extra_stop: frozenset[str] | set[str] = frozenset()) -> list[str]:
    """Значимые основы фразы: без предлогов, общих слов («тип», «поддержка») и слов из названия категории."""
    out = []
    for t in tokens(text):
        if t in STOPWORDS or t in GENERIC or _NUM.match(t):
            continue
        s = stem(t)
        if s in extra_stop or s in GENERIC_STEMS or len(s) < 2 \
                or any(len(c) >= 5 and s.startswith(c) for c in extra_stop):
            continue
        out.append(s)
    return out


def has_letter_and_digit(t: str) -> bool:
    return bool(re.search(r"[a-zа-я]", t) and re.search(r"\d", t))


def parse_value(value: str, unit: str | None) -> tuple[str, tuple] | None:
    """Значение параметра каталога -> ('num', (x, unit)) | ('pair', (a, b)) | ('range', (lo, hi, unit)) | None."""
    v = norm(value)
    if not v:
        return None
    if _NUM.match(v):
        return "num", to_base(to_number(v), unit_canon(unit))
    if m := _VAL_PAIR_RE.match(v):
        return "pair", (float(m["a"]), float(m["b"]))
    if (m := _VAL_NUM_UNIT_RE.match(v)) and (u := unit_canon(m["u"]) or (unit_canon(unit) if not m["u"] else None)):
        return "num", to_base(to_number(m["num"]), u)
    for rx in (_RANGE_RE, _UPTO_RE, _FROM_RE, _AND_MORE_RE):
        if m := rx.match(v):
            g = m.groupdict()
            raw_u = (g.get("u2") or g.get("u") or g.get("u1") or "").strip(" .")
            u = unit_canon(raw_u) or unit_canon(unit)
            if raw_u and not u:
                return None  # «1 - 2 комнаты» — неизвестная единица
            lo = to_number(g["lo"]) if g.get("lo") else 0.0
            hi = to_number(g["hi"]) if g.get("hi") else float("inf")
            (lo, base), (hi, _) = to_base(lo, u), to_base(hi, u)
            return "range", (lo, hi, base)
    return None


def query_numbers(q: str) -> tuple[list[tuple[float, str, str, str]], list[tuple[float, float, str]], set[str]]:
    """Числа с единицами и пары памяти «8/256» в запросе -> ([(x, unit, фрагмент, число)], [(a, b, фрагмент)],
    числа, занятые парами)."""
    pairs, used = [], set()
    for m in PAIR_RE.finditer(q):
        a, b = float(m["a"]), float(m["b"])
        if (a <= 32 and b >= 16 and b > a) or (m["u"] in ("тб", "tb") and b <= 2):
            mult = 1024 if (m["u"] or "") in ("тб", "tb") else 1
            pairs.append((a, b * mult, m.group(0)))
            used.update({m["a"], m["b"]})
    nums = []
    for m in NUM_UNIT_RE.finditer(q):
        canon = next(_UNITS[i][0] for i in range(len(_UNITS)) if m.group(f"u{i}") is not None)
        x, canon = to_base(to_number(m["num"]), canon)
        nums.append((x, canon, m.group(0), re.sub(r"[ \u00a0]", "", m["num"])))
    return nums, pairs, used


# --------------------------------------------------------------------------- словарь категории

@dataclass
class NumParam:
    """Числовой параметр категории: точные значения (число -> офферов), диапазоны бакетов, пары «8 ГБ/256 ГБ»."""

    name: str
    unit: str | None
    fill: float = 0.0
    total: int = 0
    exact: dict[float, int] = field(default_factory=dict)
    ranges: list[tuple[float, float, int]] = field(default_factory=list)
    pairs: set[tuple[float, float]] = field(default_factory=set)

    def contains(self, x: float) -> bool:
        return num_key(x) in self.exact or any(lo <= x <= hi for lo, hi, _ in self.ranges)

    def share(self, x: float) -> float:
        """Доля офферов с этим значением; для бакетов — доля офферов в интервале («40 - 45 см» для 45)."""
        exact = self.exact.get(num_key(x), 0) / max(self.total, 1)
        if not self.ranges:
            return exact
        if self.unit is None:
            return exact
        tot = sum(n for *_, n in self.ranges) or 1  # открытые («более 90 см») и широкие («1-200») — не свидетельство
        return max(exact, sum(n for lo, hi, n in self.ranges if lo <= x <= hi and 0 < lo and hi <= 3 * lo) / tot)

    def dist(self) -> dict[float, float]:
        s = sum(self.exact.values()) or 1
        return {k: c / s for k, c in self.exact.items()}


def _overlap(p: list[float], q: list[float]) -> float:
    return sum(min(a, b) for a, b in zip(p, q, strict=True))


def _twins(a: NumParam, b: NumParam) -> bool:
    """Одно свойство в двух параметрах фида: бакет фильтра и точное значение («Встроенная память» и «… (ROM)»,
    «Высота холодильника» = «186 - 199 см» и «Высота» = 190). Сравниваем распределения значений."""
    if a.unit != b.unit or a.unit is None:
        return False
    da, db = a.dist(), b.dist()
    if da and db and sum(min(x, db.get(k, 0.0)) for k, x in da.items()) >= 0.75:
        return True
    for exact, rng in ((da, b), (db, a)):  # распределение точных значений по интервалам бакета ~ доли бакета
        if exact and rng.ranges and not rng.exact:
            tot = sum(n for *_, n in rng.ranges) or 1
            own = [n / tot for *_, n in rng.ranges]
            mass = [sum(x for k, x in exact.items() if lo <= k <= hi) for lo, hi, _ in rng.ranges]
            if sum(mass) >= 0.9 and _overlap(own, mass) >= 0.75:
                return True
    return False


_SECONDARY = re.compile(r"карты памяти|дополнительн|в развернутом виде|фронтальн|второ|макс\w* поддерж")
_CODE_PARAM = re.compile(r"артикул|код производителя|партномер|part ?number|ean|штрихкод")
_MODEL_PARAM = re.compile(r"(модель|серия|линейка|модельный ряд|серия модели)")
_FLAG_VALUES = frozenset({"да", "нет", "есть", "отсутствует"})


@dataclass
class CategoryVocab:
    """Что из каталога категории может встретиться в запросе: бренды, токены моделей, значения и имена свойств."""

    category_id: str
    title: str
    vendor_forms: dict[str, str] = field(default_factory=dict)  # однословная форма -> бренд
    vendor_phrases: dict[tuple[str, ...], str] = field(default_factory=dict)  # многословная форма -> бренд
    model_codes: set[str] = field(default_factory=set)  # «s26», «55c6k», «4021»
    model_words: set[str] = field(default_factory=set)  # линейки: «galaxy», «pura», «magicbook», «airpods»
    model_bigrams: set[tuple[str, str]] = field(default_factory=set)  # («iphone», «17»), («honor», «600»)
    model_num_share: dict[str, float] = field(default_factory=dict)  # «17» -> доля офферов с номером модели 17
    model_like: dict[str, set[str]] = field(default_factory=dict)  # «Модель»/«Серия»/«Артикул» -> токены значений
    numeric: dict[str, NumParam] = field(default_factory=dict)
    twins: dict[str, list[str]] = field(default_factory=dict)
    phrases: list[tuple[str, frozenset[str], frozenset[str], str]] = field(default_factory=list)
    phrase_index: dict[str, list[int]] = field(default_factory=dict)
    name_keys: list[tuple[frozenset[str], set[str]]] = field(default_factory=list)  # основы имени -> параметры
    flags: set[str] = field(default_factory=set)  # свойства «Да/Нет»: в запросе называют само свойство
    count_nouns: dict[str, set[str]] = field(default_factory=dict)  # «каме» -> {«Количество камер»} (первые 4 буквы)
    color_params: list[str] = field(default_factory=list)
    num_names: dict[str, set[str]] = field(default_factory=dict)  # «высот» -> {«Высота», «Высота холодильника»}
    fill: dict[str, float] = field(default_factory=dict)
    category_stems: frozenset[str] = frozenset()

    @classmethod
    def build(cls, category_id: str, title: str, offers: Iterable[tuple[str | None, str | None, int]],
              params: Iterable[tuple[str, bool, str | None, str | None, int]], fill: dict[str, float],
              path: Iterable[str] = (), brand_aliases: dict[str, Iterable[str]] | None = None,
              max_values: int = 300, min_share: float = 0.002) -> CategoryVocab:
        """offers: (vendor, model, офферов); params: (имя, is_search, unit, value, офферов); fill: имя -> заполненность."""
        cat_stems = frozenset(stem(t) for t in tokens(" ".join([title, *path])) if len(t) > 2)
        v = cls(category_id=str(category_id), title=title, fill=dict(fill), category_stems=cat_stems)
        by_name: dict[str, list[tuple[str | None, str, int, bool]]] = defaultdict(list)
        for name, is_search, unit, value, n in params:
            if isinstance(value, str) and value.strip():
                by_name[name].append((unit if isinstance(unit, str) else None, value, int(n), bool(is_search)))
        name_stems = v._build_params(by_name, cat_stems, max_values, min_share)
        v._build_models(offers, cat_stems, brand_aliases)
        v._build_name_keys(name_stems)
        return v

    def _build_params(self, by_name: dict, cat_stems: frozenset[str], max_values: int,
                      min_share: float) -> dict[str, list[str]]:
        name_stems: dict[str, list[str]] = {}
        for name, rows in sorted(by_name.items()):
            lname = norm(name)
            if _INTERNAL_PARAM.search(lname):  # «ЗД для отчета BI/BO» — служебные сводные поля фида
                continue
            name_stems[name] = content_stems(name, cat_stems)
            if lname in COLOR_PARAMS:  # цвет ищем по корням COLOR_RE: «серия» не должна стать «серым»
                self.color_params.append(name)
            if lname.startswith("цвет"):  # «Цвет рамки», «Цвет подставки» — не то, что ищут словом «белый»
                continue
            if _CODE_PARAM.search(lname):  # артикул: только длинные коды («sm-s938b»), а не «17» или «s25»
                self.model_like[name] = {t for _, val, *_ in rows for t in tokens(val)
                                         if has_letter_and_digit(t) and len(t) >= 5}
                continue
            if _MODEL_PARAM.fullmatch(lname):
                self.model_like[name] = {t for _, val, *_ in rows for t in tokens(val)} - MODEL_STOP
                continue
            if lname.startswith(("количество ", "число ")):
                # без фильтра по названию категории: «камеры» есть и в пути «Холодильники и морозильные камеры»
                for s in content_stems(name):
                    self.count_nouns.setdefault(s[:4], set()).add(name)
            if {norm(val) for _, val, *_ in rows} <= _FLAG_VALUES:
                self.flags.add(name)
                continue
            total = sum(r[2] for r in rows)
            parsed = [(parse_value(val, unit), n, is_s) for unit, val, n, is_s in rows]
            # числовое ли свойство, решают точные значения (_search), если они есть: бакеты бывают «от 15 до 16.9»
            base = [x for x in parsed if x[2]] or parsed
            if sum(n for p, n, _ in base if p) >= 0.7 * sum(n for _, n, _ in base):
                self._add_numeric(name, [(p, n) for p, n, _ in parsed])
                continue
            for _, val, n, _ in sorted(rows, key=lambda r: (-r[2], r[1]))[:max_values]:
                if n < 2 and n < min_share * total:
                    continue
                for alt in re.split(r"\s*(?:/|,|;|\s и\s)\s*", norm(val)):
                    st = [s for s in content_stems(alt, cat_stems) if s not in MODEL_GENERIC]
                    strong = frozenset(t for t in tokens(alt) if has_letter_and_digit(t) and len(t) >= 2
                                       and not NUM_UNIT_RE.fullmatch(t) and t not in MODEL_GENERIC)
                    if (not st and not strong) or (len(st) == 1 and len(st[0]) < 3 and not strong):
                        continue  # «A», «B» — слишком неоднозначно
                    self.phrases.append((name, frozenset(st), strong, alt))
        for i, (_, st, strong, _) in enumerate(self.phrases):
            for s in st | strong:
                self.phrase_index.setdefault(s, []).append(i)
        nums = list(self.numeric.values())
        for a in nums:
            self.twins[a.name] = [b.name for b in nums if b is not a and _twins(a, b)]
            if name_stems.get(a.name):
                self.num_names.setdefault(name_stems[a.name][0], set()).add(a.name)
        return name_stems

    def _add_numeric(self, name: str, parsed: list) -> None:
        units = Counter(p[1][-1] for p, n in parsed if p and p[0] in ("num", "range") and p[1][-1])
        np_ = NumParam(name=name, unit=units.most_common(1)[0][0] if units else None, fill=self.fill.get(name, 0.0))
        for p, n in parsed:
            if not p:
                continue
            kind, data = p
            if kind == "range":
                if data[2] in (None, np_.unit):  # «от 15 до 16.9» — в единицах точного значения (дюймы)
                    np_.ranges.append((data[0], data[1], n))
                continue
            if kind == "pair":
                np_.pairs.add(data)
                np_.unit = "гб"
            for x in (data if kind == "pair" else data[:1]):
                np_.exact[num_key(x)] = np_.exact.get(num_key(x), 0) + n
            np_.total += n
        self.numeric[name] = np_

    def _build_models(self, offers: Iterable, cat_stems: frozenset[str], brand_aliases: dict | None) -> None:
        offers = [(norm(a) if isinstance(a, str) else "", norm(b) if isinstance(b, str) else "", int(n))
                  for a, b, n in offers]
        vendor_offers: Counter = Counter()
        for a, _, n in offers:
            vendor_offers[a] += n
        # коды с цифрами, слова линейки (до первого токена с цифрой), число после линейки/бренда
        word_cnt: Counter = Counter()
        for vendor, model, n in offers:
            toks = tokens(model)
            first_digit = next((i for i, t in enumerate(toks) if re.search(r"\d", t)), len(toks))
            vfirst = (tokens(vendor)[:1] or [""])[0]
            for i, t in enumerate(toks):
                prev = toks[i - 1] if i else vfirst
                nxt = toks[i + 1] if i + 1 < len(toks) else ""
                if has_letter_and_digit(t):
                    if NUM_UNIT_RE.fullmatch(t) or t in MODEL_STOP:
                        continue
                    self.model_codes.add(t)
                    if m := re.fullmatch(r"([a-z]+)(\d+[a-z]*)", t):  # «flip7» ~ «flip 7», «s25» ~ «s 25»
                        self.model_bigrams.add((m[1], m[2]))
                        if len(m[1]) >= 3 and m[1] not in MODEL_STOP:
                            word_cnt[m[1]] += n
                elif t.isalpha() and not _CYR.search(t) and i < first_digit and len(t) >= 3 \
                        and t not in MODEL_STOP and stem(t) not in cat_stems and t not in self.phrase_index \
                        and not COLOR_RE.fullmatch(t) and t != vendor:
                    word_cnt[t] += n
                elif _NUM.match(t) and len(t) >= 4 and i == 0:
                    self.model_codes.add(t)  # «4021» у ATLANT ХМ 4021-000
                if not prev.isalpha() or not (_NUM.match(t) or has_letter_and_digit(t)):
                    continue
                if _CYR.search(prev) and prev != vfirst:
                    continue  # «Телевизор 43 …» в поле model — не номер модели
                if _NUM.match(t) and (unit_canon(nxt) or to_number(t) in _MEMORY or len(t) > 4):
                    continue  # «Pro Max 256 ГБ»: 256 — память, а не номер модели
                self.model_bigrams.add((prev, t))
        self.model_words = {w for w, c in word_cnt.items() if c >= 2}
        total = sum(n for *_, n in offers) or 1
        num_cnt: Counter = Counter()
        family = self.model_words | {t for t in tokens(self.title) if t.isalpha() and not _CYR.search(t)}
        for _, model, n in offers:
            toks = tokens(model)
            for t in {t for i, t in enumerate(toks) if _NUM.match(t) and (i == 0 or toks[i - 1] in family)}:
                num_cnt[t] += n
        self.model_num_share = {t: c / total for t, c in num_cnt.items()}
        # бренды; форма, которая чаще встречается как линейка чужих моделей, — это модель (Galaxy, Redmi)
        aliases = {k: list(x) for k, x in BRAND_ALIASES.items()}
        for k, x in (brand_aliases or {}).items():
            aliases.setdefault(norm(k), []).extend(x)
        for vendor in sorted({a for a, _, _ in offers if a and a not in JUNK_VENDORS}):
            for f in {vendor, re.sub(r"[\W_]+", "", vendor), *map(norm, aliases.get(vendor, ()))}:
                if len(f) < 2 or f in MODEL_GENERIC or f in GENERIC or COLOR_RE.fullmatch(f) \
                        or word_cnt.get(f, 0) > vendor_offers[vendor]:
                    continue
                self.model_words.discard(f)
                ft = tuple(tokens(f))
                if len(ft) == 1:
                    self.vendor_forms[ft[0]] = vendor
                elif ft:
                    self.vendor_phrases[ft] = vendor

    def _build_name_keys(self, name_stems: dict[str, list[str]]) -> None:
        """Имя свойства в запросе: полная фраза имени («защита от детей») или, для флагов, отличительная основа
        (встречается не более чем в двух заметных свойствах категории: «алис», «игров», «сушк»)."""
        df: Counter = Counter()
        for name, st in name_stems.items():
            if self.fill.get(name, 0) >= 0.05:
                df.update(set(st))
        keys: dict[frozenset[str], set[str]] = defaultdict(set)
        for name, st in name_stems.items():
            if not st or len(st) > 3 or name in self.model_like or norm(name).startswith(("цвет", "количество ",
                                                                                            "число ")):
                continue
            if all(s in MODEL_SUFFIX for s in st):
                continue
            # однословное имя обычного свойства — только если слово редкое: «двер», «экран» есть у многих свойств
            if len(st) >= 2 or name in self.flags or (df[st[0]] <= 2 and st[0] not in self.phrase_index):
                keys[frozenset(st)].add(name)
            if name in self.flags and len(st) <= 2:  # «с алисой», «с шумоподавлением», «с вертикальной …»
                for s in st:  # «Mini-SIM» не должен ловить «iphone 13 mini»: суффиксы моделей — не ключ
                    if df[s] <= 2 and len(s) >= 4 and s not in self.phrase_index and s not in MODEL_SUFFIX:
                        keys[frozenset([s])].add(name)
        self.name_keys = list(keys.items())


# --------------------------------------------------------------------------- детектор упоминаний

class Detector:
    """Какие атрибуты категории названы в поисковом запросе: {атрибут: фрагмент запроса}."""

    def __init__(self, vocab: CategoryVocab):
        self.v = vocab
        self._codes_sorted = sorted(vocab.model_codes)
        self._family = vocab.model_words | set(vocab.vendor_forms)
        title_phon = {_phonetic(t) for t in tokens(vocab.title)} | {_phonetic(c) for c in vocab.category_stems}
        self._phon: dict[str, str] = {}  # «нова» -> nova, «магик» -> magic; «робот» — слово категории, не модель
        for w in sorted(vocab.model_words):
            ph = _phonetic(w)
            if len(w) >= 4 and ph not in title_phon and not any(ph.startswith(tp) for tp in title_phon if len(tp) >= 4):
                self._phon.setdefault(ph, w)
        self.unit_prior: dict[str, float] = {}  # доля явных упоминаний свойства с единицей («45 см») в категории
        self._model_numbers = {t for _, t in vocab.model_bigrams if _NUM.match(t)}
        self._vendor_first = {b: (tokens(b) or [b])[0] for b in {*vocab.vendor_forms.values(),
                                                                  *vocab.vendor_phrases.values()}}
        self._vendor_first_set = set(self._vendor_first.values())
        # латинское слово из названия категории («iPhone»): число после него — номер модели («iphone 18»)
        self._title_latin = {t for t in tokens(vocab.title) if t.isalpha() and not _CYR.search(t)}
        # вторичные свойства (карта памяти, второй экран, фронтальная камера) проигрывают основным
        self._prior = {n: 0.3 if _SECONDARY.search(norm(n)) else 1.0 for n in vocab.numeric}
        # частые значения заметных числовых свойств: «tcl 55» — диагональ, а не номер модели;
        # «сильные» (≥ 10% офферов) остаются свойству, даже если это и номер модели: «haier 55» — ещё и диагональ
        self._attr_values = {k for p in vocab.numeric.values() if p.fill >= 0.5
                             for k in p.exact if p.share(k) >= 0.03}
        self._strong_values = {k for p in vocab.numeric.values() if p.fill >= 0.8 and p.unit == "дюйм"
                               for k in p.exact if p.share(k) >= 0.10}
        self._phrase_first: dict[str, list[tuple[tuple[str, ...], str]]] = defaultdict(list)
        for ph, brand in vocab.vendor_phrases.items():
            self._phrase_first[ph[0]].append((ph, brand))
        self._name_index: dict[str, list[int]] = defaultdict(list)
        for i, (key, _) in enumerate(vocab.name_keys):
            for s in key:
                self._name_index[s].append(i)
        groups: dict[str | None, set[str]] = defaultdict(set)  # единица -> свойства (близнецы — одно)
        for name, p in vocab.numeric.items():
            groups[p.unit].add(min([name, *vocab.twins.get(name, [])]))
        self._unit_groups = groups

    def _canon(self, toks: list[str]) -> list[str]:
        """«iphone17» -> iphone 17; «air pods» -> airpods; «нова» -> nova; «s 25» / «с 25» -> s25 (если есть код)."""
        out: list[str] = []
        i = 0
        while i < len(toks):
            t = toks[i]
            nxt = toks[i + 1] if i + 1 < len(toks) else ""
            lat = _LOOKALIKE_MAP.get(t, t)
            if nxt and t.isdigit() and len(nxt) == 1 and nxt.isdigit() and self._known_decimal(f"{t}.{nxt}"):
                out.append(f"{t}.{nxt}")  # «сумка для ноутбука 15 6» -> 15.6
                i += 2
                continue
            if nxt and ((len(lat) == 1 and _NUM.match(nxt) and lat + nxt in self.v.model_codes)
                        or (t + nxt in self._family and t.isalpha() and nxt.isalpha())):
                out.append(lat + nxt if len(lat) == 1 else t + nxt)
                i += 2
                continue
            m = re.fullmatch(r"([a-z]{3,})(\d+[a-z]*)", t)
            if m and m[1] in self._family and t not in self.v.model_codes:
                out += [m[1], m[2]]
            elif len(t) >= 4 and _CYR.search(t) and t not in self.v.vendor_forms \
                    and (w := self._phon.get(_phonetic(t))):
                out.append(w)
            else:
                out.append(t)
            i += 1
        return out

    def _known_decimal(self, t: str) -> bool:
        x = num_key(float(t))
        return any(x in p.exact for p in self.v.numeric.values())

    def _code_prefix(self, t: str) -> bool:
        if len(t) < 5:
            return False
        i = bisect.bisect_left(self._codes_sorted, t)
        return i < len(self._codes_sorted) and self._codes_sorted[i].startswith(t)

    def _credit(self, x: float, unit: str | None, frag: str, found: dict, bare: bool = False) -> bool:
        """Число -> свойство, где это значение не случайно (доля ≥ 1%, без единицы ≥ 3%), с максимумом
        √доля × заполненность, и его близнецы. «1tb» — встроенная память, а не «макс. ёмкость карты памяти»."""
        v = self.v
        floor = 0.03 if bare else 0.01
        cands = [p for p in v.numeric.values() if (unit is None or p.unit == unit) and p.contains(x)
                 and (p.share(x) >= floor or (not p.exact and not bare))]
        if not cands:
            if unit is None or len(self._unit_groups.get(unit, ())) != 1:
                return False
            cands = [max((p for p in v.numeric.values() if p.unit == unit), key=lambda p: p.fill)]  # «58 дюймов»
        best = max(cands, key=lambda p: (p.share(x) + 0.01) ** 0.5 * (p.fill + 0.01) * self._prior[p.name]
                   * (1 + 4 * self.unit_prior.get(p.name, 0.0) if bare else 1))
        found.setdefault(best.name, frag)
        for name in v.twins.get(best.name, ()):
            if v.numeric[name].contains(x) or not v.numeric[name].exact:
                found.setdefault(name, frag)
        return True

    def learn_unit_prior(self, queries: Iterable[tuple[str, int]]) -> None:
        """Первый проход: какие свойства покупатели называют с единицей («посудомоечная 45 см» -> «Ширина»).
        Голое число («посудомоечная 45») потом отдаём им, а не «Максимальной температуре воды» с тем же значением."""
        found: Counter = Counter()
        for q, clicks in queries:
            nums, _, _ = query_numbers(prep(q))
            for x, unit, frag, _ in nums:
                hit: dict = {}
                if self._credit(x, unit, frag, hit):
                    for name in hit:
                        found[name] += clicks
        total = sum(found.values())
        self.unit_prior = {n: c / total for n, c in found.items()} if total else {}

    def mentions(self, query: str) -> dict[str, str]:
        v, found = self.v, {}
        qs = prep(query)
        toks = self._canon(tokens(qs))
        nums, pairs, pair_used = query_numbers(qs)
        used = [t in pair_used or bool(NUM_UNIT_RE.fullmatch(t)) for t in toks]
        # 1. бренд; canon — токен для поиска биграмм модели («хонор» -> honor, «one plus» -> oneplus)
        vendors, canon = set(), list(toks)
        for i, t in enumerate(toks):
            if t in v.vendor_forms:
                vendors.add(v.vendor_forms[t])
                found.setdefault("vendor", t)
                used[i] = True
                canon[i] = self._vendor_first.get(v.vendor_forms[t], t)
            for ph, brand in self._phrase_first.get(t, ()):
                if tuple(toks[i:i + len(ph)]) == ph:
                    vendors.add(brand)
                    found.setdefault("vendor", " ".join(ph))
                    for j in range(i, i + len(ph)):
                        used[j] = True
                        canon[j] = self._vendor_first.get(brand, t)
        # 2. модель: код («s26», «55c6k»), линейка («galaxy», «airpods»), число после линейки или бренда
        model_toks, model_nums = [], set()
        for i, t in enumerate(toks):
            if used[i]:
                continue
            prev = canon[i - 1] if i else ""
            nxt = canon[i + 1] if i + 1 < len(toks) else ""
            j = i - 1  # «airpods pro 3», «legion pro 5»: смотрим назад через «pro/max/air»
            while j > 0 and canon[j] in MODEL_GENERIC:
                j -= 1
            back = canon[j] if j >= 0 and j != i - 1 else ""
            hit, keep = False, False
            if has_letter_and_digit(t):
                hit = (t in v.model_codes or self._code_prefix(t)) and t not in MODEL_STOP
            elif t in v.model_words:
                hit = True
            elif _NUM.match(t):
                x = to_number(t)
                strong = num_key(x) in self._strong_values
                fam_prev = prev in v.model_words or prev in self._title_latin
                fam_next = nxt in v.model_words or nxt in self._title_latin
                if ((prev, t) in v.model_bigrams and fam_prev) or ((nxt, t) in v.model_bigrams and fam_next) \
                        or ((prev in self._title_latin or nxt in self._title_latin) and x < 100) \
                        or (t in self._model_numbers and nxt in MODEL_SUFFIX) \
                        or (back and t in self._model_numbers and (back in v.model_words
                                                                    or back in self._vendor_first_set)):
                    hit = True  # «iphone 17», «15 айфон», «pura 80», «iphone 18», «17 pro max», «airpods pro 3»
                elif fam_prev and x < 100:
                    hit, keep = True, strong  # «acer nitro 5»; «samsung crystal 55» — ещё и диагональ
                elif ((prev, t) in v.model_bigrams and prev not in MODEL_STOP) or (
                        (num_key(x) not in self._attr_values or v.model_num_share.get(t, 0) >= 0.05)
                        and t in self._model_numbers and prev in self._vendor_first_set):
                    hit, keep = True, strong  # «honor 600», «apple 12»; «haier 55» — ещё и диагональ
            if hit:
                model_toks.append(t)
                used[i] = not keep
                if _NUM.match(t):
                    model_nums.add(t)
        if model_toks:
            found["model"] = " ".join(model_toks)
            mt = set(model_toks)
            for name, vals in v.model_like.items():
                if vals & mt:
                    found[name] = found["model"]
        # 3. числа: пары памяти, число + единица, «двухкамерный», число + существительное, голое число
        for i in range(len(toks) - 1):  # пара через пробел: «16 512», «8 256gb» (первое — правдоподобная RAM)
            a, b = toks[i], toks[i + 1]
            mb = re.fullmatch(r"(\d+)(gb|гб|tb|тб)?", b)
            if used[i] or not (_NUM.match(a) and mb):
                continue
            x, y = to_number(a), float(mb[1]) * (1024 if mb[2] in ("tb", "тб") else 1)
            if y in _MEMORY and x <= 24 and any(p.unit == "гб" and p.share(x) >= 0.01 for p in v.numeric.values()):
                pairs.append((x, y, f"{a} {b}"))
                used[i] = used[i + 1] = True
        for a, b, frag in pairs:
            self._credit(a, "гб", frag, found)
            self._credit(b, "гб", frag, found)
            for p in v.numeric.values():
                if (a, b) in p.pairs:
                    found.setdefault(p.name, frag)
        for x, unit, frag, numstr in nums:
            if numstr not in model_nums:  # «indesit its 4180 w»: 4180 — номер модели, а не мощность
                self._credit(x, unit, frag, found)
        for i, t in enumerate(toks):
            if used[i] or t in pair_used:
                continue
            if m := _COUNT_RE.match(t):  # «двухкамерный» -> «Количество камер»
                for name in v.count_nouns.get(stem(m[2])[:4], ()):
                    found.setdefault(name, t)
                    used[i] = True
                continue
            if not _NUM.match(t) or any(t == ns for *_, ns in nums) or (i and toks[i - 1] in CPU_WORDS | VERSION_WORDS) \
                    or (len(t) > 1 and t[0] == "0" and t[1] != "."):
                continue  # «ryzen 5» — число процессора; «gefest 5100 02» — суффикс модели
            x = to_number(t)
            nxt = toks[i + 1] if i + 1 < len(toks) else ""
            noun = [n for n in v.count_nouns.get(stem(nxt)[:4], ()) if n in v.numeric and v.numeric[n].contains(x)] \
                if nxt and _CYR.search(nxt) else []
            prev_names = v.num_names.get(stem(toks[i - 1]), ()) if i else ()
            named = [v.numeric[n] for n in prev_names if v.numeric[n].contains(x) or not v.numeric[n].exact]
            if noun:  # «3 камеры», «4 конфорки»
                for name in noun:
                    found.setdefault(name, f"{t} {nxt}")
            elif named:  # «высота 150», «диагональ 55»
                best = max(named, key=lambda p: p.fill)
                for name in [best.name, *v.twins.get(best.name, ())]:
                    found.setdefault(name, f"{toks[i - 1]} {t}")
            elif x < 2 or 1900 <= x <= 2100 or x > 100_000 or not self._credit(x, None, t, found, bare=True):
                continue
            used[i] = True
        # 4. цвет
        if v.color_params and (m := COLOR_RE.search(qs)):
            for name in v.color_params:
                found.setdefault(name, m.group(0))
        # 5. текстовые значения и имена свойств; одно и то же слово запроса («беспроводные») отдаём самым
        #    заполненным свойствам (не «Тип зарядки»), флаги, названные по имени, — всегда
        q_stems = {stem(t) for t, u in zip(toks, used, strict=True) if not u and t not in STOPWORDS}
        q_strong = {t for t, u in zip(toks, used, strict=True) if not u and has_letter_and_digit(t)}
        by_key: dict[frozenset[str], dict[str, str]] = defaultdict(dict)
        seen: set[int] = set()
        for s in sorted(q_stems | q_strong):
            for i in v.phrase_index.get(s, ()):
                if i in seen:
                    continue
                seen.add(i)
                name, st, strong, alt = v.phrases[i]
                if st and st <= q_stems:
                    by_key[st].setdefault(name, alt)
                elif strong and (hit := strong & q_strong):
                    by_key[frozenset(hit)].setdefault(name, alt)
        seen.clear()
        for s in sorted(q_stems):
            for i in self._name_index.get(s, ()):
                if i in seen:
                    continue
                seen.add(i)
                key, names = v.name_keys[i]
                if key <= q_stems:
                    for name in names:
                        if name in v.flags:
                            found.setdefault(name, " ".join(sorted(key)))
                        else:
                            by_key[key].setdefault(name, " ".join(sorted(key)))
        for cands in by_key.values():
            top = max(v.fill.get(n, 0.0) for n in cands)
            for name, frag in cands.items():
                if v.fill.get(name, 0.0) >= 0.8 * top:
                    found.setdefault(name, frag)
        return found


# --------------------------------------------------------------------------- данные: кэш запросов и словари

def build_query_cache(raw: Path, out: Path, con=None) -> Path:
    """query_clicks -> queries_by_category.parquet: category_id, query, brand, clicks, n_skus (кэш, ~1 мин)."""
    from .clicks import QUERY_NORM, connect, copy_to

    path = out / "queries_by_category.parquet"
    con = con or connect(tmp=out / ".duckdb_tmp")
    copy_to(con, f"""
        WITH q AS (SELECT sku_id::VARCHAR AS sku_id, {QUERY_NORM} AS query,
                          lower(trim("toValidUTF8(sku_brand_name)")) AS brand
                   FROM read_parquet('{raw / 'query_clicks.parquet'}')),
             s AS (SELECT sku_id, category_id FROM read_parquet('{out / 'clicks_by_sku.parquet'}'))
        SELECT s.category_id, q.query, q.brand, count(*) AS clicks, count(DISTINCT q.sku_id) AS n_skus
        FROM q JOIN s USING (sku_id) WHERE q.query <> ''
        GROUP BY ALL""", path)
    return path


_TRANSLIT = dict(zip("абвгдежзийклмнопрстуфхцчшщъыьэюя",
                     ["a", "b", "v", "g", "d", "e", "zh", "z", "i", "i", "k", "l", "m", "n", "o", "p", "r", "s",
                      "t", "u", "f", "h", "ts", "ch", "sh", "sch", "", "i", "", "e", "yu", "ya"], strict=True))


def _phonetic(s: str) -> str:
    """Грубый «звуковой» ключ для сравнения латиницы и кириллицы: самсунг ~ samsung, хайер ~ haier."""
    s = "".join(_TRANSLIT.get(c, c) for c in norm(s))
    for a, b in (("sch", "sh"), ("ph", "f"), ("ck", "k"), ("x", "ks"), ("w", "v"), ("c", "k"), ("q", "k"),
                 ("y", "i"), ("j", "i"), ("ee", "i"), ("ou", "u"), ("ai", "ai")):
        s = s.replace(a, b)
    s = re.sub(r"[^a-z0-9]", "", s)
    return re.sub(r"(.)\1+", r"\1", s)


def mine_brand_aliases(con, cache: Path, min_clicks: int = 100, min_share: float = 0.6,
                       min_similarity: float = 75.0) -> dict[str, list[str]]:
    """Кириллические написания брендов из логов: токен запроса, клики по которому в основном (≥ min_share) ведут
    на товары одного бренда и который транслитерацией похож на бренд (rapidfuzz ≥ min_similarity)."""
    from rapidfuzz import fuzz

    rows = con.execute(f"""
        WITH t AS (SELECT unnest(regexp_split_to_array(query, '[^a-zа-я0-9]+')) AS tok, brand, clicks
                   FROM read_parquet('{cache}') WHERE brand <> ''),
             c AS (SELECT tok, brand, sum(clicks) AS c FROM t
                   WHERE regexp_full_match(tok, '[а-я]{{3,}}') GROUP BY 1, 2),
             tot AS (SELECT tok, sum(c) AS total FROM c GROUP BY 1)
        SELECT c.tok, c.brand, c.c, tot.total FROM c JOIN tot USING (tok)
        WHERE tot.total >= {int(min_clicks)} AND c.c >= {float(min_share)} * tot.total""").fetchall()
    out: dict[str, list[str]] = defaultdict(list)
    for tok, brand, _, _ in rows:
        b = norm(brand)
        if b in JUNK_VENDORS or tok == b:
            continue
        # кириллический бренд сравниваем как есть («янлекс» ~ «яндекс»), латинский — через транслитерацию
        sim = fuzz.ratio(tok, b) if _CYR.search(b) else fuzz.ratio(_phonetic(tok), _phonetic(b))
        if sim >= min_similarity:
            out[b].append(tok)
    return {b: sorted(set(x)) for b, x in sorted(out.items())}


def load_vocabs(out: Path, category_ids: list[str], con, brand_aliases: dict[str, list[str]] | None = None,
                max_values: int = 300) -> dict[str, CategoryVocab]:
    """Словари для выбранных категорий: значения параметров агрегируются в DuckDB (без загрузки params целиком)."""
    ids = ",".join(f"'{c}'" for c in category_ids)
    cats = con.execute(f"""SELECT id, title, path FROM read_parquet('{out / 'categories.parquet'}')
                           WHERE id IN ({ids})""").fetchall()
    offers = con.execute(f"""SELECT category_id, vendor, model, count(*) AS n FROM read_parquet('{out / 'offers.parquet'}')
                             WHERE category_id IN ({ids}) GROUP BY ALL ORDER BY ALL""").fetchall()
    params = con.execute(f"""
        WITH o AS (SELECT sku_id, category_id FROM read_parquet('{out / 'offers.parquet'}') WHERE category_id IN ({ids})),
             v AS (SELECT o.category_id, p.name, p.is_search, p.unit, p.value, count(DISTINCT p.sku_id) AS n
                   FROM read_parquet('{out / 'params.parquet'}') p JOIN o USING (sku_id)
                   WHERE NOT p.is_service AND p.value IS NOT NULL GROUP BY ALL)
        SELECT * FROM v QUALIFY row_number() OVER (PARTITION BY category_id, name, is_search
                                                   ORDER BY n DESC, value, unit) <= {int(max_values)}
        ORDER BY category_id, name, is_search, n DESC, value, unit""").fetchall()
    fill = con.execute(f"""SELECT category_id, name, fill_rate FROM read_parquet('{out / 'param_stats.parquet'}')
                           WHERE category_id IN ({ids})""").fetchall()
    by_cat_off: dict[str, list] = defaultdict(list)
    for cid, vendor, model, n in offers:
        by_cat_off[cid].append((vendor, model, n))
    by_cat_par: dict[str, list] = defaultdict(list)
    for cid, name, is_search, unit, value, n in params:
        by_cat_par[cid].append((name, is_search, unit, value, n))
    by_cat_fill: dict[str, dict] = defaultdict(dict)
    for cid, name, fr in fill:
        by_cat_fill[cid][name] = float(fr)
    return {cid: CategoryVocab.build(cid, title, by_cat_off[cid], by_cat_par[cid], by_cat_fill[cid], path=list(path),
                                     brand_aliases=brand_aliases, max_values=max_values)
            for cid, title, path in cats}


# --------------------------------------------------------------------------- веса

@dataclass
class CategoryTraffic:
    category_id: str
    title: str
    clicks: int
    n_queries: int
    any_attribute: float  # доля кликов, где назван хоть один атрибут
    any_param: float  # … хоть одна характеристика, кроме бренда и модели
    kinds: dict[str, float]  # доля кликов по типу упоминания: brand / model / number / color / text
    weights: dict[str, float]
    examples: dict[str, list[tuple[str, int, str]]]  # атрибут -> [(запрос, клики, фрагмент)]
    groups: list[list[str]]  # «близнецы» (бакет и точное значение, «Модель» и model): общий набор запросов


def score_category(vocab: CategoryVocab, queries: Iterable[tuple[str, int]], n_examples: int = 5,
                   min_weight: float = 0.0001) -> CategoryTraffic:
    """Размечает все запросы категории; вес атрибута = доля кликов запросов, где он назван."""
    queries = list(queries)
    det = Detector(vocab)
    det.learn_unit_prior(queries)
    not_param = {"vendor", "model", *vocab.model_like}
    colors = set(vocab.color_params)

    def kind(a: str) -> str:
        if a == "vendor":
            return "brand"
        if a in not_param:
            return "model"
        return "number" if a in vocab.numeric else "color" if a in colors else "text"

    total = n_q = any_clicks = param_clicks = 0
    clicks_by: Counter = Counter()
    kind_clicks: Counter = Counter()
    examples: dict[str, list] = defaultdict(list)
    matched: dict[str, set[int]] = defaultdict(set)
    for qi, (query, clicks) in enumerate(queries):
        total += clicks
        n_q += 1
        found = det.mentions(query)
        if found:
            any_clicks += clicks
            if any(a not in not_param for a in found):
                param_clicks += clicks
            for k in {kind(a) for a in found}:
                kind_clicks[k] += clicks
        for attr, frag in found.items():
            clicks_by[attr] += clicks
            if len(matched[attr]) < 200_000:
                matched[attr].add(qi)
            if len(examples[attr]) < n_examples:
                examples[attr].append((query, int(clicks), frag))
    weights = {a: round(c / total, 4) for a, c in sorted(clicks_by.items(), key=lambda kv: (-kv[1], kv[0]))
               if total and c / total >= min_weight}
    # близнецы: параметры с почти одинаковым набором запросов (Жаккар ≥ 0.8) — одно свойство
    groups, assigned = [], set()
    for a in weights:
        if a in assigned:
            continue
        grp = [a]
        assigned.add(a)
        for b in weights:
            if b in assigned or not matched[a] or not matched[b]:
                continue
            if len(matched[a] & matched[b]) / len(matched[a] | matched[b]) >= 0.8:
                grp.append(b)
                assigned.add(b)
        groups.append(grp)
    kinds = {k: round(kind_clicks[k] / max(total, 1), 4) for k in ("brand", "model", "number", "color", "text")}
    return CategoryTraffic(vocab.category_id, vocab.title, int(total), n_q, round(any_clicks / max(total, 1), 4),
                           round(param_clicks / max(total, 1), 4), kinds, weights, {a: examples[a] for a in weights},
                           groups)


def _score_job(job: tuple[CategoryVocab, list[tuple[str, int]]]) -> CategoryTraffic:
    return score_category(*job)


# запасной вариант, если config/taxonomy.yaml недоступен: смартфон = «Смартфоны» + «iPhone», ноутбук, наушники
CASE_GROUPS = {"smartphone": {"category_ids": ["761", "914"]}, "laptop": {"category_ids": ["987"]},
               "headphones": {"category_ids": ["31607"]}}
MAIN_CATEGORIES = ("761", "914", "987", "1682", "2447", "666", "31607")


def load_taxonomy_groups(path: Path) -> dict[str, dict]:
    """Категории таксономии (config/taxonomy.yaml): {имя: {title, category_ids, gold: {атрибут: [имена в фиде]}}}."""
    import yaml

    if not path.exists():
        return CASE_GROUPS
    cats = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("categories") or {}
    out = {}
    for name, c in cats.items():
        ids = [str(x) for x in (c.get("category_ids") or [])]
        if ids:
            out[name] = {"title": c.get("title"), "category_ids": ids,
                         "gold": {a: list(spec.get("gold") or []) for a, spec in (c.get("attributes") or {}).items()}}
    return out or CASE_GROUPS


def group_summary(parts: list[CategoryTraffic], gold: dict[str, list[str]] | None = None, top: int = 20) -> dict:
    """Категория таксономии из нескольких листьев YML: веса — среднее по листьям, взвешенное кликами;
    by_attribute — вес атрибута таксономии = максимум по его именам в фиде (как taxonomy.attr_traffic_weight)."""
    clicks = sum(p.clicks for p in parts) or 1
    w: Counter = Counter()
    for p in parts:
        for a, x in p.weights.items():
            w[a] += x * p.clicks
    weights = {a: round(x / clicks, 4) for a, x in w.most_common()}
    out = {"category_ids": [p.category_id for p in parts], "titles": [p.title for p in parts], "clicks": clicks,
           "n_queries": sum(p.n_queries for p in parts),
           "any_attribute": round(sum(p.any_attribute * p.clicks for p in parts) / clicks, 4),
           "any_param": round(sum(p.any_param * p.clicks for p in parts) / clicks, 4),
           "kinds": {k: round(sum(p.kinds[k] * p.clicks for p in parts) / clicks, 4) for k in parts[0].kinds},
           "weights": dict(list(weights.items())[:top])}
    if gold:
        out["by_attribute"] = {a: max((weights.get(g, 0.0) for g in names), default=0.0) for a, names in gold.items()}
    return out


def run(raw: Path, out: Path, config_path: Path, summary_path: Path, min_clicks: int = 10_000,
        refresh: bool = False, workers: int = 3, log=print) -> dict:
    """Кэш запросов -> словари категорий -> веса -> config/traffic_weights.yaml + submission/traffic.json."""
    import shutil
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    import yaml

    from . import atomic_path
    from .clicks import connect

    t0 = time.perf_counter()
    con = connect(tmp=out / ".duckdb_tmp")
    cache = out / "queries_by_category.parquet"
    if refresh or not cache.exists():
        log("  кэш запросов по категориям (query_clicks, ~1 мин) ...")
        build_query_cache(raw, out, con)
    mined = mine_brand_aliases(con, cache)
    log(f"  кириллических написаний брендов из логов: {sum(map(len, mined.values()))} у {len(mined)} брендов")
    taxo_ids = [c["id"] for c in json.loads((out / "taxonomy_candidates.json").read_text(encoding="utf-8"))["categories"]]
    by_cat = con.execute(f"""SELECT c.category_id, sum(c.clicks) AS clicks FROM read_parquet('{cache}') c
        JOIN read_parquet('{out / 'categories.parquet'}') k ON k.id = c.category_id
        WHERE k.n_offers > 0 GROUP BY 1 ORDER BY 2 DESC""").fetchall()
    total_clicks = int(con.execute(f"SELECT sum(clicks) FROM read_parquet('{cache}')").fetchone()[0])
    tax_groups = load_taxonomy_groups(config_path.parent / "taxonomy.yaml")
    with_offers = {c for c, _ in by_cat}
    tax_ids = [c for g in tax_groups.values() for c in g["category_ids"] if c in with_offers]
    chosen = list(dict.fromkeys(taxo_ids + tax_ids + [c for c, n in by_cat if n >= min_clicks]))
    log(f"  категорий: {len(chosen)} (топ-25 и листья таксономии + все с ≥ {min_clicks:,} кликов)")
    vocabs = load_vocabs(out, chosen, con, mined)
    ids = ",".join(f"'{c}'" for c in chosen)
    per_cat: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for cid, query, clicks in con.execute(f"""SELECT category_id, query, sum(clicks) AS clicks
            FROM read_parquet('{cache}') WHERE category_id IN ({ids}) GROUP BY 1, 2 ORDER BY 1, 3 DESC, 2""").fetchall():
        per_cat[cid].append((query, int(clicks)))
    con.close()
    shutil.rmtree(out / ".duckdb_tmp", ignore_errors=True)
    jobs = sorted(((vocabs[c], per_cat[c]) for c in chosen if c in vocabs and per_cat.get(c)), key=lambda j: -len(j[1]))
    log(f"  размечаем {sum(len(q) for _, q in jobs):,} пар (категория, запрос) в {workers} процессах ...")
    if workers > 1:
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
            scored = list(pool.map(_score_job, jobs, chunksize=1))
    else:
        scored = [_score_job(j) for j in jobs]
    results = {r.category_id: r for r in sorted(scored, key=lambda r: -r.clicks)}
    groups = {}
    for name, g in tax_groups.items():
        parts = [results[c] for c in g["category_ids"] if c in results]
        if parts:
            groups[name] = {"title": g.get("title"), **group_summary(parts, g.get("gold"))}
    covered = sum(r.clicks for r in results.values())
    meta = {
        "method": "вес = доля поисковых кликов категории, в запросах которых назван атрибут (значение из каталога "
                  "категории: бренд и его кириллические написания, токены модели, число с единицей или пара памяти "
                  "«8/256», голое число из частых значений параметра, текстовое значение, имя свойства-флага, цвет); "
                  "не нормировано (запрос может называть несколько атрибутов); бакет фильтра и точное значение "
                  "одного свойства получают одинаковый вес; оценка нижняя (описательные запросы без значения не "
                  "считаются). Код: src/attrx/data/traffic.py, запуск: attrx traffic",
        "date": date.today().isoformat(),
        "source": "data/raw/query_clicks.parquet (клики поиска) × data/processed/{offers,params}.parquet (каталог)",
        "total_clicks": total_clicks, "n_categories": len(results), "clicks_covered": covered,
        "clicks_covered_share": round(covered / total_clicks, 4),
        "n_queries": int(sum(r.n_queries for r in results.values())),
        "any_attribute_share": round(sum(r.any_attribute * r.clicks for r in results.values()) / max(covered, 1), 4),
        "any_param_share": round(sum(r.any_param * r.clicks for r in results.values()) / max(covered, 1), 4),
        "kinds_share": {k: round(sum(r.kinds[k] * r.clicks for r in results.values()) / max(covered, 1), 4)
                        for k in ("brand", "model", "number", "color", "text")},
    }
    yaml_doc: dict = {"_meta": {**meta, "categories": {
        cid: {"title": r.title, "clicks": r.clicks, "n_queries": r.n_queries, "any_attribute": r.any_attribute,
              "any_param": r.any_param, "kinds": r.kinds} for cid, r in results.items()}}}
    for cid, r in results.items():
        yaml_doc[cid] = dict(list(r.weights.items())[:40])
    with atomic_path(config_path) as tmp:
        tmp.write_text("# Трафик-веса атрибутов по категориям из поисковых кликов М.Видео (attrx traffic).\n"
                       "# {category_id: {атрибут: доля кликов категории, в запросах которых назван атрибут}};\n"
                       "# атрибут — vendor, model или базовое имя параметра YML. Метод и метаданные — в _meta.\n"
                       + yaml.safe_dump(yaml_doc, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")

    def summary(r: CategoryTraffic, top: int = 12) -> dict:
        items = [{"attribute": g[0], "share": r.weights[g[0]], "params": g,
                  "examples": [{"query": q, "clicks": c, "match": f} for q, c, f in r.examples[g[0]][:3]]}
                 for g in r.groups[:top]]
        return {"title": r.title, "clicks": r.clicks, "n_queries": r.n_queries, "any_attribute": r.any_attribute,
                "any_param": r.any_param, "kinds": r.kinds, "top": items}

    summ: dict = {"_meta": {**meta, "how_to_read": "share — доля кликов категории, в запросах которых назван атрибут; "
                            "params — имена параметров YML, описывающие то же свойство (бакет фильтра и точное "
                            "значение); examples — самые частые запросы с упоминанием и найденный фрагмент",
                            "brand_aliases_mined": {b: sorted(set(x)) for b, x in sorted(mined.items())}}}
    summ["_meta"]["taxonomy_groups"] = "ключи без цифр (smartphone, laptop, tv, …) — категории config/taxonomy.yaml: " \
        "веса по их листьям YML, взвешенные кликами; by_attribute — вес атрибута таксономии (максимум по его gold-именам)"
    summ.update(groups)
    for cid, r in results.items():
        summ[cid] = summary(r)
    with atomic_path(summary_path) as tmp:
        tmp.write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"categories": len(results), "clicks_covered_share": meta["clicks_covered_share"],
            "any_attribute_share": meta["any_attribute_share"], "any_param_share": meta["any_param_share"],
            "seconds": round(time.perf_counter() - t0, 1)}
