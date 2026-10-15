"""Нормализация значений к каноническому виду: единицы, числа, справочники."""
from __future__ import annotations

import re

from rapidfuzz import fuzz, process

from .taxonomy import AttrSpec, Dictionary, norm

NUM = r"(\d+(?:[.,]\d+)?)"

UNIT_PATTERNS: dict[str, list[tuple[str, float]]] = {
    # kind -> [(regex единицы, множитель к канонической единице)]
    "memory": [(r"(?:тб|tb)", 1024.0), (r"(?:гб|gb|гбайт|gbyte)", 1.0)],
    "ram": [(r"(?:тб|tb)", 1024.0), (r"(?:гб|gb|гбайт)", 1.0)],
    "storage_drive": [(r"(?:тб|tb)", 1024.0), (r"(?:гб|gb|гбайт)", 1.0)],
    "diagonal": [(r"(?:\"|''|”|″|дюйм\w*|in(?:ch)?)", 1.0), (r"(?:см|cm)", 1 / 2.54)],
    "battery": [(r"(?:мач|mah|ма·ч|ма\*ч|ма ч)", 1.0), (r"(?:ач|ah)", 1000.0)],
    "load_kg": [(r"(?:кг|kg)", 1.0)],
    "rpm": [(r"(?:об\.?\s*/\s*мин\.?|об\.?\s*в\s*мин\w*|оборот\w*\s*в\s*минуту|rpm)", 1.0)],
    "length_cm": [(r"(?:см|cm)", 1.0), (r"(?:мм|mm)", 0.1), (r"(?:м|m)", 100.0)],
    "size_mm": [(r"(?:мм|mm)", 1.0), (r"(?:см|cm)", 10.0)],
    "volume_l": [(r"(?:л|литр\w*|l)", 1.0), (r"(?:мл|ml)", 0.001)],
    "power_w": [(r"(?:квт|kw)", 1000.0), (r"(?:вт|w)", 1.0)],
    "power_kw": [(r"(?:квт|kw)", 1.0), (r"(?:вт|w)", 0.001)],
    "refresh_hz": [(r"(?:гц|hz)", 1.0)],
    "time_h": [(r"(?:ч|час\w*|h|hours?)", 1.0), (r"(?:мин\w*|min)", 1 / 60)],
    "time_min": [(r"(?:мин\w*|min)", 1.0), (r"(?:ч|час\w*|h|hours?)", 60.0)],
    "btu": [(r"(?:btu|бте)", 1.0)],
    "area_m2": [(r"(?:м2|м²|m2|кв\.?\s*м\.?|квадратн\w*\s*метр\w*)", 1.0)],
}

# Бакет фильтра каталога («6.6\" и более», «3000-5000 мАч», «от 15 до 16.9», «до 2 л») — не точное значение.
RANGE_RE = re.compile(r"\d[^\d\s]{0,4}\s*(?:-|–|—|\.\.)\s*(?:от\s*)?\d|(?:^|\s)(?:от|до|более|менее|свыше|выше|ниже)\s"
                      r"|\sи\s+(?:более|менее|выше|ниже)|[<>≤≥]", re.I)
YES = frozenset({"да", "yes", "есть", "true", "1", "+", "имеется", "присутствует"})
NO = frozenset({"нет", "no", "false", "0", "-", "отсутствует", "не поддерживается"})
NOT_A_COLOR = frozenset({"другие цвета", "другие", "другой", "другой цвет", "прочие", "комбинированный", "мультицвет", "нет"})


def compact(s: str) -> str:
    """Ключ сравнения строк: регистр, ё/е, пробелы, дефисы и пунктуация не важны («Galaxy S24-Ultra» ~ «galaxys24ultra»)."""
    return re.sub(r"[\W_]+", "", norm(str(s)))


def is_range(text: str) -> bool:
    return bool(RANGE_RE.search(str(text)))


def to_float(s: str) -> float:
    return float(s.replace(",", "."))


def fmt_number(x: float) -> float:
    return round(x, 2) if abs(x - round(x)) > 1e-6 else float(round(x))


def parse_number(text: str, kind: str | None) -> tuple[float, str] | None:
    """'256 Гб' -> (256.0, '256 Гб'); '1 ТБ' -> (1024.0, ...); '6,7"' -> (6.7, ...)."""
    t = norm(text)
    for unit_re, mult in UNIT_PATTERNS.get(kind or "", []):
        m = re.search(NUM + r"\s*" + unit_re + r"(?![a-zа-я])", t)
        if m:
            return fmt_number(to_float(m.group(1)) * mult), m.group(0)
    m = re.fullmatch(r"\s*" + NUM + r"\s*", t)
    if m:
        return fmt_number(to_float(m.group(1))), m.group(0)
    return None


def normalize_enum(text: str, dictionary: Dictionary, fuzzy_cutoff: int = 90) -> tuple[str, float] | None:
    """Точное совпадение со справочником -> 1.0; нечёткое (опечатки) -> <1.0; иначе None (не выдумываем)."""
    exact = dictionary.get(text)
    if exact:
        return exact, 1.0
    hits = dictionary.find_all(text)
    if hits:
        return hits[0][0], 0.9
    choice = process.extractOne(norm(text), list(dictionary.lookup), scorer=fuzz.ratio, score_cutoff=fuzzy_cutoff)
    if choice:
        return dictionary.lookup[choice[0]], choice[1] / 100 * 0.85
    return None


def normalize_bool(raw: str, keywords: list[str] | tuple[str, ...] = ()) -> str | None:
    """«Да»/«Нет»/«есть»/«отсутствует» -> «да»/«нет»; иначе «да», если в значении есть признак (keywords)."""
    t = norm(str(raw)).strip(" .;:")
    if t in YES:
        return "да"
    if t in NO:
        return "нет"
    return "да" if any(norm(k) in t for k in keywords) else None


def normalize_colors(text: str, colors: Dictionary) -> list[str]:
    """«черный/ серебристый» -> ['чёрный', 'серебристый']; «темно-серый матовый» -> ['серый']; «другие цвета» -> []."""
    out: list[str] = []
    for part in re.split(r"\s*(?:/|,|;|\+|\s+и\s+)\s*", norm(str(text))):
        part = part.strip(" .-")
        if not part or part in NOT_A_COLOR:
            continue
        canon = colors.get(part)
        if canon is None:
            hits = colors.find_all(part) or colors.find_all(re.sub(r"\b[а-я]+о-(?=[а-я])", "", part))
            canon = hits[0][0] if hits else None
        if canon and canon not in out:
            out.append(canon)
    return out


def normalize_value(raw: str, spec: AttrSpec, dictionaries: dict[str, Dictionary]) -> tuple[str | float, float] | None:
    """Возвращает (каноническое значение, качество нормализации 0..1) или None."""
    raw = str(raw).strip()
    if not raw:
        return None
    if spec.type == "bool":
        b = normalize_bool(raw, spec.keywords)
        return (b, 1.0) if b else None
    if spec.type == "number":
        parsed = parse_number(raw, spec.kind)
        return (parsed[0], 1.0) if parsed else None
    if spec.type == "enum" and spec.dictionary:
        return normalize_enum(raw, dictionaries[spec.dictionary])
    if spec.kind == "cpu":
        return extract_cpu_value(raw), 0.95
    return re.sub(r"\s+", " ", raw).strip(" ,.;"), 0.9


# Процессоры (ноутбуки): полные модели раньше семейств. Применяются к тексту в нижнем регистре.
CPU_PATTERNS: list[tuple[str, float]] = [  # (шаблон, уверенность: модель > семейство)
    (r"(?:intel\s+)?core\s+ultra\s*[3579]\s*-?\s*\d{3}[a-z]{0,2}(?![\w])", 1.0),  # Core Ultra 7 155H, Ultra 5-226V
    (r"(?:intel\s+)?(?:core\s+)?i[3579][\s-]+\d{4,5}(?:[a-z]{1,3}\d?)?(?![\w])", 1.0),  # i5-12450H, i3-1115G4
    (r"(?<![\w-])i[3579]-\d{4,5}(?:[a-z]{1,3}\d?)?(?![\w])", 1.0),
    (r"(?:intel\s+)?(?:core\s+)?i3-n\d{3}(?![\w])", 1.0),  # Core i3-N305
    (r"(?:intel\s+)?core\s+i?[3579]\s+\d{3}[a-z]{1,2}(?![\w])", 0.95),  # Core 5 120U (и опечатка «Core i5 120U»)
    (r"(?:intel\s+)?core\s+[3579]\s+\d{3}[a-z]{0,2}(?![\w])", 1.0),  # Intel Core 5 210H, Core 7 150U
    (r"(?:amd\s+)?ryzen\s+(?:ai\s+)?(?:max\+?\s+)?(?:pro\s+)?[3579]\s+(?:pro\s+)?(?:hx\s+)?\d{3,4}[a-z]{0,3}(?![\w])", 1.0),
    (r"(?:intel\s+)?(?:celeron|pentium(?:\s+silver|\s+gold)?)\s+[a-z]?\d{3,5}[a-z]{0,2}(?![\w])", 1.0),  # Celeron N5095
    (r"(?<![\w-])(?:intel\s+)?n(?:50|95|97|100|150|200|250|300|305|350|355)(?![\w-])", 0.95),  # Intel N100
    (r"(?<![\w-])(?:apple\s+)?m[1-5](?:\s+(?:pro|max|ultra))?(?![\w-])", 0.9),  # Apple M1–M5 (+Pro/Max/Ultra)
    (r"snapdragon\s+x\s*(?:elite|plus)?(?:\s+x\d[\w-]*)?(?![\w])", 0.95),
    (r"snapdragon\s+x1(?![\w])", 0.9),
    (r"mediatek\s+(?:kompanio|dimensity|helio)\s+\w+", 0.9),
    # только семейство (модель не указана): «Core i5», «Ryzen 7», «Ultra 5 /32/1TB»
    (r"(?:intel\s+)?core\s+ultra\s*[3579](?![\w])", 0.8),
    (r"(?<![\w-])ultra\s*[3579]\s*-\s*\d{3}[a-z]{0,2}(?![\w])", 0.9),  # «Ultra5-225H»
    (r"(?<![\w-])ultra\s*[3579](?![\w])", 0.55),  # «Ultra 7» без Core/Intel — часто название модели ноутбука
    (r"(?:intel\s+)?core\s+i[3579](?![\w-])", 0.8),
    (r"(?:amd\s+)?ryzen\s+(?:ai\s+)?[3579](?![\w-])", 0.8),
    (r"(?:intel\s+)?(?:celeron|pentium)(?![\w-])", 0.7),
]
CPU_RES = [(re.compile(p), w) for p, w in CPU_PATTERNS]


def find_cpu(low: str, start: int = 0, end: int | None = None) -> tuple[re.Match, float] | None:
    """Первый (по приоритету шаблона) процессор в low[start:end]: (совпадение, вес шаблона)."""
    end = len(low) if end is None else end
    for rx, w in CPU_RES:
        m = rx.search(low, start, end)
        if m:
            return m, w
    return None


def normalize_cpu(raw: str) -> str:
    s = re.sub(r"\s+", " ", raw).strip(" ,.;")
    s = re.sub(r"(?i)\bintel\b", "Intel", s)
    s = re.sub(r"(?i)\bcore\b", "Core", s)
    s = re.sub(r"(?i)\bamd\b", "AMD", s)
    s = re.sub(r"(?i)\bryzen\b", "Ryzen", s)
    s = re.sub(r"(?i)\bapple\b", "Apple", s)
    s = re.sub(r"(?i)\bm([1-5])\b", r"M\1", s)
    s = re.sub(r"(?i)\b(pro|max|ultra|elite|plus)\b", lambda m: m.group(1).capitalize(), s)
    s = re.sub(r"(?i)\b(i[3579])[ -]?(\d{4,5}\w*)", lambda m: f"{m.group(1).lower()}-{m.group(2).upper()}", s)
    s = re.sub(r"(?i)\b(ai|hx|hs)\b", lambda m: m.group(1).upper(), s)
    s = re.sub(r"(?i)\b(celeron|pentium|snapdragon|silver|gold)\b", lambda m: m.group(1).capitalize(), s)
    s = re.sub(r"\b([nN])(\d{2,4})\b", lambda m: f"N{m.group(2)}", s)
    s = re.sub(r"(?i)\bUltra\s*([3579])\s*-\s*(\d)", r"Ultra \1 \2", s)  # «Ultra 5-226V» -> «Ultra 5 226V»
    s = re.sub(r"(?i)\bCore\s+i([3579])\s+(\d{3}[a-z]{1,2})\b", r"Core \1 \2", s)  # «Core i5 120U» -> «Core 5 120U»
    s = re.sub(r"(?i)\b(i3)-n(\d{3})\b", r"\1-N\2", s)
    s = re.sub(r"(?i)\bUltra([3579])\b", r"Ultra \1", s)
    if re.match(r"M[1-5]\b", s):
        s = "Apple " + s
    elif re.match(r"(Core\s|i[3579]-)", s):
        s = "Intel " + ("" if s.startswith("Core") else "Core ") + s
    elif re.match(r"Ultra [3579]\b", s):
        s = "Intel Core " + s
    elif re.match(r"(Celeron|Pentium|N\d{2,4}\b)", s):
        s = "Intel " + s
    elif s.startswith("Ryzen"):
        s = "AMD " + s
    elif s.startswith("Snapdragon"):
        s = "Qualcomm " + s
    return s


def extract_cpu_value(raw: str) -> str:
    """Значение продавца -> процессор: шаблон процессора внутри длинной строки («Intel Core 5 210H (8 ядер, ...)»),
    иначе — строка без частот и пояснений в скобках."""
    hit = find_cpu(raw.lower())
    if hit:
        m = hit[0]
        return normalize_cpu(raw[m.start():m.end()])
    head = re.split(r"\s*(?:\(|—|–|,|;)\s*", str(raw), maxsplit=1)[0]
    return clean_cpu(head) or normalize_cpu(raw)


FREQ_RE = re.compile(r"\d+(?:[.,]\d+)?\s*(?:x|х|×)\s*\d+(?:[.,]\d+)?\s*(?:ггц|ghz)|\d+(?:[.,]\d+)?\s*(?:ггц|ghz|мгц|mhz)"
                     r"|\+|\b\d+\s*(?:x|х|×)\s*(?=\d)", re.I)
CPU_NOISE = re.compile(r"\b(?:intel|amd|apple|qualcomm|mediatek|samsung|hisilicon|процессор|processor|cpu|core|n-series|series|"
                       r"серии|мобильный|mobile)\b")


def clean_cpu(raw: str) -> str:
    """Значение каталога -> канон: «AMD Ryzen 3 Ryzen 3 7320U 2.4 ГГц» -> «AMD Ryzen 3 7320U»; «Apple Apple M5 Pro» -> «Apple M5 Pro»."""
    s = re.sub(r"\s+", " ", FREQ_RE.sub(" ", str(raw))).strip(" ,.;/")
    for n in (4, 3, 2, 1):  # повтор бренда/семейства («Apple Apple M4», «Intel Core Ultra 9 Intel Core Ultra 9 285HX»)
        s = re.sub(r"(?i)\b((?:\w+\s+){" + str(n - 1) + r"}\w+)\s+\1\b", r"\1", s)
    return normalize_cpu(s) if s else ""


def cpu_key(raw: str) -> str:
    """Ключ сравнения процессоров: без производителя, частот и «Core»: «Intel Core i5-12450H 2.0 ГГц» -> «i512450h»."""
    s = CPU_NOISE.sub(" ", norm(clean_cpu(raw)))
    seen: list[str] = []
    for w in s.split():
        if w not in seen:
            seen.append(w)
    return compact(" ".join(seen))


MODEL_CUT = re.compile(
    r"(?<![\w.])\d{1,2}\s*(?:gb|гб)?\s*[/+]\s*\d{1,4}(?![\d.,])\s*(?:gb|гб|tb|тб)?(?![a-zа-я])"  # 8/256GB, 12+512
    r"|(?<![\w.])\d+(?:[.,]\d+)?\s*(?:gb|гб|tb|тб|mb|мб|\"|''|”|″|дюйм\w*|inch|мач|mah|вт|w|л|кг|мм|mm|см|cm|гц|hz)(?![a-zа-я])"
    r"|[,(\[|;]|/|\s[-–—]\s", re.I)
# То же без единиц мощности: «ITS 4160 W», «WKR-2000 W» — W здесь часть кода модели (для показа), а эталон
# сравнивается через model_core (с этим срезом), поэтому ключ сравнения от этого не меняется.
MODEL_CUT_KEEP_POWER = re.compile(MODEL_CUT.pattern.replace("|вт|w|", "|"), re.I)


def model_cut_pos(low: str, pos: int = 0, end: int | None = None, keep_power: bool = False) -> int:
    """Где кончается модель в low[pos:end] (тот же срез, что в model_core): память, диагональ, единицы, «,(/», « - »."""
    end = len(low) if end is None else end
    m = (MODEL_CUT_KEEP_POWER if keep_power else MODEL_CUT).search(low, pos, end)
    return m.start() if m else end


MODEL_NOISE = re.compile(r"(?<![\w-])(?:5g|4g|lte|dual\s*sim|wi-?fi|esim|nano\s*sim|global(?:\s+version)?|ростест|"
                         r"восстановленн\w*|новый|новая)(?![\w-])", re.I)
CATEGORY_WORDS = re.compile(
    r"^(?:смартфон|телефон|ноутбук|планшет|телевизор|наушники|гарнитура|смарт-часы|умные часы|часы|пылесос|робот-пылесос|"
    r"чайник|электрочайник|холодильник|стиральная машина|сплит-система|кондиционер|аэрогриль|игровой|"
    r"беспроводные|проводные|вертикальный|электрический|мобильный|true wireless)\s+", re.I)


def model_core(raw: str, brand: str | None = None, colors: Dictionary | None = None) -> str:
    """Модель без вариативных частей: «Galaxy S25 5G 12/128GB Серый» -> «Galaxy S25»; бренд и слово категории — прочь.

    Одна функция для эталона и предсказаний — модель сравнивается по compact(model_core(...))."""
    s = re.sub(r"\s+", " ", str(raw or "")).strip()
    for _ in range(3):  # «Смартфон Samsung Galaxy ...»: слово категории и бренд в начале (в любом порядке)
        before = s
        s = CATEGORY_WORDS.sub("", s)
        if brand and norm(s).startswith(norm(brand) + " "):
            s = s[len(brand) + 1:]
        if before == s:
            break
    m = MODEL_CUT.search(s)
    s = s[: m.start()] if m else s
    if colors is not None:
        hits = colors.find_all(s)
        if hits and hits[0][2] > 0:  # цвет в начале — скорее часть названия («Black Shark»), не режем
            s = s[: hits[0][2]]
    s = MODEL_NOISE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip(" -–—,.;:/")


def display(value: str | float | None, spec: AttrSpec) -> str | None:
    if value is None:
        return None
    if isinstance(value, float):
        v = int(value) if value.is_integer() else str(value).replace(".", ",")
        return f"{v} {spec.unit}" if spec.unit else str(v)
    return str(value)
