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
}


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


def normalize_value(raw: str, spec: AttrSpec, dictionaries: dict[str, Dictionary]) -> tuple[str | float, float] | None:
    """Возвращает (каноническое значение, качество нормализации 0..1) или None."""
    raw = str(raw).strip()
    if not raw:
        return None
    if spec.type == "number":
        parsed = parse_number(raw, spec.kind)
        return (parsed[0], 1.0) if parsed else None
    if spec.type == "enum" and spec.dictionary:
        return normalize_enum(raw, dictionaries[spec.dictionary])
    if spec.kind == "cpu":
        return normalize_cpu(raw), 0.95
    return re.sub(r"\s+", " ", raw).strip(" ,.;"), 0.9


def normalize_cpu(raw: str) -> str:
    s = re.sub(r"\s+", " ", raw).strip(" ,.;")
    s = re.sub(r"(?i)\bintel\b", "Intel", s)
    s = re.sub(r"(?i)\bcore\b", "Core", s)
    s = re.sub(r"(?i)\bamd\b", "AMD", s)
    s = re.sub(r"(?i)\bryzen\b", "Ryzen", s)
    s = re.sub(r"(?i)\bapple\b", "Apple", s)
    s = re.sub(r"(?i)\bm([1-4])\b", r"M\1", s)
    s = re.sub(r"(?i)\b(pro|max|ultra|elite|plus)\b", lambda m: m.group(1).capitalize(), s)
    s = re.sub(r"(?i)\b(i[3579])[ -]?(\d{4,5}\w*)", lambda m: f"{m.group(1).lower()}-{m.group(2).upper()}", s)
    s = re.sub(r"(?i)\b(celeron|pentium|snapdragon)\b", lambda m: m.group(1).capitalize(), s)
    s = re.sub(r"\b([nN])(\d{3,4})\b", lambda m: f"N{m.group(2)}", s)
    if re.match(r"M[1-4]\b", s):
        s = "Apple " + s
    elif re.match(r"(Core\s|i[3579]-)", s):
        s = "Intel " + ("" if s.startswith("Core") else "Core ") + s
    return s


def display(value: str | float | None, spec: AttrSpec) -> str | None:
    if value is None:
        return None
    if isinstance(value, float):
        v = int(value) if value.is_integer() else str(value).replace(".", ",")
        return f"{v} {spec.unit}" if spec.unit else str(v)
    return str(value)
