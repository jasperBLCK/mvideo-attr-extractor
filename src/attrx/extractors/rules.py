"""Правила и регулярки: дешёвый, быстрый и объяснимый первый слой. Покрывает большую часть числовых атрибутов."""
from __future__ import annotations

import re

from ..normalize import NUM, fmt_number, normalize_cpu, parse_number, to_float
from ..schema import Candidate, Card
from ..taxonomy import AttrSpec, CategorySpec, Taxonomy, norm

# Правдоподобные диапазоны: защищают от «6 ГБ» в диагонали и прочих ложных срабатываний.
RANGES: dict[str, tuple[float, float]] = {
    "memory": (2, 2048),
    "ram": (1, 256),
    "storage_drive": (16, 16384),
    "diagonal": (1.0, 100.0),
    "battery": (100, 40000),
}
CATEGORY_RANGES: dict[tuple[str, str], tuple[float, float]] = {
    ("smartphone", "diagonal"): (3.5, 8.5),
    ("laptop", "diagonal"): (10.0, 18.5),
}

GB = r"(?:гб|gb|гбайт)"
TB = r"(?:тб|tb)"
RAM_HINT = r"(?:ram|озу|оперативн\w*)"
SSD_HINT = r"(?:ssd|hdd|emmc|накопител\w*|диск\w*|ufs|встроенн\w*)"

CPU_PATTERNS = [
    r"intel\s+core\s+ultra\s+\d\s+\d{3}[a-z]{0,2}",
    r"intel\s+core\s+i[3579][\s-]?\d{4,5}[a-z]{0,3}",
    r"core\s+i[3579][\s-]?\d{4,5}[a-z]{0,3}",
    r"(?<![\w-])i[3579]-\d{4,5}[a-z]{0,3}",
    r"amd\s+ryzen\s+\d\s+(?:pro\s+)?\d{4}[a-z]{0,3}",
    r"ryzen\s+\d\s+\d{4}[a-z]{0,3}",
    r"intel\s+(?:celeron|pentium(?:\s+silver|\s+gold)?|core\s+\d)\s+\w?\d{3,5}[a-z]{0,2}",
    r"intel\s+n\d{2,3}",
    r"apple\s+m[1-4](?:\s+(?:pro|max|ultra))?",
    r"(?<![\w-])m[1-4](?:\s+(?:pro|max|ultra))?(?![\w-])",
    r"snapdragon\s+x\s+(?:elite|plus)(?:\s+x\d[\w-]*)?",
]


def _cand(attr, value, raw, score, start=None, end=None) -> Candidate:
    return Candidate(attribute=attr, value=value, raw=raw.strip(), source="rules", score=score,
                     span=(start, end) if start is not None else None)


def _in_range(x: float, kind: str, category: str) -> bool:
    lo, hi = CATEGORY_RANGES.get((category, kind), RANGES.get(kind, (float("-inf"), float("inf"))))
    return lo <= x <= hi


def _numbers(text: str, unit_re: str) -> list[tuple[float, re.Match]]:
    return [(to_float(m.group(1)), m) for m in re.finditer(NUM + r"\s*" + unit_re + r"(?![a-zа-я])", text)]


def _memory_pair(text: str) -> tuple[float, float, re.Match] | None:
    """«8/256 ГБ», «8+256GB», «16 / 512 Гб» -> (RAM, storage)."""
    m = re.search(r"(\d{1,2})\s*[/+]\s*(\d{2,4}|\d)\s*(" + GB + "|" + TB + r")(?![a-zа-я])", text)
    if not m:
        return None
    second = to_float(m.group(2)) * (1024 if re.fullmatch(TB, m.group(3)) else 1)
    return to_float(m.group(1)), second, m


def extract_number(attr: str, spec: AttrSpec, title: str, text: str, category: str) -> list[Candidate]:
    out: list[Candidate] = []
    kind = spec.kind or ""
    for field_text, bonus in ((title, 0.1), (text, 0.0)):
        if kind in {"memory", "ram", "storage_drive"}:
            pair = _memory_pair(field_text)
            if pair:
                ram, storage, m = pair
                val = ram if kind == "ram" else storage
                if _in_range(val, kind, category):
                    out.append(_cand(attr, fmt_number(val), m.group(0), 0.85 + bonus, m.start(), m.end()))
                continue
            hint = RAM_HINT if kind == "ram" else SSD_HINT
            for unit_re, mult in ((GB, 1), (TB, 1024)):
                # «16 ГБ RAM» / «RAM 16 ГБ» / «SSD 512 ГБ»
                # «16 ГБ SSD 512 ГБ»: подсказка относится к следующему числу, поэтому NUM-hint не должен упираться в число
                for pat in (NUM + r"\s*" + unit_re + r"\s*(?:\w+\s){0,1}" + hint + r"(?!\s*:?\s*\d)",
                            hint + r"[\s:]*(?:\w+\s){0,1}" + NUM + r"\s*" + unit_re):
                    for m in re.finditer(pat, field_text):
                        val = to_float(m.group(1)) * mult
                        if _in_range(val, kind, category):
                            out.append(_cand(attr, fmt_number(val), m.group(0), 0.85 + bonus, m.start(), m.end()))
            if kind in {"ram", "storage_drive"} and not out:
                # Ноутбук: «8 ГБ 256 ГБ» без подписей — меньшее (<=64) это ОЗУ, большее (>=128) накопитель
                vals = [(v, m) for v, m in _numbers(field_text, GB)] + [(v * 1024, m) for v, m in _numbers(field_text, TB)]
                small = [(v, m) for v, m in vals if v <= 64]
                big = [(v, m) for v, m in vals if v >= 128]
                pick = small if kind == "ram" else big
                if len(pick) == 1 and small and big:
                    v, m = pick[0]
                    out.append(_cand(attr, fmt_number(v), m.group(0), 0.7 + bonus, m.start(), m.end()))
            if kind == "memory" and not out:
                # Смартфон: одиночный объём «256 ГБ» — скорее всего ПЗУ, если он не помечен как RAM.
                for val, m in [(v * 1, m) for v, m in _numbers(field_text, GB)] + [(v * 1024, m) for v, m in _numbers(field_text, TB)]:
                    ctx = field_text[max(0, m.start() - 25): m.end() + 25]
                    if re.search(RAM_HINT, ctx) or not _in_range(val, kind, category) or val < 16:
                        continue
                    out.append(_cand(attr, fmt_number(val), m.group(0), 0.7 + bonus, m.start(), m.end()))
        else:
            parsed_any = False
            for unit_re_match in re.finditer(NUM + r"\s*(?:\"|''|”|″|дюйм\w*|inch|мач|mah|ма·ч|ма\*ч)", field_text):
                parsed = parse_number(unit_re_match.group(0), kind)
                if parsed and _in_range(parsed[0], kind, category):
                    parsed_any = True
                    out.append(_cand(attr, parsed[0], unit_re_match.group(0), 0.85 + bonus,
                                     unit_re_match.start(), unit_re_match.end()))
            if kind == "diagonal" and not parsed_any:
                for m in re.finditer(r"(?:диагональ\w*|экран\w*|дисплей\w*)[^\d]{0,15}" + NUM, field_text):
                    val = to_float(m.group(1))
                    if _in_range(val, kind, category):
                        out.append(_cand(attr, fmt_number(val), m.group(0), 0.7 + bonus, m.start(), m.end()))
    return out


def extract_cpu(attr: str, title: str, text: str) -> list[Candidate]:
    out = []
    for field_text, bonus in ((title, 0.05), (text, 0.0)):
        for pat in CPU_PATTERNS:
            m = re.search(pat, field_text)
            if m:
                out.append(_cand(attr, normalize_cpu(m.group(0)), m.group(0), 0.85 + bonus, m.start(), m.end()))
                break
    return out


def extract_model(attr: str, card: Card, tax: Taxonomy, category: CategorySpec) -> list[Candidate]:
    """Модель = кусок названия после бренда и до памяти/цвета. Эвристика — низкая уверенность, дальше LLM/ревью."""
    title = card.title
    brands = tax.dictionaries.get("brands")
    if not brands:
        return []
    hits = brands.find_all(title)
    if not hits:
        return []
    _, _, _, end = hits[0]
    rest = title[end:]
    stop = re.search(r"(\d+\s*[/+]\s*\d+|\d+\s*(?:гб|gb|тб|tb)|,|\(|\d+(?:[.,]\d)?\s*(?:\"|дюйм))", rest, re.I)
    raw = rest[: stop.start()] if stop else rest
    colors = tax.dictionaries.get("colors")
    if colors:
        for _, _, s, _ in colors.find_all(raw):
            raw = raw[:s]
            break
    raw = re.sub(r"(?i)\b(смартфон|ноутбук|телефон|smartphone|laptop)\b", "", raw).strip(" -–,")
    raw = re.sub(r"\s+", " ", raw)
    if not raw or len(raw) > 40:
        return []
    # похоже на модель (есть цифра) — средняя уверенность; иначе только на проверку
    return [_cand(attr, raw, raw, 0.72 if re.search(r"\d", raw) else 0.55)]


def extract(card: Card, category: CategorySpec, tax: Taxonomy) -> list[Candidate]:
    title, text = norm(card.title), norm(card.title + "\n" + card.description)
    out: list[Candidate] = []
    for attr, spec in category.attributes.items():
        if spec.type == "enum" and spec.dictionary in tax.dictionaries:
            d = tax.dictionaries[spec.dictionary]
            title_hits = d.find_all(title)
            for canon, raw, s, e in title_hits[:1]:
                out.append(_cand(attr, canon, raw, 0.9, s, e))
            body_hits = d.find_all(norm(card.description))
            distinct = {h[0] for h in body_hits}
            if body_hits and not title_hits:
                # несколько разных значений в описании (например, перечисление расцветок) — снижаем уверенность
                score = 0.75 if len(distinct) == 1 else 0.45
                for canon in list(dict.fromkeys(h[0] for h in body_hits))[:3]:
                    raw = next(h[1] for h in body_hits if h[0] == canon)
                    out.append(_cand(attr, canon, raw, score))
        elif spec.type == "number":
            out += extract_number(attr, spec, title, text, category.name)
        elif spec.kind == "cpu":
            out += extract_cpu(attr, title, text)
        elif attr == "model":
            out += extract_model(attr, card, tax, category)
    return out
