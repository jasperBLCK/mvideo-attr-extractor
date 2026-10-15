"""Distant supervision для NER: эталон каталога М.Видео -> спаны в тексте карточки -> BIO-разметка.

Характеристики YML-фида заполняет контент-команда М.Видео (human-curated). Если эталонное значение встречается
в тексте карточки, этот фрагмент — бесплатная разметка токенов: «256 ГБ», «8/256», «6,7"», «черная», «Samsung»,
«Galaxy S24». Так из train-сплита (193 тыс. карточек с эталоном) получаются десятки тысяч размеченных примеров без
ручного труда — weak supervision из слайда 14 кейса, только метки даёт не правило, а каталог.

Сопоставление учитывает нормализацию (normalize.py, eval/gold.py): число с единицей в любом написании
(256 ГБ ~ 256GB ~ 256Гб, 1 ТБ = 1024 ГБ, 6,7 дюйма ~ 6.7", 5 000 мАч), пары памяти «12/256», голое число после
названия атрибута («Диагональ: 6.7»), цвета и прочие справочники с окончаниями и по-английски, бренд и модель
строкой (без регистра, пробелов и дефисов), процессор вместе с семейством. Каждый спан проверяется функцией
сравнения из метрик (eval.metrics): если NER выдаст ровно этот фрагмент, значение будет засчитано.

Неоднозначность — консервативно (маска = токены не участвуют в loss: не учим ни «атрибут», ни «не атрибут»):
- в названии другое значение того же атрибута -> атрибут этой карточки не размечается (маска);
- в одном фрагменте описания и эталонное, и другое значение («доступен в черном и белом») -> маска фрагмента;
- эталона в каталоге нет, а значение-кандидат в тексте есть (каталог неполный) -> маска, а не «O»;
- эталонное число без единицы и без названия атрибута рядом («185x60x65») -> маска;
- один фрагмент подходит двум атрибутам (материал «нержавеющая сталь» и цвет) -> маска.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..eval.gold import MEMORY_PAIR, _num_re
from ..eval.metrics import is_match, match_key
from ..normalize import UNIT_PATTERNS, compact, fmt_number
from ..schema import Card
from ..taxonomy import AttrSpec, CategorySpec, Taxonomy, norm
from .ner import CPU_ANY, NerExample, build_view, dict_regex, ntext, segments, span_value, view_scorer

MEMORY_KINDS = frozenset({"memory", "ram", "storage_drive"})
NUM_X = r"(?<![\d.,])(\d{1,3}(?: \d{3})+(?![\d.,])|\d+(?:[.,]\d+)?)"  # «5 000», «6,7», «256»
NOT_UNIT_AFTER = r"(?!\s*-?\s*[a-zа-я\"″”'%°/])"
_CARD = r"(?:\bmicro\s?sd\b|\bsd\b|\btf\b|\bкарт\w*|\bрасширени\w*|\bдо\b)[^\n]{0,15}$"
SUSPICIOUS = {  # контекст, в котором совпадение с эталоном — скорее другое значение: маска вместо метки
    "memory": re.compile(_CARD),  # «карты памяти до 1 ТБ»
    "storage_drive": re.compile(_CARD),
}
# «6,7-дюймовый», «1,7-литровый», «8-часовой»: единица приросла к слову — спан только число
HYPHEN_UNITS = {"diagonal": "дюйм", "volume_l": "литр", "time_h": "час", "time_min": "минут", "refresh_hz": "герц|гц",
                "power_w": "ватт", "memory": "гигабайт|гб", "ram": "гигабайт|гб", "storage_drive": "гигабайт|гб|терабайт"}
SLASH_BEFORE = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)?\s*/\s*$")  # «50/60 Гц», «220/240 В» — перечисление значений
# память видеокарты («RTX 4060 8GB», «8 ГБ GDDR6») — не ОЗУ и не накопитель
GPU_BEFORE = re.compile(r"(?:rtx|gtx|geforce|radeon|arc|quadro|видеокарт\w*|видеопамят\w*|графическ\w*|graphics)"
                        r"[\w\s-]{0,12}\(?\s*$")
GPU_AFTER = re.compile(r"\s*(?:g?ddr\d?\w*|видеопамят\w*|vram)")
JUNK_MODEL = re.compile(r"[\d.,xх×*]+(?:см|мм|м|л|мл|кг|г|гб|тб|шт|вт|cm|mm)?")
CPU_VENDORS = ("intel", "amd", "apple", "qualcomm", "mediatek")
CPU_TAIL = re.compile(r"[\s\-]*(?:pro|max|ultra|plus|elite|\d{3,5}[a-z]{0,3})(?![a-z0-9])")


@dataclass
class GoldAttr:
    """Эталон одного атрибута карточки: допустимые канонические значения + сырое значение фида."""

    values: list
    raw: str | None = None
    extractable: bool | None = None


@dataclass(frozen=True)
class Mention:
    start: int
    end: int
    how: str  # unit | pair | ctx | dict | word | vendor | model | cpu | kw
    suspicious: bool = False  # «до 1 ТБ» карты памяти, «50/60 Гц»: в маску, не в метки


@dataclass
class CardLabels:
    spans: list[tuple[int, int, str]]  # позитивные спаны (start, end, атрибут) в координатах full_text
    masks: list[tuple[int, int]]  # фрагменты, исключённые из loss
    status: dict[str, str]  # атрибут -> labeled | ambiguous | absent | nogold


# ------------------------------------------------------------------ упоминания значений ----------------------------
_RE_CACHE: dict[tuple, re.Pattern | None] = {}


def _cached(key: tuple, build: Callable[[], re.Pattern | None]) -> re.Pattern | None:
    if key not in _RE_CACHE:
        _RE_CACHE[key] = build()
    return _RE_CACHE[key]


def _alias_num_re(spec: AttrSpec) -> re.Pattern | None:
    """Голое число вскоре после названия атрибута: «Диагональ экрана: 6.7», «аккумулятор 5000»."""
    def build():
        al = sorted({norm(a) for a in [*spec.aliases, spec.title] if len(norm(a)) >= 4}, key=len, reverse=True)
        if not al:
            return None
        return re.compile(r"(?<![a-zа-я])(?:" + "|".join(re.escape(a) for a in al) + r")[^\d\n;]{0,25}?"
                          + NUM_X + NOT_UNIT_AFTER)
    return _cached(("alias", spec.name, spec.title, tuple(spec.aliases)), build)


def number_mentions(tn: str, spec: AttrSpec) -> list[Mention]:
    kind = spec.kind or ""
    out: list[Mention] = []
    for unit_re, _ in UNIT_PATTERNS.get(kind, []):
        pat = _cached(("unit", unit_re), lambda u=unit_re: re.compile(NUM_X + r"\s*(?:" + u + r")(?![a-zа-я])"))
        for m in pat.finditer(tn):
            num, s = m.group(1), m.start(1)
            starts = [s] + ([s + num.rindex(" ") + 1] if " " in num else [])  # «8 256 ГБ» — и «8256», и «256»
            for st in starts:
                out.append(Mention(st, m.end(), "unit", _suspicious(tn, st, kind, m.end())))
    if kind in HYPHEN_UNITS:
        pat = _cached(("hyph", kind), lambda: re.compile(NUM_X + r"-(?:" + HYPHEN_UNITS[kind] + r")[а-я]*"))
        for m in pat.finditer(tn):
            out.append(Mention(m.start(1), m.end(1), "unit", _suspicious(tn, m.start(1), kind)))
    if kind in MEMORY_KINDS:
        for m in MEMORY_PAIR.finditer(tn):
            if kind == "ram":
                out.append(Mention(m.start(1), m.end(1), "pair", _suspicious(tn, m.start(1), kind)))
            else:
                end = m.end(3) if m.group(3) else m.end(2)
                out.append(Mention(m.start(2), end, "pair", _suspicious(tn, m.start(1), kind, end)))
    ctx = _alias_num_re(spec)
    if ctx is not None:
        for m in ctx.finditer(tn):
            out.append(Mention(m.start(1), m.end(1), "ctx", _suspicious(tn, m.start(1), kind)))
    return out


def _suspicious(tn: str, start: int, kind: str, end: int | None = None) -> bool:
    before = tn[max(0, start - 30):start]
    if kind in MEMORY_KINDS:
        if GPU_BEFORE.search(before) or (end is not None and GPU_AFTER.match(tn, end)):
            return True
    elif SLASH_BEFORE.search(before):
        return True
    pat = SUSPICIOUS.get(kind)
    return bool(pat and pat.search(before))


def enum_mentions(tn: str, spec: AttrSpec, tax: Taxonomy) -> list[Mention]:
    d = tax.dictionaries.get(spec.dictionary or "")
    pat = dict_regex(d) if d is not None and spec.dictionary != "brands" else None
    return [Mention(m.start(), m.end(), "dict") for m in pat.finditer(tn)] if pat is not None else []


def word_mentions(tn: str, value: str) -> list[Mention]:
    """Значение эталона вне справочника — как слово/фраза (пробелы и дефисы — любые)."""
    words = [re.escape(w) for w in re.split(r"[\s\-]+", norm(str(value))) if w]
    if not words:
        return []
    pat = re.compile(r"(?<![a-zа-я0-9])" + r"[\s\-]*".join(words) + r"(?![a-zа-я0-9])")
    return [Mention(m.start(), m.end(), "word") for m in pat.finditer(tn)]


def vendor_mentions(tn: str, vendor: str) -> list[Mention]:
    return [Mention(m.start, m.end, "vendor") for m in word_mentions(tn, vendor)]


def brand_dict_mentions(tn: str, tax: Taxonomy) -> list[Mention]:
    """Упоминания брендов справочника (для конфликтов: «чехол для Samsung» при эталоне Borasco)."""
    d = tax.dictionaries.get("brands")
    finder = getattr(d, "find_all_raw", None)
    if finder is None:
        return []
    try:
        return [Mention(s, e, "dict") for _, _, s, e in finder(tn)]
    except Exception:  # noqa: BLE001 — API справочника брендов меняется; конфликты бренда — не критичны
        return []


def model_mentions(tn: str, strings: list[str]) -> list[Mention]:
    """Строка модели посимвольно (буквы/цифры/«+»), между символами — любые пробелы, дефисы, точки, слэши:
    «RB33A3440SA/WT» ~ «RB33A3440SAWT», «Galaxy S24-Ultra» ~ «galaxy s24 ultra»."""
    out, seen = [], set()
    for s in sorted({str(x) for x in strings if x}, key=len, reverse=True):
        chars = [c for c in norm(s) if c.isalnum() or c == "+"]
        key = "".join(chars)
        if len(chars) < 2 or len(chars) > 60 or key in seen:
            continue
        seen.add(key)
        body = r"[ \t\-_./]*".join(r"\+" if c == "+" else re.escape(c) for c in chars)
        pat = re.compile(r"(?<![a-zа-я0-9])" + body + r"(?![a-zа-я0-9])")
        out += [Mention(m.start(), m.end(), "model") for m in pat.finditer(tn)]
    return out


def cpu_mentions(tn: str, values: list) -> list[Mention]:
    """Процессор эталона в любом написании: «Intel Core i5-12450H» ~ «Core i5 12450H» ~ «i5-12450H»; эталон-семейство
    («Intel Core i5», «AMD Ryzen 5») — вместе с номером модели, если он следует («Ryzen 5 7520U»)."""
    out = []
    for v in values:
        toks = re.findall(r"[a-z0-9]+", norm(str(v)))
        while toks and toks[0] in CPU_VENDORS:
            toks.pop(0)
        core_opt = len(toks) > 1 and toks[0] == "core"
        if core_opt:
            toks.pop(0)
        if not toks:
            continue
        family = not any(len(t) >= 4 and any(ch.isdigit() for ch in t) for t in toks) and toks[0] not in (
            "m1", "m2", "m3", "m4", "m5")
        body = r"[\s\-_]*".join(map(re.escape, toks))
        suffix = r"(?:[\s\-]*\d{3,5}[a-z]{1,3}|-\d{4,5})?" if family else ""
        pat = re.compile(r"(?<![a-z0-9])(?:(?:" + "|".join(CPU_VENDORS) + r")[\s\-]*)?" + (r"(?:core[\s\-]*)?" if core_opt
                         else "") + body + suffix + r"(?![a-z0-9])")
        out += [Mention(m.start(), m.end(), "cpu") for m in pat.finditer(tn)]
    out += [Mention(m.start(), m.end(), "cpu") for m in CPU_ANY.finditer(tn)]
    return out


def kw_mentions(tn: str, spec: AttrSpec) -> list[Mention]:
    """Признак bool («инверторный», «Inverter») — слово целиком; «неинверторный»/«не инверторный» — suspicious."""
    out = []
    for k in spec.keywords:
        for m in re.finditer(re.escape(norm(k)), tn):
            s, e = m.start(), m.end()
            while s > 0 and tn[s - 1].isalnum():
                s -= 1
            while e < len(tn) and (tn[e].isalnum() or tn[e] == "-" and e + 1 < len(tn) and tn[e + 1].isalnum()):
                e += 1
            neg = tn[s:m.start()].startswith("не") or bool(re.search(r"(?:\bне|\bбез)\s+$", tn[max(0, s - 6):s]))
            out.append(Mention(s, e, "kw", neg))
    return out


def mentions(tn: str, spec: AttrSpec, gold: GoldAttr | None, tax: Taxonomy) -> list[Mention]:
    """Все кандидаты значения атрибута в тексте (эталонные и прочие)."""
    if spec.type == "number":
        return number_mentions(tn, spec)
    if spec.type == "bool":
        return kw_mentions(tn, spec)
    if spec.type == "enum" and spec.dictionary == "brands":
        out = brand_dict_mentions(tn, tax)
        if gold is not None:
            for v in gold.values:
                out += vendor_mentions(tn, str(v))
        return out
    if spec.type == "enum":
        out = enum_mentions(tn, spec, tax)
        d = tax.dictionaries.get(spec.dictionary or "")
        for v in (gold.values if gold is not None else []):
            if d is None or str(v) not in d.canon:  # эталон вне справочника — ищем как слово
                out += word_mentions(tn, str(v))
        return out
    if spec.kind == "cpu":
        return cpu_mentions(tn, gold.values if gold is not None else [])
    if spec.kind == "model":
        return model_mentions(tn, [*(gold.values if gold is not None else []), *([gold.raw] if gold and gold.raw else [])])
    return word_mentions(tn, str(gold.values[0])) if gold is not None and gold.values else []


# ------------------------------------------------------------------ разметка карточки ------------------------------
def _overlaps(s: int, e: int, spans) -> bool:
    return any(s < b and a < e for a, b, *_ in spans)


def _longest_first(ms: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Непересекающиеся спаны, длинные приоритетнее."""
    out: list[tuple[int, int]] = []
    for s, e in sorted(set(ms), key=lambda x: (-(x[1] - x[0]), x[0])):
        if not _overlaps(s, e, out):
            out.append((s, e))
    return sorted(out)


def _trim(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and not text[s].isalnum():
        s += 1
    while e > s and not (text[e - 1].isalnum() or text[e - 1] == "+"):
        e -= 1
    return s, e


def label_card(text: str, title_len: int, gold: dict[str, GoldAttr], category: CategorySpec, tax: Taxonomy,
               attrs: set[str] | None = None) -> CardLabels:
    """Позитивные спаны и маски для одной карточки (text = Card.full_text(), название — text[:title_len])."""
    tn = ntext(text)
    segs = segments(text, title_len)
    seg_starts = [s for s, _ in segs]

    def seg_of(pos: int) -> int:
        return bisect.bisect_right(seg_starts, pos) - 1

    bg = gold.get("brand")
    brand = str(bg.values[0]) if bg is not None and bg.values else None
    pos: dict[str, list[tuple[int, int]]] = {}
    neg: dict[str, list[tuple[int, int]]] = {}
    masks: list[tuple[int, int]] = []
    status: dict[str, str] = {}
    exact_brand: list[tuple[int, int]] = []
    for a, spec in category.attributes.items():
        if attrs is not None and a not in attrs:
            continue
        g = gold.get(a)
        if g is not None and spec.kind == "model" and all(JUNK_MODEL.fullmatch(compact(v)) for v in g.values):
            g = None  # «30x30см» в поле model каталога — размер, а не модель
        ms = mentions(tn, spec, g, tax)
        if g is None:  # каталог неполный: упоминания возможного значения — в маску, а не в «O»
            masks += [(m.start, m.end) for m in ms]
            if ms:
                status[a] = "nogold"
            continue
        gk = [k for k in (match_key(v, spec, tax, brand) for v in g.values) if k is not None]
        p, n = [], []
        for m in ms:
            res = span_value(text[m.start:m.end], spec, tax)
            pk = match_key(res[0], spec, tax, brand) if res is not None else None
            ok = pk is not None and bool(gk) and is_match(pk, gk, spec)
            if m.suspicious or (ok and a == "brand" and m.how != "vendor"):
                masks.append((m.start, m.end))  # синоним бренда («MacBook» -> Apple) — не метка бренда
            elif ok:
                p.append((m.start, m.end))
                if m.how == "vendor":
                    exact_brand.append((m.start, m.end))
            elif res is not None:
                n.append((m.start, m.end))
        if spec.kind == "cpu":  # «M4» при эталоне M4, а в тексте «M4 Pro» — не метка, а маска
            p2 = []
            for s, e in p:
                t = CPU_TAIL.match(tn, e)
                if t:
                    res = span_value(text[s:t.end()], spec, tax)
                    pk = match_key(res[0], spec, tax, brand) if res else None
                    if not (pk is not None and is_match(pk, gk, spec)):
                        masks.append((s, t.end()))
                        continue
                p2.append((s, e))
            p = p2
        pos[a] = sorted(set(p)) if spec.kind == "model" else _longest_first(p)  # модель — после обрезки
        neg[a] = [x for x in _longest_first(n) if not _overlaps(*x, pos[a])]
        if spec.type == "number":  # эталонное число без единицы и без названия атрибута («185x60x65») — маска
            for v in g.values:
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if fv != int(fv) or fv >= 10:
                    masks += [(m.start(), m.end()) for m in _num_re(fmt_number(fv)).finditer(tn)]

    # модель: спан режется по соседним атрибутам («Galaxy S26 Ultra 16/1Tb violet» -> «Galaxy S26 Ultra»)
    if pos.get("model"):
        others = sorted((s, e, a) for a, sp in pos.items() if a != "model" for s, e in sp
                        if not (a == "brand" and (s, e) not in exact_brand))
        spec = category.attributes["model"]
        g = gold["model"]
        gk = [k for k in (match_key(v, spec, tax, brand) for v in g.values) if k is not None]
        cut, failed = [], []
        for s, e in pos["model"]:
            for os_, oe, _ in others:
                if oe <= s or os_ >= e:
                    continue
                if os_ <= s:
                    s = oe
                else:
                    e = min(e, os_)
            s, e = _trim(text, s, e)
            if e - s >= 2:
                pk = match_key(text[s:e], spec, tax, brand)
                (cut if pk is not None and is_match(pk, gk, spec) else failed).append((s, e))
        pos["model"] = _longest_first(cut)
        masks += [x for x in failed if not _overlaps(*x, pos["model"])]
        if "brand" in pos:  # бренд-синоним внутри модели («iPhone» в «iPhone 15 Pro») — модель важнее
            pos["brand"] = [x for x in pos["brand"] if not _overlaps(*x, pos["model"])]

    # один фрагмент — два атрибута: бренд внутри другого значения («Apple» в «Apple M3») уступает ему,
    # иначе («нержавеющая сталь»: материал и цвет) — маска обоих
    flat = [(s, e, a) for a, sp in pos.items() for s, e in sp]
    clash = {(s, e, a) for s, e, a in flat for s2, e2, a2 in flat if a2 != a and s < e2 and s2 < e}
    for s, e, a in clash:
        pos[a] = [x for x in pos[a] if x != (s, e)]
        if a != "brand" and not any(a2 == "brand" and s < e2 and s2 < e for s2, e2, a2 in clash):
            masks.append((s, e))
    for s, e, a in clash:  # значение, конфликтовавшее только с брендом, возвращаем
        if a != "brand" and all(a2 == "brand" for s2, e2, a2 in clash if a2 != a and s < e2 and s2 < e):
            pos[a].append((s, e))

    spans: list[tuple[int, int, str]] = []
    final = [(s, e) for sp in pos.values() for s, e in sp]
    for a, p in pos.items():
        spec = category.attributes[a]
        n = [x for x in neg[a] if not _overlaps(*x, final)]
        pt = [x for x in p if x[0] < title_len]
        nt = [x for x in n if x[0] < title_len]
        if a == "model":
            keep = p
            for tok_ in _model_tokens(gold["model"], brand):  # части модели вне спанов («s26» в описании) — маска
                masks += [(m.start(), m.end()) for m in re.finditer(r"(?<![a-zа-я0-9])" + re.escape(tok_)
                                                                     + r"(?![a-zа-я0-9])", tn)
                          if not _overlaps(m.start(), m.end(), p)]
        elif a == "brand":
            if nt and not pt:  # в названии другой бренд, а эталонного нет — не размечаем
                masks += p + n
                keep = []
            else:
                keep = p
        elif nt:  # в названии другое значение того же атрибута
            masks += p + n
            keep = []
        else:
            keep = list(pt)
            rest_p = defaultdict(list)
            rest_n = defaultdict(list)
            for x in p:
                if x[0] >= title_len:
                    rest_p[seg_of(x[0])].append(x)
            for x in n:
                if x[0] >= title_len:
                    rest_n[seg_of(x[0])].append(x)
            for sid in set(rest_p) | set(rest_n):
                if rest_p[sid] and rest_n[sid]:
                    masks += rest_p[sid] + rest_n[sid]
                elif rest_p[sid]:
                    keep += rest_p[sid]
                elif not p:  # эталона в тексте нет, а другое значение есть — не учим «O»
                    masks += rest_n[sid]
        status[a] = "labeled" if keep else ("ambiguous" if p else "absent")
        spans += [(s, e, a) for s, e in keep]
    spans.sort()
    masks = [m for m in _merge(masks) if not any(s <= m[0] and m[1] <= e for s, e, _ in spans)]
    return CardLabels(spans, masks, status)


def _model_tokens(g: GoldAttr, brand: str | None) -> list[str]:
    """Отличительные части модели («s26», «5180», «icbnsd») — их упоминания вне размеченных спанов маскируются."""
    b = compact(brand) if brand else ""
    toks = set()
    for v in [*g.values, g.raw or ""]:
        for t in re.findall(r"[a-zа-я0-9]+", norm(str(v))):
            if t != b and len(t) >= 2 and (any(c.isdigit() for c in t) or (len(t) >= 4 and t.isascii())):
                toks.add(t)
    return sorted(toks)


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


# ------------------------------------------------------------------ датасет ----------------------------------------
def calib_part(sku: str, share: float) -> bool:
    """Отложенная часть train для калибровки уверенности (детерминированно по sku, независимо от сплита)."""
    return int(hashlib.md5(f"ner-calib:{sku}".encode()).hexdigest(), 16) % 1000 < share * 1000


def load_gold(processed: Path, split: str = "train") -> tuple[dict[str, dict[str, GoldAttr]], dict[str, str]]:
    """gold.parquet -> ({sku: {атрибут: GoldAttr}}, {sku: категория таксономии})."""
    import pyarrow.parquet as pq

    cols = ["sku_id", "category", "attribute", "alternatives", "raw_value", "extractable"]
    t = pq.read_table(processed / "gold.parquet", columns=cols, filters=[("split", "=", split)])
    gold: dict[str, dict[str, GoldAttr]] = defaultdict(dict)
    cat: dict[str, str] = {}
    d = t.to_pydict()
    for sku, c, a, alts, raw, ext in zip(d["sku_id"], d["category"], d["attribute"], d["alternatives"],
                                         d["raw_value"], d["extractable"], strict=True):
        gold[sku][a] = GoldAttr(json.loads(alts), raw if isinstance(raw, str) else None, bool(ext))
        cat[sku] = c
    return dict(gold), cat


def load_cards(processed: Path, ids: list[str]) -> list[Card]:
    import pyarrow.parquet as pq

    cols = ["sku_id", "title", "description_clean", "seller_attrs", "category_id", "category_title", "seller"]
    out = []
    for i in range(0, len(ids), 20_000):
        t = pq.read_table(processed / "cards.parquet", columns=cols, filters=[("sku_id", "in", ids[i:i + 20_000])])
        for r in t.to_pylist():
            sa = r["seller_attrs"]
            out.append(Card(id=r["sku_id"], title=r["title"] if isinstance(r["title"], str) else "",
                            description=r["description_clean"] if isinstance(r["description_clean"], str) else "",
                            seller_attrs=json.loads(sa) if isinstance(sa, str) and sa else {},
                            category=r["category_title"] if isinstance(r["category_title"], str) else None,
                            category_id=r["category_id"] if isinstance(r["category_id"], str) else None,
                            seller=r["seller"] if isinstance(r["seller"], str) else None))
    order = {s: i for i, s in enumerate(ids)}
    return sorted(out, key=lambda c: order[c.id])


def select_ids(cats: dict[str, str], n_generic: int, calib_share: float, max_examples: int | None, seed: int = 13,
               n_calib_generic: int = 2000) -> tuple[list[str], list[str]]:
    """Обучение: все карточки категорий таксономии + выборка generic (бренд/модель/цвет длинного хвоста);
    калибровка: отложенная доля тех же категорий + n_calib_generic generic. Детерминированно (seed)."""
    rnd = random.Random(seed)
    tax_ids = sorted(s for s, c in cats.items() if c != "generic")
    gen_ids = sorted(s for s, c in cats.items() if c == "generic")
    tr_tax = [s for s in tax_ids if not calib_part(s, calib_share)]
    cal_tax = [s for s in tax_ids if calib_part(s, calib_share)]
    gen_tr = [s for s in gen_ids if not calib_part(s, calib_share)]
    gen_cal = [s for s in gen_ids if calib_part(s, calib_share)]
    for ids in (tr_tax, gen_tr, gen_cal):
        rnd.shuffle(ids)
    if max_examples is not None:
        tr_tax = tr_tax[:max_examples]
        n_generic = max(0, min(n_generic, max_examples - len(tr_tax)))
    train = tr_tax + gen_tr[:n_generic]
    rnd.shuffle(train)
    return train, cal_tax + gen_cal[:n_calib_generic]


@dataclass
class Dataset:
    examples: list[NerExample]
    attributes: list[str]
    stats: dict = field(default_factory=dict)


def build_examples(cards: list[Card], gold: dict[str, dict[str, GoldAttr]], tax: Taxonomy,
                   view_chars: int | None = None, log: Callable[[str], None] = print) -> tuple[list[NerExample], dict]:
    """Карточки + эталон -> примеры NER (вид карточки, спаны и маски в его координатах) + статистика по атрибутам."""
    from .ner import VIEW_CHARS

    t0 = time.time()
    st: dict[str, Counter] = defaultdict(Counter)
    examples = []
    totals = Counter()
    for i, card in enumerate(cards):
        cname = tax.category_of_id(card.category_id) or "generic"
        cat = tax.categories[cname]
        text = card.full_text()
        g = gold.get(card.id, {})
        lab = label_card(text, len(card.title), g, cat, tax)
        view = build_view(text, len(card.title), view_scorer(cat, tax), view_chars or VIEW_CHARS)
        spans, masks = [], []
        for s, e, a in lab.spans:
            v = view.from_orig(s, e)
            if v is not None:
                spans.append((v[0], v[1], a))
            st[a]["spans_total"] += 1
            st[a]["spans_title"] += s < len(card.title)
            st[a]["spans_in_view"] += v is not None
        for s, e in lab.masks:
            v = view.from_orig(max(s, 0), e)
            if v is not None:
                masks.append(v)
        for a, ga in g.items():
            if a not in cat.attributes:
                continue
            st[a]["gold"] += 1
            st[a]["gold_extractable"] += bool(ga.extractable)
            stt = lab.status.get(a, "absent")
            st[a][stt] += 1
            st[a]["labeled_extractable"] += stt == "labeled" and bool(ga.extractable)
        for a, stt in lab.status.items():
            if stt == "nogold":
                st[a]["nogold_masked"] += 1
        totals["chars"] += len(text)
        totals["view_chars"] += len(view.text)
        totals["truncated"] += len(view.text) < len(text)
        totals["masks"] += len(masks)
        totals[f"cat:{cname}"] += 1
        examples.append(NerExample(prefix=cat.title, text=view.text, spans=spans, masks=masks, id=card.id,
                                   allowed=sorted(cat.attributes)))
        if (i + 1) % 10_000 == 0:
            log(f"distant: {i + 1}/{len(cards)} карточек, {time.time() - t0:.0f} с")
    stats = {"cards": len(cards), "seconds": round(time.time() - t0, 1),
             "truncated_share": round(totals["truncated"] / max(1, len(cards)), 4),
             "mean_chars": round(totals["chars"] / max(1, len(cards))),
             "mean_view_chars": round(totals["view_chars"] / max(1, len(cards))),
             "categories": {k[4:]: v for k, v in sorted(totals.items()) if k.startswith("cat:")},
             "attributes": {a: dict(c) | {
                 "label_rate": round(c["labeled"] / c["gold"], 4) if c["gold"] else None,
                 "label_rate_extractable": round(c["labeled_extractable"] / c["gold_extractable"], 4)
                 if c["gold_extractable"] else None} for a, c in sorted(st.items())}}
    return examples, stats


def choose_attributes(stats: dict, min_spans: int = 150) -> list[str]:
    """Метки — только атрибуты с достаточным числом размеченных спанов (остальные правила закрывают и так)."""
    return sorted(a for a, c in stats["attributes"].items() if c.get("spans_in_view", 0) >= min_spans)


def restrict(examples: list[NerExample], attrs: list[str]) -> list[NerExample]:
    """Спаны атрибутов вне набора меток -> маска (это значения, просто без своей метки)."""
    keep = set(attrs)
    out = []
    for ex in examples:
        drop = [(s, e) for s, e, a in ex.spans if a not in keep]
        out.append(NerExample(ex.prefix, ex.text, [x for x in ex.spans if x[2] in keep], ex.masks + drop, ex.id,
                              ex.allowed))
    return out


def bio_preview(ex: NerExample, tok=None, limit: int = 60) -> list[tuple[str, str]]:
    """(токен, метка) для документации: первые limit токенов примера (без токенизатора — по пробелам)."""
    from .ner import bio_labels

    if tok is None:
        offs = [(m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", ex.text)]
    else:
        offs = [tuple(o) for o in tok(ex.text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]]
    labs = bio_labels(offs, ex.spans, ex.masks)
    return [(ex.text[s:e], lab or "·") for (s, e), lab in list(zip(offs, labs, strict=True))[:limit]]


def stats_table(stats: dict) -> str:
    """Текстовая таблица статистики меток по атрибутам."""
    head = (f"{'атрибут':<14}{'эталон':>8}{'извл.':>8}{'размеч.':>9}{'% извл.':>8}{'неоднозн.':>10}{'спанов':>8}"
            f"{'в назв.':>8}{'в виде':>8}{'маска∅':>8}")
    lines = [head, "-" * len(head)]
    for a, c in stats["attributes"].items():
        rate = c.get("label_rate_extractable")
        lines.append(f"{a:<14}{c.get('gold', 0):>8}{c.get('gold_extractable', 0):>8}{c.get('labeled', 0):>9}"
                     f"{(f'{rate:.2f}' if rate is not None else '—'):>8}{c.get('ambiguous', 0):>10}"
                     f"{c.get('spans_total', 0):>8}{c.get('spans_title', 0):>8}{c.get('spans_in_view', 0):>8}"
                     f"{c.get('nogold_masked', 0):>8}")
    return "\n".join(lines)


def dataset_from_processed(processed: Path, tax: Taxonomy, *, n_generic: int = 24_000, max_examples: int | None = None,
                           calib_share: float = 0.1, seed: int = 13, view_chars: int | None = None,
                           min_spans: int = 150, log: Callable[[str], None] = print,
                           ) -> tuple[Dataset, list[Card], dict[str, dict[str, GoldAttr]]]:
    """train-сплит data/processed -> (датасет для обучения, карточки калибровки, эталон train)."""
    t0 = time.time()
    gold, cats = load_gold(processed, "train")
    tr_ids, cal_ids = select_ids(cats, n_generic, calib_share, max_examples, seed)
    log(f"distant: эталон train {len(gold)} карточек, обучение {len(tr_ids)}, калибровка {len(cal_ids)}, "
        f"{time.time() - t0:.0f} с")
    cards = load_cards(processed, tr_ids)
    examples, stats = build_examples(cards, gold, tax, view_chars, log)
    attrs = choose_attributes(stats, min_spans)
    stats["labels"] = attrs
    stats["min_spans"] = min_spans
    stats["n_generic"] = n_generic
    stats["calib_cards"] = len(cal_ids)
    examples = restrict(examples, attrs)
    stats["examples_with_spans"] = sum(1 for x in examples if x.spans)
    stats["spans"] = sum(len(x.spans) for x in examples)
    cal_cards = load_cards(processed, cal_ids)
    return Dataset(examples, attrs, stats), cal_cards, gold


# ------------------------------------------------------------------ калибровка и полный цикл -----------------------
def calibration_points(model, cards: list[Card], gold: dict[str, dict[str, GoldAttr]], tax: Taxonomy,
                       weak_only: Callable[[Card], set[str]] | None = None, batch: int = 32,
                       ) -> list[tuple[str, float, int]]:
    """(атрибут, сырая вероятность, верно ли) для сущностей NER, где эталон есть. На один (карточка, атрибут, значение)
    — одна точка с максимальной вероятностью. weak_only(card) -> атрибуты, на которых NER работает в пайплайне."""
    from ..eval.metrics import gold_matches

    pts = []
    cats = [tax.categories[tax.category_of_id(c.category_id) or "generic"] for c in cards]
    ents_all: list = []
    for i in range(0, len(cards), batch):
        ents_all += model.entities_batch(list(zip(cards[i:i + batch], cats[i:i + batch], strict=True)), tax)
    for card, cat, ents in zip(cards, cats, ents_all, strict=True):
        g = gold.get(card.id, {})
        allowed = weak_only(card) if weak_only is not None else None
        brand = str(g["brand"].values[0]) if "brand" in g and g["brand"].values else None
        for a, (v, _, p, _, _) in model.best_values(card, cat, tax, ents).items():
            if a not in g or (allowed is not None and a not in allowed):
                continue
            pts.append((a, p, int(gold_matches(v, g[a].values, cat.attributes[a], tax, brand))))
    return pts


def fit_calibration(pts: list[tuple[str, float, int]], min_per_attr: int = 100, prior: float = 10.0) -> dict:
    """Калибровка «сырая вероятность -> P(значение верное)»: изотоническая регрессия (PAV) отдельно для атрибутов
    с ≥ min_per_attr точками; для редких атрибутов — общая таблица, масштабированная на сглаженную точность атрибута
    ((верных + prior·общая) / (n + prior)): у «типа наушников» и «ёмкости» разная надёжность при той же вероятности."""
    from .ner import fit_isotonic, reliability

    xs = [p for _, p, _ in pts]
    ys = [y for _, _, y in pts]
    acc = sum(ys) / max(1, len(ys))
    by_attr: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for a, p, y in pts:
        by_attr[a].append((p, y))
    table = fit_isotonic(xs, ys) if pts else []
    per = {}
    for a, v in by_attr.items():
        if len(v) >= min_per_attr:
            per[a] = fit_isotonic([p for p, _ in v], [y for _, y in v])
        else:
            k = (sum(y for _, y in v) + prior * acc) / (len(v) + prior) / max(acc, 1e-6)
            per[a] = [[x, round(min(0.99, y * k), 4)] for x, y in table]
    return {"method": f"isotonic (PAV) по атрибутам с ≥ {min_per_attr} точками, иначе общая × точность атрибута; "
                      "сглаживание блоков (k+1)/(n+2)", "points": len(pts), "accuracy": round(acc, 4),
            "table": table, "per_attribute": per, "reliability_raw": reliability(xs, ys),
            "attr_points": {a: {"n": len(v), "accuracy": round(sum(y for _, y in v) / len(v), 4)}
                            for a, v in sorted(by_attr.items())}}


def oversample(examples: list[NerExample], tax: Taxonomy, times: int) -> list[NerExample]:
    """Карточки категорий таксономии — times раз (редкие атрибуты: глубина, загрузка, время работы), generic — один."""
    generic = tax.categories["generic"].title
    return examples + [ex for ex in examples if ex.prefix != generic] * max(0, times - 1)


def run(processed: Path, out: Path, tax: Taxonomy, *, n_generic: int = 48_000, max_examples: int | None = 60_000,
        epochs: int = 3, lr: float = 1e-4, batch_size: int = 32, device: str = "auto", threads: int = 2,
        seed: int = 13, calib_share: float = 0.1, min_spans: int = 150, base: str | None = None,
        tax_oversample: int = 2, dry_run: bool = False, log: Callable[[str], None] = print) -> dict:
    """Полный цикл: разметка train-сплита -> обучение -> калибровка на отложенной части train -> models/ner.

    Модель пишется во временную папку и подменяет out только целиком (атомарно для читающего её пайплайна);
    предыдущая версия остаётся в out.prev. Dev и holdout здесь не используются."""
    import shutil

    from . import ner

    t0 = time.time()
    ds, cal_cards, gold = dataset_from_processed(processed, tax, n_generic=n_generic, max_examples=max_examples,
                                                 calib_share=calib_share, seed=seed, min_spans=min_spans, log=log)
    log(stats_table(ds.stats))
    log(f"distant: меток {len(ds.attributes)} атрибутов ({', '.join(ds.attributes)}), примеров {len(ds.examples)}, "
        f"с разметкой {ds.stats['examples_with_spans']}, спанов {ds.stats['spans']}, {time.time() - t0:.0f} с")
    stats_path = processed / "ner_distant_stats.json"
    stats_path.write_text(json.dumps(ds.stats, ensure_ascii=False, indent=1), encoding="utf-8")
    if dry_run:
        return {"dataset": ds.stats}
    dev = ner.pick_device(device, log=log)
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    info = ner.train(oversample(ds.examples, tax, tax_oversample), ds.attributes, tmp, base=base or ner.DEFAULT_BASE,
                     epochs=epochs, lr=lr, batch_size=batch_size, seed=seed, device=dev, threads=threads, log=log,
                     extra_info={"supervision": "distant (эталон каталога, train-сплит)", "dataset": ds.stats,
                                 "tax_oversample": tax_oversample})
    t1 = time.time()
    model = ner.NerExtractor(tmp, device=dev, threads=threads)
    pts = calibration_points(model, cal_cards, gold, tax)
    info["calibration"] = fit_calibration(pts) | {"cards": len(cal_cards), "seconds": round(time.time() - t1, 1),
                                                  "slice": f"{calib_share:.0%} train (md5 'ner-calib:'+sku), "
                                                           "не участвовал в обучении",
                                                  "unit": "лучшее значение атрибута в карточке"}
    info["total_seconds"] = round(time.time() - t0, 1)
    ner.write_info(tmp, info)
    log(f"distant: калибровка на {len(cal_cards)} карточках ({len(pts)} сущностей, точность "
        f"{info['calibration']['accuracy']}), {time.time() - t1:.0f} с")
    prev = out.with_name(out.name + ".prev")
    if out.exists():
        shutil.rmtree(prev, ignore_errors=True)
        out.rename(prev)
    tmp.rename(out)
    return info
