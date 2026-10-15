"""ML-слой гибрида: NER (BIO-разметка токенов) на дообученном энкодере ruBERT-tiny2 — модель для «хвоста».

Разметка — distant supervision (extractors/ner_distant.py, docs/NER.md): значения характеристик из каталога
М.Видео находятся в тексте карточки, найденные фрагменты становятся BIO-метками — десятки тысяч размеченных
карточек без ручного труда. Прежний путь — слабая разметка от правил (weak_spans) — оставлен для маленьких наборов.

Вход модели — пара «категория таксономии» + «вид карточки» (build_view): название + самые информативные фрагменты
описания и блока продавца, ≤ 256 токенов. Одна и та же функция при обучении и в пайплайне. Softmax — только по меткам
атрибутов схемы категории (и при обучении, и при выводе). На атрибут — одно значение: упоминание с максимальной
вероятностью (средняя по токенам P(B-a) + P(I-a)) после починки границ (repair_span); вероятность калибруется
изотонической регрессией на отложенной части train -> P(значение верно) — это score кандидата для агрегатора.
torch/transformers — опциональные (pip install -e ".[ml]"): без них или без модели в models/ner слой выключен,
а NerExtractor.load не импортирует torch.
"""
from __future__ import annotations

import bisect
import json
import os
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..normalize import UNIT_PATTERNS, normalize_value
from ..schema import Candidate, Card
from ..taxonomy import AttrSpec, CategorySpec, Dictionary, Taxonomy, _inflect, norm

ML_BASE_SCORE = 0.8  # score модели без калибровки (attrx_ner.json без "calibration")
DEFAULT_BASE = "cointegrated/rubert-tiny2"
MAX_LEN = 256  # токенов на пару «категория + вид карточки»
VIEW_CHARS = 900  # бюджет вида карточки в символах: медиана 186 токенов rubert-tiny2, > 256 — у 2% (docs/NER.md)
MIN_PROB = 0.35  # сущности с меньшей сырой вероятностью не выдаём
MIN_SCORE = 0.5  # и с калиброванной P(верно) ниже — тоже: такие значения чаще неверны, чем верны
INFO_FILE = "attrx_ner.json"


def available() -> bool:
    """Установлены ли torch и transformers (без импорта: find_spec не грузит библиотеки)."""
    from importlib.util import find_spec

    return find_spec("torch") is not None and find_spec("transformers") is not None


WEIGHTS = ("model.safetensors", "pytorch_model.bin")


def has_model(path: Path | str) -> bool:
    """В папке есть обученная модель: конфиг и веса."""
    p = Path(path)
    return (p / "config.json").exists() and any((p / w).exists() for w in WEIGHTS)


# ------------------------------------------------------------------ текст: нижний регистр без сдвига смещений ------
def ntext(s: str) -> str:
    """Нижний регистр и ё->е без схлопывания пробелов: позиции символов совпадают с исходной строкой."""
    low = s.lower().replace("ё", "е")
    if len(low) == len(s):
        return low
    return "".join(ch.lower() if len(ch.lower()) == 1 else ch for ch in s).replace("ё", "е")


# ------------------------------------------------------------------ фрагменты и «вид» карточки ----------------------
SEG_SPLIT = re.compile(r"\n+|(?<=[.!?;])\s+(?=\S)|\s*•\s*")
SEG_MAX = 300


def _push(text: str, s: int, e: int, out: list[tuple[int, int]]) -> None:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    while e - s > SEG_MAX:  # длинный фрагмент — кусками по запятым/пробелам
        cut = text.rfind(", ", s, s + SEG_MAX)
        nxt = cut + 2
        if cut <= s:
            cut = text.rfind(" ", s, s + SEG_MAX)
            nxt = cut + 1
        if cut <= s:
            cut = nxt = s + SEG_MAX
        out.append((s, cut + (1 if text[cut:cut + 1] == "," else 0)))
        s = nxt
        while s < e and text[s].isspace():
            s += 1
    if e > s:
        out.append((s, e))


def segments(text: str, title_len: int) -> list[tuple[int, int]]:
    """Фрагменты Card.full_text(): название целиком, затем строки/предложения описания и пары «ключ: значение»
    блока продавца (длинные — кусками ≤ SEG_MAX символов). Смещения — в исходном тексте."""
    out: list[tuple[int, int]] = [(0, title_len)] if title_len > 0 else []
    pos = title_len
    for m in SEG_SPLIT.finditer(text, pos):
        _push(text, pos, m.start(), out)
        pos = m.end()
    _push(text, pos, len(text), out)
    return out


@dataclass
class View:
    """Сокращённый текст карточки для модели и отображение смещений «вид -> исходный текст»."""

    text: str
    pieces: list[tuple[int, int, int]]  # (начало в виде, начало в исходном тексте, длина), по возрастанию

    def to_orig(self, s: int, e: int) -> tuple[int, int] | None:
        i = bisect.bisect_right([p[0] for p in self.pieces], s) - 1
        if i < 0:
            return None
        v, o, n = self.pieces[i]
        if s >= v + n:
            return None
        return o + (s - v), o + (min(e, v + n) - v)

    def from_orig(self, s: int, e: int) -> tuple[int, int] | None:
        """Спан исходного текста -> спан вида (None, если фрагмент не попал в вид целиком)."""
        i = bisect.bisect_right([p[1] for p in self.pieces], s) - 1
        if i < 0:
            return None
        v, o, n = self.pieces[i]
        if e > o + n:
            return None
        return v + (s - o), v + (e - o)


_SCORERS: dict[tuple, Callable[[str], float]] = {}
_UNIT_ANY = None


def _unit_any() -> re.Pattern:
    global _UNIT_ANY
    if _UNIT_ANY is None:
        units = sorted({u for pats in UNIT_PATTERNS.values() for u, _ in pats}, key=len, reverse=True)
        _UNIT_ANY = re.compile(r"\d\s*-?\s*(?:" + "|".join(units) + r")(?![a-zа-я])|\d{1,2}\s*[/+]\s*\d{2,4}")
    return _UNIT_ANY


MODEL_LIKE = re.compile(r"(?<![a-z0-9])(?=[a-z0-9-]*\d)(?=[a-z0-9-]*[a-z])[a-z0-9-]{3,}")


def view_scorer(category: CategorySpec, tax: Taxonomy) -> Callable[[str], float]:
    """Оценка информативности фрагмента для категории: число с единицей, название атрибута, значение из справочника,
    «модельный» токен (латиница с цифрами). Не зависит от эталона — одинаково при обучении и в пайплайне."""
    key = (category.name, tuple(sorted(category.attributes)), len(tax.dictionaries))
    if key in _SCORERS:
        return _SCORERS[key]
    aliases = sorted({norm(a) for s in category.attributes.values() for a in [*s.aliases, s.title] if len(a) >= 4},
                     key=len, reverse=True)
    alias_re = re.compile(r"(?<![a-zа-я])(?:" + "|".join(re.escape(a) for a in aliases) + ")") if aliases else None
    dicts = [tax.dictionaries[s.dictionary] for s in category.attributes.values()
             if s.type == "enum" and s.dictionary in tax.dictionaries and s.dictionary != "brands"]
    dict_res = [dict_regex(d) for d in dicts]
    units = _unit_any()

    def score(low: str) -> float:
        sc = 2.0 * bool(units.search(low))
        sc += 2.0 * bool(alias_re and alias_re.search(low))
        sc += 1.0 * any(r.search(low) for r in dict_res if r is not None)
        sc += 1.0 * bool(MODEL_LIKE.search(low))
        return sc

    _SCORERS[key] = score
    return score


def build_view(text: str, title_len: int, scorer: Callable[[str], float] | None = None,
               max_chars: int = VIEW_CHARS) -> View:
    """Название + самые информативные фрагменты (в исходном порядке) в пределах max_chars.

    Короткая карточка (≤ max_chars) — целиком, без изменений; иначе фрагменты ранжируются scorer-ом
    (при равенстве — более ранние), набираются жадно до бюджета и склеиваются через перевод строки."""
    if len(text) <= max_chars:
        return View(text, [(0, 0, len(text))])
    segs = segments(text, title_len)
    head = [segs[0]] if title_len > 0 and segs else []
    rest = segs[len(head):]
    budget = max_chars - sum(min(e - s, max_chars) + 1 for s, e in head)
    low = ntext(text)
    scored = sorted(range(len(rest)), key=lambda i: (-(scorer(low[rest[i][0]:rest[i][1]]) if scorer else 0.0)
                                                     - (0.5 if i == 0 else 0.0), i))
    chosen: set[int] = set()
    for i in scored:
        n = rest[i][1] - rest[i][0] + 1
        if n <= budget:
            chosen.add(i)
            budget -= n
    parts, pieces, pos = [], [], 0
    for s, e in head + [rest[i] for i in sorted(chosen)]:
        e = min(e, s + max_chars)
        pieces.append((pos, s, e - s))
        parts.append(text[s:e])
        pos += e - s + 1
    return View("\n".join(parts), pieces)


# ------------------------------------------------------------------ справочники: один regex на справочник -----------
_DICT_RES: dict[tuple, re.Pattern | None] = {}


def dict_regex(d: Dictionary) -> re.Pattern | None:
    """Все синонимы справочника одним regex (длинные первыми, прилагательные в любом роде и числе)."""
    key = (d.name, id(d), len(d.lookup))
    if key not in _DICT_RES:
        syns = [s for s in sorted(d.lookup, key=len, reverse=True) if s and not re.fullmatch(r"[\d.,]+", s)]
        _DICT_RES[key] = (re.compile(r"(?<![\w-])(?:" + "|".join(_inflect(s) for s in syns) + r")(?![\w-])")
                          if syns else None)
    return _DICT_RES[key]


# ------------------------------------------------------------------ фрагмент -> значение (обучение и инференс) ------
THOUSANDS = re.compile(r"(?<=\d)[  ](?=\d{3}(?!\d))")


STRIP = " ,.;:()[]«»-–—/\\|"


def clean_raw(raw: str, number: bool = False) -> str:
    """«5 000 мАч» -> «5000 мАч» (normalize.parse_number не понимает разделитель тысяч); обрезка пунктуации по краям.
    У чисел кавычки оставляем: «6.7"» — это дюймы."""
    return THOUSANDS.sub("", re.sub(r"\s+", " ", raw)).strip(STRIP if number else STRIP + "\"'")


def span_value(raw: str, spec: AttrSpec, tax: Taxonomy) -> tuple[Any, float] | None:
    """Значение, которое пайплайн получит из фрагмента: normalize_value, а бренд вне справочника — как есть
    (открытый словарь: эталон бренда — vendor каталога, сравнивается без регистра и пунктуации)."""
    raw = clean_raw(raw, number=spec.type == "number")
    if not raw:
        return None
    res = normalize_value(raw, spec, tax.dictionaries)
    if res is None and spec.type == "enum" and spec.dictionary == "brands" and _brand_like(raw):
        return raw, 0.85
    return res


def _brand_like(raw: str) -> bool:
    return 2 <= len(raw) <= 40 and bool(re.search(r"[^\W\d_]", raw)) and len(raw.split()) <= 4


# ------------------------------------------------------------------ границы сущности -------------------------------
CPU_ANY = re.compile(
    r"(?<![a-z0-9])(?:(?:intel\s+)?core\s+(?:ultra\s+)?i?[3579](?:[\s-]*\d{3,5}[a-z]{0,3})?"
    r"|(?:intel\s+)?(?:celeron|pentium(?:\s+(?:silver|gold))?)(?:\s*[a-z]?\d{3,5}[a-z]{0,2})?"
    r"|(?:intel\s+)?n\d{2,3}|(?:amd\s+)?ryzen\s+(?:ai\s+)?(?:max\+?\s+)?(?:pro\s+)?\d(?:\s+pro)?(?:\s+(?:hx\s+)?\d{3,4}[a-z]{0,3})?"
    r"|(?:apple\s+)?m[1-5](?:\s+(?:pro|max|ultra))?|(?:qualcomm\s+)?snapdragon\s+x\d?(?:\s+(?:elite|plus))?(?:\s+x\d[\w-]*)?"
    r"|i[3579]-\d{4,5}[a-z]{0,3})(?![a-z0-9])")
_UNIT_AFTER: dict[str, re.Pattern] = {}
THOUSANDS_KINDS = frozenset({"battery", "btu", "rpm", "power_w"})  # величины, где «5 000» — одно число


def repair_span(text: str, s: int, e: int, spec: AttrSpec) -> tuple[int, int]:
    """Граница сущности -> целое значение: без перевода строки внутри; число целиком с единицей («6,|67 дюйма» ->
    «6,67 дюйма», «3"» -> «6.3"»); процессор целиком («Ryzen» -> «Ryzen 5 7520U», «12450H» -> «Core i5 12450H»)."""
    nl = text.find("\n", s, e)
    if nl != -1:
        s, e = (s, nl) if nl - s >= e - nl - 1 else (nl + 1, e)
    if spec.type == "number":
        while s > 0 and (text[s - 1].isdigit() or text[s - 1] in ".," and s >= 2 and text[s - 2].isdigit()
                         and text[s].isdigit()):
            s -= 1
        while e < len(text) and (text[e].isdigit() or text[e] in ".," and e + 1 < len(text) and text[e + 1].isdigit()
                                 and text[e - 1].isdigit()):
            e += 1
        kind = spec.kind or ""
        if kind in THOUSANDS_KINDS and re.match(r"\d{3}(?!\d)", text[s:s + 4]):  # «5 |000 мАч» -> «5 000 мАч»
            m = re.search(r"(?<![\d.,])\d{1,3}[ \u00a0]$", text[max(0, s - 5):s])
            if m:
                s -= len(m.group(0))
        if kind not in _UNIT_AFTER:
            units = "|".join(u for u, _ in UNIT_PATTERNS.get(kind, []))
            _UNIT_AFTER[kind] = re.compile(r"[ \t]*(?:" + units + r")(?![a-zа-я])") if units else re.compile(r"(?!x)x")
        if text[s:e].replace(",", "").replace(".", "").isdigit():  # голое число — с единицей, если она сразу после
            m = _UNIT_AFTER[kind].match(ntext(text[e:e + 40]))
            if m:
                e += m.end()
    elif spec.kind == "cpu":
        lo = max(0, s - 40)
        low = ntext(text[lo:e + 40])
        for m in CPU_ANY.finditer(low):
            a, b = lo + m.start(), lo + m.end()
            if a < e and s < b:
                s, e = min(s, a), max(e, b)
    while s < e and text[s] in " \t,.;:()[]«»/\\|-–—":
        s += 1
    while e > s and text[e - 1] in " \t,.;:([«/\\|-–—":
        e -= 1
    return s, e


# ------------------------------------------------------------------ слабая разметка от правил (старый путь) --------
def find_span(text: str, raw: str) -> tuple[int, int] | None:
    i = text.lower().find(raw.lower().strip())
    return (i, i + len(raw.strip())) if i >= 0 and raw.strip() else None


def weak_spans(card: Card, cands: list[Candidate], gold: dict | None, tax: Taxonomy, category: CategorySpec) -> list[tuple[int, int, str]]:
    """Спаны (start, end, attr) из кандидатов правил; при наличии gold оставляем только верные значения."""
    from ..eval.metrics import gold_matches

    text = card.full_text()
    spans = []
    brand = gold.get("brand") if gold else None
    brand = brand[0] if isinstance(brand, list) and brand else brand
    for c in cands:
        if c.source != "rules":
            continue
        spec = category.attributes[c.attribute]
        if gold is not None and not gold_matches(c.value, gold.get(c.attribute), spec, tax, brand):
            continue
        sp = c.span if c.span and text[c.span[0]:c.span[1]].lower() == c.raw.lower() else find_span(text, c.raw)
        if sp:
            spans.append((sp[0], sp[1], c.attribute))
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    out, last_end = [], -1
    for s in spans:
        if s[0] >= last_end:
            out.append(s)
            last_end = s[1]
    return out


def bio_labels(offsets: list[tuple[int, int]], spans: list[tuple[int, int, str]],
               masks: list[tuple[int, int]] = ()) -> list[str | None]:
    """Метки токенов по символьным спанам; None — не учитывается в loss (служебный токен или маска)."""
    labels: list[str | None] = []
    for s, e in offsets:
        if s == e:
            labels.append(None)
            continue
        lab: str | None = "O"
        for a, b, attr in spans:
            if s >= a and e <= b:
                lab = ("B-" if s == a else "I-") + attr
                break
            if s < b and e > a:  # токен частично в спане
                lab = ("B-" if s <= a else "I-") + attr
                break
        if lab == "O" and any(s < mb and e > ma for ma, mb in masks):
            lab = None
        labels.append(lab)
    # I- без предшествующего B- того же атрибута превращаем в B-
    prev = "O"
    for i, lab in enumerate(labels):
        if lab and lab.startswith("I-") and prev[2:] != lab[2:]:
            labels[i] = "B-" + lab[2:]
        prev = labels[i] or "O"
    return labels


# ------------------------------------------------------------------ обучение ---------------------------------------
@dataclass
class NerExample:
    """Пример для обучения: пара (prefix — название категории, text — вид карточки), спаны и маски в координатах text."""

    prefix: str
    text: str
    spans: list[tuple[int, int, str]]
    masks: list[tuple[int, int]] = field(default_factory=list)
    id: str = ""
    allowed: list[str] | None = None  # атрибуты схемы категории: softmax только по их меткам (None — по всем)


def encode(tok, prefix: str, text: str, max_len: int = MAX_LEN, **kw):
    """Пара «категория» + текст (обрезается только текст); без prefix — одиночная последовательность."""
    if prefix:
        return tok(prefix, text, truncation="only_second", max_length=max_len, return_offsets_mapping=True, **kw)
    return tok(text, truncation=True, max_length=max_len, return_offsets_mapping=True, **kw)


def label_ids(enc, ex: NerExample, l2i: dict[str, int], batch_index: int | None = None) -> list[int]:
    seq = enc.sequence_ids(batch_index) if batch_index is not None else enc.sequence_ids()
    offsets = enc["offset_mapping"][batch_index] if batch_index is not None else enc["offset_mapping"]
    text_seq = 1 if ex.prefix else 0
    offs = [(s, e) if sid == text_seq else (0, 0) for (s, e), sid in zip(offsets, seq, strict=True)]
    return [-100 if lab is None else l2i.get(lab, 0) for lab in bio_labels(offs, ex.spans, ex.masks)]


def _device(device: str | None):
    import torch

    if device in (None, "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def pick_device(device: str = "auto", min_free_gb: float = 2.5, log: Callable[[str], None] = print) -> str:
    """auto: GPU, только если torch её поддерживает (sm_XX есть в сборке) и свободно ≥ min_free_gb видеопамяти
    прямо сейчас (GPU делится с локальной LLM), иначе CPU."""
    if device != "auto":
        return device
    import torch

    if not torch.cuda.is_available():
        log("ner: CUDA недоступна -> CPU")
        return "cpu"
    major, minor = torch.cuda.get_device_capability(0)
    archs = torch.cuda.get_arch_list()
    # cubin sm_60 исполняется на sm_61 (GTX 1060): подходит та же старшая версия с младшей ≤ нашей
    if not any(a.startswith(f"sm_{major}") and a[3 + len(str(major)):].isdigit()
               and int(a[3 + len(str(major)):]) <= minor for a in archs):
        log(f"ner: сборка torch без поддержки sm_{major}{minor} ({archs}) -> CPU")
        return "cpu"
    free, total = torch.cuda.mem_get_info(0)
    if free / 2**30 < min_free_gb:
        log(f"ner: на GPU свободно {free / 2**30:.1f} из {total / 2**30:.1f} ГБ (< {min_free_gb}) -> CPU")
        return "cpu"
    log(f"ner: GPU {torch.cuda.get_device_name(0)}, свободно {free / 2**30:.1f} ГБ")
    return "cuda"


def train(examples: list, attributes: list[str], out_dir: Path, base: str = DEFAULT_BASE, epochs: int = 2,
          lr: float = 1e-4, batch_size: int = 32, seed: int = 13, max_len: int = MAX_LEN, device: str | None = "cpu",
          threads: int = 2, log: Callable[[str], None] = print, extra_info: dict | None = None,
          weight_decay: float = 0.01, warmup: float = 0.06) -> dict:
    """Дообучение токен-классификатора. examples — NerExample или (text, spans) (старый формат слабой разметки).

    Воспроизводимо: фиксированные seed (random/torch), порядок батчей из того же генератора."""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    random.seed(seed)
    torch.manual_seed(seed)
    dev = _device(device)
    if dev.type == "cpu" and threads:
        torch.set_num_threads(threads)
    exs = [x if isinstance(x, NerExample) else NerExample("", x[0], list(x[1])) for x in examples]
    labels = ["O"] + [f"{p}-{a}" for a in sorted(attributes) for p in ("B", "I")]
    l2i = {lab: i for i, lab in enumerate(labels)}
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForTokenClassification.from_pretrained(
        base, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=l2i).to(dev)
    t0 = time.time()
    data: list[tuple[list[int], list[int]]] = []
    allowed_rows: list[list[float]] = []
    restricted = any(ex.allowed is not None for ex in exs)
    for ex in exs:
        enc = encode(tok, ex.prefix, ex.text, max_len)
        data.append((enc["input_ids"], label_ids(enc, ex, l2i)))
        if restricted:
            al = set(ex.allowed) if ex.allowed is not None else None
            allowed_rows.append([0.0 if lab == "O" or al is None or lab[2:] in al else -1e4 for lab in labels])
    n_tok = sum(len(x) for x, _ in data)
    log(f"ner: {len(data)} примеров, {n_tok} токенов, токенизация {time.time() - t0:.0f} с, устройство {dev}")
    rnd = random.Random(seed)
    steps_per_epoch = (len(data) + batch_size - 1) // batch_size
    total = max(1, steps_per_epoch * epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warm = max(1, int(total * warmup))

    def lr_factor(st: int) -> float:  # линейный разогрев, затем линейное затухание до нуля
        return (st + 1) / warm if st < warm else max(0.0, (total - st) / max(1, total - warm))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    model.train()
    history, step, t1 = [], 0, time.time()
    pad = tok.pad_token_id
    for ep in range(epochs):
        idx = list(range(len(data)))
        rnd.shuffle(idx)
        # батчи из примеров близкой длины (меньше паддинга), порядок батчей — случайный
        chunks = [sorted(idx[i:i + batch_size * 50], key=lambda j: len(data[j][0]))
                  for i in range(0, len(idx), batch_size * 50)]
        batches = [c[i:i + batch_size] for c in chunks for i in range(0, len(c), batch_size)]
        rnd.shuffle(batches)
        run, n_run = 0.0, 0
        for b in batches:
            n = max(len(data[j][0]) for j in b)
            ids = torch.tensor([data[j][0] + [pad] * (n - len(data[j][0])) for j in b], device=dev)
            mask = torch.tensor([[1] * len(data[j][0]) + [0] * (n - len(data[j][0])) for j in b], device=dev)
            y = torch.tensor([data[j][1] + [-100] * (n - len(data[j][1])) for j in b], device=dev)
            logits = model(input_ids=ids, attention_mask=mask).logits
            if restricted:  # метки атрибутов вне схемы категории не конкурируют с допустимыми
                logits = logits + torch.tensor([allowed_rows[j] for j in b], device=dev)[:, None, :]
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, len(labels)), y.reshape(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            run += loss.item()
            n_run += 1
            if step % 100 == 0 or step == total:
                el = time.time() - t1
                log(f"ner: эпоха {ep + 1}/{epochs}, шаг {step}/{total}, loss {run / n_run:.4f}, "
                    f"{el / 60:.1f} мин, осталось ~{el / step * (total - step) / 60:.1f} мин")
                history.append({"step": step, "loss": round(run / n_run, 4)})
                run, n_run = 0.0, 0
    train_sec = time.time() - t1
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    info = {"base": base, "examples": len(data), "tokens": n_tok, "epochs": epochs, "lr": lr, "batch_size": batch_size,
            "seed": seed, "max_len": max_len, "view_chars": VIEW_CHARS, "prefix": any(x.prefix for x in exs),
            "schema_restricted": restricted,
            "device": str(dev), "device_name": _device_name(dev), "threads": threads if dev.type == "cpu" else None,
            "train_seconds": round(train_sec, 1), "loss": history, "labels": labels,
            "torch": torch.__version__} | (extra_info or {})
    write_info(out_dir, info)
    return info


def _device_name(dev) -> str:
    import torch

    if dev.type == "cuda":
        return torch.cuda.get_device_name(dev)
    return f"CPU ({os.cpu_count()} ядер)"


def write_info(out_dir: Path, info: dict) -> None:
    (Path(out_dir) / INFO_FILE).write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def read_info(path: Path) -> dict:
    p = Path(path) / INFO_FILE
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ------------------------------------------------------------------ калибровка уверенности -------------------------
def fit_isotonic(xs: list[float], ys: list[int]) -> list[list[float]]:
    """Изотоническая регрессия (pool adjacent violators): сырая вероятность -> доля верных.
    Узлы [x, y] для кусочно-линейной интерполяции; y сглажен (+1/+2), чтобы крайние блоки не давали 0 и 1."""
    def pav(items: list[list[float]], value) -> list[list[float]]:
        blocks: list[list[float]] = []  # [сумма y, n, x_min, x_max]
        for it in items:
            blocks.append(list(it))
            while len(blocks) > 1 and value(blocks[-2]) >= value(blocks[-1]):
                b = blocks.pop()
                blocks[-1][0] += b[0]
                blocks[-1][1] += b[1]
                blocks[-1][3] = b[3]
        return blocks

    raw = pav([[float(y), 1.0, x, x] for x, y in sorted(zip(xs, ys, strict=True))], lambda b: b[0] / b[1])
    # сглаживание может нарушить монотонность маленьких блоков — повторный PAV по сглаженным долям
    smooth = pav([[(b[0] + 1) / (b[1] + 2) * b[1], b[1], b[2], b[3]] for b in raw], lambda b: b[0] / b[1])
    return [[round((b[2] + b[3]) / 2, 4), round(b[0] / b[1], 4)] for b in smooth]


def calibrate(table: list[list[float]] | None, x: float) -> float:
    """Калиброванная P(верно) по таблице fit_isotonic (линейная интерполяция между узлами)."""
    if not table:
        return x * ML_BASE_SCORE
    xs = [k[0] for k in table]
    i = bisect.bisect_left(xs, x)
    if i == 0:
        return table[0][1]
    if i >= len(table):
        return table[-1][1]
    (x0, y0), (x1, y1) = table[i - 1], table[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y1


def reliability(xs: list[float], ys: list[int], edges=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01)) -> list[dict]:
    """Таблица надёжности: интервал уверенности -> число сущностей, средняя уверенность, доля верных."""
    rows = []
    for lo, hi in zip(edges, edges[1:], strict=False):
        sel = [(x, y) for x, y in zip(xs, ys, strict=True) if lo <= x < hi]
        if sel:
            rows.append({"from": lo, "to": min(hi, 1.0), "n": len(sel),
                         "mean_conf": round(sum(x for x, _ in sel) / len(sel), 3),
                         "accuracy": round(sum(y for _, y in sel) / len(sel), 3)})
    return rows


# ------------------------------------------------------------------ инференс ---------------------------------------
class NerExtractor:
    """Модель из models/ner: entities() — сущности в координатах Card.full_text(), extract() — кандидаты (source=ml)."""

    def __init__(self, path: Path, device: str | None = None, threads: int | None = None):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.path = Path(path)
        self.info = read_info(self.path)
        want = device or os.getenv("NER_DEVICE", "cpu")
        self.device = _device("cpu" if want.startswith("cuda") and not torch.cuda.is_available() else want)
        if self.device.type == "cpu":
            torch.set_num_threads(threads or int(os.getenv("NER_THREADS", "2")))
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForTokenClassification.from_pretrained(self.path).to(self.device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.attrs = sorted({lab[2:] for lab in self.id2label.values() if lab != "O"})
        l2i = {lab: i for i, lab in self.id2label.items()}
        self.b_cols = [l2i[f"B-{a}"] for a in self.attrs]
        self.i_cols = [l2i[f"I-{a}"] for a in self.attrs]
        self.attr_idx = {a: k for k, a in enumerate(self.attrs)}
        self.max_len = int(self.info.get("max_len", MAX_LEN))
        self.view_chars = int(self.info.get("view_chars", VIEW_CHARS))
        self.use_prefix = bool(self.info.get("prefix", False))
        self.calibration = self.info.get("calibration") or {}

    @classmethod
    def load(cls, path: Path) -> NerExtractor | None:
        """Модель или None — сразу и без импорта torch, если ML_ENABLED выключен или модели нет (CI, Docker, свежий
        клон без [ml]); torch/transformers импортируются только когда модель действительно есть."""
        if os.getenv("ML_ENABLED", "true").strip().lower() in {"0", "false", "no", "off"}:
            return None
        if not has_model(path) or not available():
            return None
        return cls(Path(path))

    def view(self, card: Card, category: CategorySpec | None, tax: Taxonomy | None) -> View:
        scorer = view_scorer(category, tax) if category is not None and tax is not None else None
        return build_view(card.full_text(), len(card.title), scorer, self.view_chars)

    def predict(self, prefix: str, text: str, allowed: set[str] | None = None) -> list[tuple[str, int, int, float]]:
        """Сущности в координатах text: (атрибут, start, end, вероятность)."""
        return self.predict_batch([prefix], [text], [allowed])[0]

    def predict_batch(self, prefixes: list[str], texts: list[str], allowed: list[set[str] | None] | None = None,
                      ) -> list[list[tuple[str, int, int, float]]]:
        """То же для пачки текстов за один прогон модели. allowed — атрибуты схемы категории каждого текста:
        метки вне схемы исключаются из softmax (как при обучении), вероятность перераспределяется на допустимые."""
        torch = self.torch
        use = self.use_prefix and any(prefixes)
        if use:
            enc = self.tok(list(prefixes), list(texts), truncation="only_second", max_length=self.max_len,
                           return_offsets_mapping=True, padding=True, return_tensors="pt")
        else:
            enc = self.tok(list(texts), truncation=True, max_length=self.max_len, return_offsets_mapping=True,
                           padding=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping").tolist()
        with torch.inference_mode():
            logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits.float()
            if allowed is not None:
                logits = logits + self.label_mask(allowed).to(logits.device)[:, None, :]
            probs = logits.softmax(-1).cpu()
        return [self._decode(probs[i], offsets[i], enc.sequence_ids(i), 1 if use else 0, texts[i])
                for i in range(len(texts))]

    def label_mask(self, allowed: list[set[str] | None]):
        """[пачка, метки]: 0 — метка допустима, -1e4 — атрибут вне схемы категории."""
        n = len(self.id2label)
        rows = []
        for al in allowed:
            if al is None:
                rows.append([0.0] * n)
            else:
                rows.append([0.0 if lab == "O" or lab[2:] in al else -1e4 for _, lab in sorted(self.id2label.items())])
        return self.torch.tensor(rows)

    def _decode(self, probs, offsets, seq, text_seq: int, text: str) -> list[tuple[str, int, int, float]]:
        """Метка токена — argmax (по меткам схемы категории); I- того же атрибута или продолжение слова продолжают
        сущность (но не через перевод строки); вероятность сущности — средняя P(B-a)+P(I-a) по её токенам."""
        pred = probs.argmax(-1).tolist()
        ents: list[list] = []
        cur: list | None = None
        for t, ((s, e), sid) in enumerate(zip(offsets, seq, strict=True)):
            lab = self.id2label[pred[t]] if sid == text_seq and s != e else "O"
            if lab == "O":
                cur = None
                continue
            kind, attr = lab.split("-", 1)
            k = self.attr_idx[attr]
            p = float(probs[t, self.b_cols[k]] + probs[t, self.i_cols[k]])
            if cur is not None and cur[0] == attr and (kind == "I" or s == cur[2]) and "\n" not in text[cur[2]:s]:
                cur[2] = e
                cur[3].append(p)
            else:
                cur = [attr, s, e, [p]]
                ents.append(cur)
        return [(a, s, e, sum(ps) / len(ps)) for a, s, e, ps in ents]

    def entities(self, card: Card, category: CategorySpec | None = None, tax: Taxonomy | None = None,
                 ) -> list[tuple[str, int, int, float]]:
        """Сущности в координатах card.full_text() (границы расширены до целых слов)."""
        return self.entities_batch([(card, category)], tax)[0]

    def entities_batch(self, items: list[tuple[Card, CategorySpec | None]], tax: Taxonomy | None = None,
                       ) -> list[list[tuple[str, int, int, float]]]:
        views = [self.view(card, cat, tax) for card, cat in items]
        preds = self.predict_batch([cat.title if cat is not None else "" for _, cat in items], [v.text for v in views],
                                   [set(cat.attributes) if cat is not None else None for _, cat in items])
        out = []
        for (card, _), view, ents in zip(items, views, preds, strict=True):
            text = card.full_text()
            res = []
            for a, s, e, p in ents:
                sp = view.to_orig(s, e)
                if sp is None:
                    continue
                s, e = sp
                while s > 0 and text[s - 1].isalnum() and text[s].isalnum():
                    s -= 1
                while e < len(text) and text[e].isalnum() and text[e - 1].isalnum():
                    e += 1
                res.append((a, s, e, p))
            out.append(res)
        return out

    def score(self, attr: str, p: float) -> float:
        """Калиброванная P(значение верное): таблица атрибута, если есть, иначе общая."""
        cal = self.calibration
        table = (cal.get("per_attribute") or {}).get(attr) or cal.get("table")
        return calibrate(table, p)

    def best_values(self, card: Card, category: CategorySpec, tax: Taxonomy,
                    ents: list[tuple[str, int, int, float]]) -> dict[str, tuple[Any, float, float, int, int]]:
        """Атрибут -> (значение, качество нормализации, вероятность, start, end): из всех упоминаний атрибута —
        значение с максимальной вероятностью (границы починены repair_span)."""
        text = card.full_text()
        best: dict[str, tuple[Any, float, float, int, int]] = {}
        for attr, s, e, p in ents:
            spec = category.attributes.get(attr)
            if spec is None:
                continue
            s, e = repair_span(text, s, e, spec)
            res = span_value(text[s:e], spec, tax) if e > s else None
            if res is not None and (attr not in best or p > best[attr][2]):
                best[attr] = (res[0], res[1], p, s, e)
        return best

    def extract(self, card: Card, category: CategorySpec, tax: Taxonomy) -> list[Candidate]:
        return self.extract_batch([(card, category)], tax)[0]

    def extract_batch(self, items: list[tuple[Card, CategorySpec]], tax: Taxonomy) -> list[list[Candidate]]:
        """Кандидаты (source=ml) для пачки карточек: один прогон модели на пачку."""
        return [self._candidates(card, cat, tax, ents)
                for (card, cat), ents in zip(items, self.entities_batch(items, tax), strict=True)]

    def _candidates(self, card: Card, category: CategorySpec, tax: Taxonomy,
                    ents: list[tuple[str, int, int, float]]) -> list[Candidate]:
        """Один кандидат на атрибут — лучшее значение; score = калиброванная P(верно) × качество нормализации."""
        text = card.full_text()
        out = []
        for attr, (value, quality, p, s, e) in self.best_values(card, category, tax, ents).items():
            if p < MIN_PROB:
                continue
            sc = self.score(attr, p) * quality
            if self.calibration and sc < MIN_SCORE:
                continue
            # цитата — ровно фрагмент текста: по ней verify.ground() выравнивает кандидата с карточкой
            out.append(Candidate(attribute=attr, value=value, raw=text[s:e].strip() or text[s:e], source="ml",
                                 score=round(min(0.99, sc), 4), span=(s, e)))
        return out
