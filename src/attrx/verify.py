"""Проверка «заземлённости»: значение должно подтверждаться текстом карточки. Главная защита от галлюцинаций LLM.

Двухшаговая проверка кандидата от модели (ml/llm):
1. **Выравнивание цитаты** (align): ищем в Card.full_text() фрагмент, который совпадает с цитатой модели с точностью
   до словоформы, пунктуации и пробелов («6,43-дюймовый» ~ «6,43-дюймовым», «Moonlit-Black» ~ «Moonlit Black»).
   Числа должны совпасть точно; слова — по общей основе или с опечаткой (rapidfuzz >= 85). Нет такого фрагмента —
   цитата выдумана, кандидат отбрасывается.
2. **Значение следует из найденного фрагмента** (а не «где-то в тексте»): число с единицей равно значению после
   нормализации единиц, значение справочника находится во фрагменте, модель/процессор содержатся во фрагменте.
Найденный фрагмент становится evidence и evidence_span (подсветка в UI).
"""
from __future__ import annotations

import re

from rapidfuzz import fuzz

from .normalize import compact, cpu_key, model_core, normalize_value, parse_number
from .schema import Candidate, Card
from .taxonomy import AttrSpec, Dictionary, lower_keep, norm

TOKEN = re.compile(r"[^\W_]+")
ALIGN_MIN = 0.85  # доля «веса» цитаты, которая должна совпасть с фрагментом текста
TRAILING_UNIT = '"”″\'%°'
# кириллица, которую пишут вместо похожей латиницы («Чип М4», «Сore i5»): сравниваем после свёртки (длина та же)
CONFUSABLE = str.maketrans("авекмнорстух", "abekmhopctyx")


def fold(s: str) -> str:
    """Нижний регистр, ё->е и кириллица-двойник -> латиница; длина строки сохраняется (смещения те же)."""
    return lower_keep(s).translate(CONFUSABLE)


# число с разделителем тысяч («5 000», «12 500») — один токен «5000» (второй проход выравнивания)
TOKEN_THOUSANDS = re.compile(r"(?<![\d.,])\d{1,3}(?:[ \u00a0\u202f\u2009]\d{3})+(?![\d])|[^\W_]+")
THOUSANDS = re.compile(r"(?<![\d.,])(\d{1,3})((?:[ \u00a0\u202f\u2009]\d{3})+)(?![\d])")


def collapse_thousands(s: str) -> str:
    """«5 000 мАч» -> «5000 мАч» (неразрывные и узкие пробелы тоже)."""
    return THOUSANDS.sub(lambda m: m.group(1) + re.sub(r"\D", "", m.group(2)), s)


def _tokens(low: str, merged: bool = False) -> list[tuple[str, int, int]]:
    if not merged:
        return [(m.group(0), m.start(), m.end()) for m in TOKEN.finditer(low)]
    return [(re.sub(r"\s", "", m.group(0)), m.start(), m.end()) for m in TOKEN_THOUSANDS.finditer(low)]


def _tok_sim(a: str, b: str) -> float:
    """Похожесть токенов цитаты и текста: числа — только точно; слова — словоформа (общая основа) или опечатка."""
    if a == b:
        return 1.0
    if any(ch.isdigit() for ch in a) or any(ch.isdigit() for ch in b):
        return 0.0
    p = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        p += 1
    if p >= 4 and p >= 0.5 * max(len(a), len(b)):  # «дюймовый» ~ «дюймовым» ~ «дюйма», «черная» ~ «черный»
        return 0.9
    if min(len(a), len(b)) >= 5:
        r = fuzz.ratio(a, b) / 100
        if r >= 0.85:  # опечатка / транслитерация с одной буквой
            return r * 0.95
    return 0.0


def align(quote: str, text: str, min_score: float = ALIGN_MIN) -> tuple[int, int, float] | None:
    """Лучший фрагмент text для цитаты quote: (start, end, score) или None, если похожего фрагмента нет.

    Жадное выравнивание токенов от каждого «якоря» (позиции, где совпал первый или второй токен цитаты):
    в тексте можно пропустить до двух лишних слов, в цитате — слово без пары (оно снижает score).
    Второй проход — числа с разделителем тысяч как один токен («5000 мАч» ~ «5 000 мАч»)."""
    best = _align(quote, text, min_score, merged=False)
    if best is None and THOUSANDS.search(quote + "\n" + text):
        best = _align(quote, text, min_score, merged=True)
    return best


def _align(quote: str, text: str, min_score: float, merged: bool) -> tuple[int, int, float] | None:
    qt = [t for t, _, _ in _tokens(fold(quote), merged)]
    if not qt:
        return None
    low = fold(text)
    tt = _tokens(low, merged)
    n = len(qt)
    best: tuple[int, int, float] | None = None
    for j0 in range(len(tt)):
        if _tok_sim(qt[0], tt[j0][0]) <= 0 and (n < 2 or _tok_sim(qt[1], tt[j0][0]) <= 0):
            continue
        i, j, got, skips, first, last = 0, j0, 0.0, 0, None, None
        unmatched_num = False
        while i < n and j < len(tt):
            s = _tok_sim(qt[i], tt[j][0])
            if s > 0:
                got += s
                first = j if first is None else first
                last = j
                i += 1
                j += 1
            elif first is not None and skips < 2 and j + 1 < len(tt) and _tok_sim(qt[i], tt[j + 1][0]) > 0:
                skips += 1  # лишнее слово в тексте («6,43-дюймовым IPS экраном» при цитате без «IPS»)
                j += 1
            else:
                unmatched_num |= any(ch.isdigit() for ch in qt[i])
                i += 1  # слово цитаты без пары
        unmatched_num |= any(any(ch.isdigit() for ch in q) for q in qt[i:])
        if first is None or unmatched_num:
            continue  # число из цитаты не найдено — не заземлено
        score = got / n - 0.05 * skips
        if score >= min_score and (best is None or score > best[2]):
            start, end = tt[first][1], tt[last][2]
            while end < len(text) and text[end] in TRAILING_UNIT:
                end += 1
            best = (start, end, round(score, 3))
            if score >= 0.999:
                break
    return best


def value_follows(value, fragment: str, spec: AttrSpec, dictionaries: dict[str, Dictionary]) -> bool:
    """Следует ли значение из фрагмента текста (а не просто «цитата где-то есть»). Сначала — та же нормализация,
    что у значения: фрагмент «Pentium Gold 7505» даёт «Intel Pentium Gold 7505» (префикс добавил нормализатор)."""
    frag = norm(fragment)
    if not frag:
        return False
    res = normalize_value(fragment, spec, dictionaries)
    if res is not None and _same_value(res[0], value):
        return True
    if spec.type == "number":
        v = float(value)
        for f in dict.fromkeys((frag, collapse_thousands(frag))):  # «5 000 мАч» = 5000
            parsed = parse_number(f, spec.kind)
            if parsed and abs(parsed[0] - v) <= 1e-6 * max(1.0, abs(v)):
                return True
            nums = [float(n.replace(",", ".")) for n in re.findall(r"\d+(?:[.,]\d+)?", f)]
            if any(abs(n - v) < 1e-6 or abs(n * 1024 - v) < 1e-6 for n in nums):
                return True
        return False
    if spec.type == "enum" and spec.dictionary in dictionaries:
        d = dictionaries[spec.dictionary]
        if d.get(frag) == value or any(h[0] == value for h in d.find_all(frag)):
            return True
        # бренд вне справочника (открытый словарь NER, новый бренд маркетплейса): засчитываем, только если он
        # дословно стоит в найденном фрагменте («Правила мебели», «Liebherr»), а не «где-то в тексте»
        key = fold(compact(str(value)))
        return spec.dictionary == "brands" and len(key) >= 2 and key in fold(compact(frag))
    if spec.type == "bool":
        return str(value) == "да" and any(norm(k) in frag for k in spec.keywords)
    key = fold(compact(str(value)))
    hay = fold(compact(frag))
    if spec.kind == "cpu":
        ck = fold(cpu_key(str(value)))
        return bool(ck) and ck in hay
    if spec.kind == "model":
        core = fold(compact(model_core(str(value))))
        return bool(key) and (key in hay or bool(core) and core in hay)
    return bool(key) and key in hay


def _same_value(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
    return fold(compact(str(a))) == fold(compact(str(b)))


def ground(cand: Candidate, card: Card, spec: AttrSpec, dictionaries: dict[str, Dictionary]) -> Candidate | None:
    """Кандидат, привязанный к фрагменту текста (evidence = фрагмент, span = его смещения), или None.

    Продавец и человек не проверяются (их значения — сами поля карточки). Для модели (ml/llm) сначала цитата
    выравнивается с текстом, затем проверяется, что значение следует именно из найденного фрагмента."""
    if cand.source in {"seller", "human"}:
        return cand
    text = card.full_text()
    quote = (cand.raw or "").strip() or str(cand.value)
    hit = align(quote, text)  # без цитаты (raw == значение) выравнивается само значение; выдуманная цитата — отказ
    if hit is None:
        return None
    start, end, _ = hit
    fragment = text[start:end]
    if not value_follows(cand.value, fragment, spec, dictionaries):
        return None
    return cand.model_copy(update={"raw": fragment, "span": (start, end)})


def grounded(cand: Candidate, card: Card, spec: AttrSpec, dictionaries: dict[str, Dictionary]) -> bool:
    """Совместимость: True, если кандидат заземлён (см. ground)."""
    return ground(cand, card, spec, dictionaries) is not None
