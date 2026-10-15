"""LLM-экстрактор: вызывается ТОЛЬКО для «хвоста» — атрибутов, которые продавец, правила и NER не закрыли уверенно.

Контракт против галлюцинаций (кейс: «не выдумывать характеристики»):
1. Промпт строится из схемы категории: название атрибута, тип, единица и справочник допустимых значений.
2. Строгий JSON по схеме (json_schema, temperature 0); справочник допустимых значений — в промпте
   (LLM_SCHEMA_ENUM=true зашивает его и в схему, но по замеру это провоцирует «вынужденный выбор» из списка).
3. Для каждого значения модель обязана дать ДОСЛОВНУЮ цитату (evidence); нет в тексте → null.
4. Значение приводится к канону (normalize.py); вне справочника — отбрасывается, а в notes пишется
   «кандидат в справочник …» с цитатой: pipeline выводит заметки в ExtractionResult.warnings (модератору).
5. Дальше pipeline проверяет заземление (verify.ground в Extractor._checked: цитата выравнивается с текстом
   карточки и значение должно из неё следовать, иначе кандидат отброшен) и порог уверенности. Значение, которое
   предложила только LLM, в каталог автоматически не попадает никогда (pipeline.UNVERIFIED_ONLY) — его проверяет модератор.
"""
from __future__ import annotations

import re
import time
from collections.abc import Sequence

from ..config import get_settings
from ..llm.client import LLMClient, LLMUnavailable
from ..normalize import normalize_value
from ..schema import Candidate, Card
from ..taxonomy import AttrSpec, CategorySpec, Taxonomy, norm
from ..verify import align

# Базовая уверенность LLM-кандидата (до веса источника 0.85 в confidence.py, заземления и калибровки).
# На автопринятие значения, которое нашла ОДНА LLM, она не влияет: такое значение уходит модератору при любой
# уверенности (явное правило pipeline.UNVERIFIED_ONLY, а не арифметика порога). Оценка важна, когда LLM согласна
# или спорит с другими источниками (noisy-OR и конфликт в confidence.aggregate_full): числа и справочники проверяемо
# выводятся из цитаты (точность заземлённых значений 95–98% на реальных карточках, docs/LLM.md) — 0.9; строки
# (модель, процессор) проверяются слабее (88–91%) — 0.75.
LLM_BASE_SCORE = 0.75
LLM_VERIFIED_SCORE = 0.9

SYSTEM = (
    "Ты — модуль извлечения характеристик товаров для каталога интернет-магазина электроники. "
    "Тебе дают текст карточки товара (название, описание, характеристики продавца) и список нужных атрибутов.\n"
    "Правила:\n"
    "1. Используй ТОЛЬКО текст карточки. Ничего не додумывай по названию модели и не используй свои знания о товаре.\n"
    "2. Если атрибута нет в тексте или он неоднозначен (например, перечислено несколько цветов) — "
    "верни null и в value, и в evidence.\n"
    "3. evidence — ДОСЛОВНАЯ цитата из карточки (обычно 2–8 слов), из которой следует значение. "
    "Копируй символы как есть: не переводи, не исправляй, не меняй регистр.\n"
    "4. Если для атрибута дан список допустимых значений и значение названо в тексте (в том числе синонимом или "
    "по-английски) — верни значение из списка; если такого значения в списке нет — верни как написано в тексте.\n"
    "5. Числа возвращай с единицей измерения: «256 ГБ», «6.7 дюйма», «5000 мА·ч», «1 ТБ».\n"
    "6. Строковые атрибуты (модель, процессор) — коротко, как в тексте, без лишних слов.\n"
    "Ответ — только JSON."
)

FEW_SHOT = (
    "\n\nПример.\nАтрибуты: battery — Ёмкость аккумулятора — число с единицей (мА·ч); "
    "color — Цвет — одно из: чёрный, белый, синий; screen — Диагональ экрана — число с единицей (дюйм).\n"
    "Карточка: «Смартфон Xiaomi Redmi 13C 4/128 ГБ. Аккумулятор ёмкостью 5000 мА·ч обеспечит день работы. "
    "Доступные цвета: чёрный, синий.»\n"
    'Ответ: {"battery": {"evidence": "Аккумулятор ёмкостью 5000 мА·ч", "value": "5000 мА·ч"}, '
    '"color": {"evidence": null, "value": null}, "screen": {"evidence": null, "value": null}}\n'
    "(цвет не определён — перечислено несколько; диагонали в тексте нет)"
)

TYPE_HINT = {"number": "число с единицей", "string": "строка, как в тексте", "enum": "значение из справочника",
             "bool": "да или нет — только если в тексте сказано явно"}
# формат строковых атрибутов как в каталоге (бренд и модель в фиде — отдельные поля): по замеру qwen3:4b без этих
# подсказок дописывал бренд к модели и отвечал «восьмиядерный процессор» вместо модели процессора
KIND_HINT = {
    "model": "без бренда, объёма памяти и цвета, например «Galaxy A55», «Redmi Note 13 Pro», «MacBook Air 13»",
    "cpu": "производитель, линейка и модель, например «Intel Core i5-1235U», «AMD Ryzen 7 8845HS», «Apple M2»; "
           "общие слова («восьмиядерный процессор») — это null",
}
# слова, по которым выбираем из длинного описания предложения, относящиеся к нужным атрибутам
KIND_WORDS = {
    "memory": ["гб", "gb", "тб", "tb", "памят"], "ram": ["гб", "gb", "оператив", "озу", "ram"],
    "storage_drive": ["гб", "gb", "тб", "tb", "ssd", "hdd", "накопит", "диск"],
    "diagonal": ["дюйм", '"', "″", "экран", "диагонал", "дисплей"],
    "battery": ["мач", "mah", "ма·ч", "ма*ч", "аккумулятор", "батаре"],
    "cpu": ["процессор", "cpu", "intel", "amd", "ryzen", "core", "apple m", "snapdragon", "mediatek"],
}


def _allowed_values(spec: AttrSpec, tax: Taxonomy, text: str, limit: int) -> tuple[list[str], bool]:
    """Допустимые значения enum для промпта: весь справочник, если он короткий, иначе — найденные в тексте.

    Возвращает (значения, полный_ли_список). Полный список можно зашить в json_schema как enum.
    """
    d = tax.dictionaries.get(spec.dictionary or "")
    if d is None:
        return [], False
    if len(d.canon) <= limit:
        return list(d.canon), True
    seen: list[str] = []
    for canon, *_ in d.find_all(text):
        if canon not in seen:
            seen.append(canon)
    return seen[:limit], False


def _attr_line(name: str, spec: AttrSpec, allowed: list[str], full: bool) -> str:
    line = f"- {name} — {spec.title} — {TYPE_HINT.get(spec.type, spec.type)}"
    if spec.unit:
        line += f" ({spec.unit})"
    if spec.type == "string" and spec.kind in KIND_HINT:
        line += ": " + KIND_HINT[spec.kind]
    if spec.type == "enum":
        if full:
            line += ": одно из: " + ", ".join(allowed)
        elif allowed:
            line += ": одно из найденных в тексте значений справочника: " + ", ".join(allowed)
        else:
            line += ": значений справочника в тексте не найдено — скорее всего null"
    return line


def _keywords(specs: Sequence[AttrSpec]) -> list[str]:
    words: list[str] = []
    for s in specs:
        words += [norm(a) for a in s.aliases] + [norm(s.title)] + KIND_WORDS.get(s.kind or "", [])
        words += [norm(k) for k in getattr(s, "keywords", None) or []]  # признаки «да» у bool-атрибутов
        if s.unit:
            words.append(norm(s.unit))
        if s.type == "enum" and s.name == "color":
            words += ["цвет", "color", "оттен"]
    return [w for w in dict.fromkeys(words) if w]


def _find(text: str, word: str, start: int = 0) -> int:
    """Позиция слова-маркера; короткие (≤3 символов: «вт», «гб») — только не внутри другого слова."""
    if len(word) > 3:
        return text.find(word, start)
    m = re.compile(r"(?<![a-zа-я])" + re.escape(word) + r"(?![a-zа-я])").search(text, start)
    return m.start() if m else -1


def fit_text(text: str, specs: Sequence[AttrSpec], budget: int) -> str:
    """Длинное описание → до budget символов: начало + предложения с упоминанием нужных атрибутов.

    Предложения берутся дословно и в исходном порядке (пропуски помечены «…»), чтобы цитата оставалась
    подстрокой карточки.
    """
    text = text.strip()
    if len(text) <= budget:
        return text
    parts = [p for p in re.split(r"(?<=[.!?…;])\s+|\n+", text) if p.strip()]
    words = _keywords(specs)

    def score(p: str) -> int:
        t = norm(p)
        return sum(w in t for w in words) + (1 if re.search(r"\d", t) else 0)

    keep: set[int] = set()
    used = 0
    for i in range(min(2, len(parts))):  # начало описания: обычно что это за товар
        if not keep or used + len(parts[i]) + 1 <= budget:
            keep.add(i)
            used += len(parts[i]) + 1
    for i in sorted(range(len(parts)), key=lambda i: (-score(parts[i]), i)):
        if i in keep or score(parts[i]) == 0:
            continue
        if used + len(parts[i]) + 1 > budget:
            continue
        keep.add(i)
        used += len(parts[i]) + 1
    out, prev = [], -1
    for i in sorted(keep):
        if prev >= 0 and i != prev + 1:
            out.append("…")
        out.append(parts[i][:budget])
        prev = i
    return " ".join(out)


def relevant_attrs(card: Card, category: CategorySpec, tax: Taxonomy, attrs: list[str]) -> list[str]:
    """Атрибуты, о которых в тексте карточки вообще что-то сказано: синоним названия, единица, слово-маркер
    или значение из справочника. Остальные LLM не может извлечь заземлённо — их не отправляем (экономия вызовов).
    Модель/серия (строка с kind=model или без kind) почти всегда в названии — её оставляем всегда.
    """
    full = card.full_text()
    text = full.lower().replace("ё", "е")  # без norm(): переводы строк — границы предложений для чисел
    out = []
    for a in attrs:
        spec = category.attributes[a]
        d = tax.dictionaries.get(spec.dictionary or "") if spec.type == "enum" else None
        if (spec.type == "string" and spec.kind in (None, "model")) or (d is not None and d.find_all(full)) \
                or _mentioned(text, _keywords([spec]), need_digit=spec.type == "number"):
            out.append(a)
    return out


def _mentioned(text: str, words: list[str], need_digit: bool, window: int = 30) -> bool:
    """Есть ли слово-маркер в тексте; для чисел — ещё и цифра рядом, в том же предложении
    (иначе заземлённого числа не будет: «батарея на весь день» LLM не спрашиваем)."""
    for w in words:
        start = _find(text, w)
        while start >= 0:
            if not need_digit:
                return True
            left = re.split(r"\n|[.!?;](?:\s|$)", text[max(0, start - window): start])[-1]
            right = re.split(r"\n|[.!?;](?:\s|$)", text[start + len(w): start + len(w) + window])[0]
            if re.search(r"\d", left + w + right):
                return True
            start = _find(text, w, start + 1)
    return False


def build_schema(attrs: list[str], enums: dict[str, list[str]] | None = None) -> dict:
    """JSON Schema ответа: {attr: {evidence, value}}. evidence первым — модель сначала цитирует, потом нормализует."""
    def item(a: str) -> dict:
        value: dict = {"type": ["string", "null"]}
        if enums and enums.get(a):
            value["enum"] = [*enums[a], None]  # справочник допустимых значений прямо в схеме
        return {
            "type": "object",
            "properties": {"evidence": {"type": ["string", "null"]}, "value": value},
            "required": ["evidence", "value"],
            "additionalProperties": False,
        }

    return {"type": "object", "properties": {a: item(a) for a in attrs}, "required": list(attrs),
            "additionalProperties": False}


def build_prompt(card: Card, category: CategorySpec, attrs: list[str], tax: Taxonomy,
                 max_input_chars: int | None = None, enum_values_max: int | None = None) -> str:
    s = get_settings()
    budget = max_input_chars or s.llm_max_input_chars
    limit = enum_values_max or s.llm_enum_values_max
    text = card.full_text()
    specs = [category.attributes[a] for a in attrs]
    lines = [_attr_line(a, sp, *_allowed_values(sp, tax, text, limit)) for a, sp in zip(attrs, specs, strict=True)]
    fmt = (f"JSON-объект с ключами {', '.join(attrs)}; значение каждого ключа — "
           '{"evidence": "дословная цитата" или null, "value": "значение" или null}')
    card_lines = [f"Название: {card.title.strip()}"]
    if (card.description or "").strip():
        card_lines.append("Описание: " + fit_text(card.description, specs, budget))
    if card.seller_attrs:
        card_lines.append("Характеристики продавца:")
        card_lines += [f"{k}: {v}" for k, v in list(card.seller_attrs.items())[:80]]
    return (f"Категория: {category.title}.\nАтрибуты (ключ — название — что вернуть):\n" + "\n".join(lines)
            + f"\nФормат ответа: {fmt}\n\nКарточка:\n" + "\n".join(card_lines))


def locate(quote: str, text: str) -> tuple[int, int] | None:
    """[start, end) цитаты в тексте карточки: точно, иначе без учёта регистра, пробелов и ё/е."""
    quote = (quote or "").strip()
    if not quote:
        return None
    i = text.find(quote)
    if i >= 0:
        return i, i + len(quote)
    pattern = r"\s+".join(re.sub("[её]", "[её]", re.escape(w)) for w in quote.split())
    m = re.search(pattern, text, re.I)
    return (m.start(), m.end()) if m else None


_NULLS = {"", "null", "none", "нет", "не указано", "не указан", "n/a", "na", "-", "—", "unknown", "неизвестно"}


def build_request(card: Card, category: CategorySpec, tax: Taxonomy, attrs: list[str], settings=None, *,
                  few_shot: bool | None = None) -> tuple[str, str, dict]:
    """(system, user, json_schema) для вызова LLM — тот же запрос использует бенчмарк (attrx.llm.bench)."""
    s = settings or get_settings()
    text = card.full_text()
    enums = {}
    for a in attrs:
        spec = category.attributes[a]
        if spec.type == "enum" and s.llm_schema_enum:
            allowed, full = _allowed_values(spec, tax, text, s.llm_enum_values_max)
            if full:
                enums[a] = allowed
    system = SYSTEM + (FEW_SHOT if (s.llm_few_shot if few_shot is None else few_shot) else "")
    user = build_prompt(card, category, attrs, tax, s.llm_max_input_chars, s.llm_enum_values_max)
    return system, user, build_schema(attrs, enums)


def unmapped_note(attr: str, spec: AttrSpec, value: str, evidence: str, text: str) -> str:
    """Ответ LLM, который не привёлся к канону, — понятной строкой для ExtractionResult.warnings.

    Значение вне справочника с цитатой из карточки — кандидат на пополнение справочника (решает модератор, петля
    обратной связи); цитаты в тексте нет — это выдумка, сообщение то же, что у незаземлённых значений."""
    value = re.sub(r"\s+", " ", value).strip()
    grounded = bool(evidence) and align(evidence, text) is not None
    if spec.type == "enum" and not grounded:
        return f"llm: отброшено незаземлённое значение {attr}={value!r}"
    quote = f"цитата «{evidence}»" if grounded else "без цитаты из текста"
    if spec.type == "enum":
        return f'кандидат в справочник {spec.dictionary}: {attr} = "{value}" (LLM, {quote}) — в справочнике нет, отброшено'
    return f'llm: {attr} = "{value}" не приводится к каноническому виду ({quote}) — отброшено'


def parse_response(data: dict, card: Card, category: CategorySpec, tax: Taxonomy, attrs: list[str],
                   notes: list[str] | None = None) -> list[Candidate]:
    """JSON модели → кандидаты в каноническом виде (заземление проверяет pipeline через verify.ground).

    notes — необязательный список: сюда пишутся значения, которые не привелись к канону (unmapped_note)."""
    text = card.full_text()
    out: list[Candidate] = []
    for a in attrs:
        item = data.get(a)
        if isinstance(item, str):  # без json_schema некоторые модели отвечают {"attr": "значение"}
            item = {"value": item, "evidence": None}
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        if value is None or str(value).strip().lower() in _NULLS:
            continue
        spec = category.attributes[a]
        evidence = str(item.get("evidence") or "").strip()
        res = normalize_value(str(value), spec, tax.dictionaries)
        if res is None:
            if notes is not None:
                notes.append(unmapped_note(a, spec, str(value), evidence, text))
            continue
        canon, quality = res
        base = LLM_VERIFIED_SCORE if spec.type in {"number", "enum"} and evidence else LLM_BASE_SCORE
        out.append(Candidate(attribute=a, value=canon, raw=evidence or str(value), source="llm",
                             score=round(base * quality, 4), span=locate(evidence, text)))
    return out


def extract(card: Card, category: CategorySpec, tax: Taxonomy, attrs: list[str], client: LLMClient, *,
            few_shot: bool | None = None, notes: list[str] | None = None,
            prefilter: bool | None = None, asked: list[str] | None = None) -> tuple[list[Candidate], list[str]]:
    """Значения для attrs (слабые/пустые/конфликтные атрибуты) → (кандидаты source=llm, предупреждения).

    Предупреждения — только об ошибках вызова (pipeline по ним понимает, что LLM не сработала).
    notes — необязательный список для заметок о значениях, которые не привелись к канону (unmapped_note).
    asked — необязательный список: сюда пишутся атрибуты, о которых LLM действительно спросили (после префильтра);
    пустой — вызова не было (и ответа из кэша тоже).
    prefilter (по умолчанию LLM_PREFILTER=true) — не спрашивать об атрибутах, которых в тексте нет (relevant_attrs).
    """
    s = getattr(client, "s", None) or get_settings()
    attrs = [a for a in attrs if a in category.attributes]
    if s.llm_prefilter if prefilter is None else prefilter:
        attrs = relevant_attrs(card, category, tax, attrs)
    if not attrs or not client.enabled:
        return [], []
    if asked is not None:
        asked += attrs
    system, user, schema = build_request(card, category, tax, attrs, s, few_shot=few_shot)
    try:
        data = client.complete_json(system, user, schema, "attributes")
    except LLMUnavailable as e:
        return [], [f"llm: {e}"]
    return parse_response(data, card, category, tax, attrs, notes), []


def extract_batch(jobs: Sequence[tuple[Card, CategorySpec, list[str]]], tax: Taxonomy, client: LLMClient, *,
                  concurrency: int | None = None, few_shot: bool | None = None,
                  info: Sequence[dict] | None = None) -> list[tuple[list[Candidate], list[str]]]:
    """extract для пачки карточек параллельно (LLM_CONCURRENCY: 1 для Ollama, 4 для API); порядок сохраняется.

    jobs = [(card, category, weak_attrs), ...] — так pipeline.Extractor.extract_many делает LLM-шаг для пачки карточек.
    info — необязательный список словарей, по одному на задачу: в каждый пишутся notes и asked (как в extract)
    и seconds — время задачи.
    """
    def run(i: int) -> tuple[list[Candidate], list[str]]:
        card, category, attrs = jobs[i]
        t = time.perf_counter()
        notes: list[str] = []
        asked: list[str] = []
        out = extract(card, category, tax, attrs, client, few_shot=few_shot, notes=notes, asked=asked)
        if info is not None:
            info[i].update(notes=notes, asked=asked, seconds=time.perf_counter() - t)
        return out

    idx = list(range(len(jobs)))
    mapper = getattr(client, "map", None)
    return mapper(run, idx, concurrency) if mapper else [run(i) for i in idx]
