"""Очистка описаний: HTML -> текст и выделение блока «Характеристики:» в seller_attrs.

Варианты в данных: обычный текст с «\\n• Ключ: значение», HTML (<br>, <p>, <ul><li>), двойное экранирование
(«&lt;br/&gt;» внутри <p>), заголовки «Характеристики», «Технические характеристики», «Основные характеристики
товара:», маркеры «•», «-», «*», «–». Блок вырезается из описания, пары ключ-значение сохраняются по порядку.
"""
from __future__ import annotations

import html
import re
import unicodedata

# --------------------------------------------------------------------------- HTML -> текст

_SCRIPT = re.compile(r"(?is)<(script|style)\b.*?</\1\s*>")
_COMMENT = re.compile(r"(?s)<!--.*?-->")
_BR = re.compile(r"(?i)<\s*br\s*/?\s*>")
_LI = re.compile(r"(?i)<\s*li\b[^<>]*>")
_LI_END = re.compile(r"(?i)</\s*li\s*>")  # следующий <li> сам начнёт строку — без пустой строки между пунктами
_BLOCK = re.compile(r"(?i)</?\s*(?:p|div|ul|ol|h[1-6]|tr|table|tbody|thead|blockquote|section|article|dl|dt|dd)\b[^<>]*>")
_CELL_END = re.compile(r"(?i)</\s*t[dh]\s*>")
# только «похожее на тег»: буква сразу после < или </ — «< 5 мм» и «<5 мм» не трогаем
_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9:-]*(?:\s[^<>]*)?/?>")

_SPACES = dict.fromkeys(map(ord, "\xa0\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\t\f\v"), " ")
_ZERO = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\xad\u200e\u200f"), None)
_MULTI_SPACE = re.compile(r" {2,}")
_MULTI_NL = re.compile(r"\n{3,}")
_DOUBLE_BULLET = re.compile(r"•\s*(?=[•\-*–])")
_LONE_BULLET = re.compile(r"(?m)^([•▪●◦‣·*–-])\n+(?=\S)")  # «•\nКлюч: значение» -> «• Ключ: значение»


def _tags_to_text(s: str) -> str:
    s = _SCRIPT.sub(" ", s)
    s = _COMMENT.sub(" ", s)
    s = _BR.sub("\n", s)
    s = _LI.sub("\n• ", s)
    s = _LI_END.sub("", s)
    s = _BLOCK.sub("\n", s)
    s = _CELL_END.sub(" ", s)
    return _TAG.sub("", s)


def normalize_ws(s: str) -> str:
    """Переводы строк -> \\n, nbsp и прочие пробелы -> пробел, без пустых хвостов и тройных переводов."""
    s = s.replace("\r\n", "\n").replace("\r", "\n").translate(_SPACES).translate(_ZERO)
    s = "\n".join(_MULTI_SPACE.sub(" ", line).strip() for line in s.split("\n"))
    s = _LONE_BULLET.sub(r"\1 ", _DOUBLE_BULLET.sub("", s))
    return _MULTI_NL.sub("\n\n", s).strip()


def clean_html(s: str | None) -> str:
    """HTML/сущности -> чистый текст с сохранением абзацев и маркеров списков."""
    if not s:
        return ""
    s = unicodedata.normalize("NFC", str(s))
    for _ in range(3):  # до трёх уровней экранирования: &amp;lt;br&amp;gt; -> &lt;br&gt; -> <br>
        if "<" in s:
            s = _tags_to_text(s)
        if "&" not in s:
            break
        u = html.unescape(s)
        if u == s:
            break
        s = u
    if "<" in s:
        s = _tags_to_text(s)
    return normalize_ws(s)


def clean_title(s: str | None) -> str:
    return " ".join(clean_html(s).split())


# --------------------------------------------------------------------------- блок «Характеристики»

_HEADER = re.compile(
    r"^(?:основные|технические|подробные|ключевые|общие|все)?\s*характеристики"
    r"(?:\s+(?:товара|модели|изделия|продукта|устройства))?\s*[:：.]?$",
    re.IGNORECASE,
)
_BULLETS = "•-*–—·▪●◦‣►✓✔"
_B = re.escape(_BULLETS)
_KV = re.compile(rf"^(?P<b>[{_B}]\s*)?(?P<k>[^:\n{_B}][^:\n]{{0,79}}?)\s*:\s*(?P<v>\S.*?)$")
_HAS_LETTER = re.compile(r"[^\W\d_]")
_INLINE_HEADER = re.compile(
    r"(?:^|(?<=[.!?…)»\"]\s))(?:основные |технические )?характеристики(?: товара| модели| изделия)?\s*:\s*",
    re.IGNORECASE,
)
_SEGMENT = re.compile(r"(?<=[.;])\s+(?=[•–-]\s*\S|[A-ZА-ЯЁ])")  # граница «пары» внутри строки
_MIN_HEADERLESS = 3  # маркированный список «• Ключ: значение» без заголовка — только от 3 строк подряд


def parse_kv(line: str) -> tuple[str, str, bool] | None:
    """«• Ключ: значение» -> (ключ, значение, был_маркер); None, если строка не похожа на пару."""
    m = _KV.match(line.strip())
    if not m:
        return None
    key = m["k"].strip(" \t-–—")
    val = m["v"].strip().rstrip(" .;,").strip()
    if not key or not val or len(key.split()) > 8 or not _HAS_LETTER.search(key) or len(val) > 500:
        return None
    if key.lower().startswith(("http", "www")):
        return None
    return key, val, m["b"] is not None


def _take_block(lines: list[str], start: int, bulleted: bool | None) -> tuple[list[tuple[str, str]], int]:
    """Пары подряд начиная со start (пустые строки между ними допустимы). -> (пары, индекс последней пары)."""
    pairs, last, j = [], -1, start
    while j < len(lines):
        line = lines[j].strip()
        if not line:
            j += 1
            continue
        kv = parse_kv(line)
        if kv is None:
            break
        if bulleted is None:
            bulleted = kv[2]
        elif kv[2] != bulleted:  # смена стиля (маркеры -> обычный текст «Важно: ...») — конец блока
            break
        pairs.append((kv[0], kv[1]))
        last = j
        j += 1
    return pairs, last


def _split_inline(line: str) -> tuple[str, list[tuple[str, str]]]:
    """«Текст. Характеристики: Материал: сталь. Длина: 14 см.» -> («Текст.», пары); от двух пар подряд."""
    m = _INLINE_HEADER.search(line)
    if not m or m.end() >= len(line):
        return line, []
    segs = _SEGMENT.split(line[m.end():])
    pairs = []
    for seg in segs:
        kv = parse_kv(seg)
        if kv is None:
            break
        pairs.append((kv[0], kv[1]))
    if len(pairs) < 2:
        return line, []
    rest = " ".join(segs[len(pairs):]).strip()
    return " ".join(x for x in (line[: m.start()].strip(), rest) if x), pairs


def split_characteristics(text: str) -> tuple[str, dict[str, str]]:
    """Вырезает блоки «Характеристики:» (и длинные маркированные списки пар без заголовка) из очищенного текста.

    -> (описание без блока, {ключ: значение} в порядке появления; повтор ключа — остаётся первое значение).
    """
    if not text:
        return "", {}
    lines = text.split("\n")
    attrs: dict[str, str] = {}
    keep: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        pairs, last = [], -1
        if _HEADER.match(line):
            pairs, last = _take_block(lines, i + 1, None)
        elif line[:1] in _BULLETS and (kv := parse_kv(line)) and kv[2]:
            pairs, last = _take_block(lines, i, True)
            if len(pairs) < _MIN_HEADERLESS:
                pairs, last = [], -1
        elif "арактеристик" in line:
            rest, inline = _split_inline(line)
            if inline:
                for k, v in inline:
                    attrs.setdefault(k, v)
                if rest:
                    keep.append(rest)
                i += 1
                continue
        if pairs:
            for k, v in pairs:
                attrs.setdefault(k, v)
            i = last + 1
            continue
        keep.append(lines[i])
        i += 1
    if not attrs:
        return text, {}
    return _MULTI_NL.sub("\n\n", "\n".join(keep)).strip(), attrs


def prepare_description(raw: str | None) -> tuple[str, dict[str, str]]:
    """Сырое описание из sku_desc -> (чистый текст без блока характеристик, seller_attrs)."""
    return split_characteristics(clean_html(raw))
