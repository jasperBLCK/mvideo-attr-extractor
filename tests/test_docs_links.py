"""Ссылки в документации ведут на существующие файлы: README.md, docs/**/*.md, .claude/context/*.md.

Проверяются относительные ссылки Markdown ([текст](путь), ![](картинка), [метка]: путь) и HTML (href/src);
внешние (http(s), mailto) и якоря (#раздел) пропускаются, у пути отрезаются #якорь и ?запрос.
Ссылки внутри блоков кода и `инлайн-кода` — не ссылки. Путь с «/» в начале — от корня репозитория."""
import re
from pathlib import Path
from urllib.parse import unquote

import pytest

from attrx.config import ROOT

DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md")), *sorted((ROOT / ".claude" / "context").glob("*.md"))]

_FENCE = re.compile(r"^[ \t]*(?P<f>```|~~~)[^\n]*\n.*?^[ \t]*(?P=f)[^\n]*$", re.M | re.S)
_CODE = re.compile(r"(`+)(?:(?!\1)[^\n])+?\1")
_INLINE = re.compile(r"!?\[(?:[^\[\]]|\[[^\]]*\])*\]\(\s*<?([^)\s>]+)>?(?:\s+[\"'(][^)]*)?\)")
_REFDEF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*<?(\S+?)>?(?:\s+.*)?$", re.M)
_HTML = re.compile(r"""\b(?:href|src)\s*=\s*["']([^"']+)["']""")
_EXTERNAL = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|//)", re.I)


def links(text: str) -> list[str]:
    text = _CODE.sub("", _FENCE.sub("", text))
    return _INLINE.findall(text) + _REFDEF.findall(text) + _HTML.findall(text)


def local_target(md: Path, link: str) -> Path | None:
    """Файл, на который указывает ссылка, или None — внешняя ссылка или якорь на этой же странице."""
    if _EXTERNAL.match(link) or link.startswith("#"):
        return None
    path = unquote(link.split("#", 1)[0].split("?", 1)[0])
    if not path:
        return None
    return ROOT / path.lstrip("/") if path.startswith("/") else md.parent / path


def test_parser_sees_links_and_skips_code():
    sample = ("[a](docs/A.md#x) ![](img/b.png) [c](https://x.ru) [d](#local) <a href=\"web/README.md\">e</a>\n"
              "`[не ссылка](nope.md)`\n```\n[в коде](nope2.md)\n```\n[ref]: CONTRIBUTING.md \"титул\"\n")
    assert links(sample) == ["docs/A.md#x", "img/b.png", "https://x.ru", "#local", "CONTRIBUTING.md", "web/README.md"]
    md = ROOT / "docs" / "X.md"
    assert local_target(md, "../README.md#раздел") == ROOT / "docs" / ".." / "README.md"
    assert local_target(md, "https://example.com/a.md") is None and local_target(md, "#якорь") is None
    assert local_target(md, "/web/README.md") == ROOT / "web" / "README.md"
    assert local_target(md, "img/%D1%84.png") == ROOT / "docs" / "img" / "ф.png"


def test_docs_have_links_to_check():
    assert ROOT / "README.md" in DOCS and len(DOCS) > 10
    assert len(links((ROOT / "README.md").read_text(encoding="utf-8"))) >= 20  # регулярка не «ослепла»


@pytest.mark.parametrize("md", DOCS, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_relative_links_resolve(md: Path):
    broken = []
    for link in links(md.read_text(encoding="utf-8")):
        target = local_target(md, link)
        if target is not None and not target.exists():
            broken.append(link)
    assert not broken, f"{md.relative_to(ROOT)}: битые ссылки {broken}"
