from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from .config import get_settings
from .schema import Card


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace("ё", "е")).strip()


@dataclass
class AttrSpec:
    name: str
    title: str
    type: str
    unit: str | None = None
    kind: str | None = None
    dictionary: str | None = None
    aliases: list[str] = field(default_factory=list)
    traffic_weight: float = 1.0


@dataclass
class CategorySpec:
    name: str
    title: str
    keywords: list[str]
    attributes: dict[str, AttrSpec]


ADJ_END = re.compile(r"(ый|ий|ой)$")
ADJ_FORMS = "(?:ый|ий|ой|ая|яя|ое|ее|ые|ие|ого|его|ую|юю)"


def _inflect(syn: str) -> str:
    """«чёрный» матчит «черная/черные/черного»: продавцы пишут прилагательные в любом роде и числе."""
    return " ".join(ADJ_END.sub("", re.escape(w)) + ADJ_FORMS if ADJ_END.search(w) and len(w) > 4 else re.escape(w)
                    for w in syn.split(" "))


class Dictionary:
    """Справочник допустимых значений: канон + синонимы. Защищает от выдуманных значений."""

    def __init__(self, name: str, mapping: dict[str, list[str]]):
        self.name = name
        self.canon = list(mapping)
        self.lookup: dict[str, str] = {}
        for canon, syns in mapping.items():
            for s in [canon, *(syns or [])]:
                self.lookup[norm(str(s))] = canon

    def add(self, synonym: str, canon: str) -> None:
        if canon not in self.canon:
            self.canon.append(canon)
        self.lookup[norm(synonym)] = canon

    def get(self, text: str) -> str | None:
        t = norm(text)
        if t in self.lookup:
            return self.lookup[t]
        hits = self.find_all(t)
        return hits[0][0] if hits and hits[0][1] == t else None

    def find_all(self, text: str) -> list[tuple[str, str, int, int]]:
        """Все вхождения синонимов в тексте: (canon, raw, start, end). Длинные синонимы приоритетнее."""
        t = norm(text)
        found: list[tuple[str, str, int, int]] = []
        taken: list[tuple[int, int]] = []
        for syn in sorted(self.lookup, key=len, reverse=True):
            for m in re.finditer(r"(?<![\w-])" + _inflect(syn) + r"(?![\w-])", t):
                if any(m.start() < e and s < m.end() for s, e in taken):
                    continue
                taken.append((m.start(), m.end()))
                found.append((self.lookup[syn], m.group(0), m.start(), m.end()))
        return sorted(found, key=lambda x: x[2])


class Taxonomy:
    def __init__(self, path: Path):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.categories: dict[str, CategorySpec] = {}
        for cname, c in data["categories"].items():
            attrs = {a: AttrSpec(name=a, **spec) for a, spec in c["attributes"].items()}
            self.categories[cname] = CategorySpec(cname, c.get("title", cname), c.get("keywords", []), attrs)
        self.dictionaries: dict[str, Dictionary] = {}
        ddir = path.parent / "dictionaries"
        for f in sorted(ddir.glob("*.yaml")):
            if f.stem == "learned":
                continue
            self.dictionaries[f.stem] = Dictionary(f.stem, yaml.safe_load(f.read_text(encoding="utf-8")) or {})
        learned = ddir / "learned.yaml"
        if learned.exists():
            for dname, pairs in (yaml.safe_load(learned.read_text(encoding="utf-8")) or {}).items():
                for syn, canon in (pairs or {}).items():
                    self.dictionaries.setdefault(dname, Dictionary(dname, {})).add(syn, canon)

    def detect_category(self, card: Card) -> tuple[str, float]:
        if card.category:
            c = norm(card.category)
            for name, spec in self.categories.items():
                if c == name or c == norm(spec.title) or any(norm(k) in c for k in spec.keywords):
                    return name, 1.0
        title, text = norm(card.title), norm(card.full_text())
        best, score = "generic", 0.0
        for name, spec in self.categories.items():
            s = sum(2.0 if norm(k) in title else 1.0 for k in spec.keywords if norm(k) in text)
            if s > score:
                best, score = name, s
        return best, (min(0.95, 0.6 + 0.15 * score) if score else 0.3)


@lru_cache(maxsize=4)
def load_taxonomy(path: str | None = None) -> Taxonomy:
    return Taxonomy(Path(path) if path else get_settings().taxonomy_path)
