"""Оркестратор: категория -> продавец -> правила -> (LLM для непокрытого) -> проверка -> уверенность -> статус."""
from __future__ import annotations

from collections.abc import Iterable, Iterator

from .confidence import aggregate
from .config import Settings, get_settings
from .extractors import llm as llm_extractor
from .extractors import rules, seller
from .llm.client import LLMClient, get_client
from .normalize import display
from .schema import AttributeValue, Card, ExtractionResult
from .taxonomy import Taxonomy, load_taxonomy
from .verify import grounded


class Extractor:
    def __init__(self, taxonomy: Taxonomy | None = None, settings: Settings | None = None, client: LLMClient | None = None):
        self.s = settings or get_settings()
        self.tax = taxonomy or load_taxonomy(str(self.s.taxonomy_path))
        self.client = client or get_client()

    def extract(self, card: Card, use_llm: bool | None = None) -> ExtractionResult:
        cat_name, cat_conf = self.tax.detect_category(card)
        category = self.tax.categories[cat_name]
        warnings: list[str] = []
        cands = seller.extract(card, category, self.tax) + rules.extract(card, category, self.tax)

        def resolve():
            by_attr = {a: [c for c in cands if c.attribute == a] for a in category.attributes}
            return {a: aggregate(cs) for a, cs in by_attr.items()}

        agg = resolve()
        llm_used = False
        want_llm = self.client.enabled if use_llm is None else (use_llm and self.client.enabled)
        if want_llm:
            weak = [a for a, (v, conf, _, conflict) in agg.items() if v is None or conf < self.s.llm_route_threshold or conflict]
            if weak:
                new, warn = llm_extractor.extract(card, category, self.tax, weak, self.client)
                warnings += warn
                for c in new:
                    if grounded(c, card, category.attributes[c.attribute], self.tax.dictionaries):
                        cands.append(c)
                    else:
                        warnings.append(f"llm: отброшено незаземлённое значение {c.attribute}={c.value!r}")
                llm_used = not warn
                agg = resolve()

        attrs = []
        for a, spec in category.attributes.items():
            value, conf, support, conflict = agg[a]
            if value is None:
                status = "missing"
            elif conf >= self.s.confidence_threshold and not (conflict and conf < 0.85):
                status = "accepted"
            else:
                status = "review"
            if conflict:
                alts = sorted({str(c.value) for c in cands if c.attribute == a and str(c.value) != str(value)})
                warnings.append(f"{a}: конфликт значений, альтернативы: {', '.join(alts)}")
            attrs.append(AttributeValue(
                attribute=a, title=spec.title, value=value, unit=spec.unit if value is not None else None,
                display=display(value, spec), confidence=conf, status=status,
                sources=sorted({c.source for c in support}), evidence=support[0].raw if support else None,
            ))
        if cat_conf < 0.6:
            warnings.append("категория определена неуверенно — карточка на проверку")
        return ExtractionResult(card_id=card.id, category=cat_name, category_confidence=round(cat_conf, 3),
                                attributes=attrs, llm_used=llm_used, warnings=warnings)

    def extract_many(self, cards: Iterable[Card], use_llm: bool | None = None) -> Iterator[ExtractionResult]:
        for card in cards:
            yield self.extract(card, use_llm=use_llm)
