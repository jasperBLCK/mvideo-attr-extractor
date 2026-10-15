"""Оркестратор: категория -> продавец -> правила -> (NER/LLM для непокрытого) -> заземление -> агрегация ->
калиброванная уверенность -> статус.

Правила статуса (в каталог попадают только надёжные значения, сомнительные — на ручную проверку):
- accepted: калиброванная уверенность >= порога (config/calibration.json, выбран на dev под точность принятых
  >= 0.95; переопределяется CONFIDENCE_THRESHOLD) и нет сильного конфликта источников;
- **значение, которое предложила ТОЛЬКО LLM, никогда не принимается автоматически** — review, пока его не
  подтвердит другой источник (продавец, правила, NER, человек), даже если цитата заземлена и уверенность высокая;
- значение, с которым спорит ДРУГОЙ источник (продавец указал одно, в тексте другое), — review;
- review: ниже порога, конфликт или только LLM; missing: значения нет.
"""
from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from .confidence import CONFLICT_ACCEPT, Aggregate, aggregate_full, load_calibrator
from .config import Settings, get_settings
from .extractors import llm as llm_extractor
from .extractors import ner, rules, seller
from .llm.client import LLMClient, get_client
from .normalize import display
from .schema import AttributeValue, Candidate, Card, ExtractionResult
from .taxonomy import (
    CategoryDecision,
    CategorySpec,
    Taxonomy,
    attr_traffic_weight,
    category_info,
    load_taxonomy,
)
from .verify import ground

ALL_SOURCES = frozenset({"seller", "rules", "ml", "llm"})


def value_key(spec):
    """Ключ «одного и того же значения» строкового атрибута для агрегации — тот же, что у метрик: модель без памяти,
    цвета и регистра, процессор без производителя и частот. Источники, записавшие одно значение по-разному, не спорят."""
    if spec.kind == "model":
        from .eval.gold import model_key

        return model_key
    if spec.kind == "cpu":
        from .normalize import cpu_key

        return cpu_key
    return None
UNVERIFIED_ONLY = frozenset({"llm"})  # источники, которых одних недостаточно для автопринятия
# extract_many с NER или LLM: карточки идут пачками — продавец и правила по очереди, один прогон NER на пачку,
# LLM-запросы пачки параллельно
BATCH = 32


@dataclass
class _State:
    """Карточка между шагами пайплайна (extract_many делает шаги NER и LLM сразу для пачки карточек)."""

    card: Card
    decision: CategoryDecision
    category: CategorySpec
    sources: frozenset[str] | set[str]
    t0: float
    cands: list[Candidate] = field(default_factory=list)
    agg: dict[str, Aggregate] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    llm_used: bool = False


class Extractor:
    def __init__(self, taxonomy: Taxonomy | None = None, settings: Settings | None = None, client: LLMClient | None = None,
                 calibrated: bool = True):
        self.s = settings or get_settings()
        self.tax = taxonomy or load_taxonomy(str(self.s.taxonomy_path))
        self.client = client or get_client()
        self.ml = ner.NerExtractor.load(self.s.ml_model_path) if self.s.ml_enabled else None
        # calibrated=False — сырые оценки (так их собирает `attrx calibrate`)
        self.calibrator = load_calibrator(str(self.s.calibration_path)) if calibrated else load_calibrator("")

    def _finalize(self, agg: Aggregate, spec, category: str, attr: str) -> tuple[float, str]:
        """(калиброванная уверенность, статус)."""
        if agg.value is None:
            return 0.0, "missing"
        conf = self.calibrator.monotone(agg, spec, category, attr) if self.calibrator.enabled else agg.confidence
        if set(agg.sources) <= UNVERIFIED_ONLY:
            return conf, "review"  # инвариант доверия: только-LLM значение не попадает в каталог без проверки
        if agg.cross_conflict:
            return conf, "review"  # источники расходятся (продавец vs текст карточки) — решает модератор
        accepted = conf >= self.s.confidence_threshold and not (agg.conflict and conf < CONFLICT_ACCEPT)
        return conf, "accepted" if accepted else "review"

    # ------------------------------------------------------------------ шаги
    @staticmethod
    def _lap(st: _State, name: str, t0: float) -> None:
        st.timings[name] = round(st.timings.get(name, 0.0) + (time.perf_counter() - t0) * 1000, 3)

    @staticmethod
    def _resolve(st: _State) -> dict[str, Aggregate]:
        by_attr: dict[str, list[Candidate]] = {a: [] for a in st.category.attributes}
        for c in st.cands:
            if c.attribute in by_attr:
                by_attr[c.attribute].append(c)
        return {a: aggregate_full(cs, value_key(st.category.attributes[a])) for a, cs in by_attr.items()}

    def _checked(self, st: _State, new: list[Candidate], source: str) -> list[Candidate]:
        """Заземление кандидатов модели: цитата выравнивается с текстом, значение должно из неё следовать."""
        out = []
        for c in new:
            g = ground(c, st.card, st.category.attributes[c.attribute], self.tax.dictionaries)
            if g is not None:
                out.append(g)
            elif source == "llm":
                st.warnings.append(f"llm: отброшено незаземлённое значение {c.attribute}={c.value!r}")
        return out

    def _weak(self, st: _State) -> list[str]:
        """Атрибуты, которые дешёвые источники не закрыли уверенно: нет значения, слабая уверенность или конфликт."""
        return [a for a, x in st.agg.items()
                if x.value is None or x.confidence < self.s.llm_route_threshold or x.conflict]

    def _want_llm(self, use_llm: bool | None, sources: Iterable[str]) -> bool:
        return "llm" in sources and (self.client.enabled if use_llm is None else (use_llm and self.client.enabled))

    def _start(self, card: Card, sources: set[str] | None) -> _State:
        """Дешёвые шаги: категория -> продавец -> правила -> агрегация."""
        t = time.perf_counter()
        decision = self.tax.classify(card)
        st = _State(card, decision, self.tax.categories[decision.name], sources or ALL_SOURCES, t0=t)
        self._lap(st, "category", t)
        if "seller" in st.sources:
            t = time.perf_counter()
            st.cands += seller.extract(card, st.category, self.tax)
            self._lap(st, "seller", t)
        if "rules" in st.sources:
            t = time.perf_counter()
            st.cands += rules.extract(card, st.category, self.tax)
            self._lap(st, "rules", t)
        st.agg = self._resolve(st)
        return st

    def _use_ner(self, sources: Iterable[str]) -> bool:
        return "ml" in sources and self.ml is not None

    def _apply_ner(self, st: _State, found: list[Candidate], weak: set[str]) -> None:
        """Кандидаты NER — только для слабых атрибутов и только заземлённые (отброшенные молча)."""
        st.cands += self._checked(st, [c for c in found if c.attribute in weak], "ml")
        st.agg = self._resolve(st)

    def _apply_llm(self, st: _State, new: list[Candidate], warn: list[str], notes: list[str], asked: list[str],
                   seconds: float) -> None:
        """Ответ LLM для карточки: заземлённые кандидаты — в агрегацию; ошибки вызова, незаземлённое и значения вне
        справочника (кандидаты на пополнение — для модератора и петли обратной связи) — в предупреждения.
        llm_used — только если LLM действительно спросили (вызов или ответ из кэша) и ошибки не было."""
        st.warnings += warn
        st.cands += self._checked(st, new, "llm")
        st.warnings += notes
        st.llm_used = bool(asked) and not warn
        st.agg = self._resolve(st)
        st.timings["llm"] = round(st.timings.get("llm", 0.0) + seconds * 1000, 3)

    def _finish(self, st: _State, wall_clock: bool = True) -> ExtractionResult:
        """Калибровка -> статус -> результат. wall_clock=False (пачка) — total = сумма этапов этой карточки."""
        t = time.perf_counter()
        decision, category, cat_name = st.decision, st.category, st.decision.name
        title_desc = rules.title_desc_conflicts(st.card, st.cands, category)
        attrs = []
        for a, spec in category.attributes.items():
            x = st.agg[a]
            conf, status = self._finalize(x, spec, cat_name, a)
            if decision.review and status == "accepted":
                status = "review"  # категория не уверена — набор атрибутов мог быть не тот: всё на проверку
            if a in title_desc and status == "accepted" and rules.TITLE_DESC_POLICY == "review":
                status = "review"  # название и описание расходятся в числе — побеждает название, решает модератор
            if x.conflict and x.value is not None:
                alts = ", ".join(str(alt["value"]) for alt in x.alternatives)
                st.warnings.append(f"{a}: конфликт значений, альтернативы: {alts}")
            best = max(x.support, key=lambda c: (c.span is not None, c.score)) if x.support else None
            attrs.append(AttributeValue(
                attribute=a, title=spec.title, value=x.value, unit=spec.unit if x.value is not None else None,
                display=display(x.value, spec), confidence=round(conf, 3), status=status, sources=x.sources,
                evidence=best.raw if best else None, evidence_span=best.span if best else None,
                alternatives=x.alternatives if x.value is not None else [],
                required=spec.required, traffic_weight=attr_traffic_weight(st.card, spec),
            ))
        self._lap(st, "aggregate", t)
        if decision.review:
            st.warnings.append(f"категория определена неуверенно ({decision.source}: {decision.title or category.title}, "
                               f"{decision.confidence:.2f}) — карточка на проверку")
        title, path = (decision.title, decision.path) if decision.title else category_info(st.card, category)
        st.timings["total"] = round((time.perf_counter() - st.t0) * 1000 if wall_clock else sum(st.timings.values()), 3)
        return ExtractionResult(card_id=st.card.id, category=cat_name, category_confidence=round(decision.confidence, 3),
                                attributes=attrs, llm_used=st.llm_used, warnings=st.warnings, category_title=title,
                                category_path=path, timings_ms=st.timings)

    # ------------------------------------------------------------------ точки входа
    def extract(self, card: Card, use_llm: bool | None = None, sources: set[str] | None = None) -> ExtractionResult:
        """sources — какие источники включить (seller, rules, ml, llm); по умолчанию все доступные."""
        st = self._start(card, sources)
        # модели — только для хвоста: атрибутов, которые продавец и правила не закрыли уверенно
        if self._use_ner(st.sources) and (weak := set(self._weak(st))):
            t = time.perf_counter()
            self._apply_ner(st, self.ml.extract(card, st.category, self.tax), weak)
            self._lap(st, "ml", t)
        if self._want_llm(use_llm, st.sources) and (weak_llm := self._weak(st)):
            t = time.perf_counter()
            notes: list[str] = []
            asked: list[str] = []
            new, warn = llm_extractor.extract(card, st.category, self.tax, weak_llm, self.client, notes=notes,
                                              asked=asked)
            self._apply_llm(st, new, warn, notes, asked, time.perf_counter() - t)
        return self._finish(st)

    def extract_many(self, cards: Iterable[Card], use_llm: bool | None = None,
                     sources: set[str] | None = None) -> Iterator[ExtractionResult]:
        """Карточки по порядку. С NER или LLM — пачками по BATCH: продавец и правила для каждой карточки, затем один
        прогон NER на слабые атрибуты всей пачки (NerExtractor.extract_batch), затем LLM-запросы пачки параллельно
        (extractors.llm.extract_batch, LLM_CONCURRENCY), затем финализация. Порядок результатов — как у входа,
        значения — те же, что у extract() по одной карточке; timings_ms["ml"] — доля карточки во времени прогона пачки."""
        src = sources or ALL_SOURCES
        want_llm = self._want_llm(use_llm, src)
        if not (want_llm or self._use_ner(src)):
            for card in cards:
                yield self.extract(card, use_llm=False, sources=sources)
            return
        chunk: list[Card] = []
        for card in cards:
            chunk.append(card)
            if len(chunk) >= BATCH:
                yield from self._extract_chunk(chunk, sources, want_llm)
                chunk = []
        if chunk:
            yield from self._extract_chunk(chunk, sources, want_llm)

    def _extract_chunk(self, cards: list[Card], sources: set[str] | None, want_llm: bool) -> list[ExtractionResult]:
        states = [self._start(card, sources) for card in cards]
        if self._use_ner(sources or ALL_SOURCES):
            todo = [(st, weak) for st in states if (weak := set(self._weak(st)))]
            if todo:
                t = time.perf_counter()
                found = self.ml.extract_batch([(st.card, st.category) for st, _ in todo], self.tax)
                share = (time.perf_counter() - t) / len(todo)
                for (st, weak), cands in zip(todo, found, strict=True):
                    t1 = time.perf_counter()
                    self._apply_ner(st, cands, weak)
                    st.timings["ml"] = round(st.timings.get("ml", 0.0) + (share + time.perf_counter() - t1) * 1000, 3)
        if want_llm:
            jobs = [(st, weak) for st in states if (weak := self._weak(st))]
            info: list[dict] = [{} for _ in jobs]
            res = llm_extractor.extract_batch([(st.card, st.category, weak) for st, weak in jobs], self.tax,
                                              self.client, info=info)
            for (st, _), (new, warn), inf in zip(jobs, res, info, strict=True):
                self._apply_llm(st, new, warn, inf.get("notes", []), inf.get("asked", []), inf.get("seconds", 0.0))
        return [self._finish(st, wall_clock=False) for st in states]
