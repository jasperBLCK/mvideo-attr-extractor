"""FastAPI-сервис: можно встроить в контур М.Видео как микросервис или запускать батчами из CLI.

Кроме извлечения здесь же живут эндпоинты для интерфейса модератора (web/):
примеры карточек, очередь проверки с приоритетом, сводка метрик из submission/ и справочник категорий.

Безопасность по умолчанию: `attrx serve` слушает 127.0.0.1 (наружу — явно `--host 0.0.0.0`, так в Docker и на ВМ).
Если задан ATTRX_TOKEN, запись решений модератора (POST /api/feedback, /api/feedback/undo) требует заголовок
X-Attrx-Token с этим значением: из журнала правок `attrx learn` учит справочники, его нельзя писать анонимно.
Без ATTRX_TOKEN (по умолчанию) всё работает как раньше — интерфейс модератора токен не шлёт.
"""
from __future__ import annotations

import csv
import hmac
import json
import mimetypes
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from .. import __version__
from ..config import ROOT, get_settings
from ..feedback import decision_action, log_correction
from ..genui.composer import compose_default, compose_llm
from ..genui.schema import UISpec
from ..io import read_predictions
from ..llm.status import llm_status
from ..pipeline import Extractor
from ..schema import Card, ExtractionResult

# Windows берёт MIME-типы из реестра и иногда отдаёт .js как text/plain — тогда ES-модули не грузятся.
for _ext, _type in ((".js", "text/javascript"), (".mjs", "text/javascript"), (".css", "text/css"),
                    (".svg", "image/svg+xml"), (".woff2", "font/woff2"), (".json", "application/json")):
    mimetypes.add_type(_type, _ext)

app = FastAPI(title="attrx — автозаполнение характеристик", version=__version__)
_extractor: Extractor | None = None

SAMPLE_DIR = ROOT / "data" / "sample"
REAL_CARDS = SAMPLE_DIR / "real_cards.jsonl"  # реальный сэмпл карточек М.Видео (готовит поток данных)
SYNTH_CARDS = SAMPLE_DIR / "cards.jsonl"  # синтетический smoke-набор
# витрина для показа: карточки, где хвост дозаполняют ML и LLM; demo_note — одна строка, что именно там видно
DEMO_CARDS = SAMPLE_DIR / "demo_cards.jsonl"
DEMO_CATEGORY = "★ Демо: ИИ дозаполняет"
SUBMISSION = ROOT / "submission"
REPORT_FILES = ("metrics", "compare", "segments", "calibration", "threshold", "cost", "traffic")
NO_CATEGORY = "Без категории"
_feedback_lock = threading.Lock()

TOKEN_ENV = "ATTRX_TOKEN"
TOKEN_HEADER = "X-Attrx-Token"
# use_llm=true, а LLM не настроена: извлечение не падает, но клиент должен об этом узнать
LLM_OFF_WARNING = "LLM не настроена — использованы правила и NER"
LLM_OFF_NO_NER_WARNING = "LLM не настроена — использованы правила и словари (NER не загружен)"


def write_token() -> str | None:
    """Токен на запись (ATTRX_TOKEN; читается при каждом запросе, .env подхватывается config.py) или None."""
    return (os.getenv(TOKEN_ENV) or "").strip() or None


def require_write_token(token: Annotated[str | None, Header(
        alias=TOKEN_HEADER, description="нужен, только если на сервере задан ATTRX_TOKEN")] = None) -> None:
    """Зависимость эндпоинтов, которые пишут в журнал решений: без ATTRX_TOKEN — открыты (по умолчанию)."""
    expected = write_token()
    if expected is None:
        return
    if not token or not hmac.compare_digest(token.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(401, f"Запись решений защищена токеном: передайте заголовок {TOKEN_HEADER} "
                                 f"со значением {TOKEN_ENV} сервера")


def extractor() -> Extractor:
    global _extractor
    if _extractor is None:
        _extractor = Extractor()
    return _extractor


def _rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


# --------------------------------------------------------------------------- модели запросов/ответов

class TextBlock(BaseModel):
    kind: Literal["title", "description", "seller"]
    start: int
    end: int


class SellerPair(BaseModel):
    key: str
    value: str
    key_span: tuple[int, int]
    value_span: tuple[int, int]


class CardText(BaseModel):
    """Card.full_text() с разметкой блоков: по этим смещениям UI подсвечивает evidence_span."""

    full: str
    blocks: list[TextBlock] = Field(default_factory=list)
    seller_pairs: list[SellerPair] = Field(default_factory=list)


class ExtractRequest(BaseModel):
    card: Card = Field(description="карточка; id необязателен — без него генерируется по тексту (auto-…)")
    use_llm: bool | None = Field(None, description="LLM для хвоста: null — по настройке сервера; true без настроенной "
                                                   "LLM — работает на правилах и NER, в result.warnings предупреждение")
    intent: str | None = None


class ExtractResponse(BaseModel):
    result: ExtractionResult
    ui: UISpec
    ui_warning: str | None = None
    text: CardText | None = None


class BatchRequest(BaseModel):
    cards: list[Card] = Field(max_length=1000)
    use_llm: bool | None = None


class UIRequest(BaseModel):
    result: ExtractionResult
    intent: str = ""


class Feedback(BaseModel):
    card_id: str
    attribute: str
    predicted: str | float | None = None
    corrected: str | float | None = None
    # формулировка значения в тексте карточки (фрагмент evidence_span; у поля продавца — без «Ключ:»)
    raw: str | None = None
    # категория таксономии карточки: по ней attrx learn выбирает справочник (type у наушников и пылесосов — разные)
    category: str | None = None


class FeedbackKey(BaseModel):
    card_id: str
    attribute: str


# --------------------------------------------------------------------------- вспомогательное

def card_text(card: Card) -> CardText:
    """Делит Card.full_text() на блоки (название / описание / поля продавца) без копирования его формата:
    каждый кусок ищется в полном тексте, и если формат full_text() поменяется — отдаём текст одним блоком."""
    full = card.full_text()
    parts = [("title", card.title), ("description", card.description),
             ("seller", "; ".join(f"{k}: {v}" for k, v in card.seller_attrs.items()))]
    blocks: list[TextBlock] = []
    pos = 0
    for kind, s in parts:
        if not s:
            continue
        i = full.find(s, pos)
        if i < 0:
            return CardText(full=full, blocks=[TextBlock(kind="description", start=0, end=len(full))])
        blocks.append(TextBlock(kind=kind, start=i, end=i + len(s)))
        pos = i + len(s)
    pairs: list[SellerPair] = []
    seller = next((b for b in blocks if b.kind == "seller"), None)
    if seller:
        p = seller.start
        for k, v in card.seller_attrs.items():
            ks, vs = str(k), str(v)
            k0, k1 = p, p + len(ks)
            v0, v1 = k1 + 2, k1 + 2 + len(vs)
            if full[k0:k1] != ks or full[v0:v1] != vs:
                pairs = []
                break
            pairs.append(SellerPair(key=ks, value=vs, key_span=(k0, k1), value_span=(v0, v1)))
            p = v1 + 2
    return CardText(full=full, blocks=blocks, seller_pairs=pairs)


def _examples_file() -> tuple[Path | None, bool]:
    """(файл, синтетика?) — реальный сэмпл, если он уже есть, иначе синтетический."""
    if REAL_CARDS.exists():
        return REAL_CARDS, False
    if SYNTH_CARDS.exists():
        return SYNTH_CARDS, True
    return None, True


_cards_cache: dict[Path, tuple[float, list[dict]]] = {}


def _load_examples(path: Path) -> list[dict]:
    """Карточки из JSONL (валидируются схемой Card; лишние поля вроде category_path сохраняются). Кэш по mtime."""
    mtime = path.stat().st_mtime
    hit = _cards_cache.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            card = Card.model_validate(raw)
        except (json.JSONDecodeError, ValidationError):
            continue
        item = card.model_dump()
        for extra in ("category_path", "url", "demo_note"):
            if extra in raw:
                item[extra] = raw[extra]
        out.append(item)
    _cards_cache[path] = (mtime, out)
    return out


def _demo_cards() -> list[dict]:
    """Карточки витрины (data/sample/demo_cards.jsonl) с пометкой demo; файла нет — пусто."""
    if not DEMO_CARDS.exists():
        return []
    return [{**c, "demo": True} for c in _load_examples(DEMO_CARDS)]


def _card_index() -> dict[str, dict]:
    """id -> карточка из всех доступных сэмплов (нужно очереди, чтобы показать цитату в контексте)."""
    idx: dict[str, dict] = {}
    for p in (SYNTH_CARDS, REAL_CARDS):  # реальные перекрывают синтетику при совпадении id
        if p.exists():
            idx.update({str(c["id"]): c for c in _load_examples(p)})
    return idx


def _predictions_path() -> Path | None:
    """Последний прогон attrx extract, иначе предсказания из сдачи (submission/)."""
    for p in (get_settings().predictions_path, SUBMISSION / "predictions.jsonl"):
        if p.exists():
            return p
    return None


def _feedback_records() -> list[dict]:
    p = get_settings().feedback_path
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _decision(rec: dict) -> dict:
    """Что сделал модератор: подтвердил, исправил или отклонил значение (то же правило, что у attrx learn)."""
    return {"action": decision_action(rec), "corrected": rec.get("corrected"), "ts": rec.get("ts")}


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _report_note() -> str | None:
    """Пояснение к набору данных из submission/REPORT.md (строки между заголовком и «Карточек:»)."""
    p = SUBMISSION / "REPORT.md"
    if not p.exists():
        return None
    lines = []
    for line in p.read_text(encoding="utf-8").splitlines()[1:]:
        if line.startswith(("Карточек:", "#")):
            break
        if line.strip():
            lines.append(line.strip())
    return " ".join(lines) or None


def _compact_traffic(t: Any) -> Any:
    """traffic.json (поток данных) весит мегабайты из-за словаря брендов: в интерфейс отдаём выжимку —
    метод и итоги, группы обязательных категорий кейса и 12 самых кликабельных категорий с примерами запросов."""
    if not isinstance(t, dict) or not any(isinstance(v, dict) and ("weights" in v or "top" in v) for v in t.values()):
        return t
    meta = t.get("_meta") if isinstance(t.get("_meta"), dict) else {}
    groups = {k: v for k, v in t.items() if k != "_meta" and isinstance(v, dict) and "weights" in v}
    cats = sorted(((k, v) for k, v in t.items() if isinstance(v, dict) and "top" in v), key=lambda kv: -(kv[1].get("clicks") or 0))
    return {
        "_meta": {k: v for k, v in meta.items() if k != "brand_aliases_mined"},
        "groups": groups,
        "categories": [{"id": k, "title": v.get("title"), "clicks": v.get("clicks"), "any_attribute": v.get("any_attribute"),
                        "top": [{"attribute": a.get("attribute"), "share": a.get("share"), "params": a.get("params"),
                                 "examples": (a.get("examples") or [])[:2]} for a in (v.get("top") or [])[:6]]}
                       for k, v in cats[:12]],
        "n_categories": len(cats),
    }


def _errors_summary(p: Path) -> dict | None:
    if not p.exists():
        return None
    with p.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_kind = Counter(r.get("kind", "") for r in rows)
    by_attr = Counter(r.get("attribute", "") for r in rows)
    return {"total": len(rows), "by_kind": dict(by_kind), "by_attribute": dict(by_attr.most_common(20)),
            "sample": rows[:30]}


_CONFLICT = re.compile(r"^(\w+): конфликт значений, альтернативы: (.*)$")


def _warning_alternatives(result: ExtractionResult) -> dict[str, list[dict]]:
    """Альтернативы из предупреждений пайплайна («storage: конфликт значений, альтернативы: 512.0»),
    если поле alternatives ещё не заполнено: модератор выбирает верное значение одним нажатием."""
    out: dict[str, list[dict]] = {}
    for w in result.warnings:
        m = _CONFLICT.match(w)
        if m:
            vals = [v.strip() for v in m.group(2).split(",") if v.strip()]
            out[m.group(1)] = [{"value": v, "confidence": None, "sources": [], "from": "warning"} for v in vals]
    return out


def _enrich(result: ExtractionResult, elapsed_ms: float | None = None) -> ExtractionResult:
    """Название категории, веса спроса и альтернативы, если пайплайн их не заполнил; общее время."""
    cat = extractor().tax.categories.get(result.category)
    if not result.category_title:
        result.category_title = cat.title if cat else None
    alts = _warning_alternatives(result)
    for a in result.attributes:
        spec = cat.attributes.get(a.attribute) if cat else None
        if spec is not None and a.traffic_weight == 1.0:  # 1.0 — значение по умолчанию в схеме
            a.traffic_weight = getattr(spec, "traffic_weight", 1.0)
        if not a.alternatives and a.attribute in alts:
            a.alternatives = alts[a.attribute]
    if elapsed_ms is not None:
        result.timings_ms.setdefault("total", round(elapsed_ms, 2))
    return result


# --------------------------------------------------------------------------- служебное

@app.get("/api/health")
def health():
    s = get_settings()
    path, synthetic = _examples_file()
    pred = _predictions_path()
    ex = extractor()
    clf = ex.tax.classifier()
    return {"status": "ok", "version": __version__, "llm_enabled": s.llm_enabled, "llm_model": s.llm_model,
            "llm_host": urlparse(s.llm_base_url).hostname, "ml_enabled": s.ml_enabled,
            # NER действительно загружен (есть веса, torch и transformers), а не просто есть папка models/ner;
            # модель грузится один раз при создании Extractor — здесь только проверка атрибута, без импорта torch
            "ml_model": ex.ml is not None,
            # провайдер, модель, доступность (проверка кэшируется на 30 с, ключ не отдаётся)
            "llm": llm_status(client=ex.client),
            "category_model": clf is not None,
            # ниже этой уверенности классификатора карточка уходит на проверку (models/category/meta.json)
            "category_threshold": clf.threshold if clf is not None else None,
            "confidence_threshold": s.confidence_threshold, "llm_route_threshold": s.llm_route_threshold,
            "examples": {"source": _rel(path) if path else None, "synthetic": synthetic},
            "predictions": _rel(pred) if pred else None,
            # запись решений модератора требует заголовок X-Attrx-Token (задан ATTRX_TOKEN); сам токен не отдаётся
            "auth": {"write_token_required": write_token() is not None, "header": TOKEN_HEADER}}


@app.get("/api/taxonomy")
def taxonomy():
    t = extractor().tax
    return {name: {"title": c.title, "attributes": {a: {"title": s.title, "type": s.type, "unit": s.unit}
                                                    for a, s in c.attributes.items()}}
            for name, c in t.categories.items()}


@app.get("/api/categories")
def categories():
    """Категории таксономии: обязательные характеристики, единицы, справочники и веса спроса (traffic_weight)."""
    t = extractor().tax
    traffic = _read_json(SUBMISSION / "traffic.json")
    out = []
    for name, c in t.categories.items():
        attrs = [{"key": a, "title": s.title, "type": s.type, "unit": s.unit,
                  "dictionary": getattr(s, "dictionary", None), "traffic_weight": getattr(s, "traffic_weight", 1.0)}
                 for a, s in c.attributes.items()]
        item = {"name": name, "title": c.title, "keywords": list(getattr(c, "keywords", []) or []), "attributes": attrs}
        if isinstance(traffic, dict) and name in traffic:
            item["traffic"] = traffic[name]
        out.append(item)
    return out


@app.get("/api/dictionary/{name}")
def dictionary(name: str, limit: int = Query(1000, ge=1, le=10000)):
    """Справочник допустимых значений (канонические формы) — подсказки модератору при исправлении."""
    d = extractor().tax.dictionaries.get(name)
    if d is None:
        raise HTTPException(404, f"Справочник {name} не найден")
    return {"name": name, "values": list(d.canon)[:limit], "synonyms": len(d.lookup)}


# --------------------------------------------------------------------------- извлечение и generative UI

def _llm_off_warning(ex: Extractor, use_llm: bool | None) -> str | None:
    """Клиент явно попросил LLM, а она не настроена (нет LLM_* / OPENAI_* / LLM_LOCAL_URL или выключена)."""
    if not use_llm or ex.client.enabled:
        return None
    return LLM_OFF_WARNING if ex.ml is not None else LLM_OFF_NO_NER_WARNING


@app.post("/api/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest):
    ex = extractor()
    t0 = time.perf_counter()
    result = _enrich(ex.extract(req.card, use_llm=req.use_llm), (time.perf_counter() - t0) * 1000)
    if warn_llm := _llm_off_warning(ex, req.use_llm):
        result.warnings.append(warn_llm)
    if req.intent:
        ui, warn = compose_llm(result, req.intent, ex.client)
    else:
        ui, warn = compose_default(result), None
    return ExtractResponse(result=result, ui=ui, ui_warning=warn, text=card_text(req.card))


@app.post("/api/extract/batch", response_model=list[ExtractionResult])
def extract_batch(req: BatchRequest):
    ex = extractor()
    results = list(ex.extract_many(req.cards, use_llm=req.use_llm))
    if warn_llm := _llm_off_warning(ex, req.use_llm):
        for r in results:
            r.warnings.append(warn_llm)
    return results


@app.post("/api/ui", response_model=ExtractResponse)
def ui(req: UIRequest):
    spec, warn = compose_llm(req.result, req.intent, extractor().client) if req.intent else (compose_default(req.result), None)
    return ExtractResponse(result=req.result, ui=spec, ui_warning=warn)


# --------------------------------------------------------------------------- примеры карточек

@app.get("/api/samples", deprecated=True)
def samples():
    """Устарел, оставлен для совместимости: список карточек целиком. Используйте /api/examples (фильтры, страницы)."""
    path, _ = _examples_file()
    return _load_examples(path) if path else []


@app.get("/api/examples")
def examples(category: str | None = None, q: str | None = None,
             limit: int = Query(50, ge=1, le=1000), offset: int = Query(0, ge=0)):
    """Примеры для витрины: реальный сэмпл (data/sample/real_cards.jsonl), если есть, иначе синтетический.

    Если есть data/sample/demo_cards.jsonl, его карточки идут первой псевдокатегорией DEMO_CATEGORY
    (у каждой demo=true и demo_note — что там дозаполняет ИИ). Без фильтра по ней в items и поиске их нет."""
    path, synthetic = _examples_file()
    demo = _demo_cards()
    demo_cat = [{"name": DEMO_CATEGORY, "count": len(demo), "demo": True}] if demo else []
    if category == DEMO_CATEGORY and demo:
        return {"source": _rel(DEMO_CARDS), "synthetic": False, "demo": True, "total": len(demo),
                "categories": demo_cat + _example_categories(path), "items": demo[offset:offset + limit]}
    if path is None:
        return {"source": None, "synthetic": True, "total": 0, "categories": demo_cat, "items": []}
    cards = _load_examples(path)
    items = cards
    if category:
        items = [c for c in items if (c.get("category") or NO_CATEGORY) == category]
    if q:
        ql = q.casefold()
        items = [c for c in items if ql in str(c.get("title", "")).casefold() or ql in str(c["id"]).casefold()]
    return {"source": _rel(path), "synthetic": synthetic, "total": len(items),
            "categories": demo_cat + _example_categories(path), "items": items[offset:offset + limit]}


def _example_categories(path: Path | None) -> list[dict]:
    if path is None:
        return []
    cats = Counter(c.get("category") or NO_CATEGORY for c in _load_examples(path))
    return [{"name": k, "count": v} for k, v in cats.most_common()]


@app.get("/api/examples/{card_id}")
def example(card_id: str, demo: bool = False):
    """Карточка по id. Демо-карточки часто копируют реальные (тот же id): demo=true — сначала витрина."""
    demo_card = next((c for c in _demo_cards() if str(c["id"]) == card_id), None)
    card = demo_card if demo and demo_card else _card_index().get(card_id) or demo_card
    if card is None:
        raise HTTPException(404, f"Карточка {card_id} не найдена в data/sample")
    return card


# --------------------------------------------------------------------------- очередь проверки и обратная связь

@app.get("/api/review", deprecated=True)
def review_queue(limit: int = 50):
    """Устарел, оставлен для совместимости: карточки со значениями на проверке. Используйте /api/review/queue
    (плоская очередь с приоритетом по спросу и решениями модератора)."""
    path = _predictions_path()
    if path is None:
        return []
    out = []
    for r in read_predictions(path):
        items = [a for a in r.attributes if a.status == "review"]
        if items:
            out.append({"card_id": r.card_id, "category": r.category, "attributes": [a.model_dump() for a in items]})
        if len(out) >= limit:
            break
    return out


@app.get("/api/review/queue")
def review_queue_v2(limit: int = Query(500, ge=1, le=5000), category: str | None = None,
                    include_missing: bool = False):
    """Плоская очередь: каждое значение на проверке — отдельная задача.

    Приоритет = вес атрибута в поиске × (1 − уверенность): сначала то, что покупатели чаще ищут
    и в чём пайплайн сильнее сомневается. Решения модератора берутся из feedback.jsonl.
    """
    s = get_settings()
    path = _predictions_path()
    empty = {"source": None, "threshold": s.confidence_threshold, "items": [], "cards": {}, "categories": [],
             "stats": {"total": 0, "done": 0, "remaining": 0, "accepted": 0, "corrected": 0, "rejected": 0}}
    if path is None:
        return empty
    tax = extractor().tax
    decisions = {(str(r.get("card_id")), r.get("attribute")): _decision(r) for r in _feedback_records()}
    cards = _card_index()
    items, used_cards, cats = [], {}, Counter()
    for r in read_predictions(path):
        cat_spec = tax.categories.get(r.category)
        _enrich(r)
        for a in r.attributes:
            if not (a.status == "review" or (include_missing and a.status == "missing" and a.required)):
                continue
            cat_title = r.category_title or (cat_spec.title if cat_spec else r.category)
            cats[cat_title] += 1
            if category and category not in (r.category, cat_title):
                continue
            weight = a.traffic_weight
            item = {**a.model_dump(), "id": f"{r.card_id}:{a.attribute}", "card_id": r.card_id,
                    "category": r.category, "category_title": cat_title, "category_path": r.category_path,
                    "priority": round(weight * (1 - a.confidence), 4),
                    "decision": decisions.get((r.card_id, a.attribute))}
            items.append(item)
            if r.card_id in cards and r.card_id not in used_cards:
                c = cards[r.card_id]
                used_cards[r.card_id] = {**c, "text": card_text(Card.model_validate(c)).model_dump()}
    items.sort(key=lambda x: (-x["priority"], x["confidence"], x["id"]))
    items = items[:limit]
    in_queue = {i["card_id"] for i in items}
    done = [i for i in items if i["decision"]]
    by_action = Counter(i["decision"]["action"] for i in done)
    return {"source": _rel(path), "threshold": s.confidence_threshold, "items": items,
            "cards": {k: v for k, v in used_cards.items() if k in in_queue},
            "categories": [{"name": k, "count": v} for k, v in cats.most_common()],
            "stats": {"total": len(items), "done": len(done), "remaining": len(items) - len(done),
                      "accepted": by_action["accepted"], "corrected": by_action["corrected"],
                      "rejected": by_action["rejected"]}}


@app.post("/api/feedback", dependencies=[Depends(require_write_token)])
def feedback(fb: Feedback):
    """Решение модератора -> outputs/feedback.jsonl. При заданном ATTRX_TOKEN нужен заголовок X-Attrx-Token."""
    with _feedback_lock:
        rec = log_correction(fb.card_id, fb.attribute, fb.predicted, fb.corrected, fb.raw, category=fb.category)
    return {**rec, "decision": _decision(rec)}


@app.post("/api/feedback/undo", dependencies=[Depends(require_write_token)])
def feedback_undo(key: FeedbackKey):
    """Отмена последнего решения по значению: удаляем последнюю запись (card_id, attribute) из feedback.jsonl."""
    p = get_settings().feedback_path
    with _feedback_lock:
        if not p.exists():
            raise HTTPException(404, "Решений ещё нет")
        lines = p.read_text(encoding="utf-8").splitlines()
        for i in range(len(lines) - 1, -1, -1):
            try:
                r = json.loads(lines[i])
            except json.JSONDecodeError:
                continue
            if str(r.get("card_id")) == key.card_id and r.get("attribute") == key.attribute:
                del lines[i]
                p.write_text("".join(x + "\n" for x in lines), encoding="utf-8")
                return {"removed": r}
    raise HTTPException(404, "Решение по этому значению не найдено")


# --------------------------------------------------------------------------- метрики

@app.get("/api/metrics")
def metrics(source: Literal["auto", "submission", "outputs"] = "auto"):
    """Метрики качества. auto — итоговые на holdout (submission/metrics.json, make report), а если их нет —
    последний локальный прогон (outputs/metrics.json: attrx demo / attrx eval); submission | outputs — явно.
    Какой файл отдан — в заголовке X-Attrx-Source."""
    files = {"submission": SUBMISSION / "metrics.json", "outputs": get_settings().predictions_path.parent / "metrics.json"}
    order = ("submission", "outputs") if source == "auto" else (source,)
    for name in order:
        data = _read_json(files[name]) if files[name].exists() else None
        if data is not None:
            return JSONResponse(data, headers={"X-Attrx-Source": name})
    raise HTTPException(404, "Метрик ещё нет: итоговые на holdout — make report (attrx report-real) -> "
                             "submission/metrics.json; быстрый прогон на 400 карточках — attrx demo -> outputs/metrics.json")


@app.get("/api/report")
def report():
    """Всё, что есть в submission/: метрики, сравнение подходов, сегменты, калибровка, порог, стоимость, трафик.

    Отсутствующие файлы перечислены в _meta.missing — интерфейс показывает честное «ещё не посчитано».
    """
    out: dict[str, Any] = {}
    meta: dict[str, Any] = {"dir": "submission", "files": {}, "missing": [], "invalid": []}
    for k in REPORT_FILES:
        p = SUBMISSION / f"{k}.json"
        if not p.exists():
            meta["missing"].append(k)
            continue
        data = _read_json(p)
        if data is None:
            meta["invalid"].append(k)
            continue
        out[k] = _compact_traffic(data) if k == "traffic" else data
        meta["files"][k] = {"path": f"submission/{k}.json", "updated": int(p.stat().st_mtime), "bytes": p.stat().st_size}
    errs = _errors_summary(SUBMISSION / "errors.csv")
    if errs is not None:
        out["errors"] = errs
    meta["note"] = _report_note()
    out["_meta"] = meta
    return out


@app.middleware("http")
async def static_revalidate(request, call_next):
    """Статика интерфейса всегда перепроверяется (ETag → 304): после обновления кода браузер не держит старые модули."""
    response = await call_next(request)
    if request.url.path.startswith("/static/") and "/assets/fonts/" not in request.url.path:
        response.headers["Cache-Control"] = "no-cache"
    return response


WEB = ROOT / "web"
if WEB.exists():
    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(Path(WEB) / "index.html", headers={"Cache-Control": "no-cache"})
