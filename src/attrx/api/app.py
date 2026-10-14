"""FastAPI-сервис: можно встроить в контур М.Видео как микросервис или запускать батчами из CLI."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..config import ROOT, get_settings
from ..feedback import log_correction
from ..genui.composer import compose_default, compose_llm
from ..genui.schema import UISpec
from ..io import read_predictions
from ..pipeline import Extractor
from ..schema import Card, ExtractionResult

app = FastAPI(title="attrx — автозаполнение характеристик", version=__version__)
_extractor: Extractor | None = None


def extractor() -> Extractor:
    global _extractor
    if _extractor is None:
        _extractor = Extractor()
    return _extractor


class ExtractRequest(BaseModel):
    card: Card
    use_llm: bool | None = None
    intent: str | None = None


class ExtractResponse(BaseModel):
    result: ExtractionResult
    ui: UISpec
    ui_warning: str | None = None


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
    raw: str | None = None


@app.get("/api/health")
def health():
    s = get_settings()
    return {"status": "ok", "version": __version__, "llm_enabled": s.llm_enabled, "llm_model": s.llm_model,
            "llm_host": urlparse(s.llm_base_url).hostname}


@app.get("/api/taxonomy")
def taxonomy():
    t = extractor().tax
    return {name: {"title": c.title, "attributes": {a: {"title": s.title, "type": s.type, "unit": s.unit}
                                                    for a, s in c.attributes.items()}}
            for name, c in t.categories.items()}


@app.post("/api/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest):
    ex = extractor()
    result = ex.extract(req.card, use_llm=req.use_llm)
    if req.intent:
        ui, warn = compose_llm(result, req.intent, ex.client)
    else:
        ui, warn = compose_default(result), None
    return ExtractResponse(result=result, ui=ui, ui_warning=warn)


@app.post("/api/extract/batch", response_model=list[ExtractionResult])
def extract_batch(req: BatchRequest):
    return list(extractor().extract_many(req.cards, use_llm=req.use_llm))


@app.post("/api/ui", response_model=ExtractResponse)
def ui(req: UIRequest):
    spec, warn = compose_llm(req.result, req.intent, extractor().client) if req.intent else (compose_default(req.result), None)
    return ExtractResponse(result=req.result, ui=spec, ui_warning=warn)


@app.get("/api/review")
def review_queue(limit: int = 50):
    path = get_settings().predictions_path
    if not path.exists():
        return []
    out = []
    for r in read_predictions(path):
        items = [a for a in r.attributes if a.status == "review"]
        if items:
            out.append({"card_id": r.card_id, "category": r.category, "attributes": [a.model_dump() for a in items]})
        if len(out) >= limit:
            break
    return out


@app.post("/api/feedback")
def feedback(fb: Feedback):
    return log_correction(fb.card_id, fb.attribute, fb.predicted, fb.corrected, fb.raw)


@app.get("/api/metrics")
def metrics():
    path = get_settings().predictions_path.parent / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "Сначала запустите: attrx eval")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/report")
def report():
    """Сравнение подходов и сегменты из последнего attrx report (submission/)."""
    d = ROOT / "submission"
    return {k: json.loads((d / f"{k}.json").read_text(encoding="utf-8")) for k in ("compare", "segments") if (d / f"{k}.json").exists()}


@app.get("/api/samples")
def samples():
    path = ROOT / "data" / "sample" / "cards.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


WEB = ROOT / "web"
if WEB.exists():
    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(Path(WEB) / "index.html")
