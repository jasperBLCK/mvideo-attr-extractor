"""ML-слой: NER (BIO-разметка токенов) на дообученном энкодере ruBERT-tiny2.

Обучающая разметка слабая (weak supervision): спаны, которые нашли правила, а если есть gold —
только те, что с ним согласны. Модель учится обобщать формулировки, которых нет в регулярках.
torch/transformers — опциональные зависимости (pip install -e ".[ml]"), без них слой просто выключен.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..normalize import normalize_value
from ..schema import Candidate, Card
from ..taxonomy import CategorySpec, Taxonomy

ML_BASE_SCORE = 0.8
DEFAULT_BASE = "cointegrated/rubert-tiny2"
MAX_LEN = 256
MIN_PROB = 0.5


def available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def find_span(text: str, raw: str) -> tuple[int, int] | None:
    i = text.lower().find(raw.lower().strip())
    return (i, i + len(raw.strip())) if i >= 0 and raw.strip() else None


def weak_spans(card: Card, cands: list[Candidate], gold: dict | None, tax: Taxonomy, category: CategorySpec) -> list[tuple[int, int, str]]:
    """Спаны (start, end, attr) из кандидатов правил; при наличии gold оставляем только верные значения."""
    from ..eval.metrics import _canon

    text = card.full_text()
    spans = []
    for c in cands:
        if c.source != "rules":
            continue
        spec = category.attributes[c.attribute]
        if gold is not None and _canon(gold.get(c.attribute), spec, tax) != _canon(c.value, spec, tax):
            continue
        sp = find_span(text, c.raw)
        if sp:
            spans.append((sp[0], sp[1], c.attribute))
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    out, last_end = [], -1
    for s in spans:
        if s[0] >= last_end:
            out.append(s)
            last_end = s[1]
    return out


def bio_labels(offsets: list[tuple[int, int]], spans: list[tuple[int, int, str]]) -> list[str | None]:
    """Метки токенов по символьным спанам; None — служебный токен (не учитывается в loss)."""
    labels: list[str | None] = []
    for s, e in offsets:
        if s == e:
            labels.append(None)
            continue
        lab = "O"
        for a, b, attr in spans:
            if s >= a and e <= b:
                lab = ("B-" if s == a else "I-") + attr
                break
            if s < b and e > a:  # токен частично в спане
                lab = ("B-" if s <= a else "I-") + attr
                break
        labels.append(lab)
    # I- без предшествующего B- превращаем в B-
    prev = "O"
    for i, lab in enumerate(labels):
        if lab and lab.startswith("I-") and prev[2:] != lab[2:]:
            labels[i] = "B-" + lab[2:]
        if lab is not None:
            prev = labels[i]
    return labels


def train(examples: list[tuple[str, list[tuple[int, int, str]]]], attributes: list[str], out_dir: Path,
          base: str = DEFAULT_BASE, epochs: int = 20, lr: float = 1e-4, batch_size: int = 8, seed: int = 13) -> dict:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    random.seed(seed)
    torch.manual_seed(seed)
    labels = ["O"] + [f"{p}-{a}" for a in sorted(attributes) for p in ("B", "I")]
    l2i = {lab: i for i, lab in enumerate(labels)}
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForTokenClassification.from_pretrained(
        base, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=l2i)
    data = []
    for text, spans in examples:
        enc = tok(text, truncation=True, max_length=MAX_LEN, return_offsets_mapping=True)
        y = [-100 if lab is None else l2i.get(lab, 0) for lab in bio_labels(enc["offset_mapping"], spans)]
        data.append((enc["input_ids"], y))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    history = []
    for _ in range(epochs):
        random.shuffle(data)
        total = 0.0
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            n = max(len(x) for x, _ in batch)
            ids = torch.tensor([x + [tok.pad_token_id] * (n - len(x)) for x, _ in batch])
            mask = torch.tensor([[1] * len(x) + [0] * (n - len(x)) for x, _ in batch])
            y = torch.tensor([lab + [-100] * (n - len(lab)) for _, lab in batch])
            loss = model(input_ids=ids, attention_mask=mask, labels=y).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            total += loss.item() * len(batch)
        history.append(round(total / max(1, len(data)), 4))
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    info = {"base": base, "examples": len(data), "epochs": epochs, "loss": history, "labels": labels}
    (out_dir / "attrx_ner.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


class NerExtractor:
    def __init__(self, path: Path):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForTokenClassification.from_pretrained(path).eval()
        self.id2label = self.model.config.id2label

    @classmethod
    def load(cls, path: Path) -> NerExtractor | None:
        if not available() or not (Path(path) / "config.json").exists():
            return None
        return cls(Path(path))

    def entities(self, text: str) -> list[tuple[str, int, int, float]]:
        enc = self.tok(text, truncation=True, max_length=MAX_LEN, return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        with self.torch.no_grad():
            probs = self.model(**enc).logits[0].softmax(-1)
        conf, idx = probs.max(-1)
        ents: list[list] = []
        for (s, e), i, p in zip(offsets, idx.tolist(), conf.tolist(), strict=True):
            if s == e:
                continue
            lab = self.id2label[i]
            if lab == "O":
                continue
            kind, attr = lab.split("-", 1)
            if kind == "I" and ents and ents[-1][0] == attr and s - ents[-1][2] <= 1:
                ents[-1][2] = e
                ents[-1][3].append(p)
            else:
                ents.append([attr, s, e, [p]])
        return [(a, s, e, sum(ps) / len(ps)) for a, s, e, ps in ents]

    def extract(self, card: Card, category: CategorySpec, tax: Taxonomy) -> list[Candidate]:
        text = card.full_text()
        out = []
        for attr, s, e, p in self.entities(text):
            if p < MIN_PROB:
                continue
            spec = category.attributes.get(attr)
            if spec is None:
                continue
            raw = text[s:e]
            res = normalize_value(raw, spec, tax.dictionaries)
            if res is None:
                continue
            value, quality = res
            out.append(Candidate(attribute=attr, value=value, raw=raw, source="ml",
                                 score=round(ML_BASE_SCORE * quality * p, 4), span=(s, e)))
        return out
