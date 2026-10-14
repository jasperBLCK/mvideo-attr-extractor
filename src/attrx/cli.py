"""CLI: attrx extract | eval | eda | serve | learn | demo | train-ner | report | split | prelabel."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ROOT, get_settings


def cmd_extract(a):
    from .io import load_cards, write_predictions
    from .pipeline import Extractor

    cards = load_cards(Path(a.input))
    ex = Extractor()
    results = write_predictions(ex.extract_many(cards, use_llm=None if a.llm is None else a.llm == "on"), Path(a.output))
    n = sum(len(r.attributes) for r in results)
    acc = sum(x.status == "accepted" for r in results for x in r.attributes)
    rev = sum(x.status == "review" for r in results for x in r.attributes)
    print(f"Карточек: {len(results)}, атрибутов: {n}, принято: {acc}, на проверку: {rev}, LLM-вызовов: {ex.client.calls}")
    print(f"-> {a.output} и {Path(a.output).with_suffix('.csv')}")


def cmd_eval(a):
    from .eval.metrics import evaluate, threshold_sweep
    from .io import load_gold, read_predictions
    from .taxonomy import load_taxonomy

    tax = load_taxonomy(str(get_settings().taxonomy_path))
    results = list(read_predictions(Path(a.pred)))
    gold = load_gold(Path(a.gold))
    m = evaluate(results, gold, tax)
    m["threshold_sweep"] = threshold_sweep(results, gold, tax)
    out = Path(a.out)
    out.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"cards={m['n_cards']} micro-F1={m['micro']['f1']} precision={m['micro']['precision']} "
          f"recall={m['micro']['recall']} coverage={m['coverage']} review_rate={m['review_rate']} "
          f"traffic-weighted F1={m['traffic_weighted_f1']}")
    print(f"{'attribute':<12}{'P':>8}{'R':>8}{'F1':>8}")
    for k, v in sorted(m["per_attribute"].items()):
        print(f"{k:<12}{v['precision']:>8.3f}{v['recall']:>8.3f}{v['f1']:>8.3f}")
    print(f"-> {out}")


def cmd_eda(a):
    from .eval.eda import eda
    from .io import load_cards

    print(eda(load_cards(Path(a.input)), Path(a.out)))


def cmd_serve(a):
    import uvicorn

    uvicorn.run("attrx.api.app:app", host=a.host, port=a.port, reload=a.reload)


def cmd_learn(a):
    from .feedback import learn_synonyms

    learned = learn_synonyms()
    print(f"Выучено синонимов: {sum(len(v) for v in learned.values())}")


def cmd_demo(a):
    sample = ROOT / "data" / "sample"
    out = get_settings().predictions_path
    cmd_extract(argparse.Namespace(input=str(sample / "cards.jsonl"), output=str(out), llm=a.llm))
    cmd_eval(argparse.Namespace(pred=str(out), gold=str(sample / "gold.jsonl"), out=str(out.parent / "metrics.json")))


def cmd_train_ner(a):
    from .extractors import ner, rules
    from .io import load_cards, load_gold
    from .pipeline import Extractor

    if not ner.available():
        sys.exit('Нужен extra [ml]: uv pip install -e ".[ml]"')
    cards = load_cards(Path(a.input))
    gold = load_gold(Path(a.gold)) if a.gold else {}
    ex = Extractor()
    examples, attrs = [], set()
    for card in cards:
        cat_name, _ = ex.tax.detect_category(card)
        cat = ex.tax.categories[cat_name]
        attrs |= set(cat.attributes)
        cands = rules.extract(card, cat, ex.tax)
        spans = ner.weak_spans(card, cands, gold.get(card.id) if gold else None, ex.tax, cat)
        examples.append((card.full_text(), spans))
    info = ner.train(examples, sorted(attrs), Path(a.out), base=a.base, epochs=a.epochs, lr=a.lr)
    n_spans = sum(len(s) for _, s in examples)
    print(f"NER обучен: {info['examples']} карточек, {n_spans} слабых спанов, loss по эпохам {info['loss']} -> {a.out}")


def cmd_report(a):
    from .eval.report import write_report
    from .io import load_cards, load_gold
    from .pipeline import Extractor

    ex = Extractor()
    res = write_report(ex, load_cards(Path(a.input)), load_gold(Path(a.gold)), ex.tax, Path(a.out), note=a.note)
    for r in res["compare"]:
        print(f"{r['mode']:<22} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
              f"coverage={r['coverage']:.3f} {r['cards_per_sec']} карт/с, LLM={r['llm_calls']}")
    print(f"ошибок: {res['errors']} -> {a.out}/REPORT.md")


def cmd_split(a):
    import hashlib

    lines = [x for x in Path(a.gold).read_text(encoding="utf-8").splitlines() if x.strip()] if a.gold.endswith(".jsonl") else None
    if lines is None:
        sys.exit("split поддерживает gold в JSONL (один объект на строку с полем id)")
    train, hold = [], []
    for line in lines:
        cid = str(json.loads(line)["id"])
        h = int(hashlib.md5(cid.encode()).hexdigest(), 16) % 1000 / 1000
        (hold if h < a.holdout else train).append(line)
    base = Path(a.gold)
    for suffix, part in (("train", train), ("holdout", hold)):
        base.with_name(f"{base.stem}_{suffix}.jsonl").write_text("\n".join(part) + "\n", encoding="utf-8")
    print(f"train={len(train)} holdout={len(hold)} (детерминированно по hash(id); holdout не трогаем при подборе правил)")


def cmd_prelabel(a):
    from .io import read_predictions

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for r in read_predictions(Path(a.pred)):
            row = {"id": r.card_id, "category": r.category,
                   "attributes": {x.attribute: x.value for x in r.attributes if x.value is not None},
                   "_check": {x.attribute: {"status": x.status, "confidence": x.confidence, "evidence": x.evidence}
                              for x in r.attributes if x.status != "accepted"}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"Черновик разметки: {n} карточек -> {out}. Человек проверяет поля из _check и удаляет _check.")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="attrx", description="Извлечение и нормализация характеристик товаров")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = get_settings()

    e = sub.add_parser("extract", help="датасет -> predictions.jsonl/.csv")
    e.add_argument("--input", "-i", required=True)
    e.add_argument("--output", "-o", default=str(s.predictions_path))
    e.add_argument("--llm", choices=["on", "off"], default=None, help="переопределить LLM_ENABLED")
    e.set_defaults(fn=cmd_extract)

    v = sub.add_parser("eval", help="метрики по gold-разметке")
    v.add_argument("--pred", default=str(s.predictions_path))
    v.add_argument("--gold", required=True)
    v.add_argument("--out", default=str(s.predictions_path.parent / "metrics.json"))
    v.set_defaults(fn=cmd_eval)

    d = sub.add_parser("eda", help="EDA датасета -> outputs/eda/eda.md")
    d.add_argument("--input", "-i", required=True)
    d.add_argument("--out", default=str(ROOT / "outputs" / "eda"))
    d.set_defaults(fn=cmd_eda)

    sv = sub.add_parser("serve", help="API + веб-интерфейс")
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--reload", action="store_true")
    sv.set_defaults(fn=cmd_serve)

    sub.add_parser("learn", help="feedback.jsonl -> config/dictionaries/learned.yaml").set_defaults(fn=cmd_learn)

    dm = sub.add_parser("demo", help="прогон на data/sample + метрики")
    dm.add_argument("--llm", choices=["on", "off"], default=None)
    dm.set_defaults(fn=cmd_demo)

    t = sub.add_parser("train-ner", help="обучить NER на слабой разметке от правил (+gold)")
    t.add_argument("--input", "-i", required=True)
    t.add_argument("--gold", default=None)
    t.add_argument("--out", default=str(s.ml_model_path))
    t.add_argument("--base", default="cointegrated/rubert-tiny2")
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--lr", type=float, default=1e-4)
    t.set_defaults(fn=cmd_train_ner)

    rp = sub.add_parser("report", help="сравнение подходов + сегменты + ошибки + предсказания -> submission/")
    rp.add_argument("--input", "-i", required=True)
    rp.add_argument("--gold", required=True)
    rp.add_argument("--out", default=str(ROOT / "submission"))
    rp.add_argument("--note", default="")
    rp.set_defaults(fn=cmd_report)

    sp = sub.add_parser("split", help="gold -> gold_train + gold_holdout (честная оценка)")
    sp.add_argument("--gold", required=True)
    sp.add_argument("--holdout", type=float, default=0.3)
    sp.set_defaults(fn=cmd_split)

    pl = sub.add_parser("prelabel", help="predictions -> черновик gold для ручной проверки (LLM-assisted разметка)")
    pl.add_argument("--pred", default=str(s.predictions_path))
    pl.add_argument("--out", default=str(ROOT / "data" / "gold_draft.jsonl"))
    pl.set_defaults(fn=cmd_prelabel)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
