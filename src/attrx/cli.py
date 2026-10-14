"""CLI: attrx extract | eval | eda | serve | learn | demo."""
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

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
