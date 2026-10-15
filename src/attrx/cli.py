"""CLI: attrx extract | eval | eda | serve | learn | demo | train-ner | report | split | prelabel | prepare |
llm-status | llm-bench | build-gold | eval-real | traffic | eda-real | train-category | build-dicts | calibrate |
report-real."""
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

    st: dict = {}
    learned = learn_synonyms(stats=st)
    print(f"Решений в журнале: {st['records']}; выучено из исправлений: {st['learned']}; справочник не учат: "
          f"принято {st['accepted']}, отклонено {st['rejected']}; пропущено: нет формулировки {st['no_text']}, "
          f"нет справочника у атрибута {st['no_dictionary']}")
    print(f"Синонимов в learned.yaml: {sum(len(v or {}) for v in learned.values())}")


DEMO_SOURCES = {"seller": "продавец", "rules": "правила", "ml": "NER", "llm": "LLM"}


def cmd_demo(a):
    """400 реальных карточек holdout (data/sample/real_cards.jsonl) против эталона каталога (real_gold.jsonl) — те же
    метрики, что у attrx eval-real. По умолчанию без LLM, NER (torch) и data/processed: офлайн, секунды, так гоняет CI."""
    from .config import Settings
    from .eval import gold, real
    from .io import load_cards, write_predictions
    from .pipeline import Extractor

    sample = ROOT / "data" / "sample"
    cards = load_cards(sample / "real_cards.jsonl")
    g, meta = gold.load_jsonl(sample / "real_gold.jsonl")
    sources = {"seller", "rules"} | ({"ml"} if a.ml == "on" else set()) | ({"llm"} if a.llm == "on" else set())
    ex = Extractor(settings=Settings(llm_enabled="llm" in sources, ml_enabled="ml" in sources))
    notes = []
    if "ml" in sources and ex.ml is None:
        sources.discard("ml")
        notes.append("NER не загружен (нужны extra [ml] и обученная models/ner) — шаг пропущен")
    if "llm" in sources and not ex.client.enabled:
        sources.discard("llm")
        notes.append("LLM не настроена (LLM_* / OPENAI_* / LLM_LOCAL_URL) — шаг пропущен")
    results, perf = real.run(ex, cards, sources)
    m = real.report(results, g, meta, ex.tax, perf, data="data/sample/real_cards.jsonl + real_gold.jsonl (holdout)",
                    sources=sorted(sources), n_cards=len(cards), llm_calls=ex.client.calls if "llm" in sources else 0)
    out = get_settings().predictions_path
    write_predictions(results, out)
    metrics_path = out.parent / "metrics.json"
    metrics_path.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")

    mi, seg = m["micro"], m["segments"]
    off = [f"без {DEMO_SOURCES[k]} (--{k} on)" for k in ("ml", "llm") if k not in sources]
    print(f"attrx demo: {m['n_cards']} реальных карточек М.Видео из holdout (data/sample/real_cards.jsonl), эталон — "
          f"характеристики каталога (data/sample/real_gold.jsonl)")
    print("Источники: " + " + ".join(DEMO_SOURCES[k] for k in DEMO_SOURCES if k in sources)
          + (", " + ", ".join(off) if off else "") + f"; {perf['seconds']} с, {perf['cards_per_sec']} карточек/с"
          + (f", LLM-вызовов: {m['info']['llm_calls']}" if "llm" in sources else ""))
    for n in notes:
        print("  ! " + n)
    print()
    print(real.table(m))
    print()
    print("Итого (в каталог идут только значения со статусом accepted; совпадение с эталоном — attrx.eval.metrics):")
    print(f"  Precision {mi['precision']:.3f} · Recall {mi['recall']:.3f} · F1 {mi['f1']:.3f} "
          f"(micro: верно {mi['tp']}, ошибок {mi['fp']}, пропущено {mi['fn']})")
    print(f"  Recall по извлекаемому эталону (значение есть в тексте карточки): {m['recall_extractable']:.3f}")
    print(f"  F1 с учётом поискового спроса: {m['traffic_weighted_f1']:.3f} · macro-F1 по атрибутам: {m['macro_f1']:.3f}")
    print(f"  Покрытие: принятые значения заполнили {m['coverage']:.1%} слотов «карточка × атрибут»; заполненность "
          f"каталога {m['catalog_fill']:.1%} -> {m['union_fill']:.1%} (+{m['new_fills']} значений, которых в каталоге нет)")
    print(f"  На проверку модератору: {m['review_rate']:.1%} значений")
    names = {"seller_type": "продавец", "size_band": "размер категории", "desc_band": "длина описания"}
    print("  F1 по сегментам: " + "; ".join(
        f"{names[k]} " + ", ".join(f"{s} {v['f1']:.3f} (n={v['n_cards']})" for s, v in seg[k].items()) for k in names))
    def rel(p: Path) -> str:
        return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)

    print(f"-> {rel(out)} (+ .csv), {rel(metrics_path)}; очередь проверки в UI (attrx serve) показывает эти предсказания")


def cmd_train_ner(a):
    from .extractors import ner, rules
    from .io import load_cards, load_gold
    from .pipeline import Extractor

    if not ner.available() and not (a.distant and a.dry_run):
        sys.exit('Нужен extra [ml]: uv pip install -e ".[ml]"')
    if a.distant:  # distant supervision: значения каталога (train-сплит) -> спаны в тексте карточек
        from .extractors import ner_distant
        from .taxonomy import load_taxonomy

        info = ner_distant.run(Path(a.processed), Path(a.out), load_taxonomy(str(get_settings().taxonomy_path)),
                               n_generic=a.generic, max_examples=a.max_examples or None, epochs=a.epochs or 3,
                               lr=a.lr, batch_size=a.batch_size, device=a.device, threads=a.threads, seed=a.seed,
                               min_spans=a.min_spans, base=a.base, tax_oversample=a.tax_oversample,
                               dry_run=a.dry_run)
        if not a.dry_run:
            print(f"NER обучен: {info['dataset']['cards']} карточек ({info['examples']} примеров за эпоху), "
                  f"{info['dataset']['spans']} спанов, {len(info['labels']) // 2} атрибутов, {info['device_name']}, "
                  f"{info['train_seconds'] / 60:.1f} мин -> {a.out}")
        return
    if not a.input:
        sys.exit("нужен --input (карточки для слабой разметки) или --distant")
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
    info = ner.train(examples, sorted(attrs), Path(a.out), base=a.base, epochs=a.epochs or 20, lr=a.lr,
                     batch_size=8, device=ner.pick_device(a.device), threads=a.threads)
    n_spans = sum(len(s) for _, s in examples)
    print(f"NER обучен: {info['examples']} карточек, {n_spans} слабых спанов, loss по эпохам {info['loss']} -> {a.out}")


def cmd_report(a):
    from .eval.report import write_report
    from .io import load_cards, load_gold
    from .pipeline import Extractor

    if Path(a.out).resolve() == (ROOT / "submission").resolve() and not a.force:
        sys.exit("submission/ — итоговый отчёт на holdout реального каталога (attrx report-real, make report); "
                 "отчёт по произвольному набору пишите в другую папку (--out) или добавьте --force")
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


def cmd_prepare(a):
    from .data.prepare import STEPS, run

    steps = tuple(x.strip() for x in a.steps.split(",") if x.strip()) if a.steps else STEPS
    if bad := [x for x in steps if x not in STEPS]:
        sys.exit(f"Неизвестные шаги: {bad}. Доступны: {', '.join(STEPS)}")
    rep = run(Path(a.raw), Path(a.out), steps)
    print(f"Готово за {rep['total_seconds']} с, пик RSS {rep['peak_rss_gb']} ГБ -> {a.out} (отчёт: prepare_report.json)")


def cmd_llm_status(a):
    from .llm.status import llm_status

    print(json.dumps(llm_status(probe=not a.no_probe), ensure_ascii=False, indent=2))


def cmd_llm_bench(a):
    from .llm.bench import run_bench

    run_bench([m.strip() for m in a.models.split(",") if m.strip()], n=a.n, base_url=a.base_url,
              reasoning_effort=a.reasoning_effort, few_shot=a.few_shot, out=Path(a.out))


def cmd_build_gold(a):
    from .eval import gold
    from .taxonomy import load_taxonomy

    processed, out = Path(a.processed), Path(a.out)
    tax = load_taxonomy(str(get_settings().taxonomy_path))
    df, summary = gold.build(processed, tax)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    rows = gold.to_jsonl_rows(df)
    for split in ("dev", "holdout"):
        gold.write_jsonl([r for r in rows if r["split"] == split], out.parent / f"gold_{split}.jsonl")
    sample = ROOT / "data" / "sample" / "real_cards.jsonl"
    if sample.exists():  # эталон для закоммиченного сэмпла реальных карточек (демо и тесты без data/processed)
        ids = [json.loads(x)["id"] for x in sample.read_text(encoding="utf-8").splitlines() if x.strip()]
        by_id = {r["id"]: r for r in rows}
        gold.write_jsonl([by_id[i] for i in ids if i in by_id], sample.with_name("real_gold.jsonl"))
    by = df.groupby(["category", "attribute"]).agg(n=("sku_id", "size"), extractable=("extractable", "mean"),
                                                   mapped=("mapped", "mean"))
    print(by.round(3).to_string())
    print(f"Эталон: {summary['cards']} карточек, {summary['values']} значений за {summary['seconds']} с -> {out}, "
          f"gold_dev.jsonl, gold_holdout.jsonl" + (", data/sample/real_gold.jsonl" if sample.exists() else ""))


def cmd_eval_real(a):
    from .config import Settings
    from .eval import gold, real
    from .pipeline import Extractor

    processed = Path(a.processed)
    sources = {x.strip() for x in a.sources.split(",") if x.strip()}
    g, meta = gold.load_parquet(processed / "gold.parquet", splits={a.split})
    ids = real.stratified_ids(meta, a.n, a.generic, seed=a.seed)
    cards = real.load_cards_by_id(ids, processed)
    ex = Extractor(settings=Settings(llm_enabled="llm" in sources, ml_enabled="ml" in sources))
    results, perf = real.run(ex, cards, sources)
    m = real.report(results, g, meta, ex.tax, perf, split=a.split, sources=sorted(sources), seed=a.seed,
                    n_cards=len(cards))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.predictions:
        from .io import write_predictions

        write_predictions(results, Path(a.predictions))
    print(real.table(m))
    print(f"-> {out}")


def cmd_traffic(a):
    from .data.traffic import run

    info = run(Path(a.raw), Path(a.processed), Path(a.config), Path(a.summary), min_clicks=a.min_clicks,
               refresh=a.refresh, workers=a.workers)
    print(f"Категорий: {info['categories']} ({info['clicks_covered_share']:.1%} кликов поиска); запросы с атрибутом: "
          f"{info['any_attribute_share']:.1%} кликов, с характеристикой кроме бренда/модели: {info['any_param_share']:.1%}; "
          f"{info['seconds']} с -> {a.config}, {a.summary}")


def cmd_eda_real(a):
    from .data.eda import run

    info = run(Path(a.processed), Path(a.img), raw=Path(a.raw))
    print(json.dumps(info, ensure_ascii=False, indent=1))


def cmd_train_category(a):
    try:
        import scipy  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError:
        sys.exit('Для обучения нужен scikit-learn: uv pip install -e ".[ml]" (для инференса достаточно numpy)')
    from .category import train_category

    train_category(Path(a.processed), Path(a.out), max_classes=a.max_classes, alpha=a.alpha, epochs=a.epochs,
                   n_jobs=a.n_jobs, seed=a.seed, search=a.search, target_precision=a.target_precision,
                   min_confidence=a.min_confidence)


def cmd_build_dicts(a):
    from .dictgen import build_brands, learn_enum_maps, learn_model_styles, write_brands_yaml, write_enum_map
    from .taxonomy import load_taxonomy

    if not a.skip_brands:
        res = build_brands(Path(a.processed), Path(a.aliases), min_offers=a.min_offers)
        write_brands_yaml(res, Path(a.out))
        st = res["stats"]
        print(f"Брендов: {st['brands']}, омонимов (_word): {len(res['word'])}, с пониженной точностью (_prior): "
              f"{len(res['prior'])}; точность выбора бренда на {st['picked']} названиях train: {st['picked_precision']} "
              f"-> {a.out}")
        load_taxonomy.cache_clear()
    if not a.skip_enums:  # после брендов: совпадения справочников внутри названия бренда не считаются
        emap = learn_enum_maps(Path(a.processed), load_taxonomy(str(get_settings().taxonomy_path)))
        write_enum_map(emap, Path(a.enum_out))
        print(f"Соответствие справочников -> {a.enum_out}")
        styles = learn_model_styles(Path(a.processed), load_taxonomy(str(get_settings().taxonomy_path)))
        out = Path(a.enum_out).with_name("model_styles.json")
        write_enum_map(styles, out)
        print(f"Стиль модели по брендам -> {out}")


def cmd_calibrate(a):
    from .config import Settings
    from .eval import calibrate, gold, real
    from .pipeline import Extractor

    processed = Path(a.processed)
    sources = {x.strip() for x in a.sources.split(",") if x.strip()}
    g, meta = gold.load_parquet(processed / "gold.parquet", splits={"dev"})  # только dev: holdout не трогаем
    ids = real.stratified_ids(meta, a.n, a.generic, seed=a.seed)
    cards = real.load_cards_by_id(ids, processed)
    ex = Extractor(settings=Settings(llm_enabled="llm" in sources, ml_enabled="ml" in sources), calibrated=False)
    results, perf = real.run(ex, cards, sources)
    cal = calibrate.calibrate(results, g, ex.tax, target=a.target, cell_floor=a.cell_floor,
                              info={"sources": sorted(sources), "cards": len(cards), "seed": a.seed, **perf})
    calibrate.write(cal, Path(a.out))
    cf = cal["cross_fit"]
    print(f"Групп: {len(cal['groups'])}, точек: {cf['points']}. Порог {cal['threshold']}: точность принятых "
          f"{cf['precision_at_threshold']} при покрытии {cf['coverage_at_threshold']}, худшая ячейка "
          f"{cf['worst_cell_at_threshold']} (2-fold на dev, ECE {cf['ece']}; "
          f"без калибровки при том же пороге: {cf['raw_precision_at_same_threshold']} / "
          f"{cf['raw_coverage_at_same_threshold']}) -> {a.out}")


def cmd_report_real(a):
    from .eval.final import run_final

    if a.split != "holdout" and Path(a.out).resolve() == (ROOT / "submission").resolve():
        sys.exit("submission/ — только для holdout; пробный прогон на dev пишите в другую папку (--out)")
    info = run_final(ROOT, Path(a.processed), Path(a.out), split=a.split, llm_n=a.llm_n, cold_n=a.cold_n, seed=a.seed)
    print(f"Готово за {info['seconds']} с: {info['n_cards']} карточек ({info['split']}), режим в сдаче «{info['final']}» "
          f"-> {a.out}/REPORT.md")


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

    dm = sub.add_parser("demo", help="400 реальных карточек holdout из data/sample + эталон каталога -> метрики "
                                     "(как eval-real), outputs/predictions.jsonl и outputs/metrics.json")
    dm.add_argument("--llm", choices=["on", "off"], default="off", help="LLM для хвоста (по умолчанию off — офлайн)")
    dm.add_argument("--ml", choices=["on", "off"], default="off", help="NER (нужны extra [ml] и models/ner)")
    dm.set_defaults(fn=cmd_demo)

    t = sub.add_parser("train-ner", help="обучить NER: --distant — distant supervision на эталоне каталога (train), "
                                          "иначе слабая разметка от правил (+gold) по --input")
    t.add_argument("--input", "-i", default=None, help="карточки для слабой разметки от правил (без --distant)")
    t.add_argument("--gold", default=None)
    t.add_argument("--out", default=str(s.ml_model_path))
    t.add_argument("--base", default="cointegrated/rubert-tiny2")
    t.add_argument("--epochs", type=int, default=None, help="эпох: по умолчанию 3 (--distant) или 20")
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--distant", action="store_true", help="разметка из значений каталога М.Видео (train-сплит)")
    t.add_argument("--processed", default=str(ROOT / "data" / "processed"))
    t.add_argument("--max-examples", type=int, default=60_000, help="карточек для обучения (0 — без ограничения)")
    t.add_argument("--generic", type=int, default=48_000, help="из них карточек вне таксономии (бренд/модель/цвет)")
    t.add_argument("--tax-oversample", type=int, default=2, help="карточки категорий таксономии — N раз за эпоху")
    t.add_argument("--min-spans", type=int, default=150, help="атрибут получает метку, если спанов не меньше")
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--device", default="auto", help="auto | cpu | cuda (auto: GPU, если свободно ≥ 2.5 ГБ)")
    t.add_argument("--threads", type=int, default=2, help="потоков torch на CPU")
    t.add_argument("--seed", type=int, default=13)
    t.add_argument("--dry-run", action="store_true", help="только разметка и её статистика, без обучения")
    t.set_defaults(fn=cmd_train_ner)

    rp = sub.add_parser("report", help="сравнение подходов + сегменты + ошибки + предсказания по своему набору "
                                       "-> outputs/report (итоговый отчёт сдачи — attrx report-real)")
    rp.add_argument("--input", "-i", required=True)
    rp.add_argument("--gold", required=True)
    rp.add_argument("--out", default=str(ROOT / "outputs" / "report"))
    rp.add_argument("--note", default="")
    rp.add_argument("--force", action="store_true", help="разрешить запись в submission/")
    rp.set_defaults(fn=cmd_report)

    sp = sub.add_parser("split", help="gold -> gold_train + gold_holdout (честная оценка)")
    sp.add_argument("--gold", required=True)
    sp.add_argument("--holdout", type=float, default=0.3)
    sp.set_defaults(fn=cmd_split)

    pl = sub.add_parser("prelabel", help="predictions -> черновик gold для ручной проверки (LLM-assisted разметка)")
    pl.add_argument("--pred", default=str(s.predictions_path))
    pl.add_argument("--out", default=str(ROOT / "data" / "gold_draft.jsonl"))
    pl.set_defaults(fn=cmd_prelabel)

    pr = sub.add_parser("prepare", help="data/raw (YML-фид, описания, клики) -> data/processed/*.parquet")
    pr.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    pr.add_argument("--out", default=str(ROOT / "data" / "processed"))
    pr.add_argument("--steps", default="", help="через запятую: yml,clicks,cards,taxonomy,sample (по умолчанию все)")
    pr.set_defaults(fn=cmd_prepare)

    ls = sub.add_parser("llm-status", help="LLM: провайдер, модель, доступность (то же, что для /api/health)")
    ls.add_argument("--no-probe", action="store_true", help="без обращения к серверу")
    ls.set_defaults(fn=cmd_llm_status)

    lb = sub.add_parser("llm-bench", help="бенчмарк LLM на реальных карточках: латентность, JSON, заземление, ₽")
    lb.add_argument("--models", required=True, help="через запятую: qwen3:4b-instruct,qwen2.5:7b-instruct")
    lb.add_argument("--n", type=int, default=30, help="карточек (половина смартфоны, половина ноутбуки)")
    lb.add_argument("--base-url", default=None, help="по умолчанию из настроек (LLM_BASE_URL / OPENAI_BASE_URL)")
    lb.add_argument("--reasoning-effort", default=None, help="none — отключить «размышления» (qwen3.x)")
    lb.add_argument("--few-shot", action="store_true")
    lb.add_argument("--out", default=str(ROOT / "outputs" / "llm_bench"))
    lb.set_defaults(fn=cmd_llm_bench)

    bg = sub.add_parser("build-gold", help="эталон из каталога М.Видео -> data/processed/gold.parquet + gold_{dev,holdout}.jsonl")
    bg.add_argument("--processed", default=str(ROOT / "data" / "processed"))
    bg.add_argument("--out", default=str(ROOT / "data" / "processed" / "gold.parquet"))
    bg.set_defaults(fn=cmd_build_gold)

    er = sub.add_parser("eval-real", help="метрики пайплайна на реальном эталоне (сплит dev; holdout — только финал)")
    er.add_argument("--processed", default=str(ROOT / "data" / "processed"))
    er.add_argument("--split", default="dev", choices=["train", "dev", "holdout"])
    er.add_argument("--n", type=int, default=5000, help="карточек категорий таксономии (если их больше — стратифицированно)")
    er.add_argument("--generic", type=int, default=1000, help="карточек прочих категорий (generic)")
    er.add_argument("--sources", default="seller,rules", help="источники: seller,rules,ml,llm")
    er.add_argument("--seed", type=int, default=0)
    er.add_argument("--out", default=str(ROOT / "data" / "processed" / "baseline_dev.json"))
    er.add_argument("--predictions", default="", help="куда сохранить предсказания (jsonl/csv), по желанию")
    er.set_defaults(fn=cmd_eval_real)

    tr = sub.add_parser("traffic", help="трафик-веса атрибутов из поисковых кликов -> config/traffic_weights.yaml + "
                                        "submission/traffic.json")
    tr.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    tr.add_argument("--processed", default=str(ROOT / "data" / "processed"))
    tr.add_argument("--config", default=str(ROOT / "config" / "traffic_weights.yaml"))
    tr.add_argument("--summary", default=str(ROOT / "submission" / "traffic.json"))
    tr.add_argument("--min-clicks", type=int, default=10_000, help="категории с не меньшим числом кликов (+ топ-25)")
    tr.add_argument("--workers", type=int, default=3)
    tr.add_argument("--refresh", action="store_true", help="пересобрать кэш queries_by_category.parquet")
    tr.set_defaults(fn=cmd_traffic)

    ed = sub.add_parser("eda-real", help="EDA реальных данных: графики docs/img/eda_*.png + "
                                         "data/processed/{eda_stats,business_stats}.json")
    ed.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ed.add_argument("--processed", default=str(ROOT / "data" / "processed"))
    ed.add_argument("--img", default=str(ROOT / "docs" / "img"))
    ed.set_defaults(fn=cmd_eda_real)

    tc = sub.add_parser("train-category", help="классификатор категории по тексту (TF-IDF + линейная модель) -> "
                                               "models/category + metrics.json")
    tc.add_argument("--processed", default=str(ROOT / "data" / "processed"))
    tc.add_argument("--out", default=str(ROOT / "models" / "category"))
    tc.add_argument("--max-classes", type=int, default=1000,
                    help="сколько листьев каталога различать явно (+ листья таксономии); остальные — «Прочее в разделе»")
    tc.add_argument("--alpha", type=float, default=2e-6, help="L2-регуляризация SGD (без --search; 2e-6 выбрано на dev)")
    tc.add_argument("--search", action="store_true", help="подобрать alpha на dev по сетке (+3–4 мин)")
    tc.add_argument("--epochs", type=int, default=5)
    tc.add_argument("--target-precision", type=float, default=0.99,
                    help="порог уверенности подбирается на dev под эту точность категории")
    tc.add_argument("--min-confidence", type=float, default=0.9,
                    help="нижняя граница порога: калиброванная уверенность, ниже которой категория — на проверку")
    tc.add_argument("--n-jobs", type=int, default=2)
    tc.add_argument("--seed", type=int, default=0)
    tc.set_defaults(fn=cmd_train_category)

    bd = sub.add_parser("build-dicts", help="справочник брендов из каталога (train-сплит) -> "
                                            "config/dictionaries/brands.yaml")
    bd.add_argument("--processed", default=str(ROOT / "data" / "processed"))
    bd.add_argument("--aliases", default=str(ROOT / "config" / "brand_aliases.yaml"), help="ручные синонимы")
    bd.add_argument("--min-offers", type=int, default=2, help="минимум офферов бренда в train")
    bd.add_argument("--out", default=str(ROOT / "config" / "dictionaries" / "brands.yaml"))
    bd.add_argument("--enum-out", default=str(ROOT / "config" / "enum_map.json"),
                    help="выученное соответствие справочников (значения каталога) по train")
    bd.add_argument("--skip-brands", action="store_true")
    bd.add_argument("--skip-enums", action="store_true")
    bd.set_defaults(fn=cmd_build_dicts)

    cb = sub.add_parser("calibrate", help="калибровка уверенности и порог автопринятия на dev -> config/calibration.json")
    cb.add_argument("--processed", default=str(ROOT / "data" / "processed"))
    cb.add_argument("--sources", default="seller,rules", help="источники: seller,rules,ml,llm")
    cb.add_argument("--n", type=int, default=5000)
    cb.add_argument("--generic", type=int, default=1000)
    cb.add_argument("--seed", type=int, default=0)
    cb.add_argument("--target", type=float, default=0.95, help="целевая точность принятых значений")
    cb.add_argument("--cell-floor", type=float, default=0.75,
                    help="минимальная точность принятых в каждой ячейке «категория/атрибут» (не прятать хвост в среднем)")
    cb.add_argument("--out", default=str(ROOT / "config" / "calibration.json"))
    cb.set_defaults(fn=cmd_calibrate)

    rr = sub.add_parser("report-real", help="ИТОГОВЫЙ отчёт на holdout реального каталога -> submission/ "
                                            "(правила, + NER на всём наборе; + LLM на подвыборке; считать один раз)")
    rr.add_argument("--processed", default=str(ROOT / "data" / "processed"))
    rr.add_argument("--split", default="holdout", choices=["dev", "holdout"], help="dev — пробный прогон в другую папку")
    rr.add_argument("--llm-n", type=int, default=600, help="подвыборка для LLM (0 — без LLM); LLM_BASE_URL/LLM_MODEL")
    rr.add_argument("--cold-n", type=int, default=1000, help="карточек вне фида для покрытия на cold start (0 — нет)")
    rr.add_argument("--seed", type=int, default=0)
    rr.add_argument("--out", default=str(ROOT / "submission"))
    rr.set_defaults(fn=cmd_report_real)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
