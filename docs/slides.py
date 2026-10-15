"""Финальная презентация attrx: docs/slides.html из результатов holdout (submission/*.json) -> PDF.

Запуск из корня репозитория:
    .venv/bin/python docs/slides.py
    chromium --headless --no-sandbox --disable-gpu --no-pdf-header-footer \\
      --print-to-pdf=docs/slides.pdf "file://$PWD/docs/slides.html"

Метрики — из submission/ (attrx report-real, отчёт v2; основной набор — весь holdout, режим в сдаче «+ NER»;
наборы compare.json: full_holdout, stratified, llm_subsample). Числа, которых нет в JSON сдачи, вынесены в константы
с источником: подвыборка LLM и карточки вне фида (REPORT.md §4, §11), новые названия (§5), dev и кандидат
исправления калибровки (docs/EVALUATION.md §9.2, §9.5), базовая линия «до» (EVALUATION.md §8.4).
Дизайн-система, навигация и схема пайплайна — из docs/checkpoint/concept.html (один визуальный язык).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submission"
OUT = ROOT / "docs" / "slides.html"
CONCEPT = ROOT / "docs" / "checkpoint" / "concept.html"
N_SLIDES = 13

# submission/REPORT.md §4 (подвыборка LLM, 662 карточки; отброшенные ответы — llm_rejected_values
# в data/processed/final_holdout.json) и §11 (1 000 карточек вне фида) — в JSON сдачи их нет
LLM = {"cards": 662, "share": 0.31, "calls": 205, "tok_in": 969, "tok_out": 52, "lat": 3.3, "p95": 5.0, "errors": 0,
       "ungrounded": 61, "llm_only": 71, "with_gold": 34, "correct_share": 0.382, "new_slots": 23,
       "auto": {"precision": 0.9579, "recall": 0.7623, "f1": 0.849}}
COLD = {"cards": 1000, "accepted": 0.543, "review": 0.116}
# REPORT.md §5: названия, которых дословно нет в train, против повторов (весь holdout, «+ NER»)
TITLES = {"new_n": 16803, "new_f1": 0.841, "new_rx": 0.922, "seen_n": 6588, "seen_f1": 0.912, "seen_rx": 0.941,
          "seen_share": 0.282}
# docs/EVALUATION.md §9.2: весь dev (23 554 карточки) тем же кодом
DEV_FULL_F1 = 0.859
# docs/EVALUATION.md §9.5: кандидат исправления калибровки — замер только на dev, в v2 не применён
CAND = {"threshold": 0.82, "precision": 0.982, "accepted": 0.855}
# базовая линия: dev (стратифицированный), продавец + правила до словарей каталога (EVALUATION.md §8.4, «F1 до»)
BASELINE_DEV = {"mvideo": 0.654, "marketplace": 0.447, "head": 0.616, "torso": 0.257, "tail": 0.227}

EXTRA_CSS = """
.kpis{display:grid;grid-template-columns:repeat(3,1fr);column-gap:28px;row-gap:22px}
.kpi{border-top:2px solid var(--ink);padding-top:10px}
.kpi b{display:block;font-size:34px;line-height:1.05;font-weight:740;letter-spacing:-.02em;white-space:nowrap}
.kpi span{display:block;font-size:13px;line-height:1.4;color:var(--ink-2);margin-top:6px}
.kpi small{display:block;font-size:11.5px;line-height:1.35;color:var(--muted);margin-top:3px}
.res-grid{display:grid;grid-template-columns:4.3fr 7.7fr;column-gap:48px;margin-top:30px;align-items:start}
.big{font-size:118px;line-height:.86;font-weight:760;letter-spacing:-.045em}
.big-cap{font-size:16px;line-height:1.42;color:var(--ink-2);margin-top:16px;max-width:400px}
.big-cap b{color:var(--ink)}
.strip{position:absolute;left:64px;right:64px;bottom:44px;display:grid;grid-template-columns:1fr 1fr;column-gap:32px;
  border-top:1px solid var(--hair);padding-top:10px;font-size:12.5px;line-height:1.45;color:var(--ink-2)}
.strip b{color:var(--ink)}
.strip.s3{grid-template-columns:1fr 1fr 1fr;column-gap:28px;font-size:12px}
.lt{width:100%;border-collapse:collapse;font-size:13.5px;line-height:1.3;margin-top:24px;font-variant-numeric:tabular-nums}
.lt th{font-size:10.5px;font-weight:650;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);text-align:right;
  padding:0 0 7px 10px;border-bottom:1px solid var(--ink);vertical-align:bottom}
.lt th:first-child,.lt td:first-child{text-align:left;padding-left:0}
.lt td{padding:6px 0 6px 10px;border-bottom:1px solid var(--hair);text-align:right;color:var(--ink-2)}
.lt td:first-child{color:var(--ink);font-weight:600}
.lt tr.sec td{border-bottom:0;padding-top:14px;font-size:11px;font-weight:650;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);text-align:left}
.lt tr.hl td{background:var(--red-wash);color:var(--ink);font-weight:700}
.lt tr.hl td:first-child{box-shadow:inset 3px 0 0 var(--red)}
.takeaways{display:grid;grid-template-columns:repeat(4,1fr);column-gap:22px;margin-top:20px}
.takeaways div{font-size:12.5px;line-height:1.42;color:var(--ink-2);border-top:2px solid var(--ink);padding-top:8px}
.takeaways b{display:block;font-size:13.5px;color:var(--ink);margin-bottom:3px}
.takeaways.t5{grid-template-columns:repeat(5,1fr);column-gap:18px;margin-top:16px}
.takeaways.t5 div{font-size:12px}
.lt.tight td{padding:4px 0 4px 10px}
.lt.tight tr.sec td{padding-top:9px}
.caveat{margin-top:10px;padding:10px 16px;border-left:3px solid var(--red);background:var(--red-wash);font-size:13px;line-height:1.45;color:var(--ink-2)}
.caveat b{color:var(--ink)}
.two{display:grid;grid-template-columns:1fr 1fr;column-gap:44px;margin-top:22px;align-items:start}
.two-w{display:grid;grid-template-columns:6fr 5fr;column-gap:40px;margin-top:20px;align-items:start}
.chart-cap{font-size:12px;line-height:1.45;color:var(--muted);margin-top:6px}
.chart-cap b{color:var(--ink-2)}
.mf{width:100%;border-collapse:collapse;font-size:13px;line-height:1.3;font-variant-numeric:tabular-nums}
.mf th{font-size:10.5px;font-weight:650;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);text-align:right;
  padding:0 0 6px 10px;border-bottom:1px solid var(--ink)}
.mf th:first-child,.mf td:first-child{text-align:left;padding-left:0}
.mf td{padding:3px 0 3px 10px;border-bottom:1px solid var(--hair);text-align:right;color:var(--ink-2)}
.mf td:first-child{color:var(--ink)}
.mf td.f{font-weight:700;color:var(--ink)}
.mf tr.grp td{font-size:11px;font-weight:650;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);padding-top:8px;border-bottom:0}
.mf td.low{color:var(--red-ink);font-weight:700}
.notes{list-style:none;margin-top:4px}
.notes li{font-size:13px;line-height:1.45;color:var(--ink-2);padding:9px 0;border-bottom:1px solid var(--hair)}
.notes li:first-child{padding-top:0}
.notes b{display:block;font-size:20px;line-height:1.15;font-weight:740;color:var(--ink);letter-spacing:-.01em;margin-bottom:2px}
.shots{display:grid;grid-template-columns:744px 1fr;column-gap:24px;margin-top:16px}
.shot img{display:block;width:100%;height:auto;border:1px solid var(--hair);border-radius:6px}
.shot p{font-size:12px;line-height:1.4;color:var(--ink-2);margin-top:6px}
.shot p b{color:var(--ink)}
.shots .col{display:flex;flex-direction:column;gap:12px}
.flow{list-style:none;counter-reset:f}
.flow li{position:relative;padding:0 0 18px 92px;min-height:62px}
.flow li > b{position:absolute;left:0;top:-4px;width:84px;font-size:34px;line-height:1;font-weight:740;letter-spacing:-.02em}
.flow li > span{display:block;font-size:15px;line-height:1.4;color:var(--ink)}
.flow li > small{display:block;font-size:12.5px;line-height:1.4;color:var(--muted);margin-top:3px}
.flow .mono{font-size:12px}
.flow li.r b{color:var(--red-ink)}
.cf{width:100%;border-collapse:collapse;font-size:15px;font-variant-numeric:tabular-nums;margin-top:8px}
.cf th{font-size:10.5px;font-weight:650;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);text-align:right;padding:0 0 7px 12px;border-bottom:1px solid var(--ink)}
.cf th:first-child,.cf td:first-child{text-align:left;padding-left:0}
.cf td{padding:7px 0 7px 12px;border-bottom:1px solid var(--hair);text-align:right}
.cf td.d{color:var(--red-ink);font-weight:700}
.verdict{margin-top:22px;padding:16px 18px;background:var(--wash);border-radius:8px;font-size:14.5px;line-height:1.5;color:var(--ink-2)}
.verdict b{color:var(--ink)}
.cost{list-style:none}
.cost li{padding:12px 0 14px;border-bottom:1px solid var(--hair)}
.cost li:first-child{padding-top:0}
.cost .who{display:flex;justify-content:space-between;align-items:baseline;font-size:14px;font-weight:650}
.cost .who em{font-style:normal;font-size:13px;font-weight:700;color:var(--red-ink)}
.cost .row{display:grid;grid-template-columns:170px 1fr 74px;column-gap:10px;align-items:center;margin-top:6px;font-size:12.5px;color:var(--ink-2)}
.cost .row i{display:block;height:12px;border-radius:0 3px 3px 0;background:var(--bar)}
.cost .row.h i{background:var(--red-ink)}
.cost .row span:last-child{text-align:right;font-weight:650;color:var(--ink);font-variant-numeric:tabular-nums}
.speed{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
.speed th{font-size:10.5px;font-weight:650;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);text-align:right;padding:0 0 7px 10px;border-bottom:1px solid var(--ink)}
.speed th:first-child,.speed td:first-child{text-align:left;padding-left:0}
.speed td{padding:6px 0 6px 10px;border-bottom:1px solid var(--hair);text-align:right;color:var(--ink-2)}
.speed td:first-child{color:var(--ink)}
.speed tr.hl td{font-weight:700;color:var(--ink);background:var(--red-wash)}
.rr{display:grid;grid-template-columns:3.4fr 8.6fr;column-gap:36px;margin-top:22px}
.steps-v{list-style:none;border-left:2px solid var(--ink)}
.steps-v li{position:relative;padding:0 0 14px 18px}
.steps-v li::before{content:"";position:absolute;left:-6px;top:4px;width:10px;height:10px;border-radius:50%;background:var(--paper);border:2px solid var(--ink)}
.steps-v li.done::before{background:var(--red);border-color:var(--red)}
.steps-v li > b{display:block;font-size:14px;color:var(--ink)}
.steps-v li > span{display:block;font-size:12.5px;line-height:1.4;color:var(--ink-2)}
.steps-v .mono{font-size:11.5px}
.rk2{width:100%;border-collapse:collapse;font-size:12.8px;line-height:1.38}
.rk2 th{font-size:10.5px;font-weight:650;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);text-align:left;padding:0 10px 7px 0;border-bottom:1px solid var(--ink)}
.rk2 td{padding:7px 10px 7px 0;border-bottom:1px solid var(--hair);vertical-align:top;color:var(--ink-2)}
.rk2 td:first-child{font-weight:700;color:var(--ink);width:150px}
.rk2 td.ev{color:var(--ink);font-weight:600;width:300px}
.fin{display:grid;grid-template-columns:repeat(3,1fr);column-gap:32px;margin-top:26px}
.fin h3{font-size:15px;font-weight:720;margin-bottom:8px;padding-top:10px;border-top:2px solid var(--ink)}
.fin ul{list-style:none}
.fin li{font-size:13px;line-height:1.42;color:var(--ink-2);padding:5px 0;border-bottom:1px solid var(--hair)}
.fin li b{color:var(--ink)}
.cover-kpis{background:var(--night-2);border:1px solid var(--night-hair);border-radius:10px;padding:22px 24px;margin-top:6px}
.cover-kpis .lblx{font-size:11px;font-weight:650;letter-spacing:.1em;text-transform:uppercase;color:var(--night-muted);margin-bottom:14px}
.cover-kpis dl{display:grid;grid-template-columns:auto 1fr;column-gap:18px;row-gap:14px;align-items:baseline}
.cover-kpis dt{font-size:30px;line-height:1;font-weight:740;letter-spacing:-.02em;color:#fff;white-space:nowrap}
.cover-kpis dd{font-size:13.5px;line-height:1.38;color:#cfcfcf}
.cover-kpis dd small{display:block;font-size:11.5px;color:var(--night-muted)}
svg .bar{fill:var(--bar)} svg .bar.hl{fill:var(--red-ink)}
svg .ax{stroke:#c3c2b7;stroke-width:1} svg .grid{stroke:#e8e8ea;stroke-width:1}
svg .refl{stroke:var(--ink);stroke-width:1} svg .refl.r{stroke:var(--red);stroke-width:1.5}
svg .tk{font-size:11px;fill:var(--muted)} svg .lb{font-size:12.5px} svg .lbm{font-size:11px;fill:var(--muted)}
svg .val{font-size:12px;font-weight:650;font-variant-numeric:tabular-nums;paint-order:stroke;stroke:var(--paper);stroke-width:4px;stroke-linejoin:round}
svg .ln2{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
svg .s-ink{stroke:var(--ink)} svg .s-gray{stroke:var(--bar)} svg .s-red{stroke:var(--red-ink)}
svg .dot{fill:var(--ink);stroke:var(--paper);stroke-width:2} svg .dot.r{fill:var(--red-ink)}
svg .dot.o{fill:var(--paper);stroke:var(--bar);stroke-width:2}
svg .val.r{fill:var(--red-ink)}
"""


def load(name: str):
    return json.loads((SUB / name).read_text(encoding="utf-8"))


def num(x: float, d: int = 3) -> str:
    """Как в submission/REPORT.md: форматирование значений JSON (4 знака) без повторного округления «вверх»."""
    return f"{x:.{d}f}".replace(".", ",")


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}".replace(".", ",") + "%"


def thou(x: int) -> str:
    return f"{int(x):,}".replace(",", "&#8239;")


def plural(n: int, one: str, few: str, many: str) -> str:
    """Форма слова после числа: 1 карточка, 2 карточки, 5 карточек."""
    n = abs(int(n)) % 100
    if 11 <= n <= 14:
        return many
    n %= 10
    return one if n == 1 else few if 2 <= n <= 4 else many


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def bar_path(x: float, y: float, w: float, h: float, r: float = 3.0) -> str:
    """Горизонтальная полоса: прямой угол у базовой линии, скругление r на конце значения."""
    r = min(r, w / 2, h / 2)
    return (f"M{x:.1f},{y:.1f}h{w - r:.1f}a{r},{r} 0 0 1 {r},{r}v{h - 2 * r:.1f}"
            f"a{r},{r} 0 0 1 -{r},{r}h-{w - r:.1f}z")


def foot(left: str, i: int) -> str:
    return f'<div class="foot"><span>{left}</span><span class="pg">{i:02d} / {N_SLIDES:02d}</span></div>'


def frame(i: int, title: str, body: str, dark: bool = False) -> str:
    cls = "slide dark" if dark else "slide"
    return (f'<div class="frame"><section class="{cls}" aria-label="Слайд {i} из {N_SLIDES}. {esc(title)}">\n'
            f"{body}\n</section></div>\n")


# ---------------------------------------------------------------- данные
def data() -> dict:
    m = load("metrics.json")
    compare = load("compare.json")
    seg = load("segments.json")
    cal = load("calibration.json")
    thr = load("threshold.json")
    cost = load("cost.json")
    tax = yaml.safe_load((ROOT / "config" / "taxonomy.yaml").read_text(encoding="utf-8"))["categories"]
    cats = []
    for cat, attrs in m["per_category"].items():
        tp = sum(v["tp"] for v in attrs.values())
        fp = sum(v["fp"] for v in attrs.values())
        fn = sum(v["fn"] for v in attrs.values())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        title = "Хвост (generic)" if cat == "generic" else tax[cat]["title"]
        cats.append({"cat": cat, "title": title, "n": max(v["n_cards"] for v in attrs.values()), "f1": f1})
    cats.sort(key=lambda c: -c["f1"])
    modes = {(c["set"], c["mode"]): c for c in compare}
    costs = {c["mode"]: c for c in cost["modes"]}
    n_generic = max(v["n_cards"] for v in m["per_category"]["generic"].values())
    return {"m": m, "modes": modes, "seg": seg, "cal": cal, "thr": thr, "costs": costs, "cats": cats, "tax": tax,
            "rub_h": cost["rub_per_hour_stand"], "generic_share": n_generic / m["n_cards"]}


def cards(n: int) -> str:
    return f"{thou(n)} {plural(n, 'карточка', 'карточки', 'карточек')}"


# ---------------------------------------------------------------- слайды
def s_cover(d: dict) -> str:
    m = d["m"]
    ap = m["accepted_precision"]
    twc = m["traffic_weighted_coverage"]
    gpu = d["costs"]
    every = gpu["LLM на каждую карточку: свой стенд (оценка)"]["cost_per_1000"]
    hyb = gpu["+ LLM · подвыборка LLM"]["cost_per_1000"]
    every_g = gpu["LLM на каждую карточку: GigaChat Lite (оценка)"]["cost_per_1000"]
    hyb_g = gpu["+ LLM через GigaChat Lite (оценка)"]["cost_per_1000"]
    lo, hi = sorted([every_g / hyb_g, every / hyb])
    nf = m["new_fills"]
    body = f"""  <div class="kicker">Точка сборки · Кейс № 3 · М.Видео · Финальная защита</div>
  <div class="cover-grid">
    <div>
      <div class="wordmark">attrx</div>
      <h1 class="cover-h">Характеристики товара — из&nbsp;текста карточки. Каждое значение с&nbsp;цитатой и&nbsp;уверенностью</h1>
      <p class="cover-lead">Гибрид для каталога М.Видео: категория по&nbsp;тексту, правила и&nbsp;словари каталога — ядро, NER&nbsp;— свободный текст, LLM&nbsp;— только подсказки модератору. Проверено на&nbsp;всём отложенном holdout: на&nbsp;нём ничего не&nbsp;настраивалось, слабые места раскрыты.</p>
    </div>
    <div class="cover-kpis" aria-label="Главные цифры на holdout">
      <div class="lblx">Holdout · {cards(m["eval_set"]["n_cards"])} · режим в&nbsp;сдаче</div>
      <dl>
        <dt>{num(ap["precision"])}</dt><dd>точность значений, принятых в&nbsp;каталог автоматически<small>95% ДИ от&nbsp;{num(ap["lower95"])}; принято {pct(ap["accepted_share"], 0)} найденного</small></dd>
        <dt>{num(m["traffic_weighted_f1"])}</dt><dd>F1 с&nbsp;учётом поискового спроса<small>micro-F1 {num(m["micro"]["f1"])}; полнота по&nbsp;значениям из&nbsp;текста {num(m["recall_extractable"])}</small></dd>
        <dt>{pct(twc["after"])}</dt><dd>поискового спроса с&nbsp;заполненным атрибутом<small>было {pct(twc["before"])}: +{thou(nf)} {plural(nf, "значение", "значения", "значений")} там, где каталог пуст</small></dd>
        <dt>{int(lo)}–{math.ceil(hi)}×</dt><dd>дешевле, чем «LLM на&nbsp;каждую карточку»<small>гибрид: {num(hyb, 2)}&nbsp;₽ на&nbsp;своей GPU, {num(hyb_g, 1)}&nbsp;₽ в&nbsp;GigaChat Lite за&nbsp;1000 карточек</small></dd>
      </dl>
    </div>
  </div>
  {foot('Команда «Ъыъ»: Makhkets, jasperBLCK · Демо: <span class="mono">attrx serve</span> → порт 8000 · все цифры — <span class="mono">submission/</span> (отчёт v2)', 1)}"""
    return frame(1, "Обложка", body, dark=True)


def s_problem(d: dict) -> str:
    focus = json.loads((ROOT / "docs" / "checkpoint" / "demand_stats.json").read_text(encoding="utf-8"))["focus_queries"]
    top = max(c for _, _, c in focus)
    rows = "\n".join(
        f'        <span class="q">«{esc(q)}»</span><span class="k">{esc(k)}</span>'
        f'<span class="b"><i style="--w:{c / top:.3f}"></i><em>{thou(c)}</em></span>' for q, k, c in focus)
    ext = pct(d["m"]["extractable_share"], 0)
    body = f"""  <div class="kicker">Проблема · деньги</div>
  <h2 class="h">Нет характеристики — нет товара в&nbsp;фильтре, а&nbsp;значит и&nbsp;в&nbsp;выручке</h2>
  <div class="p-grid" style="margin-top:26px">
    <div>
      <div class="hero-row"><div class="hero">56,5%</div>
        <div class="hero-cap">кликов из&nbsp;поиска — по&nbsp;запросам, где покупатель назвал атрибут; 24,9% — характеристику кроме бренда и&nbsp;модели</div></div>
      <div class="bars" role="img" aria-label="Клики по запросам, где названа характеристика">
{rows}
      </div>
      <p class="small" style="margin-top:8px">Числа у&nbsp;полос — клики по&nbsp;точному запросу. Доли — разметка всех запросов словарём каталога; описательные запросы без значения не&nbsp;засчитаны — оценка снизу.</p>
    </div>
    <aside class="facts" aria-label="Масштаб данных кейса">
      <div class="fact"><b>1&#8239;177&#8239;200</b><span>описаний карточек: название и&nbsp;текст продавца — вход решения</span></div>
      <div class="fact"><b>546&#8239;069</b><span>офферов каталога с&nbsp;характеристиками в&nbsp;6&#8239;963 категориях — эталон</span></div>
      <div class="fact"><b>30&#8239;991&#8239;350</b><span>кликов из&nbsp;поиска — вес атрибута в&nbsp;метриках и&nbsp;в&nbsp;очереди модератора</span></div>
      <div class="fact"><b>75%</b><span>карточек без оффера и&nbsp;без кликов: есть только текст — холодный старт</span></div>
      <div class="fact last"><b>{ext}</b><span>эталонных значений есть в&nbsp;тексте карточки (holdout): остальное не&nbsp;извлечь, только «нет в&nbsp;тексте»</span></div>
    </aside>
  </div>
  <p class="money"><b>Деньги:</b> дозаполненная характеристика → больше показов в&nbsp;фильтрах и&nbsp;атрибутной выдаче × CTR × конверсия × средний чек. CTR, конверсия и&nbsp;чек — данные М.Видео; эффект подтверждаем A/B по&nbsp;категориям.</p>
  {foot('Источники: архив кейса; доли спроса — <span class="mono">docs/EDA.md</span> §6; клики по&nbsp;запросам — <span class="mono">docs/checkpoint/demand_stats.json</span>; доля эталона в&nbsp;тексте — <span class="mono">submission/metrics.json</span>', 2)}"""
    return frame(2, "Проблема в деньгах", body)


def s_solution(d: dict) -> str:
    concept = CONCEPT.read_text(encoding="utf-8")
    fig = re.search(r'<figure class="pipe">.*?</figure>', concept, re.S)
    assert fig, "в concept.html нет схемы пайплайна"
    modes = d["modes"]
    rules = modes[("full_holdout", "правила")]
    ner = modes[("full_holdout", "+ NER")]
    body = f"""  <div class="kicker">Решение · гибридный пайплайн</div>
  <h2 class="h">Ядро закрывают дешёвые правила. LLM&nbsp;— только для остатка и&nbsp;только с&nbsp;цитатой</h2>
  {fig.group(0)}
  <div class="layers">
    <div><h4>Категория и правила</h4><p>Без id категорию даёт классификатор: 0,9998 против 0,921 у&nbsp;ключевых слов. Правила и&nbsp;4&nbsp;822 бренда каталога: F1 {num(rules["f1"])}, {thou(round(rules["cards_per_sec"]))} карточек/с на&nbsp;ядро.</p></div>
    <div><h4>NER</h4><p>ruBERT-tiny2 на&nbsp;значениях каталога, найденных в&nbsp;тексте. Только слабые атрибуты: полнота по&nbsp;извлекаемому {num(rules["recall_extractable"])} → {num(ner["recall_extractable"])}.</p></div>
    <div class="hotl"><h4>LLM</h4><p>Только слабые и&nbsp;упомянутые в&nbsp;тексте атрибуты — LLM спрашивали для&nbsp;{pct(LLM["share"], 0)} карточек. Одна LLM — только на&nbsp;проверку.</p></div>
    <div><h4>Заземление и порог</h4><p>Цитата найдена в&nbsp;тексте, значение следует из&nbsp;неё; калиброванный порог {num(d["m"]["threshold"], 2)} → точность принятых {num(d["m"]["accepted_precision"]["precision"])}.</p></div>
  </div>
  {foot('Схема = <span class="mono">src/attrx/pipeline.py</span>; классификатор — <span class="mono">docs/CATEGORY.md</span>; цифры — весь holdout, <span class="mono">submission/compare.json</span>', 3)}"""
    return frame(3, "Решение", body)


def s_demo(d: dict) -> str:
    body = f"""  <div class="kicker">Живое демо</div>
  <h2 class="h">Модератор видит значение, уверенность и&nbsp;цитату</h2>
  <div class="shots">
    <figure class="shot"><img src="img/ui_workbench_light.png" alt="Экран «Карточка»: текст карточки с подсвеченной цитатой и таблица характеристик со статусами">
      <p><b>Карточка.</b> Наведение на&nbsp;характеристику подсвечивает цитату в&nbsp;тексте; уверенность с&nbsp;отметкой порога, статус, источник, другие варианты при&nbsp;конфликте; «Принять / Исправить / Отклонить».</p></figure>
    <div class="col">
      <figure class="shot"><img src="img/ui_review_light.png" alt="Экран «Очередь проверки»">
        <p><b>Очередь:</b> первыми — частый спрос и&nbsp;низкая уверенность.</p></figure>
      <figure class="shot"><img src="img/ui_palette.png" alt="Палитра Ctrl+K: экран под запрос">
        <p><b>Ctrl/⌘K.</b> Экран под&nbsp;запрос из&nbsp;безопасных компонентов.</p></figure>
    </div>
  </div>
  {foot('Демо: <span class="mono">attrx serve</span> или <span class="mono">docker compose up</span> → порт 8000 на&nbsp;ВМ команды; Swagger — /docs; без датасета — 400 реальных карточек из&nbsp;<span class="mono">data/sample</span>', 4)}"""
    return frame(4, "Живое демо", body)


def s_result(d: dict) -> str:
    m = d["m"]
    ap = m["accepted_precision"]
    twc = m["traffic_weighted_coverage"]
    es = m["eval_set"]
    band = d["cal"]["accepted_band"]
    strat = d["modes"][("stratified", "+ NER · стратифицированный набор")]
    nf = m["new_fills"]
    body = f"""  <div class="kicker">Результат · весь holdout · {cards(es["n_cards"])}</div>
  <h2 class="h">Точность принятых {num(ap["precision"])} — на&nbsp;всём holdout, на&nbsp;котором ничего не&nbsp;настраивалось</h2>
  <div class="res-grid">
    <div>
      <div class="big">{num(ap["precision"])}</div>
      <p class="big-cap"><b>точность значений, которые attrx принял в&nbsp;каталог сам</b> (цель ≥ 0,95). 95% ДИ от&nbsp;{num(ap["lower95"])}; принято {pct(ap["accepted_share"], 0)} найденных значений, остальное — модератору с&nbsp;цитатой.</p>
    </div>
    <div class="kpis">
      <div class="kpi"><b>{num(m["micro"]["f1"])}</b><span>micro-F1 по&nbsp;всем слотам с&nbsp;эталоном</span><small>P {num(m["micro"]["precision"])} · R {num(m["micro"]["recall"])}</small></div>
      <div class="kpi"><b>{num(m["recall_extractable"])}</b><span>полнота по&nbsp;значениям, которые есть в&nbsp;тексте</span><small>в&nbsp;тексте есть {pct(m["extractable_share"])} эталона</small></div>
      <div class="kpi"><b>{num(m["traffic_weighted_f1"])}</b><span>F1 с&nbsp;учётом поискового спроса</span><small>атрибуты, по&nbsp;которым ищут, весят больше</small></div>
      <div class="kpi"><b>{pct(twc["before"])} → {pct(twc["after"])}</b><span>поискового спроса с&nbsp;заполненным атрибутом</span><small>слоты каталога {pct(m["catalog_fill"])} → {pct(m["union_fill"])}, +{thou(nf)} {plural(nf, "значение", "значения", "значений")}</small></div>
      <div class="kpi"><b>{num(m["calibration_ece_holdout"], 3)}</b><span>ошибка калибровки (ECE)</span><small>у&nbsp;порога, 0,70–0,80, верно лишь {pct(band["precision"], 0)} — слайд 9</small></div>
      <div class="kpi"><b>{pct(m["review_rate"])}</b><span>слотов уходит на&nbsp;проверку</span><small>цель ≤ 10% — чуть выше; на&nbsp;стратифицированном наборе {pct(strat["review_rate"])}</small></div>
    </div>
  </div>
  <div class="strip s3">
    <div><b>Протокол.</b> Весь holdout с&nbsp;эталоном без {es["excluded_sample_skus"]} SKU демо-сэмпла; {pct(d["generic_share"], 0)} карточек — хвост каталога. Отчёт v2: исправления выбраны на&nbsp;dev, v1 раскрыт. Весь dev → весь holdout: F1 {num(DEV_FULL_F1)} → {num(m["micro"]["f1"])}.</div>
    <div><b>Новые товары.</b> У&nbsp;{pct(1 - TITLES["seen_share"], 0)} карточек названия нет в&nbsp;train: F1 {num(TITLES["new_f1"])} против {num(TITLES["seen_f1"])} у&nbsp;повторов. Без category_id классификатор угадал категорию у&nbsp;всех 2&#8239;416 карточек стратифицированного набора.</div>
    <div><b>Холодный старт.</b> {thou(COLD["cards"])} карточек вне фида (75% каталога, эталона нет): автоматически заполнено {pct(COLD["accepted"])} слотов, на&nbsp;проверку {pct(COLD["review"])}.</div>
  </div>
  {foot('Режим в&nbsp;сдаче «+ NER», порог {} · <span class="mono">submission/metrics.json</span>, <span class="mono">REPORT.md</span> §1–3, §5, §11; dev — <span class="mono">docs/EVALUATION.md</span> §9.2'.format(num(m["threshold"], 2)), 5)}"""
    return frame(5, "Результат на holdout", body)


def ladder_row(r: dict, label: str, hl: bool = False, llm: bool = False) -> str:
    cps = "1,0" if llm else thou(round(r["cards_per_sec"]))
    cls = ' class="hl"' if hl else ""
    return (f'<tr{cls}><td>{label}</td><td>{num(r["precision"])}</td><td>{num(r["recall"])}</td>'
            f'<td>{num(r["f1"])}</td><td>{num(r["recall_extractable"])}</td><td>{num(r["traffic_weighted_f1"])}</td>'
            f'<td>{num(r["macro_f1"])}</td><td>{pct(r["coverage"])}</td><td>{pct(r["review_rate"])}</td>'
            f'<td>{cps}</td><td>{thou(round(r["llm_calls_per_1000"]))}</td></tr>')


def s_ladder(d: dict) -> str:
    md = d["modes"]
    full = [("full_holdout", "правила", "Правила и словари"),
            ("full_holdout", "продавец + правила", "+ атрибуты продавца"),
            ("full_holdout", "+ NER", "+ NER — в сдаче")]
    strat = [("stratified", "правила · стратифицированный набор", "Правила и словари"),
             ("stratified", "+ NER · стратифицированный набор", "+ NER"),
             ("stratified", "+ NER без category_id · стратифицированный набор", "+ NER без id категории (классификатор)")]
    sub = [("llm_subsample", "+ NER · подвыборка LLM", "+ NER"),
           ("llm_subsample", "+ LLM · подвыборка LLM", "+ LLM (qwen3:4b-instruct)")]
    rows_full = "\n".join(ladder_row(md[(s, k)], lab, hl=(k == "+ NER")) for s, k, lab in full)
    rows_strat = "\n".join(ladder_row(md[(s, k)], lab) for s, k, lab in strat)
    rows_sub = "\n".join(ladder_row(md[(s, k)], lab, llm=k.startswith("+ LLM")) for s, k, lab in sub)
    r0, r2 = md[("full_holdout", "правила")], md[("full_holdout", "+ NER")]
    s1 = md[("stratified", "+ NER · стратифицированный набор")]
    s2 = md[("stratified", "+ NER без category_id · стратифицированный набор")]
    n_s = s1["n_cards"]
    body = f"""  <div class="kicker">Что даёт каждый шаг · таблица из&nbsp;брифа на&nbsp;наших цифрах</div>
  <h2 class="h">Правила дают ядро, NER — хвост, локальная LLM — ноль к&nbsp;F1</h2>
  <table class="lt tight">
    <thead><tr><th>Подход</th><th>P</th><th>R</th><th>F1</th><th>R по&nbsp;извлек.</th><th>F1 по&nbsp;спросу</th><th>macro-F1</th><th>Покрытие</th><th>На проверку</th><th>Карточек/с</th><th>LLM на&nbsp;1000</th></tr></thead>
    <tbody>
      <tr class="sec"><td colspan="11">Весь holdout · {cards(r0["n_cards"])} · основной набор</td></tr>
{rows_full}
      <tr class="sec"><td colspan="11">Стратифицированный набор · {cards(n_s)}: все 12 категорий + хвост, как в&nbsp;v1</td></tr>
{rows_strat}
      <tr class="sec"><td colspan="11">Подвыборка LLM · {cards(LLM["cards"])}, все режимы на&nbsp;одних и&nbsp;тех же карточках</td></tr>
{rows_sub}
    </tbody>
  </table>
  <div class="takeaways t5">
    <div><b>Правила и словари</b>ядро: F1 {num(r0["f1"])} на&nbsp;CPU; без справочника 4&nbsp;822 брендов каталога F1 на&nbsp;dev ниже на&nbsp;0,175 (абляция)</div>
    <div><b>+ атрибуты продавца</b>≈&nbsp;0 на&nbsp;holdout: блок «Характеристики» есть лишь у&nbsp;7% карточек фида; работает вне фида</div>
    <div><b>+ NER</b>полнота {num(r0["recall"])} → {num(r2["recall"])}, macro-F1 {num(r0["macro_f1"])} → {num(r2["macro_f1"])} при той&nbsp;же точности</div>
    <div><b>Без id категории</b>та&nbsp;же F1 {num(s2["f1"])}: классификатор угадал категорию у&nbsp;всех {thou(n_s)} карточек; цена — {num(s2["ms_per_card"] - s1["ms_per_card"], 1)}&nbsp;мс на&nbsp;карточку</div>
    <div><b>+ LLM</b>F1 не&nbsp;меняется: значения одной LLM уходят на&nbsp;проверку, а&nbsp;не&nbsp;в&nbsp;каталог — слайд 10</div>
  </div>
  {foot('<span class="mono">submission/compare.json</span>, <span class="mono">REPORT.md</span> §2–4; скорость — медиана 3 замеров, NER и&nbsp;LLM на&nbsp;GPU стенда (GTX 1060); абляция — <span class="mono">docs/EVALUATION.md</span> §8.2', 6)}"""
    return frame(6, "Что даёт каждый шаг", body)


def cat_bars(cats: list[dict], overall: float) -> str:
    w, row, x0, x1 = 600, 26, 232, 540
    h = 24 + row * len(cats) + 26
    sc = x1 - x0
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="micro-F1 по категориям на holdout">']
    rx = x0 + overall * sc
    out.append(f'<line class="refl" x1="{rx:.1f}" y1="14" x2="{rx:.1f}" y2="{h - 26}"/>')
    out.append(f'<text class="lbm" x="{rx:.1f}" y="10" text-anchor="middle">весь holdout {num(overall)}</text>')
    for i, c in enumerate(cats):
        y = 22 + i * row
        hl = c["cat"] in ("smartphone", "laptop")
        bw = c["f1"] * sc
        fw = ' font-weight="700"' if hl else ""
        out.append(f'<text class="lb" x="0" y="{y + 11}"{fw}>{esc(c["title"])}</text>')
        out.append(f'<text class="lbm" x="{x0 - 8}" y="{y + 11}" text-anchor="end">{thou(c["n"])}</text>')
        out.append(f'<path class="bar{" hl" if hl else ""}" d="{bar_path(x0, y, bw, 14)}"><title>{esc(c["title"])}: F1 {num(c["f1"])}, карточек {c["n"]}</title></path>')
        out.append(f'<text class="val" x="{x0 + bw + 6:.1f}" y="{y + 11}">{num(c["f1"])}</text>')
    yb = h - 22
    out.append(f'<line class="ax" x1="{x0}" y1="16" x2="{x0}" y2="{yb}"/>')
    for t in (0, 0.5, 1.0):
        out.append(f'<text class="tk" x="{x0 + t * sc:.1f}" y="{yb + 14}" text-anchor="middle">{num(t, 1) if t else "0"}</text>')
    out.append(f'<text class="tk" x="{x0 - 8}" y="{yb + 14}" text-anchor="end">карточек</text>')
    out.append("</svg>")
    return "\n".join(out)


def s_categories(d: dict) -> str:
    m = d["m"]
    pc = m["per_category"]
    names = {"brand": "Бренд", "model": "Модель", "storage": "Память / накопитель", "screen": "Диагональ",
             "battery": "Аккумулятор", "color": "Цвет", "cpu": "Процессор", "ram": "Оперативная память"}
    groups = [("smartphone", "Смартфон", ["brand", "model", "storage", "screen", "battery", "color"]),
              ("laptop", "Ноутбук", ["brand", "cpu", "ram", "storage", "screen", "color"])]
    trs = []
    for cat, title, attrs in groups:
        k = max(v["n_cards"] for v in pc[cat].values())
        trs.append(f'<tr class="grp"><td colspan="5">{title} · {cards(k)}</td></tr>')
        for a in attrs:
            v = pc[cat][a]
            low = ' class="low"' if v["extractable_share"] < 0.6 else ""
            trs.append(f'<tr><td>{names[a]}</td><td class="f">{num(v["f1"], 2)}</td><td>{num(v["precision"], 2)}</td>'
                       f'<td{low}>{pct(v["extractable_share"], 0)}</td><td>{num(v["recall_extractable"], 2)}</td></tr>')
    table = "\n".join(trs)
    gc = pc["generic"]["color"]
    body = f"""  <div class="kicker">Качество по&nbsp;категориям и&nbsp;обязательным полям кейса</div>
  <h2 class="h">Ядро — 0,9 и&nbsp;выше; низкий F1 там, где значения нет в&nbsp;тексте</h2>
  <div class="two-w">
    <figure>
      {cat_bars(d["cats"], m["micro"]["f1"])}
      <figcaption class="chart-cap"><b>micro-F1 по&nbsp;категориям</b>, режим в&nbsp;сдаче; красным — категории кейса. Холодильники и&nbsp;аэрогрили ниже: объём камер, высота, материал редко есть в&nbsp;тексте. В&nbsp;хвосте слабее всего цвет: F1 {num(gc["f1"], 2)}, в&nbsp;тексте лишь {pct(gc["extractable_share"], 0)} эталона.</figcaption>
    </figure>
    <div>
      <table class="mf">
        <thead><tr><th>Обязательное поле</th><th>F1</th><th>P</th><th>в&nbsp;тексте</th><th>R по&nbsp;извлек.</th></tr></thead>
        <tbody>
{table}
        </tbody>
      </table>
      <p class="chart-cap" style="margin-top:8px"><b>Диагональ и&nbsp;аккумулятор смартфона</b> есть в&nbsp;тексте у&nbsp;половины карточек: полнота по&nbsp;ним 0,96–1,00, общий F1 ограничен текстом.</p>
    </div>
  </div>
  {foot('<span class="mono">submission/metrics.json</span> (<span class="mono">per_category</span>), <span class="mono">REPORT.md</span> §6–7 · «в тексте» — доля эталонных значений, которые вообще есть в&nbsp;тексте карточки', 7)}"""
    return frame(7, "По категориям", body)


def dumbbell(rows: list[tuple[str, str, float, float, int]]) -> str:
    w, x0, x1, row = 600, 190, 560, 44
    h = 30 + row * len(rows) + 30
    sc = x1 - x0
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="F1 по сегментам: базовая линия на dev и holdout">']
    yb = h - 24
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        x = x0 + t * sc
        out.append(f'<line class="grid" x1="{x:.1f}" y1="18" x2="{x:.1f}" y2="{yb}"/>')
        out.append(f'<text class="tk" x="{x:.1f}" y="{yb + 15}" text-anchor="middle">{num(t, 2) if t else "0"}</text>')
    for i, (grp, lab, before, after, n) in enumerate(rows):
        y = 34 + i * row
        if grp:
            out.append(f'<text class="lbm" x="0" y="{y - 14}" letter-spacing=".06em">{esc(grp.upper())}</text>')
        xb, xa = x0 + before * sc, x0 + after * sc
        out.append(f'<text class="lb" x="0" y="{y + 4}">{esc(lab)}</text>')
        out.append(f'<text class="lbm" x="{x0 - 10}" y="{y + 4}" text-anchor="end">{thou(n)}</text>')
        out.append(f'<line class="ln2 s-gray" x1="{xb:.1f}" y1="{y}" x2="{xa:.1f}" y2="{y}"/>')
        out.append(f'<circle class="dot o" cx="{xb:.1f}" cy="{y}" r="5"><title>{esc(lab)}: база (dev) {num(before)}</title></circle>')
        out.append(f'<circle class="dot" cx="{xa:.1f}" cy="{y}" r="6"><title>{esc(lab)}: holdout {num(after)}</title></circle>')
        out.append(f'<text class="tk" x="{xb - 9:.1f}" y="{y + 4}" text-anchor="end">{num(before)}</text>')
        out.append(f'<text class="val" x="{xa + 10:.1f}" y="{y + 4}">{num(after)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def s_fairness(d: dict) -> str:
    sg = d["seg"]
    st, sb, db = sg["seller_type"], sg["size_band"], sg["desc_band"]
    rows = [("Продавец", "М.Видео", BASELINE_DEV["mvideo"], st["mvideo"]["f1"], st["mvideo"]["n_cards"]),
            ("", "Маркетплейс", BASELINE_DEV["marketplace"], st["marketplace"]["f1"], st["marketplace"]["n_cards"]),
            ("Размер категории", "head (≥ 1000 офферов)", BASELINE_DEV["head"], sb["head"]["f1"], sb["head"]["n_cards"]),
            ("", "torso (100–999)", BASELINE_DEV["torso"], sb["torso"]["f1"], sb["torso"]["n_cards"]),
            ("", "tail (< 100)", BASELINE_DEV["tail"], sb["tail"]["f1"], sb["tail"]["n_cards"])]
    gap_s = (st["mvideo"]["f1"] - st["marketplace"]["f1"]) * 100
    gap_t = (sb["head"]["f1"] - sb["tail"]["f1"]) * 100
    hd, tl = sb["head"], sb["tail"]
    body = f"""  <div class="kicker">Справедливость · продавцы, хвост каталога, новые товары</div>
  <h2 class="h">Партнёры маркетплейса и&nbsp;редкие категории — на&nbsp;уровне М.Видео и&nbsp;крупных</h2>
  <div class="two-w">
    <figure>
      {dumbbell(rows)}
      <figcaption class="chart-cap"><b>F1 по&nbsp;сегментам.</b> ○ база: dev (стратифицированный набор), продавец + правила до&nbsp;словарей каталога · ● весь holdout, режим в&nbsp;сдаче. Справа от&nbsp;названия — карточек на&nbsp;holdout.</figcaption>
    </figure>
    <ul class="notes">
      <li><b>{num(gap_s, 1)} п.п.</b>разрыв М.Видео и&nbsp;маркетплейса (цель ≤ 5; в&nbsp;базе — 21 п.п.): справочник 4&nbsp;822 брендов всего каталога одинаково покрывает всех продавцов</li>
      <li><b>{num(gap_t, 1)} п.п.</b>разрыв крупных и&nbsp;редких категорий (цель ≤ 10): у&nbsp;tail точность {num(tl["precision"])} против {num(hd["precision"])}, полнота по&nbsp;извлекаемому {num(tl["recall_extractable"])} против {num(hd["recall_extractable"])}</li>
      <li><b>{num(db["short"]["f1"])} · {num(db["medium"]["f1"])} · {num(db["long"]["f1"])}</b>короткие, средние и&nbsp;длинные описания; полнота по&nbsp;извлекаемому {num(db["short"]["recall_extractable"])} · {num(db["medium"]["recall_extractable"])} · {num(db["long"]["recall_extractable"])}</li>
      <li><b>{num(TITLES["new_f1"])} против {num(TITLES["seen_f1"])}</b>новые названия (нет в&nbsp;train, {thou(TITLES["new_n"])} карточек) против повторов; полнота по&nbsp;извлекаемому почти та&nbsp;же: {num(TITLES["new_rx"])} против {num(TITLES["seen_rx"])}</li>
    </ul>
  </div>
  {foot('holdout — <span class="mono">submission/segments.json</span>, <span class="mono">REPORT.md</span> §5, §8; база — <span class="mono">docs/EVALUATION.md</span> §8.4 (dev); head / torso / tail — число офферов в&nbsp;листовой категории', 8)}"""
    return frame(8, "Справедливость", body)


def reliability(cal: dict) -> str:
    w, h, x0, x1, y0, y1 = 520, 292, 46, 500, 250, 14
    sx, sy = x1 - x0, y0 - y1
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Диаграмма надёжности: уверенность против доли верных на holdout">']
    for t in (0, 0.5, 1.0):
        out.append(f'<line class="grid" x1="{x0}" y1="{y0 - t * sy:.1f}" x2="{x1}" y2="{y0 - t * sy:.1f}"/>')
        out.append(f'<text class="tk" x="{x0 - 8}" y="{y0 - t * sy + 4:.1f}" text-anchor="end">{num(t, 1) if t else "0"}</text>')
        out.append(f'<text class="tk" x="{x0 + t * sx:.1f}" y="{y0 + 16}" text-anchor="middle">{num(t, 1) if t else "0"}</text>')
    out.append(f'<line class="refl" x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}"/>')
    out.append(f'<line class="refl" x1="{x0 + 10}" y1="{y1 + 12}" x2="{x0 + 34}" y2="{y1 + 12}"/>')
    out.append(f'<text class="lbm" x="{x0 + 40}" y="{y1 + 16}">идеальная калибровка: уверенность = доля верных</text>')
    pts = [(b["confidence"], b["accuracy"], b["n"], b["lo"]) for b in cal["bins"]]
    path = " ".join(f"{x0 + c * sx:.1f},{y0 - a * sy:.1f}" for c, a, _, _ in pts)
    out.append(f'<polyline class="ln2 s-ink" points="{path}" opacity=".35"/>')
    for c, a, n, lo in pts:
        r = 4 + min(7, math.sqrt(n) / 12)
        hot = abs(lo - 0.7) < 1e-9
        cx, cy = x0 + c * sx, y0 - a * sy
        out.append(f'<circle class="dot{" r" if hot else ""}" cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}"><title>уверенность {num(c)}: верно {num(a)}, значений {n}</title></circle>')
        if hot:
            out.append(f'<text class="val r" x="{x1}" y="{cy + 36:.1f}" text-anchor="end">0,7–0,8: обещано {num(c, 2)}, верно {num(a, 2)}</text>')
    out.append(f'<text class="tk" x="{(x0 + x1) / 2:.1f}" y="{h - 4}" text-anchor="middle">уверенность attrx (калибровка выучена на dev)</text>')
    out.append(f'<text class="tk" x="12" y="{(y0 + y1) / 2:.1f}" text-anchor="middle" transform="rotate(-90 12 {(y0 + y1) / 2:.1f})">доля верных на holdout</text>')
    out.append("</svg>")
    return "\n".join(out)


def threshold_chart(thr: dict, chosen: float, ap: float, cov: float) -> str:
    w, h, x0, x1, y0, y1 = 520, 292, 46, 448, 250, 14
    lo = 0.5
    sx, sy = x1 - x0, y0 - y1

    def X(t: float) -> float:
        return x0 + t / 0.95 * sx

    def Y(v: float) -> float:
        return y0 - (v - lo) / (1 - lo) * sy

    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Точность и покрытие слотов в зависимости от порога">']
    for v in (0.5, 0.75, 1.0):
        out.append(f'<line class="grid" x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}"/>')
        out.append(f'<text class="tk" x="{x0 - 8}" y="{Y(v) + 4:.1f}" text-anchor="end">{num(v, 2)}</text>')
    for t in (0, 0.25, 0.5, 0.75, 0.95):
        out.append(f'<text class="tk" x="{X(t):.1f}" y="{y0 + 16}" text-anchor="middle">{num(t, 2) if t else "0"}</text>')
    pts = [p for p in thr["points"] if p["threshold"] <= 0.95 + 1e-9]
    pp = " ".join(f"{X(p['threshold']):.1f},{Y(p['precision']):.1f}" for p in pts)
    pc = " ".join(f"{X(p['threshold']):.1f},{Y(p['coverage']):.1f}" for p in pts)
    out.append(f'<polyline class="ln2 s-ink" points="{pp}"/>')
    out.append(f'<polyline class="ln2 s-gray" points="{pc}"/>')
    last = pts[-1]
    out.append(f'<text class="val" x="{X(last["threshold"]) + 8:.1f}" y="{Y(last["precision"]) + 4:.1f}">точность</text>')
    out.append(f'<text class="val" x="{X(last["threshold"]) + 8:.1f}" y="{Y(last["coverage"]) + 4:.1f}">покрытие</text>')
    xc = X(chosen)
    out.append(f'<line class="refl r" x1="{xc:.1f}" y1="{y1 - 4}" x2="{xc:.1f}" y2="{y0}"/>')
    out.append(f'<circle class="dot r" cx="{xc:.1f}" cy="{Y(ap):.1f}" r="5"/>')
    out.append(f'<circle class="dot r" cx="{xc:.1f}" cy="{Y(cov):.1f}" r="5"/>')
    out.append(f'<text class="val" x="{xc - 8:.1f}" y="{Y(ap) - 10:.1f}" text-anchor="end">{num(ap)}</text>')
    out.append(f'<text class="val" x="{xc - 8:.1f}" y="{Y(cov) + 18:.1f}" text-anchor="end">{pct(cov)}</text>')
    out.append(f'<text class="lbm" x="{xc + 6:.1f}" y="{y1 + 6}">порог {num(chosen, 2)}</text>')
    out.append(f'<text class="tk" x="{(x0 + x1) / 2:.1f}" y="{h - 4}" text-anchor="middle">порог уверенности для автопринятия</text>')
    out.append("</svg>")
    return "\n".join(out)


def s_trust(d: dict) -> str:
    m, cal, thr = d["m"], d["cal"], d["thr"]
    ap = m["accepted_precision"]
    band = cal["accepted_band"]
    top = cal["bins"][-1]
    err_share = band["wrong"] / band["wrong_accepted_total"]
    body = f"""  <div class="kicker">Доверие · калибровка и&nbsp;порог</div>
  <h2 class="h">Точность принятых {num(ap["precision"])} выдержала holdout, но&nbsp;у&nbsp;самого порога уверенность завышена</h2>
  <div class="two" style="margin-top:14px">
    <figure>
      {reliability(cal)}
      <figcaption class="chart-cap"><b>ECE {num(cal["ece"], 3)}</b> на&nbsp;{thou(cal["n"])} значениях с&nbsp;эталоном. {pct(top["n"] / cal["n"], 0)} значений — в&nbsp;корзине 0,9–1,0: уверенность {num(top["confidence"])}, верно {num(top["accuracy"])}. Размер точки — число значений.</figcaption>
    </figure>
    <figure>
      {threshold_chart(thr, m["threshold"], ap["precision"], m["coverage"])}
      <figcaption class="chart-cap"><b>Порог {num(m["threshold"], 2)}</b> выбран на&nbsp;dev: наименьший, при котором нижняя граница точности принятых ≥ 0,95 и&nbsp;ни одна ячейка не&nbsp;ниже 0,75; на&nbsp;dev кросс-фиттингом ожидали {num(cal["fit"]["cross_fit"]["precision_at_threshold"])}. На&nbsp;holdout не&nbsp;подбирался: {num(ap["precision"])} (ДИ от&nbsp;{num(ap["lower95"])}), принято {pct(ap["accepted_share"], 0)} найденного.</figcaption>
    </figure>
  </div>
  <p class="caveat"><b>Слабое место.</b> Принятые значения с&nbsp;уверенностью 0,70–0,80 верны лишь в&nbsp;{pct(band["precision"], 0)} случаев: {thou(band["n"])} значений — {pct(band["share_of_accepted"], 0)} принятого, но&nbsp;{pct(err_share, 0)} ошибок среди принятого; {pct(band["generic_share"], 0)} из&nbsp;них — цвет и&nbsp;модель хвоста. На&nbsp;dev картина та&nbsp;же. Кандидат исправления найден только на&nbsp;dev: калибровка на&nbsp;всём dev → порог {num(CAND["threshold"], 2)}, точность принятых {num(CAND["precision"])} при&nbsp;доле автопринятия {pct(CAND["accepted"])} вместо {pct(ap["accepted_share"], 0)}. <b>Не&nbsp;применён:</b> выбор после просмотра holdout нарушил&nbsp;бы протокол; решение — по&nbsp;мониторингу на&nbsp;новых данных.</p>
  {foot('<span class="mono">submission/calibration.json</span> (<span class="mono">accepted_band</span>), <span class="mono">threshold.json</span>, <span class="mono">REPORT.md</span> §9–10; кандидат — <span class="mono">docs/EVALUATION.md</span> §9.5 (только dev)', 9)}"""
    return frame(9, "Доверие: калибровка и порог", body)


def s_llm(d: dict) -> str:
    md = d["modes"]
    ner = md[("llm_subsample", "+ NER · подвыборка LLM")]
    llm = md[("llm_subsample", "+ LLM · подвыборка LLM")]
    a = LLM["auto"]
    correct = round(LLM["with_gold"] * LLM["correct_share"])
    body = f"""  <div class="kicker">LLM · честный результат</div>
  <h2 class="h">С&nbsp;локальной 4B-моделью LLM — подсказка модератору, а&nbsp;не&nbsp;автоматический источник</h2>
  <div class="two" style="grid-template-columns:5fr 6fr">
    <ol class="flow">
      <li><b>{pct(llm["llm_share"], 0)}</b><span>карточек подвыборки ({LLM["cards"]}), где LLM вообще спросили</span><small>{LLM["calls"]} {plural(LLM["calls"], "вызов", "вызова", "вызовов")}: только слабые и&nbsp;упомянутые в&nbsp;тексте атрибуты; {LLM["tok_in"]}&nbsp;+&nbsp;{LLM["tok_out"]} {plural(LLM["tok_out"], "токен", "токена", "токенов")}, {num(LLM["lat"], 1)}&nbsp;с на&nbsp;вызов (p95 {num(LLM["p95"], 1)}&nbsp;с), ошибок {LLM["errors"]}</small></li>
      <li class="r"><b>{LLM["ungrounded"]}</b><span>{plural(LLM["ungrounded"], "ответ", "ответа", "ответов")} без опоры в&nbsp;тексте — {plural(LLM["ungrounded"], "отброшен", "отброшены", "отброшены")} проверкой заземления</span><small>цитаты нет в&nbsp;карточке или значение из&nbsp;неё не&nbsp;следует</small></li>
      <li><b>{LLM["llm_only"]}</b><span>{plural(LLM["llm_only"], "значение предложила только LLM — ушло", "значения предложила только LLM — ушли", "значений предложила только LLM — ушли")} модератору, а&nbsp;не&nbsp;в&nbsp;каталог</span><small>из&nbsp;них {LLM["new_slots"]} закрывают слот, пустой без LLM</small></li>
      <li class="r"><b>{pct(LLM["correct_share"], 0)}</b><span>верны там, где есть эталон ({correct} из&nbsp;{LLM["with_gold"]})</span><small>поэтому одна LLM не&nbsp;решает: правило <span class="mono">pipeline.UNVERIFIED_ONLY</span></small></li>
    </ol>
    <div>
      <table class="cf">
        <thead><tr><th>Подвыборка, {cards(LLM["cards"])}</th><th>P</th><th>R</th><th>F1</th></tr></thead>
        <tbody>
          <tr><td>+ NER, без LLM</td><td>{num(ner["precision"])}</td><td>{num(ner["recall"])}</td><td>{num(ner["f1"])}</td></tr>
          <tr><td>+ LLM, одна LLM → проверка (в&nbsp;сдаче)</td><td>{num(llm["precision"])}</td><td>{num(llm["recall"])}</td><td>{num(llm["f1"])}</td></tr>
          <tr><td>+ LLM, если принимать одну LLM автоматически</td><td class="d">{num(a["precision"])}</td><td>{num(a["recall"])}</td><td>{num(a["f1"])}</td></tr>
        </tbody>
      </table>
      <p class="verdict"><b>Вывод.</b> Автопринятие ответов 4B-модели добавило&nbsp;бы {num(a["recall"] - llm["recall"])} полноты ценой {num(llm["precision"] - a["precision"])} точности каталога. Правила + NER закрывают ядро и&nbsp;хвост; LLM — подсказки модератору с&nbsp;цитатой. <b>Следующий шаг:</b> тот&nbsp;же замер с&nbsp;моделью организаторов на&nbsp;ВМ (<span class="mono">make report</span>) и&nbsp;7–8B в&nbsp;vLLM — порог и&nbsp;правило доверия не&nbsp;меняются.</p>
    </div>
  </div>
  {foot('<span class="mono">submission/REPORT.md</span> §4, <span class="mono">compare.json</span> (набор <span class="mono">llm_subsample</span>); модель — qwen3:4b-instruct в&nbsp;Ollama на&nbsp;GTX 1060, кэш ответов выключен', 10)}"""
    return frame(10, "LLM: честный результат", body)


def s_cost(d: dict) -> str:
    c = d["costs"]
    pairs = [("Своя GPU (GTX 1060)", c["+ LLM · подвыборка LLM"]["cost_per_1000"],
              c["LLM на каждую карточку: свой стенд (оценка)"]["cost_per_1000"], 2),
             ("GigaChat Lite", c["+ LLM через GigaChat Lite (оценка)"]["cost_per_1000"],
              c["LLM на каждую карточку: GigaChat Lite (оценка)"]["cost_per_1000"], 1),
             ("YandexGPT Lite", c["+ LLM через YandexGPT Lite (оценка)"]["cost_per_1000"],
              c["LLM на каждую карточку: YandexGPT Lite (оценка)"]["cost_per_1000"], 1)]
    items = []
    for who, hyb, every, dd in pairs:
        items.append(f"""      <li><div class="who"><span>{who}</span><em>в&nbsp;{num(every / hyb, 1)} раза дешевле</em></div>
        <div class="row h"><span>гибрид: LLM для&nbsp;{pct(LLM["share"], 0)}&nbsp;карточек</span><i style="width:{hyb / every * 100:.1f}%"></i><span>{num(hyb, dd)}&nbsp;₽</span></div>
        <div class="row"><span>LLM на&nbsp;каждую&nbsp;карточку</span><i style="width:100%"></i><span>{num(every, dd)}&nbsp;₽</span></div></li>""")
    sp = [("Правила и словари", c["правила"]), ("+ атрибуты продавца", c["продавец + правила"]),
          ("+ NER — в сдаче (GPU)", c["+ NER"]), ("+ LLM: гибрид, своя GPU", c["+ LLM · подвыборка LLM"]),
          ("LLM на каждую карточку", c["LLM на каждую карточку: свой стенд (оценка)"])]
    trs = []
    for lab, r in sp:
        ms = r["ms_per_card"]
        ms_s = num(ms / 1000, 1) + "&nbsp;с" if ms >= 1000 else num(ms, 1) + "&nbsp;мс"
        rub = r["cost_per_1000"]
        rub_s = "&lt; 0,01&nbsp;₽" if rub < 0.01 else num(rub, 2) + "&nbsp;₽"
        cls = ' class="hl"' if "в сдаче" in lab else ""
        trs.append(f'<tr{cls}><td>{lab.replace(" ", "&nbsp;", 1)}</td><td>{ms_s}</td><td>{rub_s}</td><td>{pct(r["llm_share"], 0)}</td></tr>')
    body = f"""  <div class="kicker">Экономика</div>
  <h2 class="h">Гибрид в&nbsp;4–7 раз дешевле «LLM на&nbsp;каждую карточку», режим в&nbsp;сдаче — почти бесплатен</h2>
  <div class="two" style="grid-template-columns:6fr 5fr">
    <div>
      <span class="lbl">₽ за&nbsp;1000 карточек · полосы — внутри строки</span>
      <ul class="cost">
{chr(10).join(items)}
      </ul>
    </div>
    <div>
      <span class="lbl">Скорость и&nbsp;стоимость режимов на&nbsp;стенде команды</span>
      <table class="speed">
        <thead><tr><th>Режим</th><th>на&nbsp;карточку</th><th>₽ за&nbsp;1000</th><th>с&nbsp;LLM</th></tr></thead>
        <tbody>
{chr(10).join(trs)}
        </tbody>
      </table>
      <p class="chart-cap" style="margin-top:10px">Время — медиана 3 замеров на&nbsp;всём holdout. Свой стенд — GTX 1060: {num(d["rub_h"], 2)}&nbsp;₽/ч (амортизация + электричество) × время пайплайна. API — тарифы с&nbsp;НДС на&nbsp;14.10.2026 и&nbsp;токены замера ({LLM["tok_in"]}&nbsp;+&nbsp;{LLM["tok_out"]} на&nbsp;вызов). «LLM на&nbsp;каждую» — бенчмарк <span class="mono">attrx llm-bench</span>: все атрибуты, 6,9&nbsp;с на&nbsp;карточку.</p>
    </div>
  </div>
  {foot('<span class="mono">submission/cost.json</span>, <span class="mono">REPORT.md</span> §12, <span class="mono">config/llm_prices.yaml</span>, <span class="mono">docs/LLM.md</span> §6; провайдеры API — оценка по&nbsp;токенам замера', 11)}"""
    return frame(11, "Экономика", body)


def s_rollout(d: dict) -> str:
    m = d["m"]
    ap = m["accepted_precision"]
    sg = d["seg"]
    st, sb = sg["seller_type"], sg["size_band"]
    band = d["cal"]["accepted_band"]
    gap_s = (st["mvideo"]["f1"] - st["marketplace"]["f1"]) * 100
    gap_t = (sb["head"]["f1"] - sb["tail"]["f1"]) * 100
    body = f"""  <div class="kicker">Внедрение и&nbsp;риски</div>
  <h2 class="h">В&nbsp;прод — через тень; каждый риск кейса закрыт мерой и&nbsp;измерен</h2>
  <div class="rr">
    <ol class="steps-v">
      <li class="done"><b>0 · Офлайн-гейт</b><span>точность принятых ≥ 0,95: на&nbsp;всём holdout {num(ap["precision"])}; разрывы {num(gap_s, 1)} и&nbsp;{num(gap_t, 1)} п.п. — выполнено</span></li>
      <li><b>1 · Тень, 2–4 недели</b><span>пишем рядом, не&nbsp;в&nbsp;каталог; сверяем с&nbsp;модераторами; слой 0,70–0,80 — на&nbsp;аудит, решаем о&nbsp;новой калибровке</span></li>
      <li><b>2 · Автопринятие</b><span>выше порога — в&nbsp;PIM в&nbsp;топ-категориях по&nbsp;трафику; выборочный аудит; точность ниже гейта — категория обратно в&nbsp;тень</span></li>
      <li><b>3 · Все категории</b><span>холодный старт: LLM-подсказки на&nbsp;проверку, правки → словари и&nbsp;NER</span></li>
      <li><b>Интеграция</b><span><span class="mono">POST /api/extract/batch</span> (до&nbsp;1000) или <span class="mono">attrx extract</span>; LLM — vLLM / Ollama в&nbsp;контуре, без VPN</span></li>
    </ol>
    <table class="rk2">
      <thead><tr><th>Риск</th><th>Мера в&nbsp;решении</th><th>Что показал holdout</th></tr></thead>
      <tbody>
        <tr><td>Галлюцинации LLM</td><td>схема, справочник, цитата и&nbsp;проверка заземления, калиброванный порог; значение одной LLM — только на&nbsp;проверку</td><td class="ev">{LLM["ungrounded"]} {plural(LLM["ungrounded"], "ответ", "ответа", "ответов")} без опоры {plural(LLM["ungrounded"], "отброшен", "отброшены", "отброшены")}; {LLM["llm_only"]} {plural(LLM["llm_only"], "значение", "значения", "значений")} одной LLM — модератору (верны {pct(LLM["correct_share"], 0)})</td></tr>
        <tr><td>Стоимость на&nbsp;масштабе</td><td>LLM только для слабых и&nbsp;упомянутых атрибутов, кэш, лимиты, on-prem</td><td class="ev">гибрид в&nbsp;4–7 раз дешевле; режим в&nbsp;сдаче — &lt; 0,01&nbsp;₽ за&nbsp;1000 карточек</td></tr>
        <tr><td>Оценка не&nbsp;видит хвост</td><td>весь holdout, метрики по&nbsp;атрибуту и&nbsp;категории, head / torso / tail, полнота по&nbsp;извлекаемому</td><td class="ev">tail {num(sb["tail"]["f1"])}, torso {num(sb["torso"]["f1"])}; 77% пропусков — значения нет в&nbsp;тексте</td></tr>
        <tr><td>Дрейф, новые категории</td><td>классификатор категории, generic-схема, таксономия в&nbsp;YAML; <span class="mono">attrx monitor</span>: review, missing, PSI категорий → алерт</td><td class="ev">без category_id категория угадана у&nbsp;всех 2&#8239;416 карточек; вне фида {pct(COLD["accepted"])} слотов автоматически</td></tr>
        <tr><td>Несправедливость к&nbsp;продавцам</td><td>справочник брендов всего каталога, метрики по&nbsp;сегментам</td><td class="ev">маркетплейс {num(st["marketplace"]["f1"])} против {num(st["mvideo"]["f1"])} у&nbsp;М.Видео (было 0,447 против 0,654)</td></tr>
        <tr><td>Калибровка у&nbsp;порога</td><td>нашли сами; аудит слоя 0,70–0,80 в&nbsp;тени; кандидат — калибровка на&nbsp;всём dev (порог {num(CAND["threshold"], 2)})</td><td class="ev">слой верен на&nbsp;{pct(band["precision"], 0)}: {pct(band["share_of_accepted"], 0)} принятого, {pct(band["wrong"] / band["wrong_accepted_total"], 0)} ошибок; весь поток — {num(ap["precision"])}</td></tr>
      </tbody>
    </table>
  </div>
  {foot('<span class="mono">docs/DEPLOYMENT.md</span>, <span class="mono">docs/RISKS.md</span>, <span class="mono">docs/MONITORING.md</span>, <span class="mono">submission/REPORT.md</span> §9, §13', 12)}"""
    return frame(12, "Внедрение и риски", body)


def s_final(d: dict) -> str:
    m = d["m"]
    ap = m["accepted_precision"]
    twc = m["traffic_weighted_coverage"]
    nf = m["new_fills"]
    body = f"""  <div class="kicker">Итог</div>
  <h2 class="h">Каталог полнее, точность принятого ≥ 0,95, LLM — только под&nbsp;контролем</h2>
  <div class="fin">
    <div><h3>Что сдано</h3><ul>
      <li><b>Код и&nbsp;пайплайн</b> — пакет <span class="mono">attrx</span>, CLI, API, Docker</li>
      <li><b>README</b> — запуск за&nbsp;5 минут, матрица «требование → где»</li>
      <li><b>EDA</b> — <span class="mono">notebooks/01_eda.ipynb</span>, <span class="mono">docs/EDA.md</span></li>
      <li><b>Предсказания по&nbsp;каждой карточке</b> — <span class="mono">submission/predictions.*</span></li>
      <li><b>Отчёт и&nbsp;метрики</b> — <span class="mono">submission/REPORT.md</span>, проверка цифр — <span class="mono">make verify</span></li>
      <li><b>Презентация</b> — этот файл, <span class="mono">docs/slides.pdf</span></li>
    </ul></div>
    <div><h3>Что получает М.Видео</h3><ul>
      <li><b>{num(ap["precision"])}</b> — точность значений, которые попадают в&nbsp;каталог без человека</li>
      <li><b>{pct(twc["before"])} → {pct(twc["after"])}</b> — доля спроса с&nbsp;заполненным атрибутом; +{thou(nf)} {plural(nf, "значение", "значения", "значений")} на&nbsp;holdout</li>
      <li><b>{pct(COLD["accepted"])}</b> слотов заполняется автоматически у&nbsp;карточек вне фида (75% каталога)</li>
      <li><b>{pct(m["review_rate"])}</b> слотов — модератору, с&nbsp;цитатой и&nbsp;в&nbsp;порядке спроса</li>
      <li><b>0,83&nbsp;₽</b> за&nbsp;1000 карточек с&nbsp;LLM на&nbsp;своей GPU; без LLM — меньше копейки</li>
    </ul></div>
    <div><h3>Следующий шаг</h3><ul>
      <li><b>Пилот в&nbsp;тени</b> на&nbsp;одной категории: смартфоны или телевизоры</li>
      <li><b>Калибровка:</b> кандидат с&nbsp;dev (порог {num(CAND["threshold"], 2)}, точность принятых {num(CAND["precision"])} при&nbsp;{pct(CAND["accepted"])} автопринятия) — включаем по&nbsp;данным тени</li>
      <li><b>LLM:</b> замер с&nbsp;моделью организаторов на&nbsp;ВМ и&nbsp;7–8B в&nbsp;vLLM</li>
      <li><b>От&nbsp;М.Видео:</b> список обязательных атрибутов, CTR / конверсия / чек для&nbsp;A/B, выгрузка правок модераторов, разрешённый LLM-контур</li>
    </ul></div>
  </div>
  {foot('Репозиторий — Gitea команды · демо — ВМ команды, порт 8000 · документы: <span class="mono">docs/PRD.md</span>, <span class="mono">EVALUATION.md</span>, <span class="mono">RISKS.md</span>, <span class="mono">DEPLOYMENT.md</span>', 13)}"""
    return frame(13, "Итог", body)


HUD = """<nav class="hud" aria-label="Навигация по слайдам">
  <button type="button" id="prev" aria-label="Предыдущий слайд">←</button>
  <span class="cnt" id="cnt">1 / 13</span>
  <button type="button" id="next" aria-label="Следующий слайд">→</button>
  <button type="button" id="ovr" aria-label="Обзор всех слайдов">Обзор</button>
</nav>"""


def build() -> str:
    concept = CONCEPT.read_text(encoding="utf-8")
    css = re.search(r"<style>(.*?)</style>", concept, re.S)
    js = re.search(r"<script>(.*?)</script>", concept, re.S)
    assert css and js, "в concept.html нет <style> или <script>"
    d = data()
    slides = [s_cover(d), s_problem(d), s_solution(d), s_demo(d), s_result(d), s_ladder(d), s_categories(d),
              s_fairness(d), s_trust(d), s_llm(d), s_cost(d), s_rollout(d), s_final(d)]
    assert len(slides) == N_SLIDES
    head = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>attrx — финальная защита</title>
<meta name="description" content="Финальная презентация кейса М.Видео: автозаполнение характеристик из текста карточки, результаты на holdout.">
<!-- Сгенерировано docs/slides.py из submission/*.json (отчёт v2) — правьте генератор, а не этот файл.
     Навигация: ← → / пробел, Home / End, 1–9, O — обзор, F — полный экран. -->
"""
    return (head + "<style>" + css.group(1) + EXTRA_CSS + "</style>\n</head>\n<body>\n<main class=\"stage\" id=\"stage\">\n"
            + "".join(slides) + "</main>\n" + HUD + "\n<script>" + js.group(1) + "</script>\n</body>\n</html>\n")


if __name__ == "__main__":
    OUT.write_text(build(), encoding="utf-8")
    print(f"-> {OUT.relative_to(ROOT)} ({N_SLIDES} слайдов)")
