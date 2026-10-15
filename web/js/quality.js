// «Качество»: что даёт каждый шаг гибрида, порог точность/покрытие, калибровка, атрибуты, сегменты,
// категории, стоимость, спрос, ошибки. Всё читается из submission/*.json через /api/report.
// Файла нет — честное «ещё не посчитано» с командой, которая его создаст. Числа не выдумываются.
import { fmt, h, icon, put } from "./dom.js";
import { REPORT_FILES } from "./copy.js";
import { dataTable, dotPlot, figure, hbars, heatmap, legendItem, lineChart, reliabilityChart } from "./charts.js";

const isNum = (x) => typeof x === "number" && Number.isFinite(x);
const num = (x) => (x == null || x === "" ? null : Number(x));
const ucfirst = (t) => (t ? t.charAt(0).toUpperCase() + t.slice(1) : t);

export function mountQuality(root, app) {
  let painted = false;

  async function paint() {
    put(root, h("div", { class: "qa-loading", "aria-busy": "true" }, h("div", { class: "skel skel--title" }), h("div", { class: "skel skel--hero" }), h("div", { class: "skel skel--table" })));
    let rep;
    try {
      rep = await app.getReport(true);
    } catch (e) {
      put(root, h("div", { class: "alert alert--danger", role: "alert" }, icon("warning-circle", 16, "alert__icon"),
        h("div", { class: "alert__text" }, h("strong", null, "Отчёт не загрузился. "), e.message)));
      return;
    }
    const titles = attrTitles(app.categories || []);
    const catTitles = Object.fromEntries((app.categories || []).map((c) => [c.name, c.title]));
    const m = rep.metrics;
    put(root, 
      head(rep),
      m?.micro ? hero(rep, app.threshold) : noMetrics(),
      ladder(rep),
      h("div", { class: "qa-grid" }, thresholdSection(rep, app.threshold), calibrationSection(rep, app.threshold)),
      h("div", { class: "qa-grid" }, attributesSection(rep, titles), segmentsSection(rep)),
      categoriesSection(rep, titles, catTitles),
      trafficSection(rep, titles, catTitles),
      h("div", { class: "qa-grid" }, costSection(rep), errorsSection(rep, titles)));
    painted = true;
  }

  return {
    async show() {
      if (!painted) await paint();
    },
    refresh: paint,
  };
}

// ---------- шапка и главный вывод ----------
function head(rep) {
  const meta = rep._meta || {};
  const m = rep.metrics || {};
  const updated = Math.max(0, ...Object.values(meta.files || {}).map((f) => f.updated || 0));
  const synthetic = /синтет/i.test(meta.note || "");
  return h("header", { class: "vhead vhead--qa" },
    h("div", { class: "vhead__main" },
      h("h1", { class: "vhead__title" }, "Качество извлечения"),
      h("p", { class: "vhead__lede" }, "Метрики на размеченном эталоне. Источник: submission/, пересчитать командой make report (attrx report-real).")),
    h("dl", { class: "facts" },
      isNum(m.n_cards) ? fact("Эталон", fmt.count(m.n_cards, "карточка", "карточки", "карточек")) : null,
      m.mode ? fact("Режим в сдаче", m.mode) : null,
      updated ? fact("Обновлено", fmt.date(updated)) : null),
    meta.note ? h("div", { class: "alert " + (synthetic ? "alert--warning" : "alert--info") + " vhead__note" },
      icon(synthetic ? "warning" : "info", 16, "alert__icon"), h("p", { class: "alert__text" }, h("strong", null, "Набор данных. "), meta.note)) : null);
}

const fact = (k, v) => h("div", { class: "facts__item" }, h("dt", null, k), h("dd", null, v));

function sweepRow(rep, thr) {
  const rows = normThreshold(rep)?.rows || [];
  let best = null;
  for (const r of rows) if (!best || Math.abs(r.threshold - thr) < Math.abs(best.threshold - thr)) best = r;
  return best;
}

function hero(rep, thr) {
  const m = rep.metrics;
  const row = sweepRow(rep, thr);
  const coverage = isNum(m.coverage) ? m.coverage : row?.coverage;
  const precision = isNum(m.micro?.precision) ? m.micro.precision : row?.precision;
  return h("section", { class: "hero", "aria-label": "Главный вывод" },
    h("p", { class: "hero__line" },
      "При пороге уверенности ", h("b", null, fmt.conf(thr)), " система сама заполняет ",
      h("b", null, fmt.pct(coverage)), " значений с точностью ", h("b", null, fmt.p3(precision)), ". ",
      isNum(m.review_rate) ? ["На проверку модератору уходит ", h("b", null, fmt.pct(m.review_rate)), "."] : null),
    h("dl", { class: "hero__stats" },
      stat("F1", fmt.p3(m.micro.f1), "микро по всем значениям"),
      isNum(m.traffic_weighted_f1) ? stat("F1 с учётом спроса", fmt.p3(m.traffic_weighted_f1), "атрибуты взвешены по поисковым кликам") : null,
      isNum(m.macro_f1) ? stat("Макро F1", fmt.p3(m.macro_f1), "средний по атрибутам: редкие весят столько же") : null,
      stat("Полнота", fmt.p3(m.micro.recall), `${m.micro.tp ?? "?"} верных, ${m.micro.fn ?? "?"} пропущено`)));
}

const stat = (label, value, hint) => h("div", { class: "hero__stat", "data-tip": hint }, h("dt", null, label), h("dd", null, value));

function noMetrics() {
  return h("div", { class: "empty empty--wide" }, icon("chart-line-up", 28, "empty__icon"),
    h("h2", { class: "empty__title" }, "Метрик пока нет"),
    h("p", { class: "empty__text" }, "Запустите make report (attrx report-real) на эталоне, и здесь появятся точность, полнота, покрытие и сравнение подходов."));
}

function missing(file, what) {
  return h("div", { class: "qa-empty" }, icon("circle-dashed", 20, "qa-empty__icon"),
    h("div", null, h("p", { class: "qa-empty__title" }, "Ещё не посчитано"),
      h("p", { class: "qa-empty__text" }, ucfirst(what || REPORT_FILES[file] || file) + " появится после make report (attrx report-real): submission/" + file + ".json")));
}

function unknownFormat(file) {
  return h("div", { class: "qa-empty" }, icon("warning", 20, "qa-empty__icon"),
    h("div", null, h("p", { class: "qa-empty__title" }, "Формат не распознан"),
      h("p", { class: "qa-empty__text" }, "Интерфейс не понял структуру submission/" + file + ".json. Данные есть в файле и в /api/report.")));
}

function section(title, note, ...body) {
  return h("section", { class: "qa-sec" }, h("div", { class: "qa-sec__head" }, h("h2", { class: "qa-sec__title" }, title), note ? h("p", { class: "qa-sec__note" }, note) : null), body);
}

// ---------- сравнение подходов ----------
const LADDER_COLS = [
  { key: "precision", label: "Точность", f: fmt.p3, better: 1, delta: "p3" },
  { key: "recall", label: "Полнота", f: fmt.p3, better: 1, delta: "p3" },
  { key: "f1", label: "F1", f: fmt.p3, better: 1, delta: "p3" },
  { key: "traffic_weighted_f1", label: "F1 с учётом спроса", f: fmt.p3, better: 1, delta: "p3" },
  { key: "coverage", label: "Покрытие", f: fmt.pct, better: 1, delta: "pp" },
  { key: "review_rate", label: "На проверку", f: fmt.pct, better: -1, delta: "pp" },
  { key: "cards_per_sec", label: "Карточек в секунду", f: (x) => fmt.num(x, 1), better: 1, delta: null },
  { key: "llm_calls", label: "Вызовов LLM", f: fmt.int, better: -1, delta: null },
  { key: "cost_per_1000", label: "₽ за 1000 карточек", f: (x) => fmt.num(x, 2), better: -1, delta: null },
];

function deltaText(d, kind) {
  if (!isNum(d) || Math.abs(d) < 1e-9) return null;
  const sign = d > 0 ? "+" : "−";
  return kind === "pp" ? sign + fmt.num(Math.abs(d) * 100, 1) + " п.п." : sign + fmt.p3(Math.abs(d));
}

function ladder(rep) {
  const rows = Array.isArray(rep.compare) ? rep.compare.map((r) => ({ ...r })) : null;
  const note = "Каждая строка добавляет источник к предыдущей. Так видно, что даёт каждый шаг гибрида и сколько он стоит.";
  if (!rows?.length) return section("Что даёт каждый шаг", note, missing("compare"));
  const cost = normCost(rep);
  if (cost) for (const r of rows) {
    const c = cost.rows.find((x) => x.mode === r.mode);
    if (c && isNum(c.cost_per_1000)) r.cost_per_1000 = c.cost_per_1000;
  }
  const cols = LADDER_COLS.filter((c) => rows.some((r) => isNum(r[c.key])));
  const chosen = rep.metrics?.mode;
  const hasLlm = rows.some((r) => (r.sources || []).includes("llm") || /llm/i.test(r.mode));
  const body = rows.map((r, i) => h("tr", { class: r.mode === chosen ? "is-chosen" : null },
    h("th", { scope: "row", class: "ladder__mode" }, h("span", null, ucfirst(r.mode)), r.mode === chosen ? h("span", { class: "flag" }, "в сдаче") : null),
    cols.map((c) => {
      // дельта — только к предыдущей строке того же набора (compare.json: весь набор, затем подвыборка для LLM)
      const prev = i > 0 && (rows[i - 1].set ?? null) === (r.set ?? null) ? rows[i - 1] : null;
      const d = prev && c.delta && isNum(r[c.key]) && isNum(prev[c.key]) ? r[c.key] - prev[c.key] : null;
      const dt = deltaText(d, c.delta);
      const good = d != null && d * c.better > 0;
      return h("td", { class: "num" }, isNum(r[c.key]) ? c.f(r[c.key]) : h("span", { class: "muted" }, "нет данных"),
        dt ? h("span", { class: "delta " + (good ? "delta--good" : "delta--bad") }, dt) : null);
    })));
  if (!hasLlm) {
    body.push(h("tr", { class: "is-ghost" }, h("th", { scope: "row", class: "ladder__mode" }, "+ LLM"),
      h("td", { colspan: String(cols.length) }, "Не запускалась в этом отчёте: attrx report-real собран без подключённой LLM. С ключом (LLM_* или OPENAI_*) режим добавится автоматически.")));
  }
  return section("Что даёт каждый шаг", note,
    h("div", { class: "tbl-wrap", tabindex: "0", role: "region", "aria-label": "Таблица (прокручивается по горизонтали)" }, h("table", { class: "tbl ladder" },
      h("thead", null, h("tr", null, h("th", { scope: "col" }, "Подход"), cols.map((c) => h("th", { scope: "col", class: "num" }, c.label)))),
      h("tbody", null, body))));
}

// ---------- порог ----------
function normThreshold(rep) {
  const src = rep.threshold;
  let rows = Array.isArray(src) ? src : src && (src.points || src.curve || src.rows || src.sweep);
  if (!Array.isArray(rows)) rows = rep.metrics?.threshold_sweep;
  if (!Array.isArray(rows)) return null;
  rows = rows.map((r) => ({ ...r, threshold: num(r.threshold ?? r.t), precision: num(r.precision), coverage: num(r.coverage), recall: num(r.recall), f1: num(r.f1) }))
    .filter((r) => isNum(r.threshold)).sort((a, b) => a.threshold - b.threshold);
  return rows.length ? { rows, target: num(src?.target_precision ?? src?.target) ?? 0.95, file: rep.threshold ? "threshold.json" : "metrics.json" } : null;
}

function thresholdSection(rep, thr) {
  const d = normThreshold(rep);
  const note = "Выше порог: точнее, но меньше значений проходит автоматически. Красная линия: порог, выбранный в настройках.";
  if (!d) return section("Порог: точность и покрытие", note, missing("threshold", "Кривая порога"));
  const pts = (k) => d.rows.filter((r) => isNum(r[k])).map((r) => ({ x: r.threshold, y: r[k] }));
  const series = [
    { label: "точность", cls: "s1", values: pts("precision"), format: fmt.p3 },
    { label: "покрытие", cls: "s2", values: pts("coverage"), format: fmt.pct, labelDy: 16 },
  ].filter((sr) => sr.values.length);
  const at = d.rows.reduce((b, r) => (!b || Math.abs(r.threshold - thr) < Math.abs(b.threshold - thr) ? r : b), null);
  const chart = lineChart({
    series, marker: { x: thr, label: "порог " + fmt.conf(thr) }, refs: [{ y: d.target, label: "цель " + fmt.conf(d.target) }],
    xLabel: "порог уверенности", ariaLabel: `Точность и покрытие в зависимости от порога. При пороге ${fmt.conf(thr)}: точность ${fmt.p3(at?.precision)}, покрытие ${fmt.pct(at?.coverage)}`,
    tipRows: (x) => {
      const r = d.rows.find((q) => q.threshold === x);
      return r && isNum(r.f1) ? [h("div", { class: "tip__row" }, h("span", { class: "tip__key" }), h("strong", null, fmt.p3(r.f1)), h("span", null, "F1"))] : [];
    },
  });
  const table = dataTable(["Порог", "Точность", "Покрытие", "Полнота", "F1"],
    d.rows.map((r) => [fmt.conf(r.threshold), fmt.p3(r.precision), fmt.pct(r.coverage), fmt.p3(r.recall), fmt.p3(r.f1)]), { numeric: [1, 2, 3, 4] });
  return section("Порог: точность и покрытие", note, figure({
    title: "Как порог меняет точность и покрытие",
    legend: [legendItem("точность", "s1"), legendItem("покрытие", "s2"), legendItem("выбранный порог", "pen", "rule")],
    chart, table, note: "Источник: submission/" + d.file,
  }));
}

// ---------- калибровка ----------
function normCalibration(rep) {
  const c = rep.calibration;
  if (!c) return null;
  let bins = Array.isArray(c) ? c : c.bins || c.reliability || c.rows;
  if (!Array.isArray(bins)) return { bins: [], unknown: true };
  bins = bins.map((b) => {
    const range = Array.isArray(b.bin) ? b.bin : Array.isArray(b.range) ? b.range : null;
    const lo = num(b.lo ?? b.bin_lo ?? b.low ?? b.from ?? range?.[0]);
    const hi = num(b.hi ?? b.bin_hi ?? b.high ?? b.to ?? range?.[1]);
    const conf = num(b.confidence ?? b.mean_confidence ?? b.avg_confidence ?? b.conf);
    return { lo: isNum(lo) ? lo : isNum(conf) ? Math.floor(conf * 10) / 10 : null, hi: isNum(hi) ? hi : isNum(conf) ? Math.floor(conf * 10) / 10 + 0.1 : null,
      conf, acc: num(b.accuracy ?? b.precision ?? b.acc ?? b.observed), n: num(b.n ?? b.count ?? b.support) || 0 };
  }).filter((b) => isNum(b.acc) && isNum(b.lo) && isNum(b.hi));
  return { bins, ece: num(c.ece ?? c.ECE), unknown: !bins.length };
}

function calibrationSection(rep, thr) {
  const d = normCalibration(rep);
  const note = "Совпадает ли уверенность с реальной точностью: если столбцы идут по диагонали, порогу можно доверять.";
  if (!d) return section("Калибровка уверенности", note, missing("calibration"));
  if (d.unknown) return section("Калибровка уверенности", note, unknownFormat("calibration"));
  const table = dataTable(["Уверенность", "Точность", "Средняя уверенность", "Значений"],
    d.bins.map((b) => [fmt.conf(b.lo) + "…" + fmt.conf(b.hi), fmt.p3(b.acc), fmt.conf(b.conf), fmt.int(b.n)]), { numeric: [1, 2, 3] });
  return section("Калибровка уверенности", note, figure({
    title: "Уверенность против фактической точности" + (isNum(d.ece) ? ", ECE " + fmt.p3(d.ece) : ""),
    chart: reliabilityChart({ bins: d.bins, threshold: thr, ariaLabel: "Диаграмма надёжности уверенности" + (isNum(d.ece) ? ", ожидаемая ошибка калибровки " + fmt.p3(d.ece) : "") }),
    table, note: "Источник: submission/calibration.json",
  }));
}

// ---------- атрибуты ----------
function attrTitles(categories) {
  const map = {};
  for (const c of categories) for (const a of c.attributes || []) {
    map[a.key] = map[a.key] || { title: a.title, cats: [] };
    map[a.key].cats.push(c.title);
    if (map[a.key].title !== a.title && !map[a.key].alt) map[a.key].alt = a.title;
  }
  return map;
}

function attributesSection(rep, titles) {
  const pa = rep.metrics?.per_attribute;
  const note = "Хуже всего вверху. Красным отмечены характеристики с F1 ниже 0,90: с них начинать улучшения.";
  if (!pa || !Object.keys(pa).length) return section("По характеристикам", note, missing("metrics", "Метрики по атрибутам"));
  const items = Object.entries(pa).map(([k, v]) => ({
    key: k, label: titles[k]?.title || k, sub: titles[k]?.alt ? "другое название: «" + titles[k].alt + "»" : null, value: v.f1, v,
    tipTitle: titles[k]?.title || k, tip: `Точность ${fmt.p3(v.precision)}, полнота ${fmt.p3(v.recall)}. Верных ${v.tp}, лишних ${v.fp}, пропущено ${v.fn}.`,
  })).sort((a, b) => a.value - b.value || a.label.localeCompare(b.label, "ru"));
  const table = dataTable(["Характеристика", "Точность", "Полнота", "F1", "Верных", "Лишних", "Пропущено"],
    items.map((i) => [i.label, fmt.p3(i.v.precision), fmt.p3(i.v.recall), fmt.p3(i.v.f1), fmt.int(i.v.tp), fmt.int(i.v.fp), fmt.int(i.v.fn)]), { numeric: [1, 2, 3, 4, 5, 6] });
  return section("По характеристикам", note, figure({
    title: "F1 по каждой характеристике",
    chart: hbars(items, { emphasize: (i) => i.value < 0.9, aside: (i) => h("span", { class: "pr" }, h("span", null, "P ", fmt.num(i.v.precision, 2)), h("span", null, "R ", fmt.num(i.v.recall, 2))) }),
    table,
  }));
}

// ---------- сегменты ----------
function normSegments(rep) {
  const s = rep.segments;
  if (!s || typeof s !== "object") return null;
  const rows = [];
  const push = (label, v, group) => isNum(num(v.f1)) && rows.push({ label, group, f1: num(v.f1), coverage: num(v.coverage), precision: num(v.precision), recall: num(v.recall), n: num(v.n_cards ?? v.n ?? v.cards) });
  const entries = Array.isArray(s) ? s.map((x) => [x.segment || x.name || x.label, x]) : Object.entries(s);
  for (const [k, v] of entries) {
    if (!v || typeof v !== "object") continue;
    if ("f1" in v) push(k, v, null);
    else for (const [k2, v2] of Object.entries(v)) if (v2 && typeof v2 === "object") push(k2, v2, k);
  }
  return rows.length ? rows : null;
}

function segmentsSection(rep) {
  const rows = normSegments(rep);
  const note = "Одинаково ли хорошо для всех: продавцы М.Видео и маркетплейса, популярные и редкие категории, короткие и длинные описания.";
  if (!rows) return section("Справедливость: сегменты", note, missing("segments"));
  const vals = rows.flatMap((r) => [r.f1, r.coverage]).filter(isNum);
  const lo = Math.max(0, Math.floor((Math.min(...vals) - 0.05) * 10) / 10);
  const overall = rep.metrics?.micro?.f1;
  const groups = [...new Set(rows.map((r) => r.group))];
  const plot = groups.map((g) => h("div", { class: "dots-group" }, g ? h("h4", { class: "dots-group__title" }, g) : null,
    dotPlot(rows.filter((r) => r.group === g).map((r) => ({ ...r, sub: isNum(r.n) ? fmt.count(r.n, "карточка", "карточки", "карточек") : null,
      tip: `F1 ${fmt.p3(r.f1)}, покрытие ${fmt.pct(r.coverage)}, точность ${fmt.p3(r.precision)}` })),
    { series: [{ key: "f1", label: "F1", cls: "s1" }, { key: "coverage", label: "покрытие", cls: "s2" }], ref: overall, refLabel: "F1 по всем карточкам", domain: [lo, 1] })));
  const table = dataTable(["Сегмент", "Карточек", "Точность", "Полнота", "F1", "Покрытие"],
    rows.map((r) => [(r.group ? r.group + ": " : "") + r.label, fmt.int(r.n), fmt.p3(r.precision), fmt.p3(r.recall), fmt.p3(r.f1), fmt.pct(r.coverage)]), { numeric: [1, 2, 3, 4, 5] });
  return section("Справедливость: сегменты", note, figure({
    title: "F1 и покрытие по сегментам",
    legend: [legendItem("F1", "s1", "dot"), legendItem("покрытие", "s2", "dot"), isNum(overall) ? legendItem("F1 по всем", "ref", "rule") : null],
    chart: h("div", null, plot), table, note: "Шкала начинается с " + fmt.num(lo, 1) + ", чтобы различия были видны.",
  }));
}

// ---------- категории ----------
function categoriesSection(rep, titles, catTitles) {
  const pc = rep.metrics?.per_category;
  const note = "Цвет показывает долю ошибок: чем краснее, тем хуже. В ячейке F1. Пустая клетка: характеристики нет в категории.";
  if (!pc || !Object.keys(pc).length) return section("Категории и характеристики", note, missing("metrics", "Разрез по категориям"));
  const keys = [...new Set(Object.values(pc).flatMap((v) => Object.keys(v)))];
  const cols = keys.map((k) => ({ key: k, label: titles[k]?.title || k }));
  const rows = Object.entries(pc).map(([c, v]) => {
    const gold = Object.values(v).reduce((acc, x) => acc + (x.tp || 0) + (x.fn || 0), 0);
    return { key: c, label: catTitles[c] || c, sub: fmt.count(gold, "значение", "значения", "значений") + " в эталоне", v };
  });
  return section("Категории и характеристики", note,
    heatmap({ rows, cols, cell: (r, c) => r.v[c.key], rowLabel: "Категория" }),
    h("div", { class: "heat-legend", "aria-hidden": "true" }, h("span", null, "без ошибок"), ...[0, 1, 2, 3, 4].map((i) => h("span", { class: "heat-legend__sw heat--" + i })), h("span", null, "больше 20% ошибок")));
}

// ---------- стоимость ----------
const COST_KEYS = {
  cost_per_1000: ["cost_per_1000", "rub_per_1000", "cost_per_1000_cards", "cost_rub_per_1000", "price_per_1000"],
  ms_per_card: ["ms_per_card", "latency_ms_per_card", "latency_ms", "avg_latency_ms"],
  llm_share: ["llm_share", "llm_call_rate", "llm_calls_share", "share_llm"],
  llm_calls_per_1000: ["llm_calls_per_1000", "calls_per_1000"],
  cards_per_sec: ["cards_per_sec", "throughput"],
};

function normCost(rep) {
  const c = rep.cost;
  if (!c) return null;
  let list = Array.isArray(c) ? c : c.modes || c.rows;
  if (!Array.isArray(list) && typeof c === "object") list = Object.entries(c).filter(([, v]) => v && typeof v === "object").map(([mode, v]) => ({ mode, ...v }));
  if (!Array.isArray(list)) return { rows: [], unknown: true };
  const rows = list.map((r) => {
    const out = { mode: r.mode || r.name || r.label };
    for (const [k, alts] of Object.entries(COST_KEYS)) {
      const hit = alts.find((a) => isNum(num(r[a])));
      if (hit) out[k] = num(r[hit]);
    }
    return out;
  }).filter((r) => r.mode);
  return { rows, unknown: !rows.length, note: c.note || c.assumptions || null };
}

function costSection(rep) {
  const d = normCost(rep);
  const note = "Во что обходится каждый режим на 1000 карточек: LLM вызывается только для остатка, поэтому гибрид дешевле, чем LLM на каждую карточку.";
  if (!d) return section("Стоимость и скорость", note, missing("cost"));
  if (d.unknown) return section("Стоимость и скорость", note, unknownFormat("cost"));
  const items = d.rows.filter((r) => isNum(r.cost_per_1000)).map((r) => ({ label: ucfirst(r.mode), value: r.cost_per_1000 }));
  const max = Math.max(...items.map((i) => i.value), 0) || 1;
  const table = dataTable(["Режим", "₽ за 1000 карточек", "мс на карточку", "Доля с вызовом LLM", "Карточек в секунду"],
    d.rows.map((r) => [ucfirst(r.mode), fmt.num(r.cost_per_1000, 2), fmt.num(r.ms_per_card, 1), fmt.pct(r.llm_share), fmt.num(r.cards_per_sec, 1)]), { numeric: [1, 2, 3, 4] });
  return section("Стоимость и скорость", note, figure({
    title: "Рублей на 1000 карточек",
    chart: items.length ? hbars(items, { max, format: (v) => fmt.num(v, 2) + " ₽" }) : h("p", { class: "blk__empty" }, "В cost.json нет стоимости по режимам."),
    table, note: d.note ? String(d.note) : "Источник: submission/cost.json",
  }));
}

// ---------- спрос ----------
const KINDS = { brand: "бренд", model: "модель", text: "текстовое значение", number: "число с единицей", color: "цвет" };
const PARAM_TITLE = { vendor: "Бренд", model: "Модель" };

function trafficSection(rep, titles, catTitles) {
  const tr = rep.traffic;
  const note = "Какие характеристики покупатели называют в поисковых запросах. Эти веса определяют F1 с учётом спроса и порядок очереди проверки.";
  if (!tr) return section("Что ищут покупатели", note, missing("traffic"));
  // формат потока данных: _meta + groups (категории кейса) + categories (самые кликабельные)
  if (tr.groups || tr.categories) {
    const meta = tr._meta || {};
    const kinds = Object.entries(meta.kinds_share || {}).sort((x, y) => y[1] - x[1]);
    const lead = isNum(meta.total_clicks) ? h("p", { class: "traffic__lead" },
      "Разобрано ", h("b", null, fmt.compact(meta.total_clicks)), " кликов по ", h("b", null, fmt.compact(meta.n_queries)), " запросам. ",
      isNum(meta.any_attribute_share) ? ["В ", h("b", null, fmt.pct(meta.any_attribute_share)), " кликов покупатель назвал характеристику товара."] : null) : null;
    const kindBars = kinds.length ? figure({
      title: "Что называют в запросе", chart: hbars(kinds.map(([k, v]) => ({ label: ucfirst(KINDS[k] || k), value: v })), { max: Math.max(...kinds.map((x) => x[1])), format: fmt.pct }),
      note: "Доля кликов; запрос может называть несколько характеристик сразу.",
    }) : null;
    const groups = Object.entries(tr.groups || {}).map(([key, g]) => {
      const items = Object.entries(g.weights || {}).map(([k, w]) => ({ label: PARAM_TITLE[k] || ucfirst(k), value: num(w) }))
        .filter((i) => isNum(i.value)).sort((x, y) => y.value - x.value);
      const seen = new Set();
      const uniq = items.filter((i) => (seen.has(i.label) ? false : seen.add(i.label))).slice(0, 7);
      return uniq.length ? figure({
        title: (g.titles || [])[0] || catTitles[key] || key,
        chart: hbars(uniq, { max: Math.max(...uniq.map((i) => i.value)) || 1, format: fmt.pct }),
        note: isNum(g.clicks) ? fmt.compact(g.clicks) + " кликов" : null,
      }) : null;
    }).filter(Boolean);
    const top = (tr.categories || []).slice(0, 6);
    const table = top.length ? h("div", { class: "tbl-wrap", tabindex: "0", role: "region", "aria-label": "Таблица (прокручивается по горизонтали)" }, h("table", { class: "tbl" },
      h("thead", null, h("tr", null, h("th", { scope: "col" }, "Категория"), h("th", { scope: "col", class: "num" }, "Кликов"), h("th", { scope: "col" }, "Чаще всего называют"), h("th", { scope: "col" }, "Пример запроса"))),
      h("tbody", null, top.map((c) => {
        const ex = c.top?.[0]?.examples?.[0];
        return h("tr", null, h("th", { scope: "row" }, c.title || c.id), h("td", { class: "num" }, fmt.compact(c.clicks)),
          h("td", null, (c.top || []).slice(0, 3).map((a) => h("span", { class: "tag tag--share" }, (PARAM_TITLE[a.attribute] || a.attribute) + " " + fmt.pct(a.share)))),
          h("td", { class: "traffic__q" }, ex ? ["«" + ex.query + "» ", h("span", { class: "muted" }, fmt.compact(ex.clicks) + " кликов")] : ""));
      })))) : null;
    return section("Что ищут покупатели", note, lead,
      h("div", { class: "traffic-grid" }, kindBars, groups),
      table, h("p", { class: "chart__note" }, "Источник: submission/traffic.json" + (meta.date ? ", " + meta.date : "") + (tr.n_categories ? ", категорий: " + tr.n_categories : "")));
  }
  // простой формат {категория: {атрибут: вес}}
  const blocks = Object.entries(tr).filter(([k, v]) => k !== "_meta" && v && typeof v === "object" && !Array.isArray(v)).slice(0, 4).map(([cat, v]) => {
    const items = Object.entries(v).map(([k, w]) => ({ label: titles[k]?.title || k, value: num(typeof w === "object" ? w?.weight ?? w?.share : w) }))
      .filter((i) => isNum(i.value)).sort((a, b) => b.value - a.value).slice(0, 8);
    return items.length ? figure({ title: catTitles[cat] || cat, chart: hbars(items, { max: Math.max(...items.map((i) => i.value)) || 1, format: (x) => fmt.num(x, 2) }) }) : null;
  }).filter(Boolean);
  if (!blocks.length) return section("Что ищут покупатели", note, unknownFormat("traffic"));
  return section("Что ищут покупатели", note, h("div", { class: "traffic-grid" }, blocks), h("p", { class: "chart__note" }, "Источник: submission/traffic.json"));
}

// ---------- ошибки ----------
const KIND = { wrong: "неверное значение", extra: "лишнее значение", missed: "пропуск" };
const cellValue = (v) => (v == null || v === "" ? "пусто" : /^-?\d+(\.\d+)?$/.test(String(v)) ? fmt.num(Number(v), 2) : String(v));

function errorsSection(rep, titles) {
  const e = rep.errors;
  const note = "Каждая ошибка на эталоне построчно лежит в submission/errors.csv.";
  if (!e) return section("Разбор ошибок", note, missing("errors", "Разбор ошибок"));
  if (!e.total) return section("Разбор ошибок", note, h("p", { class: "blk__empty" }, "Ошибок на эталоне нет."));
  return section("Разбор ошибок", note,
    h("dl", { class: "facts facts--row" },
      fact("Всего", fmt.int(e.total)),
      Object.entries(e.by_kind || {}).map(([k, n]) => fact(ucfirst(KIND[k] || k), fmt.int(n)))),
    h("div", { class: "tbl-wrap", tabindex: "0", role: "region", "aria-label": "Таблица (прокручивается по горизонтали)" }, dataTable(["Карточка", "Характеристика", "Ошибка", "Предсказано", "Эталон", "Уверенность"],
      (e.sample || []).slice(0, 12).map((r) => [r.card_id, titles[r.attribute]?.title || r.attribute, KIND[r.kind] || r.kind, cellValue(r.predicted), cellValue(r.gold), r.confidence ? fmt.conf(Number(r.confidence)) : ""]),
      { numeric: [5] })));
}
