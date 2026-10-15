// Графики качества: SVG рисуется под реальную ширину контейнера (ResizeObserver), без библиотек.
// Правила (навык dataviz): одна ось Y, линии 2 px, сетка — волосяные линии, подписи текстовыми цветами,
// красный — только рабочая точка, у каждого графика есть табличный двойник и подсказка по наведению/клавишам.
import { fmt, h, icon, s, put } from "./dom.js";

let figN = 0;

/** Обёртка графика: заголовок, переключатель «график / таблица», легенда, подпись. */
export function figure({ title, note, legend, chart, table, summary }) {
  const id = "fig-" + ++figN;
  const chartWrap = h("div", { class: "chart__plot" }, chart);
  const tableWrap = h("div", { class: "chart__table", hidden: true }, table);
  const btn = table
    ? h("button", { class: "btn btn--ghost btn--sm chart__toggle", type: "button", "aria-pressed": "false", "aria-controls": id + "-t" }, icon("list-checks", 14), "Таблица")
    : null;
  tableWrap.id = id + "-t";
  btn?.addEventListener("click", () => {
    const on = btn.getAttribute("aria-pressed") !== "true";
    btn.setAttribute("aria-pressed", String(on));
    put(btn, icon(on ? "chart-line-up" : "list-checks", 14), on ? "График" : "Таблица");
    tableWrap.hidden = !on;
    chartWrap.hidden = on;
  });
  return h("figure", { class: "chart", "aria-labelledby": id },
    h("div", { class: "chart__head" }, h("h3", { class: "chart__title", id }, title), btn),
    legend ? h("div", { class: "legend" }, legend) : null,
    chartWrap, tableWrap,
    note || summary ? h("figcaption", { class: "chart__note" }, note, summary ? h("span", { class: "sr-only" }, summary) : null) : null);
}

export function legendItem(label, cls, kind = "line") {
  return h("span", { class: "legend__item" }, h("span", { class: `legend__key legend__key--${kind} ${cls}`, "aria-hidden": "true" }), label);
}

export function dataTable(head, rows, { numeric = [] } = {}) {
  return h("table", { class: "tbl" },
    h("thead", null, h("tr", null, head.map((c, i) => h("th", { scope: "col", class: numeric.includes(i) ? "num" : null }, c)))),
    h("tbody", null, rows.map((r) => h("tr", null, r.map((c, i) => (i === 0 ? h("th", { scope: "row" }, c) : h("td", { class: numeric.includes(i) ? "num" : null }, c)))))));
}

/** Перерисовка под ширину контейнера. draw(width) возвращает SVG/элемент. */
function responsive(draw, height) {
  const box = h("div", { class: "plot", style: { minHeight: height + "px" } });
  let lastW = 0;
  const ro = new ResizeObserver((entries) => {
    const w = Math.floor(entries[0].contentRect.width);
    if (!w || w === lastW) return;
    lastW = w;
    requestAnimationFrame(() => put(box, draw(w)));
  });
  ro.observe(box);
  return box;
}

const lin = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);

function tipBox() {
  return h("div", { class: "chart__tip", hidden: true, "aria-hidden": "true" });
}

function placeTip(tip, wrap, x, y) {
  tip.hidden = false;
  const w = tip.offsetWidth;
  const ww = wrap.clientWidth;
  let left = x + 14;
  if (left + w > ww - 4) left = x - w - 14;
  tip.style.transform = `translate(${Math.max(4, left)}px, ${Math.max(0, y - tip.offsetHeight / 2)}px)`;
}

/**
 * Линейный график по общей оси X (порог уверенности).
 * series: [{label, cls, values: [{x, y}]}], marker: {x, label}, refs: [{y, label}]
 */
export function lineChart({ series, height = 260, xFormat = fmt.conf, yFormat = fmt.p3, yTicks = [0, 0.25, 0.5, 0.75, 1],
  xTicks = [0, 0.2, 0.4, 0.6, 0.8, 1], marker, refs = [], xLabel, ariaLabel, tipRows }) {
  const xs = [...new Set(series.flatMap((sr) => sr.values.map((p) => p.x)))].sort((a, b) => a - b);
  return responsive((W) => {
    const m = { l: 44, r: refs.length ? 74 : 16, t: 30, b: 34 };
    const X = lin(0, 1, m.l, W - m.r);
    const Y = lin(0, 1, height - m.b, m.t);
    const svg = s("svg", { width: W, height, viewBox: `0 0 ${W} ${height}`, role: "img", "aria-label": ariaLabel, class: "svgchart" });
    for (const t of yTicks) {
      svg.append(s("line", { class: "grid", x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }),
        s("text", { class: "tick", x: m.l - 8, y: Y(t) + 4, "text-anchor": "end" }, yFormat === fmt.pct ? fmt.pct(t) : fmt.num(t, 2)));
    }
    for (const t of xTicks) svg.append(s("text", { class: "tick", x: X(t), y: height - m.b + 18, "text-anchor": "middle" }, xFormat(t)));
    svg.append(s("line", { class: "axis", x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0) }));
    if (xLabel) svg.append(s("text", { class: "tick tick--label", x: W - m.r, y: height - 4, "text-anchor": "end" }, xLabel));
    for (const r of refs) {
      svg.append(s("line", { class: "ref", x1: m.l, x2: W - m.r + 4, y1: Y(r.y), y2: Y(r.y) }),
        s("text", { class: "ref__label", x: W - m.r + 8, y: Y(r.y) + 4 }, r.label));
    }
    for (const sr of series) {
      const d = sr.values.map((p, i) => (i ? "L" : "M") + X(p.x).toFixed(1) + "," + Y(p.y).toFixed(1)).join(" ");
      svg.append(s("path", { class: "ln " + sr.cls, d }));
    }
    if (marker) {
      const mx = X(marker.x);
      svg.append(s("line", { class: "marker", x1: mx, x2: mx, y1: m.t - 6, y2: Y(0) }),
        s("text", { class: "marker__label", x: mx, y: m.t - 14, "text-anchor": "middle" }, marker.label));
      // подписи значений справа от рабочей точки; если две подписи сближаются, нижнюю сдвигаем вниз
      const left = mx > W - m.r - 80;
      let lastY = -Infinity;
      const pts = series.map((sr) => ({ sr, p: nearest(sr.values, marker.x) })).filter((x) => x.p).sort((a, b) => Y(a.p.y) - Y(b.p.y));
      for (const { sr, p } of pts) {
        const ly = Math.max(Y(p.y) + 4, lastY + 15);
        lastY = ly;
        svg.append(s("circle", { class: "dot " + sr.cls, cx: mx, cy: Y(p.y), r: 4.5 }),
          s("text", { class: "dot__label", x: mx + (left ? -10 : 10), y: ly, "text-anchor": left ? "end" : "start" }, sr.format ? sr.format(p.y) : yFormat(p.y)));
      }
    }
    // слой наведения: перекрестие привязано к ближайшему порогу
    const cross = s("line", { class: "cross", x1: 0, x2: 0, y1: m.t, y2: Y(0), visibility: "hidden" });
    const dots = series.map((sr) => s("circle", { class: "dot dot--hover " + sr.cls, r: 4, visibility: "hidden" }));
    const hit = s("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: Y(0) - m.t, fill: "transparent", class: "hit" });
    svg.append(cross, ...dots, hit);
    const wrap = h("div", { class: "plot__inner", tabindex: "0", role: "group", "aria-label": (ariaLabel || "") + ". Стрелки влево и вправо показывают значения." }, svg);
    const tip = tipBox();
    wrap.append(tip);
    let idx = -1;
    const show = (i) => {
      idx = Math.max(0, Math.min(xs.length - 1, i));
      const x = xs[idx];
      const px = X(x);
      cross.setAttribute("x1", px);
      cross.setAttribute("x2", px);
      cross.setAttribute("visibility", "visible");
      const rows = series.map((sr, k) => {
        const p = sr.values.find((v) => v.x === x);
        if (p) {
          dots[k].setAttribute("cx", px);
          dots[k].setAttribute("cy", Y(p.y));
          dots[k].setAttribute("visibility", "visible");
        }
        return h("div", { class: "tip__row" }, h("span", { class: "tip__key " + sr.cls }), h("strong", null, p ? (sr.format || yFormat)(p.y) : ""), h("span", null, sr.label));
      });
      put(tip, h("div", { class: "tip__head" }, "Порог " + xFormat(x)), ...rows, ...(tipRows ? tipRows(x) : []));
      placeTip(tip, wrap, px, (m.t + Y(0)) / 2);
    };
    const hide = () => {
      cross.setAttribute("visibility", "hidden");
      dots.forEach((d) => d.setAttribute("visibility", "hidden"));
      tip.hidden = true;
    };
    hit.addEventListener("pointermove", (e) => {
      const r = svg.getBoundingClientRect();
      const x = (e.clientX - r.left - m.l) / (W - m.l - m.r);
      let best = 0;
      xs.forEach((v, i) => Math.abs(v - x) < Math.abs(xs[best] - x) && (best = i));
      show(best);
    });
    hit.addEventListener("pointerleave", hide);
    wrap.addEventListener("focus", () => show(idx < 0 ? xs.findIndex((v) => marker && v >= marker.x) : idx));
    wrap.addEventListener("blur", hide);
    wrap.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        e.preventDefault();
        show(idx + (e.key === "ArrowRight" ? 1 : -1));
      } else if (e.key === "Escape") hide();
    });
    return wrap;
  }, height);
}

function nearest(values, x) {
  let best = null;
  for (const p of values) if (!best || Math.abs(p.x - x) < Math.abs(best.x - x)) best = p;
  return best;
}

/** Диаграмма надёжности: столбец = бин уверенности, высота = фактическая точность, диагональ = идеальная калибровка. */
export function reliabilityChart({ bins, threshold, height = 260, ariaLabel }) {
  return responsive((W) => {
    const m = { l: 44, r: 16, t: 26, b: 34 };
    const X = lin(0, 1, m.l, W - m.r);
    const Y = lin(0, 1, height - m.b, m.t);
    const svg = s("svg", { width: W, height, viewBox: `0 0 ${W} ${height}`, role: "img", "aria-label": ariaLabel, class: "svgchart" });
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      svg.append(s("line", { class: "grid", x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }), s("text", { class: "tick", x: m.l - 8, y: Y(t) + 4, "text-anchor": "end" }, fmt.num(t, 2)));
    }
    for (const t of [0, 0.2, 0.4, 0.6, 0.8, 1]) svg.append(s("text", { class: "tick", x: X(t), y: height - m.b + 18, "text-anchor": "middle" }, fmt.conf(t)));
    svg.append(s("text", { class: "tick tick--label", x: W - m.r, y: height - 4, "text-anchor": "end" }, "уверенность модели"));
    const wrap = h("div", { class: "plot__inner" });
    const tip = tipBox();
    for (const b of bins) {
      if (!b.n) continue;
      const x0 = X(b.lo) + 1;
      const x1 = X(b.hi) - 1;
      const w = Math.max(2, Math.min(x1 - x0, 48));
      const cx = (x0 + x1) / 2;
      const y = Y(b.acc);
      const r = Math.min(4, w / 2, Y(0) - y);
      const left = cx - w / 2;
      const d = `M${left},${Y(0)} V${y + r} Q${left},${y} ${left + r},${y} H${left + w - r} Q${left + w},${y} ${left + w},${y + r} V${Y(0)} Z`;
      const bar = s("path", { class: "bar bar--1", d, tabindex: "0", "aria-label": `Уверенность от ${fmt.conf(b.lo)} до ${fmt.conf(b.hi)}: точность ${fmt.p3(b.acc)}, значений ${b.n}` });
      const showTip = () => {
        bar.classList.add("is-hover");
        put(tip, h("div", { class: "tip__head" }, `Уверенность от ${fmt.conf(b.lo)} до ${fmt.conf(b.hi)}`),
          h("div", { class: "tip__row" }, h("strong", null, fmt.p3(b.acc)), h("span", null, "фактическая точность")),
          b.conf != null ? h("div", { class: "tip__row" }, h("strong", null, fmt.conf(b.conf)), h("span", null, "средняя уверенность")) : null,
          h("div", { class: "tip__row" }, h("strong", null, fmt.int(b.n)), h("span", null, fmt.plural(b.n, "значение", "значения", "значений"))));
        placeTip(tip, wrap, cx, y);
      };
      const hideTip = () => {
        bar.classList.remove("is-hover");
        tip.hidden = true;
      };
      bar.addEventListener("pointerenter", showTip);
      bar.addEventListener("pointerleave", hideTip);
      bar.addEventListener("focus", showTip);
      bar.addEventListener("blur", hideTip);
      svg.append(bar);
    }
    svg.append(s("line", { class: "diag", x1: X(0), y1: Y(0), x2: X(1), y2: Y(1) }),
      s("text", { class: "ref__label", x: X(0.04), y: Y(0.04) - 22, transform: `rotate(${-Math.atan2(Y(0) - Y(1), X(1) - X(0)) * 180 / Math.PI} ${X(0.04)} ${Y(0.04) - 22})` }, "идеальная калибровка"));
    if (threshold != null) {
      svg.append(s("line", { class: "marker", x1: X(threshold), x2: X(threshold), y1: m.t - 6, y2: Y(0) }),
        s("text", { class: "marker__label", x: X(threshold), y: m.t - 12, "text-anchor": "middle" }, "порог " + fmt.conf(threshold)));
    }
    svg.append(s("line", { class: "axis", x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0) }));
    wrap.append(svg, tip);
    return wrap;
  }, height);
}

/** Горизонтальные полосы (HTML): одна серия, выделение — красным, остальное — графитом. */
export function hbars(items, { max = 1, format = fmt.p3, emphasize = () => false, aside } = {}) {
  return h("ul", { class: "hbars" }, items.map((it) => h("li", { class: "hbar" + (emphasize(it) ? " is-weak" : ""), tabindex: "0", "data-tip": it.tip || null, "data-tip-title": it.tipTitle || null },
    h("span", { class: "hbar__label" }, it.label, it.sub ? h("small", null, it.sub) : null),
    h("span", { class: "hbar__track", "aria-hidden": "true" }, h("span", { class: "hbar__fill", style: { "--v": String(Math.max(0, Math.min(1, it.value / max))) } })),
    h("span", { class: "hbar__num" }, format(it.value)),
    aside ? h("span", { class: "hbar__aside" }, aside(it)) : null)));
}

/** Точечная диаграмма по сегментам (HTML): две метки на одной шкале 0..1 и опорная линия общего значения. */
export function dotPlot(rows, { series, ref, domain = [0, 1], refLabel }) {
  const pctOf = (v) => ((v - domain[0]) / (domain[1] - domain[0])) * 100;
  const range = domain[1] - domain[0];
  const step = range > 0.5 ? 0.25 : range > 0.2 ? 0.1 : 0.05;
  const ticks = [];
  for (let v = Math.ceil(domain[0] / step - 1e-9) * step; v <= domain[1] + 1e-9; v += step) ticks.push(Math.round(v * 100) / 100);
  return h("div", { class: "dots" },
    h("div", { class: "dots__axis", "aria-hidden": "true" }, h("span"), h("span", { class: "dots__ticks" }, ticks.map((v) => h("span", { style: { left: pctOf(v) + "%" } }, fmt.conf(v))))),
    rows.map((r) => h("div", { class: "dots__row", tabindex: "0", "data-tip": r.tip || null, "data-tip-title": r.label },
      h("span", { class: "dots__label" }, r.label, r.sub ? h("small", null, r.sub) : null),
      h("span", { class: "dots__track" },
        ref != null ? h("span", { class: "dots__ref", style: { left: pctOf(ref) + "%" }, "aria-hidden": "true" }) : null,
        series.map((sr) => (r[sr.key] == null ? null : h("span", { class: "dots__dot " + sr.cls, style: { left: pctOf(r[sr.key]) + "%" }, "aria-hidden": "true" }))),
        h("span", { class: "sr-only" }, series.map((sr) => sr.label + " " + fmt.p3(r[sr.key])).join(", "))))));
}

/** Тепловая карта как таблица: цвет = доля ошибок (1 − F1), число = F1. Таблица сама себе текстовый двойник. */
export function heatmap({ rows, cols, cell, rowLabel, colLabel }) {
  const step = (err) => (err <= 0.02 ? 0 : err <= 0.05 ? 1 : err <= 0.1 ? 2 : err <= 0.2 ? 3 : 4);
  return h("div", { class: "heat-wrap", tabindex: "0", role: "region", "aria-label": "Тепловая карта: категории и характеристики" }, h("table", { class: "heat" },
    h("thead", null, h("tr", null, h("th", { scope: "col", class: "heat__corner" }, rowLabel), cols.map((c) => h("th", { scope: "col", class: "heat__col" }, c.label)))),
    h("tbody", null, rows.map((r) => h("tr", null, h("th", { scope: "row", class: "heat__row" }, r.label, r.sub ? h("small", null, r.sub) : null),
      cols.map((c) => {
        const v = cell(r, c);
        if (!v) return h("td", { class: "heat__cell is-none" }, h("span", { class: "sr-only" }, "нет данных"));
        return h("td", { class: "heat__cell heat--" + step(1 - v.f1), "data-tip": `Точность ${fmt.p3(v.precision)}, полнота ${fmt.p3(v.recall)}, ошибок ${(v.fp || 0) + (v.fn || 0)}`, "data-tip-title": `${r.label}: ${c.label}`, tabindex: "0" },
          fmt.conf(v.f1));
      })))),
    colLabel ? h("caption", { class: "heat__caption" }, colLabel) : null));
}
