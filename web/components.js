// Реестр компонентов generative UI. LLM (или подбор по ключевым словам) выбирает тип и параметры
// из фиксированного набора (src/attrx/genui/schema.py), а данные компоненты берут только из result:
// модель не может показать характеристику, которой нет. Текст выводится через textContent.
import { fmt, h, icon, shownValue, put, valueText } from "./js/dom.js";
import { attributeList, meter, statusMark } from "./js/attrs.js";
import { snippet } from "./js/evidence.js";
import { SOURCE, STATUS } from "./js/copy.js";

export { h } from "./js/dom.js";

let blkN = 0;
function blk(type, title, body, note) {
  const id = "blk-" + ++blkN;
  return h("section", { class: "blk blk--" + type, "aria-labelledby": title ? id : null },
    title ? h("h3", { class: "blk__title", id }, title) : null,
    note ? h("p", { class: "blk__note" }, note) : null,
    body);
}

function pick(r, c, ctx) {
  let list = r.attributes;
  const f = c.filter && c.filter !== "all" ? c.filter : ctx?.filter || null;
  if (f) list = list.filter((a) => a.status === f);
  if (c.attributes?.length) list = list.filter((a) => c.attributes.includes(a.attribute));
  return list;
}

const titleOf = (r, key) => r.attributes.find((a) => a.attribute === key)?.title || key;
const unitOf = (r, key) => r.attributes.find((a) => a.attribute === key)?.unit || null;
const altText = valueText;
const pyValue = (v) => String(v).trim().replace(/^['"]|['"]$/g, "");

/** Предупреждения пайплайна → понятные записи. Формат строк задаёт src/attrx/pipeline.py. */
export function parseWarnings(r) {
  const out = [];
  for (const w of r.warnings || []) {
    let m = /^llm: отброшено незаземл[её]нное значение (\w+)=(.+)$/u.exec(w);
    if (m) {
      out.push({ kind: "rejected", attr: m[1], value: pyValue(m[2]) });
      continue;
    }
    m = /^(\w+): конфликт значений, альтернативы: (.*)$/u.exec(w);
    if (m) {
      out.push({ kind: "conflict", attr: m[1], alts: m[2].split(",").map((x) => x.trim()).filter(Boolean) });
      continue;
    }
    if (/категория определена неуверенно/u.test(w)) out.push({ kind: "category", text: "Категория определена неуверенно: проверьте, к какой категории относится товар" });
    else out.push({ kind: "other", text: w.replace(/^llm:\s*/u, "LLM: ") });
  }
  return out;
}

const ALERT_ICON = { info: "info", success: "check-circle", warning: "warning", danger: "warning-circle" };

export const registry = {
  alert: (c) => h("div", { class: "alert alert--" + (c.level || "info"), role: c.level === "danger" ? "alert" : null },
    icon(ALERT_ICON[c.level] || "info", 16, "alert__icon"), h("p", { class: "alert__text" }, c.text || "")),

  summary(c, r, ctx) {
    const n = (s) => r.attributes.filter((a) => a.status === s).length;
    const req = r.attributes.filter((a) => a.required !== false);
    const filled = req.filter((a) => a.status !== "missing").length;
    const order = ["accepted", "review", "missing"];
    const bar = h("div", { class: "sum__bar", role: "img",
      "aria-label": order.map((s) => STATUS[s].label + " " + n(s)).join(", ") + " из " + r.attributes.length },
    order.map((s) => (n(s) ? h("span", { class: "sum__seg sum__seg--" + s, style: { flexGrow: String(n(s)) } }) : null)));
    const items = order.map((s) => h("button", {
      class: "sum__item sum__item--" + s, type: "button", "aria-pressed": String(ctx?.filter === s),
      "data-tip": ctx?.filter === s ? "Показать все характеристики" : "Показать только «" + STATUS[s].label + "»",
      onclick: () => ctx?.setFilter?.(ctx.filter === s ? null : s),
    }, icon(STATUS[s].icon, 16), h("span", { class: "sum__num" }, fmt.int(n(s))), h("span", { class: "sum__label" }, STATUS[s].label)));
    return blk("summary", c.title, h("div", { class: "sum" }, bar,
      h("div", { class: "sum__row" }, h("div", { class: "sum__items", role: "group", "aria-label": "Фильтр по статусу" }, items),
        h("p", { class: "sum__aside" }, "Обязательных заполнено ", h("strong", null, filled + " из " + req.length)))), c.text);
  },

  attribute_table(c, r, ctx) {
    const list = pick(r, c, ctx);
    const body = list.length
      ? [h("div", { class: "ahead", "aria-hidden": "true" },
        h("span", null, "Характеристика"), h("span", null, "Значение"), h("span", null, "Уверенность"), h("span", null, "Статус"), h("span", null, "Источник")),
      attributeList(list, ctx)]
      : h("p", { class: "blk__empty" }, "Нет характеристик с таким статусом.");
    return blk("table", c.title || "Характеристики", body, c.text || (ctx?.hint && list.length ? ctx.hint : null));
  },

  confidence_chart(c, r, ctx) {
    const list = pick(r, { ...c, filter: c.filter || "all" }, null).filter((a) => a.status !== "missing");
    const thr = ctx?.threshold ?? 0.7;
    const body = list.length
      ? h("div", { class: "cbars", style: { "--t": String(thr) } },
        h("div", { class: "cbars__axis", "aria-hidden": "true" }, h("span", null, "0"), h("span", { class: "cbars__thr" }, "порог " + fmt.conf(thr)), h("span", null, "1")),
        list.map((a) => h("div", { class: "cbar cbar--" + a.status, onpointerenter: () => ctx?.focusEvidence?.(a.attribute, { scroll: true }) },
          h("span", { class: "cbar__label" }, a.title),
          h("span", { class: "cbar__track", "aria-hidden": "true" }, h("span", { class: "cbar__fill", style: { "--v": String(a.confidence) } })),
          h("span", { class: "cbar__num" }, fmt.conf(a.confidence)))))
      : h("p", { class: "blk__empty" }, "Нет найденных значений.");
    return blk("confidence", c.title || "Уверенность относительно порога", body,
      c.text || "Ниже красной линии значение не принимается автоматически и уходит на проверку.");
  },

  evidence(c, r, ctx) {
    const list = pick(r, c, null).filter((a) => a.status !== "missing");
    const items = list.map((a) => {
      const loc = ctx?.located?.get(a.attribute);
      return h("li", { class: "evi", onpointerenter: () => ctx?.focusEvidence?.(a.attribute, { scroll: true }) },
        h("div", { class: "evi__head" }, h("span", { class: "evi__attr" }, a.title), h("span", { class: "evi__val" }, shownValue(a) ?? ""), statusMark(a.status, ctx?.threshold, { withTip: false })),
        loc && ctx?.full ? h("blockquote", { class: "evi__quote" }, snippet(ctx.full, loc, 48))
          : h("p", { class: "evi__none" }, a.evidence ? "«" + a.evidence + "»" : "Цитаты нет"));
    });
    return blk("evidence", c.title || "Откуда взяты значения", items.length ? h("ul", { class: "evlist" }, items) : h("p", { class: "blk__empty" }, "Цитат нет."), c.text);
  },

  review_queue(c, r, ctx) {
    const list = pick(r, { ...c, filter: c.filter || "review" }, null);
    if (!list.length) return blk("review", c.title || "На проверку", h("p", { class: "blk__empty" }, "Проверять нечего: всё принято автоматически."), c.text);
    const ul = attributeList(list, ctx);
    queueMicrotask(() => ul._rows?.[0]?.open(true));
    return blk("review", c.title || "На проверку", ul, c.text);
  },

  conflicts(c, r, ctx) {
    const wanted = c.attributes?.length ? new Set(c.attributes) : null;
    const items = [];
    for (const a of r.attributes) {
      if (!a.alternatives?.length || (wanted && !wanted.has(a.attribute))) continue;
      items.push(h("li", { class: "cf", onpointerenter: () => ctx?.focusEvidence?.(a.attribute, { scroll: true }) },
        icon("scales", 18, "cf__icon"),
        h("div", { class: "cf__body" },
          h("p", { class: "cf__title" }, a.title, ": источники расходятся"),
          h("div", { class: "cf__alts" },
            h("span", { class: "chip chip--chosen", "data-tip": "Выбрано пайплайном" }, icon("check", 12), shownValue(a) ?? "нет", " ", h("small", null, a.sources.map((s) => SOURCE[s]?.label || s).join(", "))),
            a.alternatives.map((alt) => h("span", { class: "chip" }, altText(alt.value, alt.unit || a.unit), " ",
              h("small", null, (alt.sources || []).map((s) => SOURCE[s]?.label || s).join(", "), alt.confidence != null ? " " + fmt.conf(alt.confidence) : "")))))));
    }
    const seen = new Set(r.attributes.filter((a) => a.alternatives?.length).map((a) => a.attribute));
    for (const w of parseWarnings(r)) {
      if (w.attr && wanted && !wanted.has(w.attr)) continue;
      if (w.kind === "conflict" && !seen.has(w.attr)) {
        items.push(h("li", { class: "cf", onpointerenter: () => ctx?.focusEvidence?.(w.attr, { scroll: true }) }, icon("scales", 18, "cf__icon"),
          h("div", { class: "cf__body" }, h("p", { class: "cf__title" }, titleOf(r, w.attr), ": источники расходятся"),
            h("div", { class: "cf__alts" }, h("span", { class: "cf__note" }, "Не выбраны:"), w.alts.map((v) => h("span", { class: "chip" }, altText(v, unitOf(r, w.attr))))))));
      } else if (w.kind === "rejected") {
        items.push(h("li", { class: "cf cf--rejected" }, icon("prohibit", 18, "cf__icon"),
          h("div", { class: "cf__body" },
            h("p", { class: "cf__title" }, "LLM предложила ", h("s", null, "«" + w.value + "»"), " для «" + titleOf(r, w.attr) + "»"),
            h("p", { class: "cf__text" }, "Этого нет в тексте карточки, значение отброшено проверкой цитаты. Так видно, что модель не может дописать характеристику от себя."))));
      } else if (w.kind === "category" || w.kind === "other") {
        items.push(h("li", { class: "cf cf--note" }, icon("warning", 18, "cf__icon"), h("div", { class: "cf__body" }, h("p", { class: "cf__text" }, w.text))));
      }
    }
    return blk("conflicts", c.title || "Спорные и отброшенные значения",
      items.length ? h("ul", { class: "cflist" }, items) : h("p", { class: "blk__empty" }, "Источники не спорят, отброшенных значений нет."), c.text);
  },

  metrics(c, r, ctx) {
    const body = h("div", { class: "mm" }, h("p", { class: "blk__empty" }, "Загрузка метрик…"));
    (ctx?.getReport ? ctx.getReport() : Promise.reject(new Error("нет данных"))).then((rep) => {
      const m = rep?.metrics;
      if (!m?.micro) {
        put(body, h("p", { class: "blk__empty" }, "Метрик ещё нет: запустите make report (attrx report-real) на эталоне."));
        return;
      }
      const cell = (label, v, f = fmt.p3) => h("div", { class: "mm__cell" }, h("span", { class: "mm__num" }, f(v)), h("span", { class: "mm__label" }, label));
      put(body, 
        h("div", { class: "mm__row" }, cell("точность", m.micro.precision), cell("полнота", m.micro.recall), cell("F1", m.micro.f1),
          cell("F1 по спросу", m.traffic_weighted_f1), cell("покрытие", m.coverage, fmt.pct)),
        h("p", { class: "mm__foot" }, "Эталон: " + fmt.count(m.n_cards || 0, "карточка", "карточки", "карточек") + (rep._meta?.note ? ". " + rep._meta.note.split(".")[0] : ""),
          h("a", { class: "link", href: "#/quality" }, "Открыть «Качество»")));
    }).catch(() => put(body, h("p", { class: "blk__empty" }, "Метрики недоступны.")));
    return blk("metrics", c.title || "Качество на эталоне", body, c.text);
  },
};

/** Рендер спецификации экрана. Неизвестные типы пропускаются (схема на сервере их и так не пропустит). */
export function render(root, spec, result, ctx) {
  root.replaceChildren();
  root.classList.toggle("rbody--grid", spec.layout === "grid");
  for (const c of spec.components) {
    const fn = registry[c.type];
    if (fn) root.append(fn(c, result, ctx));
  }
  return root;
}

export { meter, statusMark };
