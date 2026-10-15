// Строка характеристики: имя ··· значение, уверенность со шкалой и засечкой порога, статус, источники.
// Раскрывается в карточку решения: цитата в контексте, нормализация, альтернативы, «Принять / Исправить / Отклонить».
import { fmt, h, icon, shownValue, uid, put, valueText } from "./dom.js";
import { CONFIDENCE_TIP, DECISION, SOURCE, STATUS, t } from "./copy.js";
import { snippet } from "./evidence.js";
import { kbd } from "./ui.js";

export function meter(conf, threshold, status) {
  const v = Math.max(0, Math.min(1, Number(conf) || 0));
  return h("span", { class: "meter meter--" + status, "data-tip": t(CONFIDENCE_TIP, threshold), "data-tip-title": "Уверенность " + fmt.conf(v) },
    h("span", { class: "meter__track", "aria-hidden": "true" },
      h("span", { class: "meter__fill", style: { "--v": String(v) } }),
      threshold != null ? h("span", { class: "meter__notch", style: { "--t": String(threshold) } }) : null),
    h("span", { class: "meter__num" }, h("span", { class: "sr-only" }, "уверенность "), status === "missing" ? "0" : fmt.conf(v)));
}

export function statusMark(status, threshold, { withTip = true } = {}) {
  const st = STATUS[status] || STATUS.missing;
  return h("span", { class: "st st--" + status, "data-tip": withTip ? t(st.tip, threshold) : null },
    icon(st.icon, 16, "st__icon"), h("span", { class: "st__label" }, st.label));
}

export function sourceTags(sources = []) {
  return h("span", { class: "tags" }, sources.map((src) => {
    const s = SOURCE[src] || { label: src, tip: "" };
    return h("span", { class: "tag", dataset: { src }, "data-tip": s.tip, "data-tip-title": "Источник: " + s.label }, s.label);
  }));
}

const altValue = (alt, unit) => valueText(alt?.value, unit);

function whyText(a, threshold) {
  if (a.status === "missing") return "Ни один источник не нашёл значение в тексте карточки";
  const conf = fmt.conf(a.confidence);
  const thr = fmt.conf(threshold);
  if (a.status === "accepted") {
    return a.sources.length > 1 ? `Уверенность ${conf} не ниже порога ${thr}, источники согласны` : `Уверенность ${conf} не ниже порога ${thr}`;
  }
  const parts = [];
  if (a.alternatives?.length) parts.push("источники предлагают разные значения");
  if (a.confidence < threshold) parts.push(`уверенность ${conf} ниже порога ${thr}`);
  const text = parts.join(", ") || "есть конфликт источников";
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/**
 * Список характеристик. ctx:
 *  threshold, full, located: Map, decisions: Map,
 *  focusEvidence(attr, {scroll}), blurEvidence(), showInText(attr),
 *  decide(a, action, corrected) -> Promise, undo(a) -> Promise, suggestions(a) -> Promise<string[]>
 */
export function attributeList(attrs, ctx, { compact = false } = {}) {
  const list = h("ul", { class: "attrs" + (compact ? " attrs--compact" : ""), role: "list" });
  const rows = attrs.map((a) => attributeRow(a, ctx, list));
  list.append(...rows.map((r) => r.el));
  list.addEventListener("pointerleave", () => ctx.blurEvidence?.());
  list._rows = rows;
  return list;
}

export function attributeRow(a, ctx, list) {
  const id = uid("attr");
  const value = shownValue(a);
  const li = h("li", { class: "arow", dataset: { attr: a.attribute, status: a.status } });
  const main = h("button", { class: "arow__main", type: "button", "aria-expanded": "false", "aria-controls": id + "-d", id },
    h("span", { class: "arow__name" },
      h("span", { class: "arow__title" }, a.title || a.attribute),
      a.required === false ? h("span", { class: "arow__opt" }, "доп.") : null,
      h("span", { class: "arow__leader", "aria-hidden": "true" })),
    h("span", { class: "arow__val" + (value == null ? " is-empty" : "") }, value ?? "не найдено"),
    h("span", { class: "arow__conf" }, meter(a.confidence, ctx.threshold, a.status)),
    h("span", { class: "arow__status" }, statusMark(a.status, ctx.threshold)),
    h("span", { class: "arow__src" }, sourceTags(a.sources)));
  const detail = h("div", { class: "arow__detail", id: id + "-d", hidden: true, role: "region", "aria-labelledby": id });
  li.append(main, detail);

  let built = false;
  const row = {
    el: li,
    a,
    open(state = true) {
      if (state && list) list._rows?.forEach((r) => r !== row && r.open(false));
      main.setAttribute("aria-expanded", String(state));
      li.classList.toggle("is-open", state);
      detail.hidden = !state;
      if (state && !built) {
        buildDetail(row, detail, ctx);
        built = true;
      }
      if (state) ctx.focusEvidence?.(a.attribute, { scroll: true });
    },
    isOpen: () => li.classList.contains("is-open"),
    focus: () => main.focus(),
    rebuild() {
      built = false;
      if (row.isOpen()) {
        buildDetail(row, detail, ctx);
        built = true;
      }
      paintDecision(li, ctx.decisions?.get(a.attribute));
    },
    act(action) {
      if (!row.isOpen()) row.open(true);
      detail.querySelector(`[data-action="${action}"]`)?.click();
    },
  };
  main.addEventListener("click", () => row.open(!row.isOpen()));
  li.addEventListener("pointerenter", () => ctx.focusEvidence?.(a.attribute, { scroll: true }));
  main.addEventListener("focus", () => ctx.focusEvidence?.(a.attribute, { scroll: true }));
  paintDecision(li, ctx.decisions?.get(a.attribute));
  return row;
}

function paintDecision(li, d) {
  li.classList.toggle("is-decided", Boolean(d));
  li.querySelector(".arow__decided")?.remove();
  if (!d) return;
  const meta = DECISION[d.action];
  li.querySelector(".arow__status")?.append(h("span", { class: "arow__decided", "data-tip": meta.label }, icon(meta.icon, 14), h("span", { class: "sr-only" }, meta.label)));
}

function buildDetail(row, detail, ctx) {
  const a = row.a;
  const loc = ctx.located?.get(a.attribute);
  const value = shownValue(a);
  const facts = [];

  const quote = h("div", { class: "detail__quote" },
    h("span", { class: "detail__label" }, "Цитата"),
    loc && ctx.full
      ? h("blockquote", { class: "detail__snippet" }, snippet(ctx.full, loc))
      : h("p", { class: "detail__none" }, a.evidence ? `«${a.evidence}» в тексте карточки дословно не найдена` : "Цитаты нет: значение не найдено в тексте"),
    loc && !loc.exact ? h("p", { class: "detail__approx" }, "Место в тексте найдено поиском по цитате: пайплайн не передал точные смещения.") : null,
    loc && ctx.showInText ? h("button", { class: "btn btn--ghost btn--sm detail__jump", type: "button", onclick: () => ctx.showInText(a.attribute) }, icon("text-align-left", 14), "Показать в тексте") : null);

  if (a.evidence && value != null && String(a.evidence).trim().toLowerCase() !== String(value).trim().toLowerCase()) {
    facts.push(h("div", { class: "fact" }, h("span", { class: "detail__label" }, "Нормализация"),
      h("span", { class: "fact__val" }, h("span", { class: "fact__raw" }, "«" + a.evidence + "»"), icon("arrow-right", 14, "fact__arrow"), h("strong", null, value))));
  }
  facts.push(h("div", { class: "fact" }, h("span", { class: "detail__label" }, "Статус"), h("span", { class: "fact__val" }, whyText(a, ctx.threshold))));
  if (a.traffic_weight != null) {
    facts.push(h("div", { class: "fact" }, h("span", { class: "detail__label" }, "Спрос в поиске"),
      h("span", { class: "fact__val", "data-tip": "Вес характеристики по поисковым запросам покупателей. Чем выше, тем важнее заполнить её точно: по ней фильтруют." },
        "вес " + fmt.num(a.traffic_weight, 2), a.required === false ? ", необязательная" : ", обязательная")));
  }

  const alts = a.alternatives?.length
    ? h("div", { class: "detail__alts" }, h("span", { class: "detail__label" }, "Другие варианты"),
      h("div", { class: "alts" }, a.alternatives.map((alt) => h("button", {
        class: "alt", type: "button", "data-tip": "Выбрать этот вариант как верный",
        onclick: () => submit("corrected", alt.value),
      }, h("span", { class: "alt__val" }, altValue(alt, a.unit)),
      h("span", { class: "alt__meta" }, (alt.sources || []).map((s) => SOURCE[s]?.label || s).join(", "), alt.confidence != null ? " " + fmt.conf(alt.confidence) : "")))))
    : null;

  // решение модератора
  const input = h("input", { class: "input fix__input", type: "text", "aria-label": "Верное значение для «" + (a.title || a.attribute) + "»", value: value ?? "", autocomplete: "off" });
  const listId = uid("dl");
  const datalist = h("datalist", { id: listId });
  input.setAttribute("list", listId);
  const fix = h("form", { class: "fix", hidden: true, onsubmit: (e) => { e.preventDefault(); const v = input.value.trim(); if (v) submit("corrected", v); } },
    input, datalist,
    h("button", { class: "btn btn--primary btn--sm", type: "submit" }, "Сохранить"),
    h("button", { class: "btn btn--quiet btn--sm", type: "button", onclick: () => toggleFix(false) }, "Отмена"));
  input.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      e.stopPropagation();
      toggleFix(false);
    }
  });

  const missing = a.status === "missing";
  const actions = h("div", { class: "decision" },
    missing ? null : h("button", { class: "btn btn--" + (a.status === "review" ? "primary" : "quiet") + " btn--sm", type: "button", dataset: { action: "accept" }, onclick: () => submit("accepted", a.value) },
      icon("check", 14), "Принять", kbd("A")),
    h("button", { class: "btn btn--quiet btn--sm", type: "button", dataset: { action: "edit" }, onclick: () => toggleFix(true) },
      icon("pencil-simple", 14), missing ? "Заполнить" : "Исправить", kbd("E")),
    missing ? null : h("button", { class: "btn btn--quiet btn--sm", type: "button", dataset: { action: "reject" }, onclick: () => submit("rejected", null) },
      icon("prohibit", 14), "Отклонить", kbd("R")));

  const decided = h("p", { class: "decided", hidden: true });

  function toggleFix(state) {
    fix.hidden = !state;
    actions.hidden = state;
    if (state) {
      input.focus();
      input.select();
      ctx.suggestions?.(a).then((vals) => put(datalist, ...(vals || []).slice(0, 300).map((v) => h("option", { value: v }))));
    } else {
      actions.querySelector('[data-action="edit"]')?.focus();
    }
  }

  function paint() {
    const d = ctx.decisions?.get(a.attribute);
    decided.hidden = !d;
    actions.hidden = Boolean(d);
    fix.hidden = true;
    if (!d) return;
    const meta = DECISION[d.action];
    put(decided, icon(meta.icon, 16), h("span", null, meta.label,
      d.action === "corrected" ? ": «" + d.corrected + "»" : d.action === "accepted" && value ? ": «" + value + "»" : ""),
    h("span", { class: "decided__where" }, "записано в feedback.jsonl"),
    h("button", { class: "btn btn--ghost btn--sm", type: "button", onclick: undo }, icon("arrow-u-up-left", 14), "Отменить"));
  }

  async function submit(action, corrected) {
    const prev = ctx.decisions?.get(a.attribute);
    ctx.decisions?.set(a.attribute, { action, corrected });
    paint();
    paintDecision(row.el, ctx.decisions?.get(a.attribute));
    decided.querySelector("button")?.focus();
    try {
      await ctx.decide(a, action, corrected);
    } catch (e) {
      if (prev) ctx.decisions.set(a.attribute, prev);
      else ctx.decisions?.delete(a.attribute);
      paint();
      paintDecision(row.el, ctx.decisions?.get(a.attribute));
      ctx.onError?.(e, () => submit(action, corrected));
    }
  }

  async function undo() {
    const prev = ctx.decisions?.get(a.attribute);
    ctx.decisions?.delete(a.attribute);
    paint();
    paintDecision(row.el, null);
    actions.querySelector("button")?.focus();
    try {
      await ctx.undo(a);
    } catch (e) {
      if (prev) ctx.decisions.set(a.attribute, prev);
      paint();
      paintDecision(row.el, prev);
      ctx.onError?.(e);
    }
  }

  put(detail, h("div", { class: "detail" },
    quote,
    h("div", { class: "detail__facts" }, facts),
    alts,
    h("div", { class: "detail__act" }, actions, fix, decided)));
  paint();
}
