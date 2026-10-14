// Реестр компонентов generative UI. LLM выбирает тип и параметры, данные берём только из result.
// Весь текст выводится через textContent — никакого innerHTML с данными.
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v);
  }
  for (const c of children.flat()) if (c != null) el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
}

const STATUS = { accepted: "принято", review: "проверить", missing: "нет в тексте" };
const SOURCE = { seller: "продавец", rules: "правила", ml: "ML", llm: "LLM", human: "модератор" };
export const pct = (x) => Math.round(x * 100) + "%";
const card = (title, ...body) => h("div", { class: "c" }, title ? h("h3", {}, title) : null, ...body);
const pick = (r, c) => {
  let a = r.attributes;
  if (c.filter && c.filter !== "all") a = a.filter((x) => x.status === c.filter);
  if (c.attributes) a = a.filter((x) => c.attributes.includes(x.attribute));
  return a;
};
const fillClass = (v) => "bar__fill " + (v >= 0.7 ? "ok" : v >= 0.4 ? "mid" : "");

export const registry = {
  alert: (c) => h("div", { class: "alert alert--" + (c.level || "info") }, c.text || ""),

  summary: (c, r) => {
    const n = (s) => r.attributes.filter((a) => a.status === s).length;
    return card(null, h("div", { class: "summary" },
      h("span", { class: "summary__cat" }, r.category),
      h("span", { class: "muted" }, "категория определена с уверенностью " + pct(r.category_confidence)),
      r.llm_used ? h("span", { class: "chip chip--review" }, "LLM") : null,
      h("div", { class: "summary__kpi" },
        h("div", { class: "kpi" }, h("b", {}, n("accepted")), h("span", {}, "принято")),
        h("div", { class: "kpi" }, h("b", {}, n("review")), h("span", {}, "на проверку")),
        h("div", { class: "kpi" }, h("b", {}, n("missing")), h("span", {}, "нет в тексте")))),
      c.text ? h("p", { class: "muted" }, c.text) : null);
  },

  attribute_table: (c, r) => card(c.title || "Характеристики",
    h("table", { class: "specs" }, pick(r, c).map((a) => h("tr", {},
      h("td", { class: "k" }, h("span", {}, a.title)),
      h("td", { class: "v" }, a.display ?? "—",
        h("span", { class: "chip chip--" + a.status }, STATUS[a.status] + " · " + pct(a.confidence)),
        h("span", { class: "src" }, a.sources.map((s) => SOURCE[s] || s).join(", ")),
        a.evidence ? h("span", { class: "ev" }, "«" + a.evidence + "»") : null))))),

  confidence_chart: (c, r) => card(c.title || "Уверенность",
    h("div", { class: "bars" }, pick(r, c).map((a) => h("div", { class: "bar" },
      h("span", {}, a.title),
      h("div", { class: "bar__track" }, h("div", { class: fillClass(a.confidence), style: "width:" + pct(a.confidence) })),
      h("b", {}, pct(a.confidence)))))),

  evidence: (c, r) => card(c.title || "Откуда взяли значения",
    h("ul", { class: "evidence" }, pick(r, c).filter((a) => a.evidence).map((a) =>
      h("li", {}, h("b", {}, a.title + ": "), a.display, " ← ", h("q", {}, a.evidence))))),

  review_queue: (c, r, ctx) => card(c.title || "На проверку",
    pick(r, { ...c, filter: c.filter || "review" }).map((a) => reviewRow(r.card_id, a, ctx))),

  metrics: (c, r, ctx) => card(c.title || "Качество на gold-разметке",
    h("p", { class: "muted" }, c.text || "Сводные метрики — во вкладке «Метрики»."),
    h("button", { class: "btn btn--ghost", onclick: () => document.querySelector('[data-tab="metrics"]').click() }, "Открыть метрики")),
};

export function reviewRow(cardId, a, ctx) {
  const input = h("input", { value: a.value ?? "", placeholder: "верное значение" });
  const btn = h("button", { class: "btn btn--ghost", onclick: async () => {
    await ctx.feedback({ card_id: cardId, attribute: a.attribute, predicted: a.value, corrected: input.value, raw: a.evidence });
    btn.textContent = "Сохранено"; btn.disabled = true;
  } }, "Подтвердить");
  return h("div", { class: "rq" },
    h("div", { class: "rq__name" }, h("b", {}, a.title), h("span", { class: "chip chip--" + a.status }, pct(a.confidence)),
      a.evidence ? h("div", { class: "muted" }, "«" + a.evidence + "»") : null),
    input, btn);
}

export function render(root, spec, result, ctx) {
  root.replaceChildren();
  root.className = "ui-root" + (spec.layout === "grid" ? " grid" : "");
  root.append(h("div", { class: "ui-title" }, h("h2", {}, spec.title),
    h("span", {}, spec.generated_by === "llm" ? "интерфейс собран LLM" : "шаблонный интерфейс")));
  for (const c of spec.components) {
    const fn = registry[c.type];
    if (fn) root.append(fn(c, result, ctx));
  }
}
