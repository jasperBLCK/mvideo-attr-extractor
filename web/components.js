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
const SOURCE_ORDER = ["seller", "rules", "ml", "llm", "human"];
export const pct = (x) => Math.round(x * 100) + "%";
const card = (title, ...body) => h("div", { class: "c" }, title ? h("h3", {}, title) : null, ...body);
const LOW = 0.7;

// Отбор атрибутов по параметрам компонента: filter → attributes → sort → limit
const pick = (r, c) => {
  let a = [...r.attributes];
  if (c.filter === "low_confidence") a = a.filter((x) => x.confidence < LOW || x.status !== "accepted");
  else if (c.filter && c.filter !== "all") a = a.filter((x) => x.status === c.filter);
  if (c.attributes) a = a.filter((x) => c.attributes.includes(x.attribute));
  if (c.sort === "confidence_asc") a.sort((x, y) => x.confidence - y.confidence);
  if (c.sort === "confidence_desc") a.sort((x, y) => y.confidence - x.confidence);
  if (c.sort === "name") a.sort((x, y) => x.title.localeCompare(y.title, "ru"));
  if (c.limit) a = a.slice(0, c.limit);
  return a;
};
const level = (v) => (v >= LOW ? "ok" : v >= 0.4 ? "mid" : "low");
const chip = (a) => h("span", { class: "chip chip--" + a.status }, STATUS[a.status] + " · " + pct(a.confidence));
const emptyNote = (text) => h("p", { class: "muted empty-note" }, text);
const bar = (a) => h("div", { class: "bar" },
  h("span", {}, a.title),
  h("div", { class: "bar__track" }, h("div", { class: "bar__fill " + level(a.confidence), style: "width:0", "data-w": pct(a.confidence) })),
  h("b", {}, pct(a.confidence)));

// Разбор предупреждений пайплайна «attr: конфликт значений, альтернативы: a, b»
const conflictsOf = (r) => {
  const out = [];
  for (const w of r.warnings) {
    const m = /^(\w+): конфликт значений, альтернативы: (.+)$/.exec(w);
    if (m) out.push({ attribute: m[1], alternatives: m[2].split(", ") });
  }
  return out;
};

export const registry = {
  alert: (c) => h("div", { class: "alert alert--" + (c.level || "info") }, c.title ? h("b", {}, c.title + " ") : null, c.text || ""),

  summary: (c, r) => {
    const n = (s) => r.attributes.filter((a) => a.status === s).length;
    return card(null, h("div", { class: "summary" },
      h("span", { class: "plate" }, r.category),
      h("span", { class: "muted" }, "категория определена с уверенностью " + pct(r.category_confidence)),
      r.llm_used ? h("span", { class: "chip chip--review" }, "LLM") : null,
      h("div", { class: "summary__kpi" },
        h("div", { class: "kpi" }, h("b", {}, n("accepted")), h("span", {}, "принято")),
        h("div", { class: "kpi" }, h("b", {}, n("review")), h("span", {}, "на проверку")),
        h("div", { class: "kpi" }, h("b", {}, n("missing")), h("span", {}, "нет в тексте")))),
      c.text ? h("p", { class: "muted" }, c.text) : null);
  },

  kpi: (c, r) => {
    const n = (s) => r.attributes.filter((a) => a.status === s).length;
    const mean = r.attributes.length ? r.attributes.reduce((s, a) => s + a.confidence, 0) / r.attributes.length : 0;
    const tile = (v, label, cls, key) => h("div", { class: "tile tile--" + cls + (c.filter === key ? " is-focus" : "") },
      h("b", { class: "tile__num", "data-count": typeof v === "number" ? v : null }, typeof v === "number" ? "0" : v), h("span", {}, label));
    return card(c.title, h("div", { class: "tiles" },
      tile(n("accepted"), "заполнено автоматически", "ok", "accepted"),
      tile(n("review"), "на проверку", "mid", "review"),
      tile(n("missing"), "нет в тексте", "low", "missing"),
      tile(pct(mean), "средняя уверенность", "plain", "all")),
      c.text ? h("p", { class: "muted" }, c.text) : null);
  },

  attribute_table: (c, r) => {
    const rows = pick(r, c);
    const compact = c.variant === "compact";
    return card(c.title || "Характеристики", rows.length ? h("table", { class: "specs" }, rows.map((a) => h("tr", {},
      h("td", { class: "k" }, h("span", {}, a.title)),
      h("td", { class: "v" }, a.display ?? "—", chip(a),
        compact ? null : h("span", { class: "src" }, a.sources.map((s) => SOURCE[s] || s).join(", ")),
        !compact && a.evidence ? h("span", { class: "ev" }, "«" + a.evidence + "»") : null)))) : emptyNote("Под этот фильтр ничего не попало"),
      c.text ? h("p", { class: "muted" }, c.text) : null);
  },

  attribute_cards: (c, r) => {
    const rows = pick(r, c);
    return card(c.title, rows.length ? h("div", { class: "acards" }, rows.map((a) => h("div", { class: "acard acard--" + a.status },
      h("span", { class: "acard__k" }, a.title),
      h("b", { class: "acard__v" }, a.display ?? "—"),
      h("div", { class: "acard__meta" }, chip(a), h("span", { class: "src" }, a.sources.map((s) => SOURCE[s] || s).join(", "))),
      a.evidence && c.variant !== "compact" ? h("q", { class: "acard__ev" }, a.evidence) : null))) : emptyNote("Под этот фильтр ничего не попало"),
      c.text ? h("p", { class: "muted" }, c.text) : null);
  },

  confidence_chart: (c, r) => {
    const rows = pick(r, c);
    return card(c.title || "Уверенность", rows.length ? h("div", { class: "bars" }, rows.map(bar)) : emptyNote("Нет данных"),
      c.text ? h("p", { class: "muted" }, c.text) : null);
  },

  evidence: (c, r) => {
    const rows = pick(r, c).filter((a) => a.evidence);
    return card(c.title || "Откуда взяли значения", rows.length ? h("ul", { class: "evidence" }, rows.map((a) =>
      h("li", {}, h("b", {}, a.title + ": "), a.display ?? "—", h("span", { class: "arrow" }, "←"), h("q", {}, a.evidence), chip(a))))
      : emptyNote("Для выбранных характеристик цитат нет — значения не из текста"),
      c.text ? h("p", { class: "muted" }, c.text) : null);
  },

  // Текст карточки с подсветкой цитат: строим через split/append, без innerHTML
  highlight: (c, r, ctx) => {
    const text = [ctx.card?.title, ctx.card?.description].filter(Boolean).join("\n\n");
    const rows = pick(r, c).filter((a) => a.evidence);
    if (!text) return card(c.title || "Текст карточки", emptyNote("Текст карточки недоступен"));
    const marks = [];
    const low = text.toLowerCase();
    for (const a of rows) {
      const i = low.indexOf(String(a.evidence).toLowerCase());
      if (i >= 0) marks.push({ s: i, e: i + a.evidence.length, a });
    }
    marks.sort((x, y) => x.s - y.s);
    const out = h("p", { class: "hl" });
    let pos = 0;
    for (const m of marks) {
      if (m.s < pos) continue;
      out.append(text.slice(pos, m.s), h("mark", { class: "hl__m hl__m--" + level(m.a.confidence), title: m.a.title + " · " + pct(m.a.confidence) }, text.slice(m.s, m.e)));
      pos = m.e;
    }
    out.append(text.slice(pos));
    return card(c.title || "Текст карточки", out, h("p", { class: "muted hl__legend" }, marks.length + " цитат подсвечено: зелёный — уверенно, жёлтый — спорно, красный — слабо"));
  },

  conflicts: (c, r) => {
    const cf = conflictsOf(r);
    const multi = r.attributes.filter((a) => a.sources.length > 1 && !cf.some((x) => x.attribute === a.attribute));
    const byKey = Object.fromEntries(r.attributes.map((a) => [a.attribute, a]));
    const rows = cf.map(({ attribute, alternatives }) => {
      const a = byKey[attribute] || { title: attribute, display: "—", confidence: 0, status: "review", sources: [] };
      return h("div", { class: "cf" },
        h("div", { class: "cf__name" }, h("b", {}, a.title), chip(a)),
        h("div", { class: "cf__vals" },
          h("span", { class: "cf__v cf__v--win" }, a.display ?? "—", h("small", {}, "выбрано · " + a.sources.map((s) => SOURCE[s] || s).join(", "))),
          alternatives.map((alt) => h("span", { class: "cf__v" }, alt, h("small", {}, "альтернатива")))));
    });
    return card(c.title || "Расхождения источников",
      rows.length ? rows : h("div", { class: "alert alert--success" }, "Конфликтов между продавцом и текстом не найдено"),
      multi.length ? h("p", { class: "muted" }, "Подтверждено двумя источниками: " + multi.map((a) => a.title).join(", ")) : null,
      c.text ? h("p", { class: "muted" }, c.text) : null);
  },

  source_breakdown: (c, r) => {
    const counts = {};
    for (const a of pick(r, c)) for (const s of a.sources) counts[s] = (counts[s] || 0) + 1;
    const total = Object.values(counts).reduce((x, y) => x + y, 0) || 1;
    const keys = SOURCE_ORDER.filter((k) => counts[k]);
    return card(c.title || "Источники значений",
      h("div", { class: "stack" }, keys.map((k) => h("div", { class: "stack__seg stack__seg--" + k, style: "flex-grow:" + counts[k], title: SOURCE[k] + ": " + counts[k] }))),
      h("div", { class: "legend" }, keys.map((k) => h("span", {}, h("i", { class: "dot dot--" + k }), SOURCE[k] + " · " + counts[k] + " (" + pct(counts[k] / total) + ")"))),
      c.text ? h("p", { class: "muted" }, c.text) : null);
  },

  review_queue: (c, r, ctx) => {
    const rows = pick(r, { ...c, filter: c.filter || "review" });
    return card(c.title || "На проверку", rows.length ? rows.map((a) => reviewRow(r.card_id, a, ctx)) : h("div", { class: "alert alert--success" }, "Очередь пуста"),
      c.text ? h("p", { class: "muted" }, c.text) : null);
  },

  checklist: (c, r) => {
    const rows = pick(r, { ...c, filter: c.filter || "low_confidence" });
    return card(c.title || "Чеклист модератора", rows.length ? h("ul", { class: "checks" }, rows.map((a) => {
      const box = h("input", { type: "checkbox", id: "chk-" + a.attribute });
      const li = h("li", {}, box, h("label", { for: "chk-" + a.attribute }, h("b", {}, a.title), " — ", a.status === "missing" ? "найти значение вручную" : "сверить «" + (a.display ?? "—") + "»", chip(a)));
      box.addEventListener("change", () => li.classList.toggle("is-done", box.checked));
      return li;
    })) : h("div", { class: "alert alert--success" }, "Всё заполнено — проверять нечего"),
      c.text ? h("p", { class: "muted" }, c.text) : null);
  },

  metrics: (c, r, ctx) => card(c.title || "Качество на gold-разметке",
    h("p", { class: "muted" }, c.text || "Сводные метрики — во вкладке «Метрики»."),
    h("button", { class: "btn btn--ghost", onclick: () => document.querySelector('[data-tab="metrics"]').click() }, "Открыть метрики")),
};

export function reviewRow(cardId, a, ctx) {
  const input = h("input", { value: a.value ?? "", placeholder: "верное значение" });
  const btn = h("button", { class: "btn btn--ghost", onclick: async () => {
    btn.disabled = true;
    await ctx.feedback({ card_id: cardId, attribute: a.attribute, predicted: a.value, corrected: input.value, raw: a.evidence });
    btn.textContent = "Сохранено"; btn.classList.add("is-done");
  } }, "Подтвердить");
  return h("div", { class: "rq" },
    h("div", { class: "rq__name" }, h("b", {}, a.title), h("span", { class: "chip chip--" + a.status }, pct(a.confidence)),
      a.evidence ? h("div", { class: "muted" }, "«" + a.evidence + "»") : null),
    input, btn);
}

export function render(root, spec, result, ctx) {
  root.replaceChildren();
  root.className = "ui-root" + (spec.layout === "grid" ? " grid" : "");
  root.append(h("div", { class: "ui-title", style: "--i:0" },
    h("div", {}, h("h2", {}, spec.title), spec.subtitle ? h("p", { class: "muted" }, spec.subtitle) : null),
    h("span", { class: "ui-title__by" }, spec.generated_by === "llm" ? "интерфейс собран LLM" : "шаблонный интерфейс")));
  spec.components.forEach((c, i) => {
    const fn = registry[c.type];
    if (!fn) return;
    const el = fn(c, result, ctx);
    el.style.setProperty("--i", i + 1);
    if (c.span === 2) el.classList.add("span-2");
    root.append(el);
  });
  // анимации «после появления»: бары растут, счётчики набегают
  requestAnimationFrame(() => {
    root.querySelectorAll(".bar__fill").forEach((f) => (f.style.width = f.dataset.w));
    root.querySelectorAll("[data-count]").forEach((el) => countUp(el, +el.dataset.count));
  });
}

function countUp(el, to, ms = 600) {
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return (el.textContent = to);
  const t0 = performance.now();
  const step = (t) => {
    const k = Math.min(1, (t - t0) / ms), e = 1 - Math.pow(1 - k, 3);
    el.textContent = Math.round(to * e);
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
