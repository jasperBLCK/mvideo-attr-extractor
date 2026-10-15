import { h, render, reviewRow, toast } from "./components.js?v=3";

const $ = (s) => document.querySelector(s);
const api = async (path, body) => {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  if (!r.ok) throw new Error(r.status + " " + (await r.text()));
  return r.json();
};
const ctx = { feedback: (fb) => api("/api/feedback", fb), card: null };
let lastResult = null;
let samples = [];

// Волна от точки нажатия на кнопках (как на mvideo.ru): элемент удаляется по окончании анимации
document.addEventListener("pointerdown", (e) => {
  const b = e.target.closest(".btn, .search__btn, .tab, .presets button");
  if (!b || b.disabled) return;
  const r = b.getBoundingClientRect(), d = Math.max(r.width, r.height);
  const s = h("span", { class: "ripple", style: `width:${d}px;height:${d}px;left:${e.clientX - r.left - d / 2}px;top:${e.clientY - r.top - d / 2}px` });
  s.addEventListener("animationend", () => s.remove());
  b.append(s);
});

document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
  document.querySelectorAll(".tab, .tabpane").forEach((x) => x.classList.remove("is-active"));
  t.classList.add("is-active");
  $("#tab-" + t.dataset.tab).classList.add("is-active");
  if (t.dataset.tab === "review") loadReview();
  if (t.dataset.tab === "metrics") loadMetrics();
}));

function readCard() {
  const seller = {};
  for (const line of $("#seller").value.split("\n")) {
    const i = line.indexOf(":");
    if (i > 0) seller[line.slice(0, i).trim()] = line.slice(i + 1).trim();
  }
  return { id: $("#title").dataset.id || "manual", title: $("#title").value, description: $("#description").value,
           seller_attrs: seller, category: $("#category").value || null };
}

// Состояние «собираем экран»: скелет вместо контента, кнопка с индикатором
function busy(on, label) {
  const btn = $(".search__btn");
  btn.disabled = on; btn.classList.toggle("is-busy", on);
  $("#card-form button[type=submit]").disabled = on;
  if (on) $("#ui-root").replaceChildren(h("div", { class: "skeleton" }, h("div", { class: "spinner" }), h("p", {}, label || "Собираем экран…"),
    h("div", { class: "sk sk--w" }), h("div", { class: "sk" }), h("div", { class: "sk sk--tall" })));
}

async function run(label, call) {
  busy(true, label);
  try { show(await call()); }
  catch (e) { $("#ui-root").replaceChildren(h("div", { class: "alert alert--danger" }, "Ошибка: " + e.message)); }
  finally { busy(false); }
}

function show(resp) {
  lastResult = resp.result;
  $("#ui-warning").hidden = !resp.ui_warning;
  $("#ui-warning").textContent = resp.ui_warning || "";
  render($("#ui-root"), resp.ui, resp.result, ctx);
}

$("#card-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const intent = $("#intent").value.trim() || null;
  ctx.card = readCard();
  run("Извлекаем характеристики…", () => api("/api/extract", { card: ctx.card, use_llm: $("#use-llm").checked, intent }));
});

$("#intent-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  document.querySelector('[data-tab="card"]').click();
  if (!lastResult) return $("#card-form").requestSubmit();
  run("LLM собирает экран под запрос…", () => api("/api/ui", { result: lastResult, intent: $("#intent").value }));
});

$("#presets").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-intent]");
  if (!b) return;
  $("#intent").value = b.dataset.intent;
  document.querySelectorAll("#presets button").forEach((x) => x.classList.toggle("is-active", x === b));
  $("#intent-form").requestSubmit();
});

$("#sample").addEventListener("change", () => {
  const s = samples.find((x) => x.id === $("#sample").value);
  if (!s) return;
  $("#title").value = s.title; $("#title").dataset.id = s.id;
  $("#description").value = s.description || "";
  $("#seller").value = Object.entries(s.seller_attrs || {}).map(([k, v]) => k + ": " + v).join("\n");
  $("#category").value = s.category || "";
  $("#card-form").requestSubmit();
});

async function loadReview() {
  const root = $("#review-root");
  const items = await api("/api/review");
  root.replaceChildren(...(items.length ? items.map((it) => h("div", { class: "c", style: "margin-top:12px" },
    h("h3", {}, "Карточка " + it.card_id + " · " + it.category), it.attributes.map((a) => reviewRow(it.card_id, a, ctx))))
    : [h("p", { class: "muted" }, "Пусто. Запустите attrx extract — сюда попадут значения с низкой уверенностью.")]));
}

function sweepChart(rows) {
  const W = 600, H = 220, P = 30, NS = "http://www.w3.org/2000/svg";
  const x = (t) => P + t * (W - 2 * P), y = (v) => H - P - v * (H - 2 * P);
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", "0 0 " + W + " " + H); svg.setAttribute("class", "sweep");
  [["precision", "#e30613"], ["coverage", "#1e1e1e"], ["f1", "#00c3b5"]].forEach(([k, color], i) => {
    const p = document.createElementNS(NS, "path");
    p.setAttribute("d", rows.map((r, j) => (j ? "L" : "M") + x(r.threshold) + "," + y(r[k])).join(" "));
    p.setAttribute("fill", "none"); p.setAttribute("stroke", color); p.setAttribute("stroke-width", "3");
    const t = document.createElementNS(NS, "text");
    t.setAttribute("x", P + 10 + i * 110); t.setAttribute("y", 16); t.setAttribute("fill", color); t.setAttribute("font-size", "14");
    t.textContent = "— " + k;
    svg.append(p, t);
  });
  return svg;
}

async function loadMetrics() {
  const root = $("#metrics-root");
  let m;
  try { m = await api("/api/metrics"); } catch { return root.replaceChildren(h("div", { class: "panel" }, "Метрик нет. Запустите: attrx demo или attrx eval --gold ...")); }
  const kpi = (v, l) => h("div", { class: "c kpi" }, h("b", {}, (v * 100).toFixed(1) + "%"), h("span", {}, l));
  root.replaceChildren(
    h("div", { class: "metrics-grid" }, kpi(m.micro.precision, "Precision (micro)"), kpi(m.micro.recall, "Recall (micro)"),
      kpi(m.micro.f1, "F1 (micro)"), kpi(m.traffic_weighted_f1, "Traffic-weighted F1"), kpi(m.coverage, "Coverage"), kpi(m.review_rate, "Доля на проверку")),
    h("div", { class: "grid2 is-active", style: "grid-template-columns:1fr 1fr" },
      h("div", { class: "c" }, h("h3", {}, "По атрибутам"), h("table", { class: "t" },
        h("tr", {}, h("th", {}, "Атрибут"), h("th", {}, "Precision"), h("th", {}, "Recall"), h("th", {}, "F1")),
        Object.entries(m.per_attribute).map(([k, v]) => h("tr", {}, h("td", {}, k), h("td", {}, v.precision.toFixed(3)), h("td", {}, v.recall.toFixed(3)), h("td", {}, v.f1.toFixed(3)))))),
      h("div", { class: "c" }, h("h3", {}, "Порог уверенности: точность и покрытие"), sweepChart(m.threshold_sweep),
        h("p", { class: "muted" }, "Выбираем CONFIDENCE_THRESHOLD под целевую точность бизнеса; всё ниже порога уходит модератору."))),
    h("div", { class: "c", style: "margin-top:16px" }, h("h3", {}, "По категориям"), Object.entries(m.per_category).map(([cat, attrs]) =>
      h("p", {}, h("b", {}, cat), " — ", Object.entries(attrs).map(([a, v]) => a + ": F1 " + v.f1.toFixed(2)).join(" · ")))));
  let rep = {};
  try { rep = await api("/api/report"); } catch { return; }
  const f3 = (x) => x.toFixed(3);
  const table = (title, cols, rows) => h("div", { class: "c", style: "margin-top:16px" }, h("h3", {}, title),
    h("table", { class: "t" }, h("tr", {}, cols.map((c) => h("th", {}, c))), rows));
  if (rep.compare) root.append(table("Сравнение подходов (attrx report)", ["Режим", "Precision", "Recall", "F1", "Coverage", "Карточек/с", "LLM-вызовов"],
    rep.compare.map((r) => h("tr", {}, h("td", {}, h("b", {}, r.mode)), h("td", {}, f3(r.precision)), h("td", {}, f3(r.recall)),
      h("td", {}, f3(r.f1)), h("td", {}, f3(r.coverage)), h("td", {}, String(r.cards_per_sec)), h("td", {}, String(r.llm_calls))))));
  if (rep.segments) root.append(table("Качество по сегментам карточек", ["Сегмент", "Карточек", "Precision", "Recall", "F1", "Coverage"],
    Object.entries(rep.segments).map(([k, v]) => h("tr", {}, h("td", {}, k), h("td", {}, String(v.n_cards)), h("td", {}, f3(v.precision)),
      h("td", {}, f3(v.recall)), h("td", {}, f3(v.f1)), h("td", {}, f3(v.coverage))))));
}

(async () => {
  const hl = await api("/api/health");
  $("#health").classList.toggle("is-on", hl.llm_enabled);
  $("#health span").textContent = hl.llm_enabled ? "LLM · " + hl.llm_model + " @ " + hl.llm_host : "LLM выключена · только правила";
  samples = await api("/api/samples");
  for (const s of samples) $("#sample").append(h("option", { value: s.id }, s.id + " · " + s.title.slice(0, 50)));
})();
