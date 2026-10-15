// Точка входа консоли модератора attrx. Без сборки: ES-модули отдаёт FastAPI из web/.
// Разделы: #/card (проверка карточки), #/review (очередь), #/quality (качество). Ctrl/⌘ K — палитра.
import { get } from "./js/api.js";
import { $, $$, h, icon, isTyping, keyOf, put } from "./js/dom.js";
import { mountWorkbench } from "./js/workbench.js";
import { mountReview } from "./js/review.js";
import { mountQuality } from "./js/quality.js";
import { initPalette } from "./js/palette.js";
import { initHelp } from "./js/help.js";
import { getTheme, hideTip, initTooltips, isMac, nextTheme, onThemeChange, setTheme, themeLabel, toast } from "./js/ui.js";

const app = {
  health: null,
  threshold: 0.7,
  categories: [],
  isMac,
  views: {},
  route: null,
  _report: null,
  _dicts: new Map(),

  /** Отчёт из submission/ (кэш; fresh — перечитать). */
  getReport(fresh = false) {
    if (!app._report || fresh) app._report = get("/api/report").catch((e) => { app._report = null; throw e; });
    return app._report;
  },

  /** Значения справочника для подсказок при исправлении (справочник допустимых значений). */
  async suggestions(category, attribute) {
    const cat = app.categories.find((c) => c.name === category);
    const dict = cat?.attributes.find((a) => a.key === attribute)?.dictionary;
    if (!dict) return [];
    if (!app._dicts.has(dict)) app._dicts.set(dict, get("/api/dictionary/" + encodeURIComponent(dict)).then((d) => d.values).catch(() => []));
    return app._dicts.get(dict);
  },

  setQueueCount(n) {
    const el = $("#queue-count");
    el.textContent = n > 0 ? String(n) : "";
    el.hidden = !(n > 0);
    el.setAttribute("aria-label", n > 0 ? n + " на проверку" : "");
  },

  async refreshQueueCount() {
    try {
      const q = await get("/api/review/queue?limit=5000");
      app.setQueueCount(q.stats.remaining);
    } catch {
      /* счётчик не критичен */
    }
  },

  async go(route, params = {}) {
    const qs = new URLSearchParams(params).toString();
    if (app.route === route && !qs) return;
    const hash = "#/" + route + (qs ? "?" + qs : "");
    if (location.hash !== hash) {
      location.hash = hash;
      await new Promise((r) => setTimeout(r, 0));
    } else await show(route, params);
  },
};

function parseHash() {
  const [path, qs] = location.hash.replace(/^#\/?/, "").split("?");
  const route = ["card", "review", "quality"].includes(path) ? path : "card";
  return { route, params: Object.fromEntries(new URLSearchParams(qs || "")) };
}

async function show(route, params) {
  hideTip();
  app.route = route;
  for (const v of $$(".view")) v.hidden = v.dataset.view !== route;
  for (const a of $$(".nav__link")) {
    if (a.dataset.route === route) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  }
  document.title = { card: "Карточка", review: "Очередь проверки", quality: "Качество" }[route] + " · attrx";
  await app.views[route].show(params);
}

function paintLlmStatus() {
  const hl = app.health;
  const btn = $("#llm-status");
  const on = Boolean(hl?.llm_enabled);
  btn.dataset.state = on ? "on" : "off";
  $(".llm-status__text", btn).textContent = on ? "LLM: " + (hl.llm_model || "подключена") : "LLM выключена";
  btn.dataset.tip = on ? `Модель ${hl.llm_model} на ${hl.llm_host}. Вызывается только для непокрытых характеристик.` : "Работают продавец, правила и ML. Как подключить LLM, написано в справке.";
}

function paintThemeButton() {
  const v = getTheme();
  const btn = $("#theme");
  btn.setAttribute("aria-label", "Тема: " + themeLabel(v) + ". Нажмите, чтобы сменить");
  btn.dataset.tip = "Тема: " + themeLabel(v);
  put(btn, icon({ system: "circle-half", light: "sun", dark: "moon" }[v], 18));
}

async function boot() {
  initTooltips();
  setTheme(getTheme());
  paintThemeButton();
  onThemeChange(paintThemeButton);
  $("#theme").addEventListener("click", () => {
    const v = nextTheme();
    toast("Тема: " + themeLabel(v), { timeout: 1600 });
  });
  put($(".cmdk__keys"), h("kbd", { class: "kbd" }, isMac ? "⌘" : "Ctrl"), h("kbd", { class: "kbd" }, "K"));

  try {
    app.health = await get("/api/health");
    app.threshold = app.health.confidence_threshold ?? 0.7;
  } catch (e) {
    document.body.classList.add("is-offline");
    put($("#main"), h("div", { class: "empty empty--wide", role: "alert" }, icon("warning-circle", 28, "empty__icon"),
      h("h1", { class: "empty__title" }, "Сервер не отвечает"),
      h("p", { class: "empty__text" }, "Интерфейс не смог связаться с API: " + e.message + ". Запустите attrx serve и обновите страницу.")));
    return;
  }
  app.categories = await get("/api/categories").catch(() => []);
  paintLlmStatus();

  app.palette = initPalette(app);
  app.help = initHelp(app);
  app.views.card = app.workbench = mountWorkbench($("#view-card"), app);
  app.views.review = mountReview($("#view-review"), app);
  app.views.quality = mountQuality($("#view-quality"), app);

  $("#open-palette").addEventListener("click", () => app.palette.open());
  $("#open-help").addEventListener("click", () => app.help.open());
  $("#llm-status").addEventListener("click", () => app.help.open("help-llm"));

  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && keyOf(e) === "k") {
      e.preventDefault();
      app.palette.isOpen() ? app.palette.close() : app.palette.open();
      return;
    }
    if (app.palette.isOpen() || app.help.isOpen() || e.defaultPrevented) return;
    if (!isTyping(e) && e.key === "?" ) {
      e.preventDefault();
      app.help.open();
      return;
    }
    if (!isTyping(e) && (e.key === "/" || e.code === "Slash") && !e.shiftKey) {
      e.preventDefault();
      app.palette.open();
      return;
    }
    const v = app.views[app.route];
    if (v?.onKey && v.onKey(e)) e.preventDefault();
  });

  window.addEventListener("hashchange", () => {
    const { route, params } = parseHash();
    show(route, params);
  });
  const { route, params } = parseHash();
  await show(route, params);
  if (route !== "review") app.refreshQueueCount();
}

boot();
