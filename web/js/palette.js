// Палитра команд (Ctrl/⌘ K): собрать экран под задачу (generative UI), найти карточку, перейти, сменить тему.
// Запрос уходит в POST /api/ui: LLM выбирает компоненты из фиксированного реестра, без LLM — подбор по словам.
import { get } from "./api.js";
import { debounce, h, icon, put } from "./dom.js";
import { INTENTS } from "./copy.js";
import { kbd, setTheme, themeLabel } from "./ui.js";

export function initPalette(app) {
  const dialog = document.getElementById("palette");
  const input = h("input", { class: "palette__input", type: "text", role: "combobox", "aria-expanded": "true", "aria-controls": "palette-list",
    "aria-autocomplete": "list", placeholder: "Что показать? Например: только спорные значения…", autocomplete: "off", spellcheck: "false", "aria-label": "Команда или запрос" });
  const list = h("div", { class: "palette__list", id: "palette-list", role: "listbox", "aria-label": "Команды" });
  const llmNote = h("span", { class: "palette__llm" });
  dialog.append(h("div", { class: "palette__box" },
    h("div", { class: "palette__field" }, icon("magnifying-glass", 18, "palette__glass"), input, kbd("Esc")),
    list,
    h("div", { class: "palette__foot" }, llmNote, h("span", { class: "palette__keys" }, kbd("↑"), kbd("↓"), " выбрать", kbd("Enter"), " выполнить"))));

  let options = [];
  let active = 0;
  let cards = [];
  let lastQ = "";

  const searchCards = debounce(async (q) => {
    if (q.length < 2) {
      cards = [];
      paint();
      return;
    }
    try {
      const res = await get("/api/examples?limit=6&q=" + encodeURIComponent(q));
      if (input.value.trim() !== q) return;
      cards = res.items;
      paint();
    } catch {
      cards = [];
    }
  }, 160);

  function build(q) {
    const ql = q.toLowerCase();
    const has = (t) => !ql || t.toLowerCase().includes(ql);
    const out = [];
    const compose = (intent) => async () => {
      close();
      await app.go("card");
      app.workbench.runIntent(intent);
    };
    if (q) out.push({ group: "Собрать экран для текущей карточки", icon: "list-checks", label: "Собрать экран: «" + q + "»", hint: "Enter", run: compose(q), primary: true });
    for (const it of INTENTS.filter(has)) out.push({ group: "Собрать экран для текущей карточки", icon: "list-checks", label: it, run: compose(it) });
    for (const c of cards) out.push({ group: "Карточки", icon: "file-text", label: c.title, hint: c.category || "", run: () => { close(); location.hash = "#/card?id=" + encodeURIComponent(c.id); } });
    const nav = [
      { label: "Карточка", icon: "file-text", route: "card", hint: "проверка одной карточки" },
      { label: "Очередь проверки", icon: "tray", route: "review", hint: "решения по сомнительным значениям" },
      { label: "Качество", icon: "chart-line-up", route: "quality", hint: "метрики на эталоне" },
    ];
    for (const n of nav.filter((n) => has(n.label) || has(n.hint))) out.push({ group: "Перейти", icon: n.icon, label: n.label, hint: n.hint, run: () => { close(); app.go(n.route); } });
    const view = [
      { label: "Тема: " + themeLabel("system"), icon: "monitor", run: () => setTheme("system") },
      { label: "Тема: " + themeLabel("light"), icon: "sun", run: () => setTheme("light") },
      { label: "Тема: " + themeLabel("dark"), icon: "moon", run: () => setTheme("dark") },
      { label: "Справка: как читать экран", icon: "question", run: () => { close(); app.help.open(); } },
    ];
    for (const v of view.filter((v) => has(v.label))) out.push({ group: "Вид", ...v, run: v.run.length ? v.run : () => { v.run(); close(); } });
    return out;
  }

  function paint() {
    const q = input.value.trim();
    options = build(q);
    if (q !== lastQ) active = 0;
    lastQ = q;
    active = Math.min(active, Math.max(0, options.length - 1));
    const groups = [];
    for (const [i, o] of options.entries()) {
      let g = groups.find((x) => x.name === o.group);
      if (!g) groups.push((g = { name: o.group, items: [] }));
      g.items.push(h("div", {
        class: "palette__opt" + (o.primary ? " is-primary" : ""), role: "option", id: "pal-" + i, "aria-selected": String(i === active),
        onpointermove: () => { if (active !== i) setActive(i); }, onclick: () => o.run(),
      }, icon(o.icon, 16, "palette__opt-icon"), h("span", { class: "palette__opt-label" }, o.label), o.hint ? h("span", { class: "palette__opt-hint" }, o.hint) : null));
    }
    put(list, ...(groups.length ? groups.map((g) => h("div", { class: "palette__group", role: "group", "aria-label": g.name },
      h("div", { class: "palette__group-title", "aria-hidden": "true" }, g.name), g.items))
      : [h("p", { class: "palette__empty" }, "Ничего не нашлось. Нажмите Enter, чтобы собрать экран по этому запросу.")]));
    input.setAttribute("aria-activedescendant", options.length ? "pal-" + active : "");
  }

  function setActive(i) {
    active = (i + options.length) % options.length;
    list.querySelectorAll(".palette__opt").forEach((el, j) => el.setAttribute("aria-selected", String(j === active)));
    input.setAttribute("aria-activedescendant", "pal-" + active);
    list.querySelector("#pal-" + active)?.scrollIntoView({ block: "nearest" });
  }

  input.addEventListener("input", () => {
    paint();
    searchCards(input.value.trim());
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (options.length) setActive(active + (e.key === "ArrowDown" ? 1 : -1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      options[active]?.run();
    } else if (e.key === "Home" || e.key === "End") {
      if (!options.length) return;
      e.preventDefault();
      setActive(e.key === "Home" ? 0 : options.length - 1);
    }
  });
  dialog.addEventListener("click", (e) => e.target === dialog && close());

  function open(prefill = "") {
    put(llmNote, app.health?.llm_enabled
      ? [icon("lightning", 14), "Экран соберёт LLM из готовых компонентов"]
      : [icon("info", 14), "LLM не подключена: экран соберётся по ключевым словам"]);
    input.value = prefill;
    cards = [];
    active = 0;
    lastQ = null;
    paint();
    if (!dialog.open) dialog.showModal();
    input.focus();
  }

  function close() {
    if (dialog.open) dialog.close();
  }

  return { open, close, isOpen: () => dialog.open };
}
