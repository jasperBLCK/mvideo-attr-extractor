// «Очередь проверки»: значения ниже порога и конфликты источников, по одному на задачу.
// Порядок: вес характеристики в поиске × (1 − уверенность). Клавиатура: J/K, A, E, R, U.
import { get, post } from "./api.js";
import { cleanPath, fmt, h, icon, isTyping, keyOf, shownValue, put, valueText } from "./dom.js";
import { cardText, evidenceValue, locate, renderDocument, snippet } from "./evidence.js";
import { meter, sourceTags, statusMark } from "./attrs.js";
import { DECISION, EMPTY, LOOP, SOURCE, t } from "./copy.js";
import { announce, kbd, toast } from "./ui.js";

export function mountReview(root, app) {
  const st = { items: [], cards: {}, stats: null, source: null, sel: 0, category: "", withMissing: false, loaded: false, history: [], error: null };

  const progressText = h("span", { class: "progress__text" });
  const progressFill = h("span", { class: "progress__fill" });
  const progress = h("div", { class: "progress", role: "group", "aria-label": "Прогресс проверки" },
    progressText, h("span", { class: "progress__track", "aria-hidden": "true" }, progressFill));
  const catSelect = h("select", { class: "select__control", "aria-label": "Категория" });
  const missToggle = h("input", { type: "checkbox", class: "check__input", id: "rv-missing" });
  const sourceNote = h("span", { class: "qtools__src" });
  const list = h("ul", { class: "qlist", role: "listbox", tabindex: "0", "aria-label": "Значения на проверку" });
  const main = h("section", { class: "queue__main", "aria-live": "polite", "aria-atomic": "false" });
  const queue = h("div", { class: "queue" }, h("aside", { class: "queue__side" }, list), main);

  root.append(
    h("header", { class: "vhead" },
      h("div", { class: "vhead__main" },
        h("h1", { class: "vhead__title" }, "Очередь проверки"),
        h("p", { class: "vhead__lede" }, "Сначала значения, по которым чаще фильтруют покупатели и в которых система сильнее сомневается.")),
      progress),
    h("div", { class: "qtools" },
      h("label", { class: "select select--sm" }, h("span", { class: "sr-only" }, "Категория"), catSelect, icon("caret-down", 14, "select__caret")),
      h("label", { class: "check", for: "rv-missing" }, missToggle, h("span", null, "Добавить пропуски обязательных")),
      sourceNote),
    queue);

  async function load(itemId) {
    try {
      const q = new URLSearchParams({ limit: "1000" });
      if (st.category) q.set("category", st.category);
      if (st.withMissing) q.set("include_missing", "true");
      const res = await get("/api/review/queue?" + q);
      st.items = res.items;
      st.cards = res.cards || {};
      st.stats = res.stats;
      st.source = res.source;
      st.error = null;
      if (!catSelect.options.length || !st.category) {
        put(catSelect, h("option", { value: "" }, "Все категории"), ...(res.categories || []).map((c) => h("option", { value: c.name }, c.name + " (" + c.count + ")")));
        catSelect.value = st.category;
      }
      const byId = itemId ? st.items.findIndex((i) => i.id === itemId) : -1;
      const first = st.items.findIndex((i) => !i.decision);
      st.sel = byId >= 0 ? byId : first >= 0 ? first : 0;
      st.loaded = true;
    } catch (e) {
      st.error = e;
    }
    paint();
  }

  // ---------- рендер ----------
  function paint() {
    paintProgress();
    put(sourceNote, st.source ? h("span", { "data-tip": "Откуда взяты предсказания: последний прогон attrx extract, иначе файл сдачи. " + st.source },
      icon("file-text", 14), h("span", { class: "qtools__path", translate: "no" }, st.source.split("/").slice(-2).join("/"))) : null);
    if (st.error) {
      queue.classList.add("is-empty");
      list.replaceChildren();
      put(main, h("div", { class: "alert alert--danger", role: "alert" }, icon("warning-circle", 16, "alert__icon"),
        h("div", { class: "alert__text" }, h("strong", null, "Очередь не загрузилась. "), st.error.message,
          h("div", { class: "alert__actions" }, h("button", { class: "btn btn--quiet btn--sm", type: "button", onclick: load }, icon("arrow-clockwise", 14), "Повторить")))));
      return;
    }
    const empty = !st.items.length;
    queue.classList.toggle("is-empty", empty);
    if (empty) {
      list.replaceChildren();
      put(main, emptyState(st.source ? EMPTY.queueClean : EMPTY.queueNoRun));
      return;
    }
    paintList();
    paintFocus();
    revealInList(st.sel, true);
  }

  /** Прокрутка только списка (не всей страницы), чтобы выбранная задача была видна. */
  function revealInList(i, center = false) {
    const side = list.parentElement;
    const el = list.children[i];
    if (!side || !el) return;
    const top = el.offsetTop;
    const bottom = top + el.offsetHeight;
    if (center) side.scrollTop = top - side.clientHeight / 2 + el.offsetHeight / 2;
    else if (top < side.scrollTop + 8) side.scrollTop = top - 8;
    else if (bottom > side.scrollTop + side.clientHeight - 8) side.scrollTop = bottom - side.clientHeight + 8;
  }

  function paintProgress() {
    const total = st.items.length;
    const done = st.items.filter((i) => i.decision).length;
    put(progressText, "Проверено ", h("strong", null, done + " из " + total));
    progressFill.style.setProperty("--v", total ? String(done / total) : "0");
    progress.hidden = !total;
    app.setQueueCount(total - done);
  }

  function paintList() {
    put(list, ...st.items.map((it, i) => {
      const card = st.cards[it.card_id];
      const d = it.decision;
      const li = h("li", {
        class: "qitem" + (d ? " is-done" : ""), role: "option", id: "q-" + i, "aria-selected": String(i === st.sel),
        onclick: () => selectItem(i, true),
      },
      h("span", { class: "qitem__top" },
        d ? icon(DECISION[d.action].icon, 14, "qitem__icon is-done") : icon(it.status === "missing" ? "circle-dashed" : "seal-question", 14, "qitem__icon"),
        h("span", { class: "qitem__attr" }, it.title),
        h("span", { class: "qitem__conf" }, it.status === "missing" ? "пусто" : fmt.conf(it.confidence))),
      h("span", { class: "qitem__val" }, d ? DECISION[d.action].label + (d.action === "corrected" ? ": " + d.corrected : "") : shownValue(it) ?? "не найдено"),
      h("span", { class: "qitem__card" }, card?.title || "карточка " + it.card_id));
      return li;
    }));
    list.setAttribute("aria-activedescendant", "q-" + st.sel);
  }

  function selectItem(i, focusList = false) {
    if (i < 0 || i >= st.items.length) return;
    st.sel = i;
    const q = new URLSearchParams({ item: st.items[i].id });
    if (st.category) q.set("category", st.category);
    history.replaceState(null, "", "#/review?" + q);
    list.querySelectorAll(".qitem").forEach((el, j) => el.setAttribute("aria-selected", String(j === i)));
    list.setAttribute("aria-activedescendant", "q-" + i);
    revealInList(i);
    paintFocus();
    if (focusList) list.focus({ preventScroll: true });
  }

  function paintFocus() {
    const it = st.items[st.sel];
    if (!it) return;
    const card = st.cards[it.card_id];
    const value = shownValue(it);
    const d = it.decision;
    const thr = app.threshold;
    const path = cleanPath(it.category_path).length ? cleanPath(it.category_path) : [it.category_title || it.category];

    const input = h("input", { class: "input", type: "text", id: "rv-fix", value: value ?? "", autocomplete: "off", "aria-label": "Верное значение" });
    const dl = h("datalist", { id: "rv-fix-list" });
    input.setAttribute("list", "rv-fix-list");
    const fix = h("form", { class: "fix fix--lg", hidden: true, onsubmit: (e) => { e.preventDefault(); if (input.value.trim()) decide("corrected", input.value.trim()); } },
      input, dl,
      h("button", { class: "btn btn--primary", type: "submit" }, "Сохранить", kbd("Enter")),
      h("button", { class: "btn btn--quiet", type: "button", onclick: () => showFix(false) }, "Отмена", kbd("Esc")));
    input.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        showFix(false);
      }
    });

    const missing = it.status === "missing";
    const actions = h("div", { class: "focus__actions" },
      missing ? null : h("button", { class: "btn btn--primary", type: "button", dataset: { action: "accept" }, onclick: () => decide("accepted", it.value) },
        icon("check", 16), "Принять", value ? h("span", { class: "btn__val" }, "«" + value + "»") : null, kbd("A")),
      h("button", { class: "btn btn--quiet", type: "button", dataset: { action: "edit" }, onclick: () => showFix(true) }, icon("pencil-simple", 16), missing ? "Заполнить" : "Исправить", kbd("E")),
      missing ? null : h("button", { class: "btn btn--quiet", type: "button", dataset: { action: "reject" }, onclick: () => decide("rejected", null) }, icon("prohibit", 16), "Отклонить", kbd("R")));

    function showFix(on) {
      fix.hidden = !on;
      actions.hidden = on;
      if (on) {
        input.focus();
        input.select();
        app.suggestions(it.category, it.attribute).then((vals) => put(dl, ...(vals || []).slice(0, 300).map((v) => h("option", { value: v }))));
      } else {
        actions.querySelector('[data-action="edit"]')?.focus();
      }
    }

    const alts = it.alternatives?.length
      ? h("div", { class: "focus__alts" }, h("span", { class: "detail__label" }, "Другие источники предлагают"),
        h("div", { class: "alts" }, it.alternatives.slice(0, 9).map((alt, n) => h("button", { class: "alt", type: "button", onclick: () => decide("corrected", alt.value) },
          h("span", { class: "alt__val" }, valueText(alt.value, it.unit)),
          h("span", { class: "alt__meta" }, (alt.sources || []).map((s) => SOURCE[s]?.label || s).join(", "), alt.confidence != null ? " " + fmt.conf(alt.confidence) : ""),
          kbd(String(n + 1))))))
      : null;

    const decided = d ? h("div", { class: "focus__decided" }, icon(DECISION[d.action].icon, 16),
      h("span", null, DECISION[d.action].label, d.action === "corrected" ? ": «" + d.corrected + "»" : d.action === "accepted" && value ? ": «" + value + "»" : ""),
      h("span", { class: "decided__where" }, "записано в feedback.jsonl"),
      h("button", { class: "btn btn--ghost btn--sm", type: "button", onclick: () => undo(st.sel) }, icon("arrow-u-up-left", 14), "Отменить", kbd("U"))) : null;

    // текст карточки с подсвеченной цитатой
    let docBlock;
    if (card) {
      const text = card.text || cardText(card);
      const loc = locate(text.full, it);
      const located = new Map(loc ? [[it.attribute, loc]] : []);
      const doc = renderDocument({ card, text, located, titleTag: "h3" });
      const sheetEl = h("article", { class: "sheet doc doc--queue", "aria-label": "Текст карточки", tabindex: "0" }, doc.el);
      docBlock = h("div", { class: "focus__doc" },
        h("div", { class: "focus__doc-head" }, h("h3", { class: "focus__h3" }, "Текст карточки"),
          !loc ? h("span", { class: "focus__doc-note" }, it.evidence ? "цитата «" + it.evidence + "» в тексте не найдена" : "цитаты нет")
            : !loc.exact ? h("span", { class: "focus__doc-note" }, "место найдено поиском по цитате «" + it.evidence + "»") : null),
        sheetEl);
      // высота листа = остаток окна (на широком экране), чтобы прокрутка могла вывести цитату в видимую часть;
      // подсветка — после загрузки шрифтов, иначе подмена шрифта собьёт смещение строк
      document.fonts.ready.then(() => requestAnimationFrame(() => {
        if (matchMedia("(min-width: 1041px)").matches) {
          const top = sheetEl.getBoundingClientRect().top + scrollY;
          sheetEl.style.maxHeight = Math.max(260, innerHeight - top - 24) + "px";
        }
        doc.highlight(it.attribute, { scroll: true, scroller: sheetEl });
      }));
    } else {
      docBlock = h("div", { class: "focus__doc" }, h("h3", { class: "focus__h3" }, "Цитата"),
        it.evidence ? h("blockquote", { class: "detail__snippet" }, snippet(it.evidence, { start: 0, end: it.evidence.length })) : h("p", { class: "detail__none" }, "Цитаты нет"),
        h("p", { class: "focus__doc-note" }, "Текста карточки нет в data/sample, поэтому показана только цитата."));
    }

    put(main, h("div", { class: "focus" },
      h("div", { class: "focus__ctx" },
        h("ol", { class: "crumbs__list" }, path.map((p, i) => h("li", { class: "crumbs__item" + (i === path.length - 1 ? " is-last" : "") }, p))),
        h("span", { class: "focus__card" }, card?.seller ? card.seller + ", " : "", "карточка " + it.card_id),
        h("span", { class: "focus__pos" }, st.sel + 1 + " из " + st.items.length)),
      h("h2", { class: "focus__attr" }, it.title),
      h("div", { class: "focus__prop" },
        h("span", { class: "focus__value" + (value == null ? " is-empty" : "") }, value ?? "не найдено"),
        h("span", { class: "focus__signals" }, meter(it.confidence, thr, it.status), statusMark(it.status, thr), sourceTags(it.sources))),
      h("p", { class: "focus__why" }, whyLine(it, thr)),
      alts,
      d ? decided : h("div", { class: "focus__act" }, actions, fix),
      h("p", { class: "focus__keys" }, kbd("J"), kbd("K"), " следующее и предыдущее"),
      docBlock));
  }

  function whyLine(it, thr) {
    const bits = [];
    if (it.status === "missing") bits.push("Обязательная характеристика не найдена в тексте");
    else {
      if (it.alternatives?.length) bits.push("Источники предлагают разные значения");
      if (it.confidence < thr) bits.push(`Уверенность ${fmt.conf(it.confidence)} ниже порога ${fmt.conf(thr)}`);
    }
    bits.push(`вес в поиске ${fmt.num(it.traffic_weight, 2)}`);
    return bits.join(". ").replace(". вес", ". Вес") + ".";
  }

  function emptyState(copy) {
    const thr = app.threshold;
    return h("div", { class: "empty" },
      icon("tray", 28, "empty__icon"),
      h("h2", { class: "empty__title" }, copy.title),
      h("p", { class: "empty__text" }, t(copy.text, thr)),
      h("ol", { class: "loop", "aria-label": "Как работает петля обратной связи" }, LOOP.map((s) => h("li", { class: "loop__step" },
        h("strong", { class: "loop__title" }, s.title), h("span", { class: "loop__text" }, t(s.text, thr))))),
      h("a", { class: "btn btn--quiet", href: "#/card" }, icon("file-text", 16), "Открыть карточку"));
  }

  // ---------- решения ----------
  async function decide(action, corrected) {
    const i = st.sel;
    const it = st.items[i];
    if (!it || it.decision) return;
    it.decision = { action, corrected };
    st.history.push(i);
    const verb = { accepted: "Принято", corrected: "Исправлено", rejected: "Отклонено" }[action];
    const nextIdx = st.items.findIndex((x, j) => j > i && !x.decision);
    const fallback = st.items.findIndex((x) => !x.decision);
    paintProgress();
    paintList();
    const target = nextIdx >= 0 ? nextIdx : fallback;
    if (target >= 0) selectItem(target, true);
    else allDone();
    announce(`${verb}: ${it.title}.` + (target >= 0 ? ` Дальше: ${st.items[target].title}` : " Очередь пройдена"));
    toast(`${verb}: ${it.title}`, { tone: "success", action: "Отменить", onAction: () => undo(i), timeout: 4000 });
    try {
      const card = st.cards[it.card_id];
      const raw = evidenceValue(card ? card.text || cardText(card) : null, it);
      await post("/api/feedback", { card_id: String(it.card_id), category: it.category, attribute: it.attribute, predicted: it.value, corrected, raw });
    } catch (e) {
      it.decision = null;
      st.history = st.history.filter((x) => x !== i);
      paintProgress();
      paintList();
      selectItem(i, true);
      toast("Решение не сохранено: " + e.message, { tone: "danger", action: "Повторить", onAction: () => { selectItem(i); decide(action, corrected); } });
    }
  }

  async function undo(i = st.history[st.history.length - 1]) {
    const it = st.items[i];
    if (i == null || !it?.decision) return;
    const prev = it.decision;
    it.decision = null;
    st.history = st.history.filter((x) => x !== i);
    paintProgress();
    paintList();
    selectItem(i, true);
    announce("Решение отменено: " + it.title);
    try {
      await post("/api/feedback/undo", { card_id: String(it.card_id), attribute: it.attribute });
    } catch (e) {
      it.decision = prev;
      paintProgress();
      paintList();
      paintFocus();
      toast("Не удалось отменить: " + e.message, { tone: "danger" });
    }
  }

  function allDone() {
    put(main, emptyState(EMPTY.queueDone));
    list.setAttribute("aria-activedescendant", "");
  }

  catSelect.addEventListener("change", () => {
    st.category = catSelect.value;
    load();
  });
  missToggle.addEventListener("change", () => {
    st.withMissing = missToggle.checked;
    load();
  });
  list.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      selectItem(st.sel + (e.key === "ArrowDown" ? 1 : -1), true);
    } else if (e.key === "Home" || e.key === "End") {
      e.preventDefault();
      selectItem(e.key === "Home" ? 0 : st.items.length - 1, true);
    }
  });

  return {
    async show(params = {}) {
      if (params.category != null) st.category = params.category;
      await load(params.item);
      catSelect.value = st.category;
    },
    onKey(e) {
      if (isTyping(e) || e.metaKey || e.ctrlKey || e.altKey || !st.items.length) return false;
      const k = keyOf(e);
      const it = st.items[st.sel];
      if (k === "j" || k === "k") {
        selectItem(st.sel + (k === "j" ? 1 : -1), true);
        return true;
      }
      if (k === "u") {
        undo();
        return true;
      }
      if (!it || it.decision) return false;
      const btn = (a) => main.querySelector(`[data-action="${a}"]`);
      if (k === "a" && btn("accept")) return btn("accept").click(), true;
      if (k === "e" && btn("edit")) return btn("edit").click(), true;
      if (k === "r" && btn("reject")) return btn("reject").click(), true;
      if (/^[1-9]$/.test(k) && it.alternatives?.[Number(k) - 1]) {
        decide("corrected", it.alternatives[Number(k) - 1].value);
        return true;
      }
      return false;
    },
    reload: load,
  };
}
