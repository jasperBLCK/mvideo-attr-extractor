// «Карточка»: слева текст карточки (как лист на столе), справа характеристики. Наведение на характеристику
// подсвечивает цитату в тексте, наведение на цитату подсвечивает характеристику.
import { get, post } from "./api.js";
import { capitalize, cleanPath, fmt, h, icon, isTyping, keyOf, reducedMotion, put } from "./dom.js";
import { cardText, evidenceValue, locateAll, renderDocument } from "./evidence.js";
import { render as renderSpec } from "../components.js";
import { SOURCE } from "./copy.js";
import { announce, kbd, toast } from "./ui.js";

const STAGES = { category: "категория", seller: "продавец", rules: "правила", ml: "ML", llm: "LLM", verify: "проверка", total: "всего" };

export function mountWorkbench(root, app) {
  const st = {
    cats: [], category: null, items: [], index: -1, synthetic: true, source: null,
    card: null, result: null, text: null, spec: null, defaultSpec: null, intent: null, specWarning: null, composing: false,
    useLlm: false, loading: false, error: null, editing: false, filter: null, ready: false,
    decisions: new Map(), located: new Map(), doc: null, req: 0,
  };

  // ---------- каркас ----------
  const catSelect = h("select", { class: "select__control", id: "wb-cat", "aria-label": "Категория примеров" });
  const prevBtn = h("button", { class: "icon-btn", type: "button", "aria-label": "Предыдущая карточка", "data-tip": "Предыдущая карточка" }, icon("caret-left", 16));
  const nextBtn = h("button", { class: "icon-btn", type: "button", "aria-label": "Следующая карточка", "data-tip": "Следующая карточка" }, icon("caret-right", 16));
  const pos = h("span", { class: "pager__pos", "aria-live": "polite" });
  const ownBtn = h("button", { class: "btn btn--quiet btn--sm", type: "button" }, icon("pencil-simple", 14), "Своя карточка");
  const srcbar = h("div", { class: "srcbar" },
    h("label", { class: "select" }, h("span", { class: "sr-only" }, "Категория"), catSelect, icon("caret-down", 14, "select__caret")),
    h("div", { class: "pager" }, prevBtn, pos, nextBtn),
    ownBtn);

  const sheet = h("article", { class: "sheet doc", "aria-label": "Текст карточки", tabindex: "0" });
  const llmInput = h("input", { type: "checkbox", role: "switch", class: "switch__input", id: "wb-llm" });
  const llmHint = h("p", { class: "srcfoot__hint", id: "wb-llm-hint" });
  llmInput.setAttribute("aria-describedby", "wb-llm-hint");
  const srcfoot = h("div", { class: "srcfoot" },
    h("label", { class: "switch", for: "wb-llm" }, llmInput, h("span", { class: "switch__track", "aria-hidden": "true" }), h("span", { class: "switch__label" }, "Дозаполнить через LLM")),
    llmHint);

  const crumbs = h("div", { class: "crumbs" });
  const metaRow = h("div", { class: "rhead__meta" });
  const composeBtn = h("button", { class: "btn btn--quiet btn--sm", type: "button", onclick: () => app.palette.open() },
    icon("list-checks", 14), "Собрать экран", kbd(app.isMac ? "⌘" : "Ctrl", "K"));
  const rerunBtn = h("button", { class: "icon-btn", type: "button", "aria-label": "Извлечь заново", "data-tip": "Извлечь заново" }, icon("arrow-clockwise", 16));
  const rhead = h("header", { class: "rhead" },
    h("div", { class: "rhead__top" }, crumbs, h("div", { class: "rhead__actions" }, composeBtn, rerunBtn)),
    metaRow);
  const genbar = h("div", { class: "genbar", hidden: true });
  const body = h("div", { class: "rbody", "aria-busy": "false" });
  const resultCol = h("div", { class: "bench__result" }, rhead, genbar, body);

  root.append(
    h("h1", { class: "sr-only" }, "Карточка товара: извлечение характеристик"),
    h("div", { class: "bench" },
      h("div", { class: "bench__source" }, srcbar, sheet, srcfoot),
      resultCol));

  // ---------- данные ----------
  async function loadCategory(name) {
    st.category = name;
    const q = name ? "&category=" + encodeURIComponent(name) : "";
    const ex = await get("/api/examples?limit=500" + q);
    st.items = ex.items;
    st.synthetic = ex.synthetic;
    st.source = ex.source;
    if (!st.cats.length) st.cats = ex.categories;
  }

  async function init(params = {}) {
    try {
      const all = await get("/api/examples?limit=1");
      st.cats = all.categories;
      st.synthetic = all.synthetic;
      st.source = all.source;
      put(catSelect, ...st.cats.map((c) => h("option", { value: c.name }, c.name + " (" + c.count + ")")));
      let target = null;
      if (params.id) target = await get("/api/examples/" + encodeURIComponent(params.id)).catch(() => null);
      const preferred = st.cats.find((c) => /смартфон/i.test(c.name))?.name || st.cats[0]?.name || null;
      const cat = target ? target.category || "Без категории" : preferred;
      if (!cat) {
        paintNoExamples();
        return;
      }
      catSelect.value = cat;
      await loadCategory(cat);
      const idx = target ? Math.max(0, st.items.findIndex((c) => String(c.id) === String(target.id))) : 0;
      select(idx);
    } catch (e) {
      st.error = e;
      paintError();
    } finally {
      st.ready = true;
    }
  }

  function select(i) {
    if (i < 0 || i >= st.items.length) return;
    st.index = i;
    setEditing(false);
    paintPager();
    const card = st.items[i];
    history.replaceState(null, "", "#/card?id=" + encodeURIComponent(card.id));
    extract(card);
  }

  async function extract(card) {
    const req = ++st.req;
    st.card = card;
    st.loading = true;
    st.error = null;
    st.located = new Map();
    st.text = cardText(card);
    st.decisions = new Map();
    st.filter = null;
    paintDoc();
    // скелетон только если ответ задерживается: без LLM извлечение занимает миллисекунды и мигать не нужно
    const slow = setTimeout(() => req === st.req && st.loading && paintLoading(), 160);
    try {
      const res = await post("/api/extract", { card, use_llm: st.useLlm && app.health.llm_enabled }, { timeout: 120000 });
      if (req !== st.req) return;
      st.result = res.result;
      st.text = res.text || cardText(card);
      st.defaultSpec = res.ui;
      st.spec = res.ui;
      st.intent = null;
      st.specWarning = null;
      st.located = locateAll(st.text.full, st.result.attributes);
      paintDoc();
      paintHead();
      paintBody();
      const n = st.result.attributes.filter((a) => a.status === "review").length;
      announce(`Извлечено ${fmt.count(st.result.attributes.length, "характеристика", "характеристики", "характеристик")}, на проверку ${n}`);
    } catch (e) {
      if (req !== st.req) return;
      st.error = e;
      paintError();
    } finally {
      clearTimeout(slow);
      if (req === st.req) {
        st.loading = false;
        body.setAttribute("aria-busy", "false");
        sheet.classList.remove("is-loading");
      }
    }
  }

  // ---------- левая колонка: документ ----------
  function paintPager() {
    const n = st.items.length;
    pos.textContent = n ? `${st.index + 1} из ${n}` : "";
    prevBtn.disabled = st.index <= 0;
    nextBtn.disabled = st.index >= n - 1;
  }

  function paintDoc() {
    if (st.editing) return;
    const c = st.card;
    if (!c) return;
    const meta = h("div", { class: "doc__meta" },
      c.seller ? h("span", { class: "doc__metaitem", "data-tip": "Продавец карточки" }, icon("storefront", 14), c.seller) : null,
      c.category ? h("span", { class: "doc__metaitem", "data-tip": "Категория, указанная в данных" }, icon("tag", 14), c.category) : null,
      h("span", { class: "doc__metaitem doc__id" }, "id " + c.id),
      h("button", { class: "btn btn--ghost btn--sm doc__edit", type: "button", onclick: () => openEditor(true) }, icon("pencil-simple", 14), "Изменить текст"));
    st.doc = renderDocument({ card: c, text: st.text, located: st.located });
    st.doc.onHover((attrs) => hotRows(attrs));
    put(sheet, meta, st.doc.el,
      st.synthetic ? h("p", { class: "doc__note" }, icon("info", 14), "Синтетический пример из data/sample/cards.jsonl. Реальные карточки появятся после подготовки сэмпла (real_cards.jsonl).") : null);
    sheet.classList.toggle("is-loading", st.loading);
    sheet.scrollTop = 0;
  }

  function hotRows(attrs) {
    body.querySelectorAll(".arow.is-hot, .evi.is-hot, .cbar.is-hot").forEach((r) => r.classList.remove("is-hot"));
    if (!attrs) return;
    for (const a of attrs) body.querySelectorAll(`.arow[data-attr="${CSS.escape(a)}"]`).forEach((r) => r.classList.add("is-hot"));
  }

  // предупреждение о несохранённом тексте своей карточки при уходе со страницы
  window.addEventListener("beforeunload", (e) => {
    if (st.editing && sheet.querySelector("form.edit") && [...sheet.querySelectorAll(".edit .input")].some((x) => x.value.trim())) e.preventDefault();
  });

  function setEditing(on) {
    st.editing = on;
    resultCol.inert = on;
    resultCol.classList.toggle("is-stale", on);
    sheet.classList.toggle("is-editing", on);
  }

  function openEditor(prefill) {
    setEditing(true);
    const c = prefill && st.card ? st.card : { title: "", description: "", seller_attrs: {}, category: "", seller: "" };
    const title = h("input", { class: "input", id: "ed-title", name: "title", required: true, value: c.title || "", autocomplete: "off" });
    const desc = h("textarea", { class: "input textarea", id: "ed-desc", name: "description", rows: "7" });
    desc.value = c.description || "";
    const seller = h("textarea", { class: "input textarea", id: "ed-seller", name: "seller_attrs", rows: "3", placeholder: "Цвет: чёрный\nВстроенная память: 256 ГБ…" });
    seller.value = Object.entries(c.seller_attrs || {}).map(([k, v]) => k + ": " + v).join("\n");
    const cat = h("input", { class: "input", id: "ed-cat", name: "category", value: c.category || "", list: "ed-cats", autocomplete: "off" });
    const shop = h("input", { class: "input", id: "ed-shop", name: "seller", value: c.seller || "", autocomplete: "off" });
    const cats = h("datalist", { id: "ed-cats" }, (app.categories || []).map((x) => h("option", { value: x.title })));
    const titleError = h("p", { class: "field__error", id: "ed-title-error", hidden: true }, "Введите название товара: без него характеристики не извлечь.");
    title.setAttribute("aria-describedby", "ed-title-error");
    title.addEventListener("input", () => {
      title.removeAttribute("aria-invalid");
      titleError.hidden = true;
    });
    const field = (id, label, control, hint) => h("div", { class: "field" },
      h("label", { class: "field__label", for: id }, label), control, hint ? h("p", { class: "field__hint", id: id + "-hint" }, hint) : null);
    const form = h("form", {
      class: "edit", onsubmit: (e) => {
        e.preventDefault();
        if (!title.value.trim()) {
          title.setAttribute("aria-invalid", "true");
          titleError.hidden = false;
          title.focus();
          return;
        }
        const attrs = {};
        for (const line of seller.value.split("\n")) {
          const i = line.indexOf(":");
          if (i > 0) attrs[line.slice(0, i).trim()] = line.slice(i + 1).trim();
        }
        setEditing(false);
        st.index = -1;
        paintPager();
        pos.textContent = "своя карточка";
        extract({ id: "manual-" + Date.now().toString(36), title: title.value.trim(), description: desc.value.trim(),
          seller_attrs: attrs, category: cat.value.trim() || null, seller: shop.value.trim() || null });
      },
    },
    h("h2", { class: "edit__title" }, prefill ? "Изменить текст карточки" : "Своя карточка"),
    h("p", { class: "edit__lede" }, "Вставьте текст как есть. Система найдёт характеристики и покажет, из какой фразы взято каждое значение."),
    field("ed-title", "Название товара", h("div", null, title, titleError)),
    field("ed-desc", "Описание", desc),
    field("ed-seller", "Поля продавца", seller, "По одному на строку: «Ключ: значение»"),
    h("div", { class: "field-row" },
      field("ed-cat", "Категория", h("div", null, cat, cats), "Можно не указывать: определим по тексту"),
      field("ed-shop", "Продавец", shop, "Необязательно")),
    h("div", { class: "edit__actions" },
      h("button", { class: "btn btn--primary", type: "submit" }, "Извлечь характеристики"),
      h("button", { class: "btn btn--quiet", type: "button", onclick: () => { setEditing(false); paintDoc(); } }, "Отмена")));
    for (const [ctl, hint] of [[seller, "ed-seller-hint"], [cat, "ed-cat-hint"], [shop, "ed-shop-hint"]]) ctl.setAttribute("aria-describedby", hint);
    put(sheet, form);
    sheet.scrollTop = 0;
    title.focus();
  }

  function paintLlm() {
    const on = Boolean(app.health?.llm_enabled);
    llmInput.disabled = !on;
    llmInput.checked = on && st.useLlm;
    srcfoot.classList.toggle("is-disabled", !on);
    put(llmHint, on
      ? "Только для характеристик, которые не закрыли продавец, правила и ML. Значение без цитаты из текста отбрасывается."
      : h("span", null, "LLM не подключена: без неё работают продавец, правила и ML. Чтобы включить, задайте LLM_BASE_URL и LLM_MODEL (или OPENAI_API_KEY) в .env и перезапустите attrx serve."));
  }

  // ---------- правая колонка ----------
  function paintHead() {
    const r = st.result;
    const path = cleanPath(r.category_path).length ? cleanPath(r.category_path) : [r.category_title || r.category];
    // «неуверенно» решает пайплайн (порог классификатора категории) и пишет предупреждение — своего порога здесь нет
    const lowCat = (r.warnings || []).some((w) => /^категория определена неуверенно/u.test(w));
    const catThr = app.health?.category_threshold;
    const catTip = "Насколько уверенно определена категория. " + (catThr != null
      ? "Если классификатор по тексту уверен меньше чем на " + fmt.conf(catThr) + ", все значения карточки уходят на проверку."
      : "Если категория определена неуверенно, все значения карточки уходят на проверку.");
    put(crumbs,
      path.length > 1 ? h("ol", { class: "crumbs__list", "aria-label": "Путь в каталоге" }, path.slice(0, -1).map((p) => h("li", { class: "crumbs__item" }, p))) : null,
      h("div", { class: "crumbs__titlerow" },
        h("h2", { class: "crumbs__title" }, path[path.length - 1]),
        h("span", { class: "crumbs__conf" + (lowCat ? " is-low" : ""), tabindex: "0", "data-tip": catTip },
          "уверенность в категории ", fmt.conf(r.category_confidence))));
    const tm = r.timings_ms || {};
    put(metaRow, trace(r, tm), h("span", { class: "meta", "data-tip": "Время извлечения на сервере" }, icon("clock", 14), fmt.ms(tm.total ?? Object.values(tm).reduce((x, y) => x + y, 0))));
  }

  /** Ход гибрида по этой карточке: какие источники сработали, какие не понадобились или выключены. */
  function trace(r, tm) {
    const n = (src) => r.attributes.filter((a) => a.sources.includes(src)).length;
    const hl = app.health || {};
    const vals = (k) => fmt.count(k, "значение", "значения", "значений");
    const stage = (key, label, state, text, tip) => h("li", { class: "trace__stage trace__stage--" + state, "data-tip": tip + (tm[key] != null ? " Время: " + fmt.ms(tm[key]) + "." : ""), "data-tip-title": label },
      h("span", { class: "trace__label" }, label), h("span", { class: "trace__val" }, text));
    const hasSeller = Object.keys(st.card?.seller_attrs || {}).length > 0;
    const stages = [
      n("seller") ? stage("seller", "Продавец", "used", vals(n("seller")), "Значения из полей продавца.")
        : stage("seller", "Продавец", "idle", hasSeller ? "не совпало" : "полей нет", hasSeller ? "Поля продавца есть, но не сопоставились с атрибутами категории." : "Продавец не заполнил блок характеристик."),
      stage("rules", "Правила", n("rules") ? "used" : "idle", n("rules") ? vals(n("rules")) : "ничего", "Регулярные выражения и справочники работают всегда."),
      !hl.ml_model ? stage("ml", "ML", "off", "нет модели", "NER-модель не обучена в этом окружении (attrx train-ner).")
        : n("ml") ? stage("ml", "ML", "used", vals(n("ml")), "NER дозаполнил хвост.") : stage("ml", "ML", "idle", "не понадобилась", "Продавец и правила закрыли всё, что смогли; NER не добавил значений."),
      !hl.llm_enabled ? stage("llm", "LLM", "off", "выключена", "LLM не подключена, дорогой шаг не вызывается.")
        : r.llm_used ? stage("llm", "LLM", "used", vals(n("llm")), "LLM дозаполнила непокрытые характеристики, каждое значение проверено цитатой.")
          : stage("llm", "LLM", "idle", st.useLlm ? "не понадобилась" : "не включена", st.useLlm ? "Дешёвые методы справились, вызов LLM не потребовался." : "Включите «Дозаполнить через LLM» слева."),
    ];
    return h("ol", { class: "trace", "aria-label": "Какие источники сработали" }, stages);
  }

  function ctx() {
    return {
      threshold: app.threshold,
      hint: matchMedia("(max-width: 1040px)").matches
        ? "Нажмите на характеристику: откроется цитата из текста карточки и кнопки решения."
        : "Наведите на характеристику: в тексте слева подсветится цитата, из которой взято значение. Enter раскрывает решение, J и K переключают строки.",
      get filter() { return st.filter; },
      setFilter(f) {
        st.filter = f;
        paintBody();
        body.querySelector(".sum__item[aria-pressed='true']")?.focus() || body.querySelector(".sum__item--" + (f || "accepted"))?.focus();
      },
      full: st.text?.full || "",
      located: st.located,
      decisions: st.decisions,
      focusEvidence: (attr, { scroll = true } = {}) => st.doc?.highlight(attr, { scroll, scroller: sheet }),
      blurEvidence: () => st.doc?.highlight(null),
      showInText(attr) {
        const m = st.doc?.first(attr);
        if (!m) return;
        st.doc.highlight(attr, { scroll: false });
        m.scrollIntoView({ block: "center", behavior: reducedMotion() ? "auto" : "smooth" });
        m.classList.add("is-flash");
        setTimeout(() => m.classList.remove("is-flash"), 900);
      },
      decide: (a, action, corrected) => post("/api/feedback", { card_id: String(st.card.id), category: st.result.category, attribute: a.attribute, predicted: a.value, corrected, raw: evidenceValue(st.text, a) })
        .then((rec) => {
          const verb = { accepted: "Принято", corrected: "Исправлено", rejected: "Отклонено" }[action];
          announce(verb + ": " + a.title);
          app.refreshQueueCount();
          return rec;
        }),
      undo: (a) => post("/api/feedback/undo", { card_id: String(st.card.id), attribute: a.attribute }).then(() => app.refreshQueueCount()),
      suggestions: (a) => app.suggestions(st.result.category, a.attribute),
      getReport: app.getReport,
      onError: (e, retry) => toast("Решение не сохранено: " + e.message, { tone: "danger", action: retry ? "Повторить" : null, onAction: retry }),
    };
  }

  function paintBody() {
    if (!st.result) return;
    genbar.hidden = !st.intent && !st.composing;
    if (st.composing) {
      put(genbar, h("span", { class: "genbar__text" }, icon("list-checks", 16), "Сборка экрана по запросу «" + st.intent + "»…"));
    } else if (st.intent) {
      const byLlm = st.spec.generated_by === "llm";
      put(genbar, 
        h("div", { class: "genbar__text" }, icon("list-checks", 16),
          h("span", null, h("strong", null, "Экран по запросу «" + st.intent + "»"),
            h("span", { class: "genbar__how" }, byLlm ? "собран LLM из безопасных компонентов" : st.specWarning || "собран по ключевым словам"))),
        h("button", { class: "btn btn--ghost btn--sm", type: "button", onclick: resetSpec }, icon("arrow-u-up-left", 14), "Стандартный вид"));
    }
    renderSpec(body, st.spec, st.result, ctx());
    body.setAttribute("aria-busy", "false");
  }

  function paintLoading() {
    body.setAttribute("aria-busy", "true");
    sheet.classList.add("is-loading");
    genbar.hidden = true;
    if (!st.result) {
      put(crumbs, h("span", { class: "skel skel--text", style: { width: "220px" } }));
      put(metaRow, h("span", { class: "skel skel--text", style: { width: "300px" } }));
    }
    put(body, h("div", { class: "skel-block", "aria-hidden": "true" },
      h("div", { class: "skel skel--sum" }),
      Array.from({ length: 6 }, () => h("div", { class: "skel-row" }, h("span", { class: "skel skel--text" }), h("span", { class: "skel skel--text skel--short" }), h("span", { class: "skel skel--bar" })))),
    h("p", { class: "sr-only" }, "Извлечение характеристик…"));
  }

  function paintError() {
    body.setAttribute("aria-busy", "false");
    sheet.classList.remove("is-loading");
    put(body, h("div", { class: "alert alert--danger", role: "alert" }, icon("warning-circle", 16, "alert__icon"),
      h("div", { class: "alert__text" }, h("strong", null, "Не удалось извлечь характеристики. "), st.error?.message || "",
        h("div", { class: "alert__actions" }, h("button", { class: "btn btn--quiet btn--sm", type: "button", onclick: () => st.card && extract(st.card) }, icon("arrow-clockwise", 14), "Повторить")))));
  }

  function paintNoExamples() {
    put(sheet, h("div", { class: "empty empty--sheet" }, icon("file-text", 28, "empty__icon"),
      h("h2", { class: "empty__title" }, "Примеров карточек нет"),
      h("p", { class: "empty__text" }, "Положите карточки в data/sample/real_cards.jsonl или вставьте свою."),
      h("button", { class: "btn btn--primary", type: "button", onclick: () => openEditor(false) }, "Своя карточка")));
  }

  async function runIntent(intent) {
    if (!st.result) {
      toast("Сначала выберите карточку: экран собирается под её характеристики.");
      return;
    }
    st.intent = intent;
    st.composing = true;
    paintBody();
    try {
      const res = await post("/api/ui", { result: st.result, intent }, { timeout: 90000 });
      st.spec = res.ui;
      st.specWarning = res.ui_warning;
    } catch (e) {
      st.intent = null;
      toast("Не удалось собрать экран: " + e.message, { tone: "danger" });
    } finally {
      st.composing = false;
      paintBody();
      body.querySelector(".blk__title")?.scrollIntoView({ block: "nearest" });
    }
  }

  function resetSpec() {
    st.spec = st.defaultSpec;
    st.intent = null;
    st.specWarning = null;
    paintBody();
  }

  // ---------- события ----------
  catSelect.addEventListener("change", async () => {
    await loadCategory(catSelect.value);
    select(0);
  });
  prevBtn.addEventListener("click", () => select(st.index - 1));
  nextBtn.addEventListener("click", () => select(st.index + 1));
  ownBtn.addEventListener("click", () => openEditor(false));
  rerunBtn.addEventListener("click", () => st.card && extract(st.card));
  llmInput.addEventListener("change", () => {
    st.useLlm = llmInput.checked;
    if (st.card && !st.editing) extract(st.card);
  });

  function rows() {
    return [...body.querySelectorAll(".attrs")].flatMap((l) => l._rows || []);
  }

  return {
    async show(params) {
      paintLlm();
      if (!st.ready) await init(params);
      else if (params.id && String(params.id) !== String(st.card?.id)) {
        st.ready = false;
        await init(params);
      }
    },
    runIntent,
    hasResult: () => Boolean(st.result),
    onKey(e) {
      if (isTyping(e) || e.metaKey || e.ctrlKey || e.altKey || st.editing) return false;
      const list = rows();
      if (!list.length) return false;
      const k = keyOf(e);
      const cur = list.findIndex((r) => r.el.contains(document.activeElement));
      const open = list.find((r) => r.isOpen());
      if (k === "j" || k === "arrowdown" || k === "k" || k === "arrowup") {
        if ((k === "arrowdown" || k === "arrowup") && cur < 0) return false;
        const d = k === "j" || k === "arrowdown" ? 1 : -1;
        const next = list[Math.max(0, Math.min(list.length - 1, (cur < 0 ? -1 : cur) + d))];
        next.focus();
        return true;
      }
      const target = cur >= 0 ? list[cur] : open;
      if (!target) return false;
      if (k === "a" || k === "e" || k === "r") {
        target.act({ a: "accept", e: "edit", r: "reject" }[k]);
        return true;
      }
      return false;
    },
    refreshLlm: paintLlm,
    cardTitle: () => st.card?.title,
  };
}

export { capitalize };
