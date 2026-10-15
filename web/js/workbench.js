// «Карточка»: слева текст карточки (как лист на столе), справа характеристики. Наведение на характеристику
// подсвечивает цитату в тексте, наведение на цитату подсвечивает характеристику.
import { get, post } from "./api.js";
import { capitalize, cleanPath, fmt, h, icon, isTyping, keyOf, reducedMotion, put } from "./dom.js";
import { cardText, evidenceValue, locateAll, renderDocument } from "./evidence.js";
import { render as renderSpec } from "../components.js";
import { fingerprint, lru } from "./cache.js";
import { announce, kbd, toast } from "./ui.js";

// порядок гибрида: значение «добавил» первый источник по этому порядку, остальные его подтвердили
const PIPELINE = ["seller", "rules", "ml", "llm"];
const LLM_REJECTED = /^llm: отброшено незаземл[её]нное значение /u;

export function mountWorkbench(root, app) {
  const st = {
    cats: [], category: null, items: [], index: -1, synthetic: true, source: null,
    card: null, result: null, text: null, spec: null, defaultSpec: null, intent: null, specWarning: null, composing: false,
    useLlm: false, loading: false, error: null, editing: false, filter: null, ready: false,
    decisions: new Map(), located: new Map(), doc: null, req: 0,
    // кэш и ход запросов: key — карточка + режим LLM текущего результата; resultCard — чья это выдача
    key: null, llmMode: false, llmAuto: false, resultCard: null, ctrl: null,
    composeReq: 0, composeCtrl: null, composeError: null, pendingIntent: null, screenAt: null, restored: false, llmPending: false,
  };
  // извлечение — только в памяти (без LLM это миллисекунды); собранные экраны дороже (LLM, 5–20 с) — ещё и в sessionStorage
  const cache = {
    extract: lru("extract", { max: 60 }),
    screens: lru("screens", { max: 40, persist: true }),
    last: lru("last-screen", { max: 80, persist: true }),
    decisions: new Map(),
  };
  const cardKey = (card) => String(card.id) + "|" + fingerprint(card);
  const resultKey = (card, llm) => cardKey(card) + "|" + (llm ? "llm" : "base");
  const screenKey = (key, intent) => key + "|" + intent.trim().toLowerCase();

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
  const rerunBtn = h("button", { class: "icon-btn", type: "button", "aria-label": "Извлечь заново", "data-tip": "Извлечь заново, не глядя в кэш" }, icon("arrow-clockwise", 16));
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
      if (params.id) target = await get("/api/examples/" + encodeURIComponent(params.id) + (params.demo ? "?demo=true" : "")).catch(() => null);
      // демо-витрина (data/sample/demo_cards.jsonl) открывается первой; без неё — смартфоны, как раньше
      const demoCat = st.cats.find((c) => c.demo)?.name;
      const preferred = demoCat || st.cats.find((c) => /смартфон/i.test(c.name))?.name || st.cats[0]?.name || null;
      const cat = target ? (target.demo && demoCat) || target.category || "Без категории" : preferred;
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

  const llmReady = () => {
    const l = app.health?.llm;
    return l ? Boolean(l.enabled && l.reachable) : Boolean(app.health?.llm_enabled);
  };
  const wantLlm = () => st.useLlm && Boolean(app.health?.llm_enabled);

  function select(i) {
    if (i < 0 || i >= st.items.length) return;
    st.index = i;
    setEditing(false);
    paintPager();
    const card = st.items[i];
    // демо-карточка показывает, как ИИ дозаполняет хвост: LLM включается сама, если доступна;
    // на обычной карточке возвращается выбор пользователя
    if (card.demo && !st.useLlm && llmReady()) {
      st.useLlm = true;
      st.llmAuto = true;
    } else if (!card.demo && st.llmAuto) {
      st.useLlm = false;
      st.llmAuto = false;
    }
    paintLlm();
    extract(card);
  }

  const cardHash = (card) => "#/card?id=" + encodeURIComponent(card.id) + (card.demo ? "&demo=1" : "");

  /** Адрес и ссылка «Карточка» в шапке ведут на текущую карточку: вернулись из другой вкладки — она же. */
  function syncHash(card) {
    const hash = cardHash(card);
    document.querySelector('.nav__link[data-route="card"]')?.setAttribute("href", hash);
    if (app.route === "card" && location.hash !== hash) history.replaceState(null, "", hash);
  }

  async function extract(card, { force = false } = {}) {
    const llm = wantLlm();
    const key = resultKey(card, llm);
    st.ctrl?.abort(); // прежнее извлечение больше не нужно
    dropCompose(); // экран собирался для прежнего результата: ответ ляжет в кэш, но не на экран
    const req = ++st.req;
    const hit = force ? null : cache.extract.get(key);
    st.card = card;
    st.loading = !hit;
    st.error = null;
    st.composeError = null;
    st.pendingIntent = null;
    st.located = new Map();
    st.text = cardText(card);
    const ck = cardKey(card);
    if (!cache.decisions.has(ck)) cache.decisions.set(ck, new Map());
    st.decisions = cache.decisions.get(ck);
    st.filter = null;
    syncHash(card);
    paintDoc();
    if (hit) {
      paintActions();
      apply(hit, key, llm);
      return;
    }
    // старый результат до появления скелетона не должен принимать решения за новую карточку
    for (const el of body.children) el.inert = true;
    paintActions();
    const ctrl = (st.ctrl = new AbortController());
    // скелетон только если ответ задерживается: без LLM извлечение занимает миллисекунды и мигать не нужно
    const slow = setTimeout(() => req === st.req && st.loading && paintLoading("extract"), 160);
    try {
      const res = await post("/api/extract", { card, use_llm: llm }, { timeout: 120000, signal: ctrl.signal });
      const entry = { result: res.result, text: res.text || null, ui: res.ui };
      cache.extract.set(key, entry);
      if (req !== st.req) return;
      st.loading = false;
      apply(entry, key, llm);
    } catch (e) {
      if (req !== st.req) return;
      st.loading = false;
      st.error = e;
      if (st.resultCard !== st.card) {
        // выдача прежней карточки не должна стать основой экрана для этой
        st.result = null;
        st.key = null;
        st.resultCard = null;
      }
      paintError();
    } finally {
      clearTimeout(slow);
      if (req === st.req) {
        st.loading = false;
        st.ctrl = null;
        stopTicker();
        paintActions();
        sheet.classList.remove("is-loading");
      }
    }
  }

  /** Показать результат извлечения (свежий или из кэша) и последний собранный для него экран. */
  function apply(entry, key, llm) {
    st.result = entry.result;
    st.resultCard = st.card;
    st.text = entry.text || cardText(st.card);
    st.defaultSpec = entry.ui;
    st.key = key;
    st.llmMode = llm;
    st.located = locateAll(st.text.full, st.result.attributes);
    restoreScreen();
    paintDoc();
    paintHead();
    paintBody();
    const n = st.result.attributes.filter((a) => a.status === "review").length;
    announce(`Извлечено ${fmt.count(st.result.attributes.length, "характеристика", "характеристики", "характеристик")}, на проверку ${n}`
      + (st.intent ? `. Экран «${st.intent}» восстановлен` : ""));
    if (st.pendingIntent) {
      const intent = st.pendingIntent;
      st.pendingIntent = null;
      runIntent(intent);
    }
  }

  function restoreScreen() {
    const intent = cache.last.get(st.key);
    const hit = intent ? cache.screens.get(screenKey(st.key, intent)) : null;
    if (hit) setScreen(intent, hit, true);
    else {
      st.spec = st.defaultSpec;
      st.intent = null;
      st.specWarning = null;
      st.screenAt = null;
      st.restored = false;
    }
  }

  function setScreen(intent, entry, restored) {
    st.intent = intent;
    st.spec = entry.spec;
    st.specWarning = entry.warning || null;
    st.screenAt = entry.at || null;
    st.restored = restored;
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
    put(sheet, meta,
      c.demo_note ? h("p", { class: "doc__demo" }, icon("lightning", 14), h("span", null, h("strong", null, "Демо. "), c.demo_note)) : null,
      st.doc.el,
      st.synthetic && !c.demo ? h("p", { class: "doc__note" }, icon("info", 14), "Синтетический пример из data/sample/cards.jsonl. Реальные карточки появятся после подготовки сэмпла (real_cards.jsonl).") : null);
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
        if (st.llmAuto) {
          st.useLlm = false;
          st.llmAuto = false;
          paintLlm();
        }
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
      ? [st.llmAuto && st.useLlm ? h("strong", null, "Включено для демо-карточки. ") : null,
        "Только для характеристик, которые не закрыли продавец, правила и ML. Значение без цитаты из текста отбрасывается."]
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

  /** Ход гибрида по этой карточке: какие источники сработали, сколько добавили ML и LLM, что не понадобилось. */
  function trace(r, tm) {
    const found = r.attributes.filter((a) => a.value != null && a.sources?.length);
    const origin = (a) => PIPELINE.find((src) => a.sources.includes(src));
    const n = (src) => found.filter((a) => a.sources.includes(src)).length;
    const added = (src) => found.filter((a) => origin(a) === src).length;
    const confirmed = (src) => found.filter((a) => a.sources.includes(src) && origin(a) !== src).length;
    const hl = app.health || {};
    const vals = (k) => fmt.count(k, "значение", "значения", "значений");
    const stage = (key, label, state, text, tip) => h("li", { class: "trace__stage trace__stage--" + state, "data-tip": tip + (tm[key] != null ? " Время: " + fmt.ms(tm[key]) + "." : ""), "data-tip-title": label },
      h("span", { class: "trace__label" }, label), h("span", { class: "trace__val" }, text));
    // «+1 значение, подтвердила 2, отброшено 1»: вклад модели виден прямо в строке
    const contribution = (add, conf, rej = 0) => [
      add ? h("strong", { class: "trace__plus" }, "+" + vals(add)) : null,
      [conf ? "подтвердила " + fmt.int(conf) : null, rej ? "отброшено " + fmt.int(rej) : null].filter(Boolean).map((t, i) => (add || i ? ", " : "") + t),
    ];
    const hasSeller = Object.keys(st.card?.seller_attrs || {}).length > 0;
    const others = (k) => fmt.count(k, "значение", "значения", "значений") + " других источников";
    const sentence = (parts) => parts.filter(Boolean).join(", ") + ".";

    const mlAdd = added("ml");
    const mlConf = confirmed("ml");
    const ml = !hl.ml_model ? stage("ml", "ML", "off", "нет модели", "NER-модель не обучена в этом окружении (attrx train-ner).")
      : mlAdd || mlConf ? stage("ml", "ML", "used", contribution(mlAdd, mlConf),
        "NER разметил хвост: " + sentence([mlAdd ? `добавил ${vals(mlAdd)}, которых не нашли продавец и правила` : null, mlConf ? `подтвердил ${others(mlConf)}` : null])
        + " Каждое значение заземлено цитатой.")
        : tm.ml != null ? stage("ml", "ML", "idle", "ничего не добавила", "NER разметил непокрытые характеристики, но заземлённых значений не нашёл.")
          : stage("ml", "ML", "idle", "не понадобилась", "Продавец и правила закрыли всё уверенно: NER не запускался.");

    const llmAdd = added("llm");
    const llmConf = confirmed("llm");
    const llmRej = (r.warnings || []).filter((w) => LLM_REJECTED.test(w)).length;
    // ошибка вызова — предупреждение «llm: …» без ответа; отброшенные значения — это ответ, а не ошибка
    const llmFailed = !r.llm_used && (r.warnings || []).some((w) => /^llm: /u.test(w) && !LLM_REJECTED.test(w));
    const llm = !hl.llm_enabled ? stage("llm", "LLM", "off", "выключена", "LLM не подключена, дорогой шаг не вызывается.")
      : r.llm_used ? (llmAdd || llmConf || llmRej
        ? stage("llm", "LLM", llmAdd || llmConf ? "used" : "idle", contribution(llmAdd, llmConf, llmRej),
          "LLM: " + sentence([llmAdd ? `дозаполнила ${vals(llmAdd)} (значение только от LLM всегда уходит на проверку)` : null,
            llmConf ? `подтвердила ${others(llmConf)}` : null,
            llmRej ? `отброшено предложений: ${fmt.int(llmRej)}, их нет в тексте карточки` : null])
          + " Каждое принятое значение проверено цитатой.")
        : stage("llm", "LLM", "idle", "ничего не добавила", "LLM спросили про непокрытые характеристики, но в тексте карточки их нет."))
        : llmFailed ? stage("llm", "LLM", "idle", "не ответила", "Вызов LLM не удался: значения взяты из дешёвых источников. Подробности в блоке «Спорные и отброшенные значения».")
          : tm.llm != null ? stage("llm", "LLM", "idle", "не понадобилась", "Непокрытые характеристики в тексте карточки не упоминаются: спрашивать LLM не о чем, вызова не было.")
            : st.llmMode ? stage("llm", "LLM", "idle", "не понадобилась", "Дешёвые методы закрыли всё уверенно, вызов LLM не потребовался.")
              : stage("llm", "LLM", "idle", "не включена", "Включите «Дозаполнить через LLM» слева.");

    const stages = [
      n("seller") ? stage("seller", "Продавец", "used", vals(n("seller")), "Значения из полей продавца.")
        : stage("seller", "Продавец", "idle", hasSeller ? "не совпало" : "полей нет", hasSeller ? "Поля продавца есть, но не сопоставились с атрибутами категории." : "Продавец не заполнил блок характеристик."),
      stage("rules", "Правила", n("rules") ? "used" : "idle", n("rules") ? vals(n("rules")) : "ничего", "Регулярные выражения и справочники работают всегда."),
      ml,
      llm,
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
    stopTicker(); // панель хода запроса заменяется результатом
    paintGenbar();
    renderSpec(body, st.spec, st.result, ctx());
    body.setAttribute("aria-busy", "false");
  }

  function paintGenbar() {
    genbar.classList.toggle("genbar--error", Boolean(st.composeError));
    if (st.composeError) {
      const { intent, message } = st.composeError;
      genbar.hidden = false;
      put(genbar,
        h("div", { class: "genbar__text", role: "alert" }, icon("warning-circle", 16),
          h("span", null, h("strong", null, "Экран по запросу «" + intent + "» не собран"), h("span", { class: "genbar__how" }, message + ". Показан прежний экран."))),
        h("div", { class: "genbar__actions" },
          h("button", { class: "btn btn--quiet btn--sm", type: "button", onclick: () => runIntent(intent, { force: true }) }, icon("arrow-clockwise", 14), "Повторить"),
          h("button", { class: "btn btn--ghost btn--sm", type: "button", onclick: () => { st.composeError = null; paintGenbar(); } }, "Скрыть")));
      return;
    }
    genbar.hidden = !st.intent;
    if (!st.intent) return;
    const byLlm = st.spec.generated_by === "llm";
    const when = st.restored && st.screenAt
      ? " · восстановлен без запроса, собран в " + new Date(st.screenAt).toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" })
      : "";
    put(genbar,
      h("div", { class: "genbar__text" }, icon("list-checks", 16),
        h("span", null, h("strong", null, "Экран по запросу «" + st.intent + "»"),
          h("span", { class: "genbar__how" }, (byLlm ? "собран LLM из безопасных компонентов" : st.specWarning || "собран по ключевым словам") + when))),
      h("div", { class: "genbar__actions" },
        h("button", { class: "btn btn--ghost btn--sm", type: "button", "data-tip": "Собрать этот экран заново, не глядя в кэш", onclick: () => runIntent(st.intent, { force: true }) },
          icon("arrow-clockwise", 14), "Пересобрать"),
        h("button", { class: "btn btn--ghost btn--sm", type: "button", onclick: resetSpec }, icon("arrow-u-up-left", 14), "Стандартный вид")));
  }

  /** Кнопки в шапке результата: пока идёт запрос, нажать повторно нельзя, на кнопке — индикатор. */
  function paintActions() {
    const composing = Boolean(st.composing);
    composeBtn.disabled = st.loading || composing;
    composeBtn.classList.toggle("is-busy", composing);
    composeBtn.setAttribute("aria-busy", String(composing));
    put(composeBtn, composing ? h("span", { class: "spin spin--sm", "aria-hidden": "true" }) : icon("list-checks", 14),
      composing ? "Собираем экран…" : "Собрать экран", composing ? null : kbd(app.isMac ? "⌘" : "Ctrl", "K"));
    rerunBtn.disabled = st.loading;
    rerunBtn.classList.toggle("is-busy", st.loading);
    rerunBtn.setAttribute("aria-busy", String(st.loading));
  }

  // ---------- ход запроса: этапы, время, отмена ----------
  let ticker = 0;
  function stopTicker() {
    clearInterval(ticker);
    ticker = 0;
  }

  /** Этапы с примерным началом (мс от старта). Сервер отвечает одним ответом, поэтому ход примерный:
   *  дешёвые шаги проходят быстро, дальше индикатор стоит на самом долгом шаге до ответа. */
  function busySteps(kind) {
    if (kind === "compose") {
      const llm = Boolean(app.health?.llm_enabled);
      return [["Метаданные карточки", 0], [llm ? "LLM выбирает компоненты" : "Подбор по словам", 300], ["Проверка схемы", Infinity]];
    }
    const llm = st.llmPending;
    return [["Категория", 0], ["Продавец и правила", 250], app.health?.ml_model ? ["ML", 600] : null,
      llm ? ["LLM", 1100] : null, ["Собираем экран", llm ? Infinity : 1500]].filter(Boolean);
  }

  function busyPanel(kind) {
    const compose = kind === "compose";
    const llm = compose ? Boolean(app.health?.llm_enabled) : st.llmPending;
    const steps = busySteps(kind);
    const t0 = performance.now();
    const time = h("span", { class: "busy__time", "aria-hidden": "true" });
    const items = steps.map(([label]) => h("li", { class: "busy__step" }, h("span", { class: "busy__mark" }), h("span", null, label)));
    const title = compose ? "Собираем экран по запросу «" + st.composing.intent + "»" : "Извлекаем характеристики";
    const hint = compose
      ? (llm ? "LLM получает только метаданные (ключи, статусы, уверенность) и выбирает компоненты из фиксированного словаря. Обычно 5–20 с."
        : "Компоненты подбираются по ключевым словам запроса.")
      : (llm ? "С LLM обычно 3–10 с: модель дозаполняет только то, что не закрыли продавец, правила и ML, и каждое значение проверяется цитатой."
        : "Обычно меньше секунды.");
    const cancel = () => (compose ? st.composeCtrl?.abort() : st.ctrl?.abort());
    const el = h("section", { class: "busy", "aria-label": title },
      h("div", { class: "busy__head" },
        h("span", { class: "spin", "aria-hidden": "true" }),
        h("p", { class: "busy__title", role: "status" }, title + "…"),
        time,
        h("button", { class: "btn btn--ghost btn--sm", type: "button", onclick: cancel }, icon("x", 14), "Отменить")),
      h("ol", { class: "busy__steps", "aria-label": "Этапы", "data-tip": compose ? null : "Этапы показаны примерно: сервер присылает результат одним ответом. Точное время каждого этапа появится в строке над таблицей." }, items),
      h("p", { class: "busy__hint" }, hint));
    let shown = -1;
    const tick = () => {
      const ms = performance.now() - t0;
      if (ms >= 1000) time.textContent = fmt.int(Math.floor(ms / 1000)) + " с";
      let active = 0;
      steps.forEach(([, at], i) => { if (ms >= at) active = i; });
      if (active === shown) return;
      shown = active;
      items.forEach((li, i) => {
        li.className = "busy__step" + (i < active ? " is-done" : i === active ? " is-active" : "");
        put(li.firstChild, i < active ? icon("check", 12) : i === active ? h("span", { class: "spin spin--xs" }) : null);
        if (i === active) li.setAttribute("aria-current", "step");
        else li.removeAttribute("aria-current");
      });
    };
    tick();
    stopTicker();
    ticker = setInterval(tick, 250);
    return el;
  }

  /** Скелетон таблицы характеристик: та же сетка, что у настоящей, мерцание выключается при reduced-motion. */
  function skeleton() {
    const widths = [["58%", "44%"], ["72%", "30%"], ["46%", "52%"], ["64%", "38%"], ["52%", "60%"], ["68%", "26%"]];
    return h("div", { class: "skel-block", "aria-hidden": "true" },
      h("div", { class: "skel-sum" }, h("span", { class: "skel skel--line" }),
        h("div", { class: "skel-sum__row" }, [0, 1, 2].map(() => h("span", { class: "skel skel--pill" })))),
      h("div", { class: "ahead skel-ahead" }, ["Характеристика", "Значение", "Уверенность", "Статус", "Источник"].map((t) => h("span", null, t))),
      widths.map(([a, b]) => h("div", { class: "skel-arow" },
        h("span", { class: "skel skel--text", style: { width: a } }), h("span", { class: "skel skel--text", style: { width: b } }),
        h("span", { class: "skel skel--bar" }), h("span", { class: "skel skel--text", style: { width: "70%" } }), h("span", { class: "skel skel--chip" }))));
  }

  function paintLoading(kind) {
    body.setAttribute("aria-busy", "true");
    genbar.hidden = true;
    if (kind === "extract") {
      st.llmPending = wantLlm();
      sheet.classList.add("is-loading");
      // результат другой карточки в шапке вводил бы в заблуждение
      if (st.resultCard !== st.card) {
        put(crumbs, h("span", { class: "skel skel--text", style: { width: "220px", height: "22px" } }));
        put(metaRow, h("span", { class: "skel skel--text", style: { width: "300px" } }));
      }
    }
    put(body, busyPanel(kind), skeleton());
  }

  function paintError() {
    stopTicker();
    body.setAttribute("aria-busy", "false");
    sheet.classList.remove("is-loading");
    genbar.hidden = true;
    if (st.resultCard !== st.card) {
      put(crumbs);
      put(metaRow);
    }
    const aborted = Boolean(st.error?.aborted);
    put(body, h("div", { class: "alert alert--" + (aborted ? "info" : "danger"), role: "alert" }, icon(aborted ? "info" : "warning-circle", 16, "alert__icon"),
      h("div", { class: "alert__text" },
        h("strong", null, aborted ? "Извлечение остановлено. " : "Не удалось извлечь характеристики. "),
        aborted ? "Запрос отменён, результат не показан." : st.error?.message || "",
        h("div", { class: "alert__actions" },
          h("button", { class: "btn btn--quiet btn--sm", type: "button", onclick: () => st.card && extract(st.card) }, icon("arrow-clockwise", 14), "Повторить"),
          wantLlm() ? h("button", { class: "btn btn--ghost btn--sm", type: "button", onclick: () => { st.useLlm = false; st.llmAuto = false; paintLlm(); if (st.card) extract(st.card); } }, "Без LLM") : null))));
  }

  function paintNoExamples() {
    put(sheet, h("div", { class: "empty empty--sheet" }, icon("file-text", 28, "empty__icon"),
      h("h2", { class: "empty__title" }, "Примеров карточек нет"),
      h("p", { class: "empty__text" }, "Положите карточки в data/sample/real_cards.jsonl или вставьте свою."),
      h("button", { class: "btn btn--primary", type: "button", onclick: () => openEditor(false) }, "Своя карточка")));
  }

  /** Прекратить показ сборки экрана. Запрос не обрывается: ответ ляжет в кэш и откроется, когда вернётесь. */
  function dropCompose() {
    st.composeReq++;
    st.composing = null;
    st.composeCtrl = null;
    paintActions();
  }

  /** Собрать экран под запрос. Готовый экран для этой карточки, режима LLM и запроса берётся из кэша;
   *  force — «Пересобрать», мимо кэша. Пока идёт извлечение, запрос ждёт его конца. */
  async function runIntent(raw, { force = false } = {}) {
    const intent = String(raw || "").trim();
    if (!intent) return;
    if (st.loading) {
      st.pendingIntent = intent;
      toast("Экран «" + intent + "» соберётся сразу после извлечения.");
      return;
    }
    if (!st.result) {
      toast("Сначала выберите карточку: экран собирается под её характеристики.");
      return;
    }
    const key = st.key;
    const sk = screenKey(key, intent);
    st.composeError = null;
    const hit = force ? null : cache.screens.get(sk);
    if (hit) {
      st.composeCtrl?.abort();
      dropCompose();
      cache.last.set(key, intent);
      setScreen(intent, hit, true);
      paintBody();
      announce("Экран «" + intent + "» открыт без нового запроса");
      return;
    }
    st.composeCtrl?.abort(); // новый запрос заменяет прежний для этой же карточки
    const token = ++st.composeReq;
    const ctrl = (st.composeCtrl = new AbortController());
    st.composing = { intent };
    paintActions();
    const slow = setTimeout(() => token === st.composeReq && st.composing && paintLoading("compose"), 160);
    try {
      const res = await post("/api/ui", { result: st.result, intent }, { timeout: 90000, signal: ctrl.signal });
      const entry = { spec: res.ui, warning: res.ui_warning, at: Date.now() };
      cache.screens.set(sk, entry);
      cache.last.set(key, intent);
      if (token !== st.composeReq || st.key !== key) return; // ушли на другую карточку: экран ждёт в кэше
      setScreen(intent, entry, false);
      announce("Экран «" + intent + "» собран");
    } catch (e) {
      if (token !== st.composeReq || st.key !== key) return;
      if (e.aborted) announce("Сборка экрана отменена");
      else st.composeError = { intent, message: e.message };
    } finally {
      clearTimeout(slow);
      if (token === st.composeReq) {
        stopTicker();
        st.composing = null;
        st.composeCtrl = null;
        paintActions();
        if (st.key === key) {
          paintBody();
          (body.querySelector(".blk__title") || genbar).scrollIntoView({ block: "nearest" });
        }
      }
    }
  }

  function resetSpec() {
    st.spec = st.defaultSpec;
    st.intent = null;
    st.specWarning = null;
    st.restored = false;
    cache.last.delete(st.key);
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
  rerunBtn.addEventListener("click", () => st.card && extract(st.card, { force: true }));
  llmInput.addEventListener("change", () => {
    st.useLlm = llmInput.checked;
    st.llmAuto = false;
    paintLlm();
    if (st.card && !st.editing) extract(st.card);
  });

  function rows() {
    return [...body.querySelectorAll(".attrs")].flatMap((l) => l._rows || []);
  }

  return {
    async show(params) {
      paintLlm();
      if (!st.ready) {
        await init(params);
        return;
      }
      const same = (c) => c && String(c.id) === String(params.id) && Boolean(c.demo) === Boolean(params.demo);
      if (!params.id || same(st.card)) {
        // вернулись из другой вкладки: карточка, результат и собранный экран остались как были
        if (st.card && app.route === "card") syncHash(st.card);
        return;
      }
      const i = st.items.findIndex(same);
      if (i >= 0) {
        select(i);
        return;
      }
      st.ready = false;
      await init(params);
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
