// Справка «Как читать экран»: статусы, источники, уверенность, цитаты, горячие клавиши, подключение LLM.
import { h, icon, put } from "./dom.js";
import { CONFIDENCE_TIP, SHORTCUTS, SOURCE, STATUS, t } from "./copy.js";
import { meter, sourceTags, statusMark } from "./attrs.js";
import { isMac, kbd } from "./ui.js";

export function initHelp(app) {
  const dialog = document.getElementById("help");
  const body = h("div", { class: "drawer__body" });
  dialog.append(h("div", { class: "drawer__box" },
    h("header", { class: "drawer__head" },
      h("h2", { class: "drawer__title", id: "help-title" }, "Как читать экран"),
      h("button", { class: "icon-btn", type: "button", "aria-label": "Закрыть справку", onclick: () => dialog.close() }, icon("x", 18))),
    body));
  dialog.addEventListener("click", (e) => e.target === dialog && dialog.close());

  function paint() {
    const thr = app.threshold;
    const llm = app.health || {};
    put(body, 
      h("p", { class: "help__lede" }, "attrx читает текст карточки (название, описание, поля продавца), находит характеристики, приводит их к виду справочника и для каждого значения показывает уверенность, источник и цитату из текста."),

      sec("help-status", "Статусы значения",
        Object.keys(STATUS).map((k) => row(statusMark(k, thr, { withTip: false }), t(STATUS[k].tip, thr)))),

      sec("help-conf", "Уверенность",
        row(meter(0.82, thr, "accepted"), "Шкала от 0 до 1 и число. " + t(CONFIDENCE_TIP, thr).replace("Уверенность от 0 до 1. ", "")),
        row(meter(0.55, thr, "review"), "Ниже засечки значение не принимается автоматически. Насколько уверенность совпадает с реальной точностью, видно на графике калибровки во вкладке «Качество».")),

      sec("help-src", "Источники значения",
        Object.keys(SOURCE).map((k) => row(sourceTags([k]), SOURCE[k].tip)),
        h("p", { class: "help__p" }, "Порядок дешёвый → дорогой: продавец и правила работают всегда, ML и LLM подключаются только для характеристик, которые не закрыты уверенно.")),

      sec("help-quote", "Цитаты и подсветка",
        h("p", { class: "help__p" }, "Наведите на характеристику (или выберите её клавишами J и K), и фрагмент текста, из которого взято значение, подсветится ", h("mark", { class: "ev is-active" }, "красным маркером"), ". Наведите на подсвеченный фрагмент в тексте, и подсветится характеристика."),
        h("p", { class: "help__p" }, "Текст карточки набран книжным шрифтом: так везде отмечены слова самого продавца, а не интерфейса. Значения LLM без цитаты в тексте отбрасываются, это видно в блоке «Спорные и отброшенные значения».")),

      sec("help-loop", "Решения и обратная связь",
        h("p", { class: "help__p" }, "Кнопки «Принять», «Исправить», «Отклонить» записывают решение в outputs/feedback.jsonl. Команда attrx learn превращает исправления в синонимы справочников, и в следующий раз такие значения проходят автоматически. Решение можно отменить.")),

      sec("help-genui", "Собрать экран под задачу",
        h("p", { class: "help__p" }, "В палитре ", kbd(isMac ? "⌘" : "Ctrl", "K"), " опишите, что нужно увидеть, например «только спорные значения с цитатами». LLM выбирает компоненты из фиксированного набора и не пишет HTML, а данные подставляются из результата извлечения. Без LLM экран собирается по ключевым словам запроса.")),

      sec("help-keys", "Горячие клавиши",
        h("table", { class: "keys" }, h("tbody", null, SHORTCUTS.map((s) => h("tr", null,
          h("td", { class: "keys__k" }, kbd(...(isMac && s.mac ? s.mac : s.keys)), s.alt ? [" или ", kbd(...s.alt)] : null),
          h("td", null, s.label)))))),

      sec("help-llm", "LLM",
        llm.llm_enabled
          ? h("p", { class: "help__p" }, "Подключена модель ", h("strong", null, llm.llm_model || ""), " на ", llm.llm_host || "", ". Она вызывается только для непокрытых характеристик, ответы кэшируются.")
          : h("p", { class: "help__p" }, "Сейчас LLM не подключена, работают продавец, правила и ML. Чтобы подключить OpenAI-совместимую модель (в том числе локальную, данные не уходят из контура), задайте в .env переменные LLM_BASE_URL, LLM_MODEL и LLM_API_KEY (или OPENAI_BASE_URL и OPENAI_API_KEY) и перезапустите attrx serve.")));
  }

  return {
    open(sectionId) {
      paint();
      if (!dialog.open) dialog.showModal();
      if (sectionId) dialog.querySelector("#" + sectionId)?.scrollIntoView({ block: "start" });
      dialog.querySelector(".drawer__head .icon-btn")?.focus();
    },
    isOpen: () => dialog.open,
  };
}

function sec(id, title, ...content) {
  return h("section", { class: "help__sec", id, "aria-labelledby": id + "-t" }, h("h3", { class: "help__title", id: id + "-t" }, title), content);
}

function row(visual, text) {
  return h("div", { class: "help__row" }, h("div", { class: "help__vis" }, visual), h("p", { class: "help__text" }, text));
}
