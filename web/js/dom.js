// Помощники DOM и форматирование чисел. Текст из данных выводится только через textContent:
// innerHTML с данными нигде не используется, поэтому значения из карточек не могут внедрить разметку.

export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  setAttrs(el, attrs);
  return append(el, children);
}

function setAttrs(el, attrs) {
  if (!attrs) return;
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k === "style" && typeof v === "object") {
      // кастомные свойства (--v, --t) задаются только через setProperty
      for (const [p, val] of Object.entries(v)) {
        if (p.startsWith("--")) el.style.setProperty(p, val);
        else el.style[p] = val;
      }
    }
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, String(v));
  }
}

export function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false || c === "") continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

const SVG_NS = "http://www.w3.org/2000/svg";

export function s(tag, attrs, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v);
      else if (k === "text") el.textContent = v;
      else el.setAttribute(k, String(v));
    }
  }
  return append(el, children);
}

/** Иконка из спрайта assets/icons.svg (Phosphor, MIT). Декоративная: aria-hidden. */
export function icon(name, size = 16, cls = "") {
  const svg = s("svg", { class: ("i " + cls).trim(), width: size, height: size, viewBox: "0 0 256 256",
    "aria-hidden": "true", focusable: "false" });
  svg.append(s("use", { href: "/static/assets/icons.svg#" + name }));
  return svg;
}

/** Замена содержимого. В отличие от replaceChildren, пропускает null/false (иначе в DOM попадает текст «null»). */
export function put(el, ...children) {
  el.replaceChildren();
  return append(el, children);
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function clear(el) {
  el.replaceChildren();
  return el;
}

export function debounce(fn, ms) {
  let t = 0;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

let uidN = 0;
export const uid = (p = "u") => p + "-" + (++uidN).toString(36);

export const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Клавиша без учёта раскладки: в русской раскладке J — это «о», поэтому смотрим на физическую клавишу (e.code). */
export function keyOf(e) {
  if (/^Key[A-Z]$/.test(e.code)) return e.code.slice(3).toLowerCase();
  if (/^Digit[0-9]$/.test(e.code)) return e.code.slice(5);
  return (e.key || "").toLowerCase();
}

/** Пользователь печатает в поле — горячие клавиши не перехватываем. */
export function isTyping(e) {
  const t = e.target;
  if (!(t instanceof HTMLElement)) return false;
  return t.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName);
}

// ---------- числа по-русски: десятичная запятая, неразрывные пробелы ----------
const nf = (min, max) => new Intl.NumberFormat("ru-RU", { minimumFractionDigits: min, maximumFractionDigits: max });
const NF0 = nf(0, 0);
const NF01 = nf(0, 1);
const NF2 = nf(2, 2);
const NF3 = nf(3, 3);
const PCT = new Intl.NumberFormat("ru-RU", { style: "percent", maximumFractionDigits: 1 });
const DATE = new Intl.DateTimeFormat("ru-RU", { day: "numeric", month: "long", hour: "2-digit", minute: "2-digit" });
const COMPACT = new Intl.NumberFormat("ru-RU", { notation: "compact", maximumFractionDigits: 1 });

const isNum = (x) => typeof x === "number" && Number.isFinite(x);

export const fmt = {
  conf: (x) => (isNum(x) ? NF2.format(x) : ""),
  p3: (x) => (isNum(x) ? NF3.format(x) : ""),
  pct: (x) => (isNum(x) ? PCT.format(x) : ""),
  int: (x) => (isNum(x) ? NF0.format(x) : ""),
  compact: (x) => (isNum(x) ? COMPACT.format(x) : ""),
  num: (x, d = 1) => (isNum(x) ? nf(0, d).format(x) : ""),
  ms(x) {
    if (!isNum(x)) return "";
    if (x < 10) return NF01.format(x) + " мс";
    if (x < 1000) return NF0.format(x) + " мс";
    return NF01.format(x / 1000) + " с";
  },
  date: (ts) => (isNum(ts) ? DATE.format(new Date(ts * 1000)) : ""),
  /** Склонение: plural(5, "значение", "значения", "значений") */
  plural(n, one, few, many) {
    const a = Math.abs(n) % 100;
    const b = a % 10;
    if (a > 10 && a < 20) return many;
    if (b === 1) return one;
    if (b >= 2 && b <= 4) return few;
    return many;
  },
  count: (n, one, few, many) => NF0.format(n) + " " + fmt.plural(n, one, few, many),
};

/** Отображаемое значение атрибута: display из пайплайна, иначе value + единица. */
export function shownValue(a) {
  if (a.display != null && a.display !== "") return String(a.display);
  if (a.value == null || a.value === "") return null;
  const v = typeof a.value === "number" ? fmt.num(a.value, 2) : String(a.value);
  return a.unit ? v + " " + a.unit : v;
}

export const capitalize = (t) => (t ? t.charAt(0).toUpperCase() + t.slice(1) : t);

/** Путь в каталоге без пустых и повторяющихся подряд уровней («Ноутбуки › Ноутбуки» → «Ноутбуки»). */
export const cleanPath = (path) => (path || []).filter((p, i, a) => p && p !== a[i - 1]);

/** Значение альтернативы: 512 и «512.0» → «512 ГБ», строки как есть. */
export function valueText(v, unit) {
  const str = String(v ?? "").trim();
  if (typeof v === "number" || /^-?\d+(\.\d+)?$/.test(str)) return fmt.num(Number(str), 2) + (unit ? "\u00a0" + unit : "");
  return str;
}
