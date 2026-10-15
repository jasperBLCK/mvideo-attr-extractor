// Общие слои интерфейса: подсказки, уведомления, озвучка для экранных дикторов, тема.
import { h, icon, reducedMotion, put } from "./dom.js";

// ---------- подсказки: [data-tip] (+ необязательный [data-tip-title]) ----------
let tipEl = null;
let tipFor = null;
let tipTimer = 0;

export function initTooltips() {
  tipEl = document.getElementById("tip");
  const target = (e) => (e.target instanceof Element ? e.target.closest("[data-tip]") : null);
  document.addEventListener("pointerover", (e) => {
    const t = target(e);
    if (!t || t === tipFor) return;
    clearTimeout(tipTimer);
    tipTimer = setTimeout(() => showTip(t), tipFor ? 0 : 480);
  });
  document.addEventListener("pointerout", (e) => {
    const t = target(e);
    if (t && !(e.relatedTarget instanceof Node && t.contains(e.relatedTarget))) hideTip();
  });
  document.addEventListener("focusin", (e) => {
    const t = target(e);
    if (t && t.matches(":focus-visible")) showTip(t);
  });
  document.addEventListener("focusout", () => hideTip());
  document.addEventListener("keydown", (e) => e.key === "Escape" && hideTip());
  document.addEventListener("scroll", () => hideTip(), { passive: true, capture: true });
}

function showTip(t) {
  if (!tipEl || !t.isConnected || !t.dataset.tip) return;
  tipFor = t;
  put(tipEl, 
    t.dataset.tipTitle ? h("strong", { class: "tip__title" }, t.dataset.tipTitle) : null,
    h("span", null, t.dataset.tip),
  );
  tipEl.hidden = false;
  const r = t.getBoundingClientRect();
  const w = tipEl.offsetWidth;
  const ht = tipEl.offsetHeight;
  let x = r.left + r.width / 2 - w / 2;
  x = Math.max(8, Math.min(x, innerWidth - w - 8));
  let y = r.top - ht - 8;
  if (y < 8) y = r.bottom + 8;
  tipEl.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
  t.setAttribute("aria-describedby", "tip");
}

export function hideTip() {
  clearTimeout(tipTimer);
  if (tipFor) tipFor.removeAttribute("aria-describedby");
  tipFor = null;
  if (tipEl) tipEl.hidden = true;
}

// ---------- озвучка изменений ----------
export function announce(text) {
  const live = document.getElementById("live");
  if (!live) return;
  live.textContent = "";
  requestAnimationFrame(() => (live.textContent = text));
}

// ---------- уведомления ----------
export function toast(message, { tone = "neutral", action, onAction, timeout = 5200 } = {}) {
  const root = document.getElementById("toasts");
  if (!root) return () => {};
  const ic = { success: "check-circle", danger: "warning-circle", neutral: "info" }[tone] || "info";
  let closed = false;
  const close = () => {
    if (closed) return;
    closed = true;
    el.classList.add("is-leaving");
    setTimeout(() => el.remove(), reducedMotion() ? 0 : 160);
  };
  const el = h("div", { class: "toast toast--" + tone },
    icon(ic, 16, "toast__icon"),
    h("span", { class: "toast__text" }, message),
    action ? h("button", { class: "toast__action", type: "button", onclick: () => { onAction?.(); close(); } }, action) : null,
    h("button", { class: "toast__close", type: "button", "aria-label": "Закрыть уведомление", onclick: close }, icon("x", 14)));
  root.append(el);
  if (timeout) setTimeout(close, timeout);
  return close;
}

// ---------- тема: системная / светлая / тёмная ----------
const THEME_KEY = "attrx.theme";
const THEME_LABEL = { system: "как в системе", light: "светлая", dark: "тёмная" };
const listeners = new Set();

export function getTheme() {
  try {
    const v = localStorage.getItem(THEME_KEY);
    return v === "light" || v === "dark" ? v : "system";
  } catch {
    return "system";
  }
}

export function setTheme(v) {
  try {
    if (v === "system") localStorage.removeItem(THEME_KEY);
    else localStorage.setItem(THEME_KEY, v);
  } catch {
    /* приватный режим: тема живёт до перезагрузки */
  }
  if (v === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = v;
  // цвет панели браузера следует за выбранной темой
  for (const m of document.querySelectorAll('meta[name="theme-color"]')) {
    if (!m.dataset.media) m.dataset.media = m.getAttribute("media") || "";
    if (v === "system") m.setAttribute("media", m.dataset.media);
    else m.setAttribute("media", m.dataset.media.includes(v) ? "all" : "not all");
  }
  listeners.forEach((fn) => fn(v));
}

export const themeLabel = (v) => THEME_LABEL[v] || THEME_LABEL.system;
export const onThemeChange = (fn) => listeners.add(fn);

export function nextTheme() {
  const order = ["system", "light", "dark"];
  const v = order[(order.indexOf(getTheme()) + 1) % order.length];
  setTheme(v);
  return v;
}

matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => listeners.forEach((fn) => fn(getTheme())));

export const isMac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);

export function kbd(...keys) {
  return h("span", { class: "kbds", translate: "no" }, keys.map((k) => h("kbd", { class: "kbd" }, k)));
}
