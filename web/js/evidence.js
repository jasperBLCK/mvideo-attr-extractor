// Цитаты в тексте карточки. Каждое значение связано со словами, из которых оно взято:
// evidence_span [start, end) указывает в Card.full_text(); если его нет, ищем саму цитату в тексте
// (без учёта регистра, ё = е, любые пробелы). Найденные фрагменты размечаются <mark class="ev">.
import { h, reducedMotion } from "./dom.js";

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

function quoteRegex(q, bounded) {
  const t = String(q).trim();
  const words = t.split(/\s+/).filter(Boolean).map((w) => escapeRe(w).replace(/[её]/giu, "[её]"));
  if (!words.length) return null;
  // границы слова: цитата «2» не должна находиться внутри «42 дБ»
  const pre = bounded && /^[\p{L}\p{N}]/u.test(t) ? "(?<![\\p{L}\\p{N}])" : "";
  const post = bounded && /[\p{L}\p{N}]$/u.test(t) ? "(?![\\p{L}\\p{N}])" : "";
  return new RegExp(pre + words.join("\\s+") + post, "iu");
}

/** {start, end, exact} или null. exact = смещения пришли из пайплайна, а не найдены по тексту цитаты. */
export function locate(full, a) {
  const span = a.evidence_span;
  if (Array.isArray(span) && span.length === 2) {
    const [s, e] = span;
    if (Number.isInteger(s) && Number.isInteger(e) && s >= 0 && s < e && e <= full.length) return { start: s, end: e, exact: true };
  }
  if (!a.evidence) return null;
  let m = quoteRegex(a.evidence, true)?.exec(full);
  if (!m && String(a.evidence).trim().length >= 4) m = quoteRegex(a.evidence, false)?.exec(full);
  return m ? { start: m.index, end: m.index + m[0].length, exact: false } : null;
}

/** Формулировка значения для обратной связи (POST /api/feedback, поле raw): фрагмент evidence_span, а у поля
 *  продавца — только значение без «Ключ:». attrx learn делает из неё синоним справочника. */
export function evidenceValue(text, a) {
  const full = text?.full || "";
  const loc = full ? locate(full, a) : null;
  if (!loc) return a.evidence || null;
  const pair = (text.seller_pairs || []).find((p) => loc.start <= p.value_span[0] && loc.end >= p.value_span[1] && loc.start >= p.key_span[0]);
  return pair ? pair.value : full.slice(loc.start, loc.end);
}

export function locateAll(full, attrs) {
  const out = new Map();
  for (const a of attrs) {
    if (a.status === "missing" && !a.evidence) continue;
    const loc = locate(full, a);
    if (loc) out.set(a.attribute, loc);
  }
  return out;
}

/** Разбиение текста на отрезки с набором атрибутов, чьи цитаты их покрывают. */
function segments(len, ranges) {
  const cuts = new Set([0, len]);
  for (const r of ranges) {
    cuts.add(r.start);
    cuts.add(r.end);
  }
  const pts = [...cuts].sort((a, b) => a - b);
  const out = [];
  for (let i = 0; i < pts.length - 1; i++) {
    const s = pts[i];
    const e = pts[i + 1];
    if (s >= e) continue;
    out.push({ start: s, end: e, attrs: ranges.filter((r) => r.start <= s && r.end >= e).map((r) => r.attr) });
  }
  return out;
}

/** Текст карточки на клиенте, если сервер не прислал разметку (например, экран из очереди). */
export function cardText(card) {
  const seller = Object.entries(card.seller_attrs || {}).map(([k, v]) => k + ": " + v).join("; ");
  const parts = [["title", card.title], ["description", card.description], ["seller", seller]].filter(([, v]) => v);
  let pos = 0;
  const blocks = [];
  for (const [kind, v] of parts) {
    blocks.push({ kind, start: pos, end: pos + v.length });
    pos += v.length + 1;
  }
  return { full: parts.map(([, v]) => v).join("\n"), blocks, seller_pairs: [] };
}

/** Короткий фрагмент вокруг цитаты: «…до [цитата] после…». */
export function snippet(full, loc, radius = 64) {
  let s = Math.max(0, loc.start - radius);
  let e = Math.min(full.length, loc.end + radius);
  if (s > 0) {
    const sp = full.indexOf(" ", s);
    if (sp > -1 && sp < loc.start) s = sp + 1;
  }
  if (e < full.length) {
    const sp = full.lastIndexOf(" ", e);
    if (sp > loc.end) e = sp;
  }
  const clean = (t) => t.replace(/\s+/g, " ");
  return h("span", { class: "quote" },
    s > 0 ? "…" : null,
    clean(full.slice(s, loc.start)),
    h("mark", { class: "ev is-active" }, clean(full.slice(loc.start, loc.end))),
    clean(full.slice(loc.end, e)),
    e < full.length ? "…" : null);
}

/**
 * Рендер карточки как документа с размеченными цитатами.
 * Возвращает ручку: highlight(attr) подсвечивает цитату и прокручивает к ней, onHover(cb) — обратная связь.
 */
export function renderDocument({ card, text, located, titleTag = "h2" }) {
  const full = text.full;
  const ranges = [...located.entries()].map(([attr, l]) => ({ attr, start: l.start, end: l.end }));
  const segs = segments(full.length, ranges);
  const marks = new Map();

  const piece = (s, e) => {
    const out = [];
    for (const seg of segs) {
      if (seg.end <= s || seg.start >= e) continue;
      const a = Math.max(s, seg.start);
      const b = Math.min(e, seg.end);
      const str = full.slice(a, b);
      if (!seg.attrs.length) {
        out.push(str);
        continue;
      }
      const m = h("mark", { class: "ev", dataset: { attrs: seg.attrs.join(" ") } }, str);
      for (const at of seg.attrs) {
        if (!marks.has(at)) marks.set(at, []);
        marks.get(at).push(m);
      }
      out.push(m);
    }
    return out;
  };

  const blocks = text.blocks?.length ? text.blocks : [{ kind: "description", start: 0, end: full.length }];
  const body = [];
  for (const b of blocks) {
    if (b.kind === "title") {
      body.push(h(titleTag, { class: "doc__title" }, piece(b.start, b.end)));
    } else if (b.kind === "description") {
      body.push(h("div", { class: "doc__body" }, piece(b.start, b.end)));
    } else if (b.kind === "seller") {
      const pairs = text.seller_pairs || [];
      body.push(h("section", { class: "doc__seller", "aria-label": "Поля продавца" },
        h("h3", { class: "doc__seller-title" }, "Поля продавца"),
        pairs.length
          ? h("dl", { class: "kv" }, pairs.map((p) => h("div", { class: "kv__row" },
            h("dt", { class: "kv__k" }, piece(p.key_span[0], p.key_span[1])),
            h("dd", { class: "kv__v" }, piece(p.value_span[0], p.value_span[1])))))
          : h("p", { class: "doc__plain" }, piece(b.start, b.end))));
    }
  }

  const root = h("div", { class: "doc__text" }, body);
  let active = null;
  let hoverCb = null;

  root.addEventListener("pointerover", (e) => {
    const m = e.target instanceof Element ? e.target.closest("mark.ev") : null;
    if (m && hoverCb) hoverCb(m.dataset.attrs.split(" "));
  });
  root.addEventListener("pointerleave", () => hoverCb && hoverCb(null));

  return {
    el: root,
    marks,
    /** Подсветить цитату атрибута. scroll: прокрутить контейнер так, чтобы цитата была видна. */
    highlight(attr, { scroll = true, scroller = null } = {}) {
      if (active === attr) return;
      root.querySelectorAll("mark.ev.is-active").forEach((m) => m.classList.remove("is-active"));
      root.classList.toggle("has-active", Boolean(attr && marks.has(attr)));
      active = attr;
      const list = attr ? marks.get(attr) : null;
      if (!list) return;
      list.forEach((m) => m.classList.add("is-active"));
      if (scroll) scrollToMark(list[0], scroller);
    },
    clear() {
      this.highlight(null);
    },
    onHover(cb) {
      hoverCb = cb;
    },
    has: (attr) => marks.has(attr),
    first: (attr) => marks.get(attr)?.[0] || null,
  };
}

/** Прокрутка внутри своего контейнера (не всей страницы), только если цитата не видна.
 *  Учитывается только та часть контейнера, что попадает в окно: длинный лист может уходить за нижний край. */
export function scrollToMark(mark, scroller) {
  const box = scroller || mark.closest(".doc");
  if (!box) return;
  const mr = mark.getBoundingClientRect();
  const br = box.getBoundingClientRect();
  const visTop = Math.max(br.top, 0);
  const visBottom = Math.min(br.bottom, window.innerHeight);
  if (visBottom - visTop < 60) return;
  const pad = Math.min(48, (visBottom - visTop) / 4);
  if (mr.top >= visTop + pad && mr.bottom <= visBottom - pad) return;
  const target = visTop + (visBottom - visTop) / 3;
  const top = box.scrollTop + (mr.top - target);
  box.scrollTo({ top: Math.max(0, top), behavior: reducedMotion() ? "auto" : "smooth" });
}
