// Обращения к API attrx (src/attrx/api/app.py). Ошибки приводятся к понятному русскому тексту.

export class ApiError extends Error {
  constructor(message, { status = 0, detail = null, aborted = false } = {}) {
    super(message);
    this.status = status;
    this.detail = detail;
    this.aborted = aborted; // запрос отменили сами (signal), а не сервер или сеть
  }
}

function detailText(data) {
  if (!data) return "";
  if (typeof data === "string") return data.slice(0, 200);
  if (typeof data.detail === "string") return data.detail;
  if (Array.isArray(data.detail)) return data.detail.map((d) => d.msg).filter(Boolean).join("; ");
  return "";
}

/** signal — внешняя отмена (пользователь нажал «Отменить» или ушёл на другую карточку). */
export async function api(path, { method = "GET", body, timeout = 30000, signal } = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeout);
  const cancel = () => ctrl.abort();
  if (signal?.aborted) cancel();
  else signal?.addEventListener("abort", cancel, { once: true });
  try {
    const r = await fetch(path, {
      method,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: ctrl.signal,
    });
    const raw = await r.text();
    let data = null;
    try {
      data = raw ? JSON.parse(raw) : null;
    } catch {
      data = raw;
    }
    if (!r.ok) throw new ApiError(detailText(data) || "Сервер ответил с ошибкой " + r.status, { status: r.status, detail: data });
    return data;
  } catch (e) {
    if (e instanceof ApiError) throw e;
    if (signal?.aborted) throw new ApiError("Запрос отменён", { aborted: true });
    if (e.name === "AbortError") throw new ApiError("Сервер не ответил за " + Math.round(timeout / 1000) + " с");
    throw new ApiError("Нет связи с сервером. Проверьте, что запущен attrx serve");
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", cancel);
  }
}

export const get = (path, opts) => api(path, opts);
export const post = (path, body, opts) => api(path, { ...opts, method: "POST", body });
