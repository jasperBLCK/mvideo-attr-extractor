// Кэш на клиенте: результаты извлечения и собранные экраны. Вернулись на карточку или вкладку —
// всё восстанавливается без нового запроса; «Извлечь заново» и «Пересобрать» идут мимо кэша.
// Map в памяти с вытеснением самых старых (LRU); persist — копия в sessionStorage, живёт до закрытия вкладки.

export function lru(name, { max = 50, persist = false } = {}) {
  const map = new Map();
  const key = "attrx.cache." + name;
  if (persist) {
    try {
      for (const [k, v] of JSON.parse(sessionStorage.getItem(key) || "[]")) map.set(k, v);
    } catch {
      /* приватный режим или битая запись: начинаем с пустого */
    }
  }
  const save = () => {
    if (!persist) return;
    try {
      sessionStorage.setItem(key, JSON.stringify([...map]));
    } catch {
      /* квота: кэш в памяти работает и без копии */
    }
  };
  return {
    get(k) {
      if (!map.has(k)) return undefined;
      const v = map.get(k);
      map.delete(k);
      map.set(k, v);
      return v;
    },
    set(k, v) {
      map.delete(k);
      map.set(k, v);
      while (map.size > max) map.delete(map.keys().next().value);
      save();
      return v;
    },
    delete(k) {
      map.delete(k);
      save();
    },
  };
}

/** Короткий отпечаток текста карточки: демо-карточка и реальная с тем же id (или правленый текст) не смешиваются. */
export function fingerprint(card) {
  const s = JSON.stringify([card?.title, card?.description, card?.seller_attrs, card?.category]);
  let x = 5381;
  for (let i = 0; i < s.length; i++) x = ((x << 5) + x + s.charCodeAt(i)) | 0;
  return (x >>> 0).toString(36);
}
