"""Генерация справочников из каталога М.Видео (`attrx build-dicts`): воспроизводимо, только по train-сплиту.

Бренды (config/dictionaries/brands.yaml):
1. Канон — `vendor` офферов YML-фида train-сплита (md5(sku_id) % 100 < 80; dev/holdout не смотрим), бренды с
   >= min_offers офферов; написание — самое частое в каталоге («HONOR», «realme»). Заглушки («нет бренда», «No Name»,
   «###») — не бренды.
2. Синонимы: ручные (config/brand_aliases.yaml: кириллица «Самсунг», суббренды «Redmi» -> Xiaomi) + транслитерация:
   кириллический vendor, совпавший по транслиту с латинским («Атлант» ~ «ATLANT»), становится синонимом более частого.
3. Защита от омонимов — по названиям карточек train: бренд считается выбранным, если он первое упоминание бренда
   в названии вне контекста совместимости («для iPhone»). Бренды, которые при этом чаще ошибочны, чем верны
   («СМ» из «45 см», «Wireless» из «True Wireless», «Comfort», «Blue»), попадают в `_word` и в тексте не ищутся.
   Для брендов с заметной долей ошибок сохраняется `_prior` — точность (основа уверенности правила).
"""
from __future__ import annotations

import itertools
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from .taxonomy import BrandDictionary, compact_key, lower_keep, norm, pick_mention

PLACEHOLDER_BRANDS = frozenset({
    "нетбренда", "безбренда", "noname", "nobrand", "нет", "no", "нетмарки", "безмарки", "unknown", "неизвестно",
    "неизвестен", "другой", "другие", "прочие", "generic", "oem", "china", "китай", "brand", "бренд", "na", "none",
})
PRIOR_P0, PRIOR_A, PRIOR_KEEP = 0.99, 5.0, 0.95  # априорная точность бренда, её вес, порог записи в _prior
CYR = re.compile(r"[а-я]")
LAT = re.compile(r"[a-z]")
TRANSLIT: dict[str, tuple[str, ...]] = {
    "а": ("a",), "б": ("b",), "в": ("v", "w"), "г": ("g",), "д": ("d",), "е": ("e",), "ж": ("zh", "j"), "з": ("z",),
    "и": ("i",), "й": ("y", "i", "j"), "к": ("k", "c"), "л": ("l",), "м": ("m",), "н": ("n",), "о": ("o",),
    "п": ("p",), "р": ("r",), "с": ("s", "c"), "т": ("t",), "у": ("u",), "ф": ("f",), "х": ("h", "kh", "x"),
    "ц": ("c", "ts", "tz"), "ч": ("ch",), "ш": ("sh",), "щ": ("sch", "sh"), "ъ": ("",), "ы": ("y", "i"), "ь": ("",),
    "э": ("e",), "ю": ("yu", "ju", "u"), "я": ("ya", "ja", "ia"),
}


def vendor_key(vendor: str) -> str:
    """Ключ vendor: без регистра/пунктуации и без пояснения в скобках («Western Digital (WD)» ~ «Western Digital»)."""
    k = compact_key(re.sub(r"\s*\([^)]*\)", " ", str(vendor)))
    return k or compact_key(vendor)


def is_placeholder(vendor: str) -> bool:
    """«нет бренда», «No Name», «###» — в каталоге так помечают товар без бренда: это не значение бренда."""
    k = compact_key(vendor)
    return not k or k in PLACEHOLDER_BRANDS  # пустой ключ — «###», «-»


def translit_variants(key: str, limit: int = 64) -> set[str]:
    """Варианты латинского написания кириллического ключа: «атлант» -> {"atlant"}, «бош» -> {"bosh"}."""
    parts = [TRANSLIT.get(ch, (ch,)) for ch in key]
    out: set[str] = set()
    for combo in itertools.islice(itertools.product(*parts), limit):
        out.add("".join(combo))
    return out


def _offers_train(processed: Path) -> list[tuple[str, str]]:
    import pyarrow.parquet as pq

    from .data import split_of

    t = pq.read_table(processed / "offers.parquet", columns=["sku_id", "vendor"]).to_pylist()
    return [(r["sku_id"], r["vendor"]) for r in t if r["vendor"] and split_of(r["sku_id"]) == "train"]


def _train_titles(processed: Path) -> list[tuple[str, str]]:
    """(название карточки, vendor) карточек train-сплита с оффером."""
    import duckdb

    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")
    return con.execute(f"""
        SELECT c.title, o.vendor FROM read_parquet('{processed / 'cards.parquet'}') c
        JOIN read_parquet('{processed / 'offers.parquet'}') o USING (sku_id)
        WHERE c.split = 'train' AND c.has_offer AND o.vendor IS NOT NULL""").fetchall()


def build_brands(processed: Path, aliases_path: Path | None = None, min_offers: int = 2, rounds: int = 3,
                 log=print) -> dict[str, Any]:
    """-> {"mapping": {канон: [синонимы]}, "word": [...], "prior": {...}, "stats": {...}}."""
    spell: dict[str, Counter] = defaultdict(Counter)
    for _, vendor in _offers_train(processed):
        v = re.sub(r"\s+", " ", str(vendor)).strip().strip("`'\"")
        k = vendor_key(v)
        if k and not is_placeholder(v):
            spell[k][v] += 1
    counts = {k: sum(c.values()) for k, c in spell.items()}
    canon_of = {k: c.most_common(1)[0][0] for k, c in spell.items() if counts[k] >= min_offers}
    log(f"brands: {len(spell)} vendor в train, {len(canon_of)} с >= {min_offers} офферов")

    # ключ -> ключ канона (слияние синонимов)
    parent: dict[str, str] = {k: k for k in canon_of}
    names = dict(canon_of)  # ключ канона -> написание
    manual = yaml.safe_load(aliases_path.read_text(encoding="utf-8")) if aliases_path and aliases_path.exists() else {}
    extra_syn: dict[str, set[str]] = defaultdict(set)
    for canon, syns in (manual or {}).items():
        ck = compact_key(canon)
        if ck not in names:
            names[ck] = str(canon)
            parent[ck] = ck
        for s in syns or []:
            sk = compact_key(s)
            if not sk or sk == ck:
                continue
            if sk in names and sk != ck:  # суббренд-vendor каталога («POCO») сливается с каноном («Xiaomi»)
                names.pop(sk)
            parent[sk] = ck
            extra_syn[ck].add(str(s))
    # транслитерация: кириллический vendor ~ латинский -> синоним более частого
    latin = {k for k in names if LAT.search(k) and not CYR.search(k)}
    n_translit = 0
    for k in [k for k in names if CYR.search(k) and not LAT.search(k)]:
        hit = next((v for v in sorted(translit_variants(k)) if v in latin and v != k), None)
        if hit is None or k not in names:
            continue
        a, b = (hit, k) if counts.get(hit, 0) >= counts.get(k, 0) else (k, hit)
        extra_syn[a].add(names[b])
        names.pop(b)
        parent[b] = a
        n_translit += 1
    log(f"brands: ручных синонимов {sum(len(v) for v in extra_syn.values()) - n_translit}, транслитерацией {n_translit}")

    def root(k: str) -> str:
        while parent.get(k, k) != k:
            k = parent[k]
        return k

    mapping: dict[str, list[str]] = {}
    for k, name in names.items():
        syns = set(extra_syn.get(k, set())) | set(spell.get(k, ()))  # ручные + все написания vendor этого ключа
        for x in canon_of:  # слитые ключи (суббренды, транслит)
            if x != k and root(x) == k:
                syns |= set(spell[x])
        syns.add(re.sub(r"\s*\([^)]*\)", " ", name).strip())  # «Western Digital (WD)» -> «Western Digital»
        ck = compact_key(name)
        uniq: dict[str, str] = {}  # одно написание на нормализованную форму («POCO», «Poco» -> одно)
        for s in sorted((s for s in syns if s and compact_key(s) != ck), key=lambda x: (x != x.lower(), x)):
            uniq.setdefault(norm(s), s)
        mapping[name] = sorted(uniq.values(), key=str.lower)

    # защита от омонимов: статистика «выбранного» упоминания на названиях train
    titles = _train_titles(processed)
    word: set[str] = set()
    stats: dict[str, Counter] = {}
    for rnd in range(rounds):
        bd = BrandDictionary("brands", mapping, {"word": sorted(word)})
        stats = defaultdict(Counter)
        for title, vendor in titles:
            if is_placeholder(vendor):
                continue
            gold = bd.lookup.get(re.sub(r"\s+", " ", vendor.lower().replace("ё", "е")).strip())
            gold = gold or bd.keys().get(compact_key(vendor)) or bd.keys().get(vendor_key(vendor))
            m = pick_mention(bd.mentions(lower_keep(title or "")))
            if m is None:
                continue
            stats[m.canon]["n"] += 1
            stats[m.canon]["ok"] += m.canon == gold
        new = {c for c, s in stats.items() if s["n"] >= 3 and (s["ok"] + 1) / (s["n"] + 2) < 0.5}
        log(f"brands: раунд {rnd + 1}: омонимов {len(word | new)} (+{len(new - word)})")
        if new <= word:
            break
        word |= new
    # точность с априорным средним (P0 с весом A псевдонаблюдений): бренд без ошибок на 5 названиях — не «0.86»
    prior = {c: round(p, 2) for c, s in stats.items()
             if c not in word and (p := (s["ok"] + PRIOR_A * PRIOR_P0) / (s["n"] + PRIOR_A)) < PRIOR_KEEP}
    n = sum(s["n"] for s in stats.values())
    ok = sum(s["ok"] for s in stats.values())
    return {"mapping": mapping, "word": sorted(word, key=str.lower), "prior": dict(sorted(prior.items())),
            "stats": {"brands": len(mapping), "train_titles": len(titles), "picked": n,
                      "picked_precision": round(ok / n, 4) if n else None, "min_offers": min_offers}}


def write_brands_yaml(res: dict[str, Any], path: Path) -> None:
    st = res["stats"]
    head = [
        "# Справочник брендов. СГЕНЕРИРОВАН командой `attrx build-dicts` (src/attrx/dictgen.py) — руками не править:",
        "# ручные синонимы — в config/brand_aliases.yaml, затем перегенерировать.",
        "# Источник: vendor офферов YML-фида М.Видео, только train-сплит (md5(sku_id) % 100 < 80), бренды с >= "
        f"{st['min_offers']} офферами;",
        "# канон — самое частое написание в каталоге; синонимы — ручные + кириллица по транслиту («Атлант» ~ ATLANT).",
        f"# Брендов: {st['brands']}. Проверка на {st['train_titles']} названиях train: бренд товара = первое упоминание",
        "# вне контекста совместимости («для iPhone»), точность выбора "
        f"{st['picked_precision']} на {st['picked']} названиях.",
        "# _word  — омонимы слов и единиц («СМ» из «45 см», «Wireless», «Blue»): в тексте не ищутся, только в поле продавца.",
        f"# _prior — точность бренда как выбранного упоминания (train, априорно {PRIOR_P0} с весом {PRIOR_A:g}), если ниже "
        f"{PRIOR_KEEP}: основа уверенности правила.",
    ]

    def line(key: str, value: Any) -> str:
        return yaml.safe_dump({key: value}, allow_unicode=True, default_flow_style=None, width=10_000).strip()

    lines = [line("_word", res["word"])]
    lines.append("_prior:")
    lines += ["  " + yaml.safe_dump({k: v}, allow_unicode=True, default_flow_style=False).strip()
              for k, v in res["prior"].items()]
    lines += [line(k, res["mapping"][k]) for k in sorted(res["mapping"], key=lambda s: (s.lower(), s))]
    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text("\n".join(head) + "\n" + text, encoding="utf-8")
    tmp.replace(path)


# ------------------------------------------------------------------ соответствие справочников ----------------------
ENUM_MIN_N = 5  # минимум карточек train на ключ


def _enum_train_cards(processed: Path, tax, generic_limit: int) -> list[tuple[Any, str, dict[str, list]]]:
    """(Card, категория таксономии, {атрибут-справочник: допустимые значения эталона}) карточек train."""
    import json

    import duckdb

    from .schema import Card

    pairs = [(c, a) for c, cat in tax.categories.items() for a, spec in cat.attributes.items()
             if spec.type == "enum" and spec.dictionary != "brands"]
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")
    con.execute("SET threads=2")
    where = " OR ".join(f"(g.category = '{c}' AND g.attribute = '{a}')" for c, a in pairs)
    rows = con.execute(f"""
        SELECT g.sku_id, g.category, g.attribute, g.alternatives FROM read_parquet('{processed / 'gold.parquet'}') g
        WHERE g.split = 'train' AND ({where})""").fetchall()
    gold: dict[str, dict[str, list]] = {}
    cat_of: dict[str, str] = {}
    for sku, cat, attr, alts in rows:
        gold.setdefault(sku, {})[attr] = json.loads(alts)
        cat_of[sku] = cat
    gen = sorted((s for s, c in cat_of.items() if c == "generic"), key=lambda s: hashlib_key(s))[:generic_limit]
    keep = {s for s, c in cat_of.items() if c != "generic"} | set(gen)
    ids = pd_frame(sorted(keep))
    con.register("ids", ids)
    cards = con.execute(f"""
        SELECT c.sku_id, c.title, c.description_clean, c.seller_attrs, c.category_id
        FROM read_parquet('{processed / 'cards.parquet'}') c JOIN ids USING (sku_id)""").fetchall()
    out = []
    for sku, title, desc, sa, cid in cards:
        card = Card(id=sku, title=title or "", description=desc or "", seller_attrs=json.loads(sa or "{}"),
                    category_id=cid)
        out.append((card, cat_of[sku], gold[sku]))
    return out


def hashlib_key(s: str) -> str:
    import hashlib

    return hashlib.md5(f"enum:{s}".encode()).hexdigest()


def pd_frame(ids: list[str]):
    import pandas as pd

    return pd.DataFrame({"sku_id": ids})


def learn_enum_maps(processed: Path, tax, generic_limit: int = 40_000, log=print) -> dict[str, Any]:
    """Соответствие «что нашёл справочник -> значение каталога» по train: для каждого ключа EnumHits
    (позиция + набор найденных значений / первый синоним / первое значение) — самое частое значение эталона
    и доля карточек, где оно верно. Так правила перенимают соглашения каталога («ручной (handstick)» ->
    «вертикальный», «Bluetooth TWS» -> «True Wireless») и получают точность для уверенности, без ручных правок."""
    from .extractors import rules

    cards = _enum_train_cards(processed, tax, generic_limit)
    log(f"enum: {len(cards)} карточек train с эталоном атрибутов-справочников")
    tally: dict[tuple[str, str], dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    n_key: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for card, cat_name, gold in cards:
        cat = tax.categories[cat_name]
        ctx = rules.Ctx.of(card, cat_name)
        brand_cands, _ = rules.extract_brand(ctx, "brand", tax) if "brand" in cat.attributes else ([], None)
        skip = [c.span for c in brand_cands if c.span]
        for attr, values in gold.items():
            spec = cat.attributes.get(attr)
            if spec is None or not values:
                continue
            eh = rules.enum_hits(ctx, spec, tax.dictionaries[spec.dictionary], skip)
            if eh is None:
                continue
            for key in eh.keys():
                n_key[(cat_name, attr)][key] += 1
                for v in set(values):
                    tally[(cat_name, attr)][key][v] += 1
    out: dict[str, dict[str, dict[str, list]]] = {}
    n_entries = 0
    for (cat_name, attr), keys in sorted(tally.items()):
        entries = {}
        for key, cnt in sorted(keys.items()):
            n = n_key[(cat_name, attr)][key]
            if n < ENUM_MIN_N:
                continue
            best, c = max(cnt.items(), key=lambda kv: (kv[1], kv[0]))
            entries[key] = [best, round((c + 0.5) / (n + 1), 3), n]
        if entries:
            out.setdefault(cat_name, {})[attr] = entries
            n_entries += len(entries)
    log(f"enum: {n_entries} ключей соответствия по {sum(len(v) for v in out.values())} атрибутам")
    return {"_meta": {"source": "gold.parquet (train) + cards.parquet (train)", "min_n": ENUM_MIN_N,
                      "generic_limit": generic_limit, "cards": len(cards),
                      "format": "{категория: {атрибут: {ключ: [значение каталога, точность, n]}}}; ключ = "
                                "позиция|набор значений, позиция|~синоним, позиция|=значение (rules.EnumHits.keys)"},
            **out}


def write_enum_map(res: dict[str, Any], path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(res, ensure_ascii=False, indent=1, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


# ------------------------------------------------------------------ стиль модели по брендам ------------------------
def learn_model_styles(processed: Path, tax, min_n: int = 8, log=print) -> dict[str, Any]:
    """Как у бренда записана модель в каталоге — по train: «после бренда» (кусок названия после бренда, как
    rules.extract_model) или «артикул» (отдельное 5–7-значное число в названии: Brauberg «…, 880015»).
    Для каждого бренда с >= min_n карточками train — доля верных у обоих способов; в rules уверенность модели
    = точность способа для этого бренда (вместо общей эвристики «буквы+цифры»)."""
    import duckdb

    from .eval.gold import has_model_chars, model_key
    from .extractors import rules
    from .normalize import model_core
    from .schema import Card

    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")
    rows = con.execute(f"""
        SELECT c.sku_id, c.title, c.category_id, o.vendor, o.model FROM read_parquet('{processed / 'cards.parquet'}') c
        JOIN read_parquet('{processed / 'offers.parquet'}') o USING (sku_id)
        WHERE c.split = 'train' AND c.has_offer AND o.model IS NOT NULL AND o.vendor IS NOT NULL""").fetchall()
    colors = tax.dictionaries.get("colors")
    st: dict[str, Counter] = defaultdict(Counter)
    for sku, title, cid, vendor, model in rows:
        gold = model_core(model, vendor, colors)
        if not has_model_chars(gold):
            continue
        gk = model_key(gold)
        cat_name = tax.category_of_id(cid) or "generic"
        card = Card(id=sku, title=title or "", category_id=cid)
        ctx = rules.Ctx.of(card, cat_name)
        _, pick = rules.extract_brand(ctx, "brand", tax)
        if pick is None:
            continue
        b = pick.canon
        st[b]["n"] += 1
        after = rules.extract_model(ctx, "model", tax, pick, styles={})
        st[b]["after"] += bool(after) and model_key(after[0].value) == gk
        art = rules.find_article(ctx)
        st[b]["article"] += art is not None and model_key(ctx.text[art[0]:art[1]]) == gk
    out = {}
    for b, c in sorted(st.items()):
        if c["n"] < min_n:
            continue
        pa, pr = (c["after"] + 0.5) / (c["n"] + 1), (c["article"] + 0.5) / (c["n"] + 1)
        if pa < 0.9 or pr > pa:  # хранить только бренды, где общая эвристика ошибается
            out[b] = {"n": c["n"], "after": round(pa, 3), "article": round(pr, 3)}
    log(f"model styles: {len(rows)} карточек train, {len(st)} брендов, особых: {len(out)}")
    return {"_meta": {"source": "offers.model + cards.title (train)", "min_n": min_n,
                      "format": "{бренд: {n, after: точность «после бренда», article: точность «артикул»}}"}, **out}
