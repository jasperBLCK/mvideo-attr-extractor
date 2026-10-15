# Где что хранится

Кто пишет, кто читает и как пересобрать каждый файл. Размеры и строки — на 15.10. Подробности о данных кейса — [DATA.md](DATA.md).

## Сводка

| Место | В git | Кто пишет | Объём |
|---|---|---|---|
| `data/raw/` | нет | `scripts/download_data.sh` (`make data`) | 3,5 ГБ (архив 1,16 ГБ + распакованное) |
| `data/processed/` | нет | `attrx prepare`, `build-gold`, `traffic`, `eval-real`, `eda-real`, `train-ner --distant` | ~550 МБ |
| `data/sample/` | да | руками (синтетика) и `attrx prepare` / `build-gold` (реальный сэмпл) | ~1,7 МБ |
| `config/` | да | руками и генераторами (`build-dicts`, `calibrate`, `traffic`, `learn`) | ~570 КБ |
| `models/category/` | да (исключение в `.gitignore`) | `attrx train-category` | 7,8 МБ |
| `models/ner/` | нет (`models/*`) | `attrx train-ner --distant` | — |
| `outputs/` | нет (кроме `.gitkeep`) | `extract`, `demo`, `eval`, API, LLM-клиент, `llm-bench` | — |
| `submission/` | да | `attrx report`, `attrx traffic` | — |
| `.claude/context/` | да | оркестратор и агенты | — |

## data/raw — архив организаторов

| Файл | Размер | Что | Читают |
|---|---:|---|---|
| `cu_ws.zip` | 1,16 ГБ | исходный архив | `download_data.sh` |
| `skus.pkl` | 1,57 ГБ | YML-фид М.Видео: 546 069 офферов, 6 963 категории. **12 ГБ RAM при загрузке** — читает только шаг `yml` | `attrx prepare` |
| `sku_desc.parquet` | 352 МБ | 1 177 200 описаний: `sku_id, title, description` | `prepare`, `eda-real` (доля HTML), фолбэк `llm-bench` |
| `query_clicks.parquet` | 651 МБ | 30 991 350 кликов поиска | `prepare` (clicks), `traffic`, `docs/checkpoint/demand_stats.py` |
| `case_brief.pdf` | 0,55 МБ | презентация кейсодателя | люди |

Пересборка: `make data` (докачка, повторный запуск ничего не перекачивает).

## data/processed — подготовленные данные (parquet zstd, запись атомарная)

| Файл | Кто пишет | Колонки / содержимое | Строк / размер | Кто читает |
|---|---|---|---|---|
| `categories.parquet` | `prepare` (yml) | id, parent_id, title, path, path_ids, depth, is_leaf, n_offers, n_offers_total | 6 963 / 0,3 МБ | шаг cards, `taxonomy.category_info` (путь для UI), `Taxonomy.classify` (известный лист), `train-category` |
| `offers.parquet` | `prepare` (yml) | sku_id, available, name, model, vendor, category_id, n_category_ids, seller, n_params, n_raw_params, category_title | 546 069 / 20 МБ | cards, `build-gold`, `build-dicts`, `traffic`, `io.load_offer_params` |
| `params.parquet` | `prepare` (yml) | sku_id, name (без `_search`), is_search, unit, value, is_service | 20 975 731 / 62 МБ | `build-gold`, taxonomy, sample, `traffic`, `io.load_offer_params` |
| `clicks_by_sku.parquet` | `prepare` (clicks) | sku_id, clicks, n_queries, mean_position, seo_id, subject_id, price, sku_name, sku_brand, category_id, category_source | 332 753 / 11 МБ | cards (категория по seo_id), EDA |
| `clicks_by_category.parquet` | `prepare` (clicks) | category_id, clicks, n_skus, clicks_with_offer | 4 002 / 25 КБ | `eval/metrics.load_traffic_weights` (доля кликов категории; без файла — `_meta.categories` из `config/traffic_weights.yaml`, те же числа), EDA |
| `top_queries.parquet` | `prepare` (clicks) | category_id, query, clicks, n_skus (топ-300 на категорию) | 291 063 / 2,9 МБ | шаг taxonomy, EDA |
| `cards.parquet` | `prepare` (cards) | sku_id, title, description_clean, seller_attrs (JSON), n_seller_attrs, category_id, category_title, category_path (JSON), category_source, seller, seller_type, has_offer, split, description_len | 1 177 200 / 401 МБ | почти всё: `build-gold`, `eval-real`, `calibrate`, `train-category`, `train-ner`, `build-dicts`, `io.load_processed_cards` |
| `param_stats.parquet` | `prepare` (taxonomy) | category_id, name, n_offers_with, fill_rate, has_search, has_bucket, unit, top_value | 82 424 / 0,9 МБ | EDA, проектирование таксономии |
| `taxonomy_candidates.json` | `prepare` (taxonomy) | 25 топ-категорий: заполненность, извлекаемость, значения, ключи продавцов, запросы | 1,5 МБ | таксономия, шаг sample, `category._provisional_leaves` |
| `prepare_report.json` | `prepare` | статистика и время шагов, пик RSS | 3 КБ | документация |
| `queries_by_category.parquet` | `traffic` (кэш, `--refresh`) | category_id, query, brand, clicks, n_skus | 2 385 980 / 39 МБ | `traffic` |
| `gold.parquet` | `build-gold` | sku_id, split, category, category_id, attribute, value, value_num, alternatives, raw_value, raw_unit, unit, source, mapped, extractable, seller_type, size_band, desc_band | 626 116 / 21 МБ | `eval-real`, `calibrate`, `ner_distant.load_gold` (train), `dictgen.learn_enum_maps` (train) |
| `gold_dev.jsonl`, `gold_holdout.jsonl` | `build-gold` | `{id, category, split, attributes, extractable, сегменты}` | 7,3 / 7,4 МБ | `io.load_gold`, `eval.gold.load_jsonl` |
| `baseline_dev.json`, `rules_dev.json` | `eval-real --out …` | метрики: micro, macro, traffic-weighted, per_attribute, per_category, segments, threshold_sweep, perf, info | ~43 КБ | [docs/EVALUATION.md](../../docs/EVALUATION.md) §7–8 |
| `*_dev.json`, `*_holdout.json` (гибрид) | `eval-real --sources … --out …` | то же | — | финальный отчёт |
| `eda_stats.json`, `business_stats.json` | `eda-real` | цифры EDA и бизнес-раздела | 28 / 6 КБ | `docs/EDA.md` |
| `ner_distant_stats.json` | `train-ner --distant` (и `--dry-run`) | статистика разметки по атрибутам | — | поток NER |

## data/sample — в git, для демо и тестов без датасета

| Файл | Строк | Откуда | Кто читает |
|---|---:|---|---|
| `cards.jsonl` | 20 | руками, синтетика | юнит-тесты, `make eda / ner / report`, запасные примеры API |
| `gold.jsonl` | 20 | руками | юнит-тесты |
| `real_cards.jsonl` | 400 | `attrx prepare --steps sample` — карточки **holdout** (60 смартфонов, 60 ноутбуков, остальные из 22 топ-категорий) | `attrx demo` (и CI), `/api/examples` (витрина UI), `attrx llm-bench`, `tests/test_eval_gold.py`, `tests/test_demo.py` |
| `real_params.jsonl` | 400 | тот же шаг: параметры оффера | эталон `llm-bench` |
| `demo_cards.jsonl` | 3 | руками: копии карточек из `real_cards.jsonl` + `demo_note` (что там дозаполняет ИИ) | `/api/examples` — первая псевдокатегория «★ Демо: ИИ дозаполняет»; нет файла — её нет |
| `real_gold.jsonl` | 400 | `attrx build-gold` | `attrx demo` (метрики как у `eval-real`), тесты |

## config — конфиги вместо кода

| Файл | Руками / генерируется | Что внутри | Кто читает |
|---|---|---|---|
| `taxonomy.yaml` | руками | 12 категорий + `generic`: `category_ids`, `keywords`, атрибуты (`type`, `unit`, `kind`, `dictionary`, `aliases`, `gold`, `traffic_weight`, `keywords` у bool) | `Taxonomy`, `build-gold`, классификатор (отображение листьев), метрики |
| `dataset.yaml` | руками | маппинг колонок датасета → `Card` | `io.load_cards` (`extract`, `eval`, `report`, `eda`) |
| `dictionaries/brands.yaml` | **генерируется** `attrx build-dicts` | 4 822 бренда из `vendor` офферов train + синонимы; `_word` (омонимы), `_prior` (точность) | `BrandDictionary` |
| `dictionaries/colors.yaml` | руками (по значениям каталога) | 17 базовых цветов, RU/EN-синонимы | правила, нормализация, метрики |
| `dictionaries/{chambers,connection_types,defrost,headphone_types,materials,resolutions,tv_os,vacuum_types,washer_types}.yaml` | руками | канон → синонимы | enum-атрибуты |
| `dictionaries/learned.yaml` | **генерируется** `attrx learn` из `outputs/feedback.jsonl` | `{справочник: {синоним: канон}}`; пока не создан; попадёт в git | `Taxonomy` (добавляется к справочникам) |
| `brand_aliases.yaml` | руками | ручные синонимы брендов («Самсунг», Redmi/POCO → Xiaomi) | вход `build-dicts` |
| `enum_map.json` | **генерируется** `build-dicts` | `{категория: {атрибут: {ключ находки: [значение каталога, точность, n]}}}` | `rules.extract_enum` |
| `model_styles.json` | **генерируется** `build-dicts` | `{бренд: {n, after, article}}` — как бренд пишет модель | `rules.extract_model` |
| `calibration.json` | **генерируется** `attrx calibrate` (dev) | кривые по группам, порог, проверка кросс-фиттингом | `Calibrator`, `config._default_threshold` |
| `traffic_weights.yaml` | **генерируется** `attrx traffic` | `{category_id: {vendor \| model \| имя param: доля кликов}}` + `_meta`; 255 категорий | `attr_traffic_weight`, `eval/metrics` |
| `llm_prices.yaml` | руками (с источниками тарифов) | ₽ за 1000 токенов, on-prem ₽/ч | `llm/cost.py`, `llm-bench` |

## models

| Папка | В git | Файлы | Кто пишет |
|---|---|---|---|
| `models/category/` | да | `model.npz` (6,6 МБ, веса int8), `vocab.json.gz` (0,9 МБ), `meta.json` (классы, температуры, порог), `metrics.json` (все метрики) | `attrx train-category` |
| `models/ner/` | нет | веса и токенизатор HF (`config.json` — признак модели для `NerExtractor.load`) + `attrx_ner.json` (параметры обучения и калибровка) | `attrx train-ner --distant`: пишет в `models/ner.tmp`, подменяет целиком, прежняя версия — `models/ner.prev` |

## outputs — рабочие файлы (gitignored)

| Файл | Кто пишет | Кто читает |
|---|---|---|
| `predictions.jsonl`, `predictions.csv` | `attrx extract`, `attrx demo` (`PREDICTIONS_PATH`) | `/api/review/queue`, `/api/review` (раньше, чем `submission/`), `attrx eval`, `attrx prelabel` |
| `metrics.json` | `attrx eval`, `attrx demo` (формат `eval-real`: micro, per_category, segments, threshold_sweep, perf, info) | `/api/metrics` (раньше, чем `submission/`) |
| `feedback.jsonl` | `POST /api/feedback` (`{ts, card_id, category, attribute, predicted, corrected, raw}`; `raw` — фрагмент evidence_span, у поля продавца без «Ключ:»; в старых записях нет `category`); `POST /api/feedback/undo` переписывает файл | очередь (решения), `attrx learn` (учит только исправления) |
| `llm_cache.sqlite` | LLM-клиент (`CACHE_PATH`): таблицы `cache(k, v)` и `usage(day, calls, prompt_tokens, completion_tokens)` | клиент (кэш, суточный лимит), `llm_status.today` |
| `llm_bench/` | `attrx llm-bench` (`<модель>.json`) и ручные прогоны с логами | сводка — `docs/llm_bench.json` |
| `eda/` | `attrx eda` (старый EDA: `eda.md`, `eda.png`) | — |

## submission — артефакты для проверяющих (в git)

| Файл | Кто пишет | Состояние на 15.10 |
|---|---|---|
| `predictions.jsonl`, `predictions.csv`, `metrics.json`, `compare.json`, `segments.json`, `errors.csv`, `REPORT.md` | `attrx report -i <карточки> --gold <эталон>` (в `predictions` — лучший режим по traffic-weighted F1) | синтетика от 14.10 (20 карточек) — перегенерировать на holdout |
| `traffic.json` | `attrx traffic` | актуален (1,7 МБ) |
| `calibration.json`, `threshold.json`, `cost.json` | **генератора в коде нет**; их читают `/api/report` («Качество») и README | нет; форматы — [PLAN.md](PLAN.md) |

## Сгенерированное в docs/ и notebooks/

`docs/img/eda_*.png` — `attrx eda-real`; `docs/img/ui_*.png` — скриншоты (agent-browser); `docs/llm_bench.json` — сводка
прогонов `llm-bench` (собирает поток LLM); `docs/checkpoint/demand_stats.json` — `docs/checkpoint/demand_stats.py`;
`docs/checkpoint/concept.pdf`, `docs/slides.pdf` — печать HTML через chromium; `notebooks/01_eda.ipynb` — исполняется на месте (`make eda-nb`).

## Пересборка с нуля (порядок важен)

```bash
make data                                    # data/raw
attrx prepare                                # ~8 мин, пик 11,8 ГБ RAM → data/processed + data/sample/real_{cards,params}.jsonl
attrx build-gold                             # 2–3 мин → gold.parquet, gold_{dev,holdout}.jsonl, data/sample/real_gold.jsonl
attrx traffic                                # ~5 мин → config/traffic_weights.yaml, submission/traffic.json
attrx build-dicts                            # ~5 мин, нужен gold.parquet → brands.yaml, enum_map.json, model_styles.json
attrx train-category                         # ~6 мин → models/category
attrx train-ner --distant                    # extra [ml] → models/ner
attrx calibrate --sources seller,rules,ml,llm  # только dev → config/calibration.json (seller,rules — ~20 с; с LLM дольше, кэш помогает)
attrx eval-real --out data/processed/rules_dev.json   # dev; holdout — один финальный прогон
attrx eda-real                               # 1–2 мин → docs/img/eda_*.png, eda_stats.json, business_stats.json
```
`attrx report` пишет в `submission/` по любому набору карточек и эталону в формате `io.load_gold`.
