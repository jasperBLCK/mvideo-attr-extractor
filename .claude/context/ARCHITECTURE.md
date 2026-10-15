# Архитектура attrx — рабочая карта

Для жюри и ревьюера — [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md). Здесь: кто за что отвечает, как текут данные,
куда расширять и что нельзя ломать. Шаги пайплайна, пороги и отказы — [ORCHESTRATOR.md](ORCHESTRATOR.md);
файлы данных — [STORAGE.md](STORAGE.md); дерево репозитория — [PROJECT_MAP.md](PROJECT_MAP.md). Сверено с кодом 15.10.

## 1. Компоненты

| Компонент | Отвечает за | Точки входа |
|---|---|---|
| `schema.py` | контракт данных (pydantic): `Card`, `Candidate`, `AttributeValue`, `ExtractionResult` | `Card.full_text()`, `ExtractionResult.as_flat()` |
| `config.py` | `ROOT` (ATTRX_ROOT → репо рядом с пакетом → cwd), чтение `.env` (не перекрывает окружение), `Settings` из env, выбор источника LLM | `get_settings()`, `llm_source()` |
| `taxonomy.py` | загрузка и валидация `config/taxonomy.yaml` и справочников; `Dictionary` / `BrandDictionary` (поиск по окнам токенов); решение о категории; трафик-вес атрибута; путь категории для UI | `load_taxonomy()`, `Taxonomy.classify()`, `attr_traffic_weight()`, `category_info()` |
| `category.py` | классификатор категории по тексту: TF-IDF → линейные SVM «раздел → лист», temperature scaling; инференс на numpy, обучение на scikit-learn | `CategoryClassifier.load/predict/is_review`, `train_category()` |
| `extractors/seller.py` | поля продавца → атрибуты по `aliases`, значение → канон | `extract()` |
| `extractors/rules.py` | правила и справочники по одной строке `full_text()`: бренд, модель, память, диагональ, батарея, габариты, величины, enum, bool, процессор | `extract()` |
| `extractors/ner.py` | ML-слой: токен-классификатор ruBERT-tiny2 (BIO), «вид» карточки, декодирование, калибровка сущностей | `NerExtractor.load/extract`, `train()` |
| `extractors/ner_distant.py` | distant supervision: эталон каталога (train) → спаны и маски → обучение → калибровка → `models/ner` | `run()` (`attrx train-ner --distant`) |
| `extractors/llm.py` | промпт и JSON-схема из схемы категории, префильтр атрибутов, разбор ответа, нормализация | `extract()`, `build_request()`, `parse_response()` |
| `normalize.py` | канон: единицы (`UNIT_PATTERNS`), числа, enum (+ нечёткий), цвета, процессор, `model_core`, `display` | `normalize_value()` |
| `verify.py` | заземление кандидатов ml/llm: выравнивание цитаты с текстом + «значение следует из фрагмента» | `ground()`, `align()` |
| `confidence.py` | агрегация кандидатов (noisy-OR, конфликт, альтернативы) и калибровка (PAV-кривые по группам) | `aggregate_full()`, `load_calibrator()`, `group_keys()` |
| `pipeline.py` | оркестратор: категория → источники → маршрутизация → заземление → калибровка → статус | `Extractor.extract/extract_many` |
| `llm/` | `client.py` — любой OpenAI-совместимый API (откат формата, предохранитель, лимиты, учёт токенов); `cache.py` — SQLite-кэш и суточный счётчик; `status.py`; `cost.py` — ₽ на 1000 карточек; `bench.py` — бенчмарк | `get_client()`, `llm_status()`, `per_1000_cards()`, `run_bench()` |
| `eval/` | `gold.py` — эталон из каталога; `metrics.py` — сравнение и метрики; `real.py` — прогон на сплите; `calibrate.py` — калибровка и порог; `report.py` — отчёт в `submission/`; `eda.py` — старый EDA по списку `Card` | `attrx build-gold / eval-real / calibrate / report / eda` |
| `data/` | `data/raw` → `data/processed`: `yml.py`, `text.py` (HTML и блок «Характеристики:»), `clicks.py`, `taxo.py`, `prepare.py`; трафик-веса `traffic.py`; EDA `eda.py`; сплиты `split_of()` | `attrx prepare / traffic / eda-real` |
| `dictgen.py` | по train: справочник брендов, `enum_map.json`, `model_styles.json` | `attrx build-dicts` |
| `feedback.py` | журнал решений модератора (с категорией карточки), синонимы только из исправлений: справочник — по схеме категории, синоним — формулировка значения без «Ключ:» | `log_correction()`, `decision_action()`, `learn_synonyms()` |
| `genui/` | словарь компонентов `UISpec` (`schema.py`), сборка экрана: шаблон / по ключевым словам / LLM (`composer.py`) | `compose_default/intent/llm` |
| `api/app.py` | FastAPI: извлечение, generative UI, примеры, очередь, обратная связь, отчёт; раздаёт `web/` | `attrx serve` |
| `io.py` | датасет по маппингу `config/dataset.yaml`, загрузка `data/processed`, запись и чтение предсказаний | `load_cards()`, `write_predictions()` |
| `cli.py` | 20 команд `attrx …` (список — [PROJECT_MAP.md](PROJECT_MAP.md)) | `main()` |
| `web/` | UI модератора без сборки: «Карточка», «Очередь», «Качество», палитра Ctrl/⌘K | `index.html`, `app.js` |

## 2. Поток данных

```
ОФЛАЙН (при обновлении данных)
data/raw  (архив организаторов: skus.pkl, sku_desc.parquet, query_clicks.parquet)
  └─ attrx prepare ─► data/processed/*.parquet (cards, offers, params, categories, clicks_*, top_queries ...)
       ├─ attrx build-gold ───────► gold.parquet, gold_{dev,holdout}.jsonl, data/sample/real_gold.jsonl
       ├─ attrx build-dicts (train) ► config/dictionaries/brands.yaml, config/enum_map.json, config/model_styles.json
       ├─ attrx traffic ──────────► config/traffic_weights.yaml, submission/traffic.json
       ├─ attrx train-category ───► models/category/   (train → dev: подбор и калибровка → holdout: отчёт)
       ├─ attrx train-ner --distant ► models/ner/       (train; калибровка на отложенной части train)
       ├─ attrx calibrate (dev) ──► config/calibration.json (кривые + порог)
       └─ attrx eval-real ────────► data/processed/<имя>.json (dev; holdout — один финальный прогон)

ОНЛАЙН (на карточку)
Card ─► Extractor.extract: категория → продавец → правила → агрегация → [NER] → [LLM] → заземление → калибровка → статус
ExtractionResult ─┬► attrx extract ─► outputs/predictions.jsonl/.csv
                  ├► POST /api/extract (+UISpec +CardText) ─► web «Карточка»
                  └► GET /api/review/queue (из predictions) ─► web «Очередь»
Решение модератора ─► POST /api/feedback ─► outputs/feedback.jsonl ─► attrx learn ─► config/dictionaries/learned.yaml
attrx report ─► submission/*.json, predictions, REPORT.md ─► GET /api/report ─► web «Качество»
```

## 3. Зависимости модулей

```
schema   ◄── почти все модули
config   ◄── taxonomy, category, io, feedback, llm/*, pipeline, api, cli
normalize ─► taxonomy          verify ─► normalize, taxonomy          confidence ─► schema, taxonomy
taxonomy ─(лениво)─► category  (category.py не импортирует taxonomy.py)
extractors/{seller,rules,llm} ─► normalize, taxonomy   (llm ─► llm.client)
extractors/{ner,ner_distant}  ─► eval.metrics, eval.gold  — разметка NER сверяется той же функцией, что и метрики
pipeline ─► taxonomy, extractors/*, verify, confidence, llm.client
eval/*   ─► normalize, taxonomy, confidence (calibrate), dictgen (is_placeholder)
api ─► pipeline, genui, feedback, io      genui ─► llm.client      cli ─► всё (импорты внутри команд)
data/*   ─► только pandas / pyarrow / duckdb: от пайплайна не зависит
```
Циклов нет: `taxonomy → category` и `dictgen ↔ eval.gold` — импорты внутри функций.

## 4. Точки расширения

**Новая категория**
1. `config/taxonomy.yaml`: `title`, `category_ids` (листья YML; один лист — в одной категории, иначе `ValueError`), `keywords`, `attributes`.
2. Классификатор категории: если листья уже есть среди классов модели — переобучать не нужно (отображение лист → категория
   строится при загрузке; `clf.unmapped_leaves` покажет пропущенные), иначе `attrx train-category`.
3. `attrx build-gold` → `attrx build-dicts` → `attrx traffic` → `attrx calibrate` → `attrx eval-real` (dev).
4. Правила по паре (категория, атрибут): `rules.ATTR_RANGES`, `EXTRA_HINTS`, `NEG_BEFORE`, `NEG_AFTER`.

**Новый атрибут**
- `type: enum` — нужен `config/dictionaries/<dictionary>.yaml` (иначе ошибка загрузки); `number` — `kind` из
  `normalize.UNIT_PATTERNS` (новая величина: `UNIT_PATTERNS` + `rules.RANGES`); `string` — `kind: model | cpu` или свободная
  строка; `bool` — обязательны `keywords`.
- `aliases` — ключи продавца; `gold` — `vendor` / `model` / имена параметров фида для эталона; `traffic_weight` — вес по умолчанию.
- Правила выбираются в `rules.extract` по `kind`/`type`. NER получит метку только после переобучения и при ≥ `--min-spans` спанах.

**Новый источник (экстрактор)**
1. Выдаёт `Candidate(source=…, span=…)`; `span` — смещения в `Card.full_text()`; источник добавить в `schema.Source`.
2. Вес — `confidence.SOURCE_WEIGHT`; подключение — `pipeline.Extractor.extract` до финализации; модельные источники — через
   `checked()` (= `verify.ground`).
3. Нельзя верить в одиночку — добавить в `pipeline.UNVERIFIED_ONLY`; плюс `ALL_SOURCES`, режим в `eval/report.MODES`.
4. Переобучить калибровку: `attrx calibrate --sources …`.

**Компонент generative UI**: `genui/schema.py` (`ComponentType` + описание в `COMPONENTS` — уходит в промпт) →
`genui/composer.py` (`_INTENTS`) → `web/components.js` (`registry`) → стили → `tests/test_api_ui.py` → `docs/GENUI.md`.

**LLM-провайдер**: только env (`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`); тариф — `config/llm_prices.yaml`;
подпись — `llm.client.provider_of`, `llm.status.PROVIDER_TITLES`.

**Синонимы**: ручные справочники `config/dictionaries/*.yaml` (ключи с `_` — метаданные); бренды — только через
`config/brand_aliases.yaml` + `attrx build-dicts`; из правок — `attrx learn` → `config/dictionaries/learned.yaml`.

## 5. Инварианты (не ломать)

| # | Инвариант | Где держится | Тест |
|---|---|---|---|
| 1 | Значение только от LLM никогда не `accepted` | `pipeline.UNVERIFIED_ONLY` → `Extractor._finalize`; то же правило в `eval/calibrate._accepted` | `tests/test_grounding.py::test_llm_only_value_goes_to_review` |
| 2 | Кандидаты ml и llm проходят заземление (цитата выровнена с текстом, значение следует из фрагмента), иначе отброшены; у llm — с предупреждением | `pipeline.checked` → `verify.ground` | `tests/test_grounding.py`, `tests/test_llm.py::test_hallucinated_value_rejected` |
| 3 | `evidence_span` и `Candidate.span` — `[start, end)` в `Card.full_text()` = title `\n` description `\n` «k: v; …». Формат менять только вместе с `rules.Ctx`, `seller.seller_spans`, `api.card_text`, `ner.build_view` | `schema.Card.full_text` | `tests/test_api_ui.py::test_card_text_blocks_point_into_full_text` |
| 4 | Поиск — в строке той же длины, что текст (`taxonomy.lower_keep`, `verify.fold`, `ner.ntext`): смещения не сдвигаются | — | `tests/test_grounding.py` |
| 5 | Сплиты только `attrx.data.split_of` (md5(sku_id) % 100: train < 80 ≤ dev < 90 ≤ holdout). Train — словари, NER, классификатор; dev — настройка и калибровка; holdout — один финальный прогон | `data/__init__.py`, `dictgen`, `ner_distant`, `cli.cmd_calibrate` (только dev) | — |
| 6 | `data/sample/real_cards.jsonl` — 400 карточек **holdout**: демо и тесты, не настройка правил и промптов | `data/prepare.step_sample` | — |
| 7 | Одна нормализация для эталона и предсказаний: `metrics.match_key`, `normalize.model_core`, `clean_cpu` / `cpu_key` | `eval/metrics.py`, `eval/gold.py` | `tests/test_eval_gold.py` |
| 8 | Одни ключи калибровки при обучении и применении | `confidence.group_keys` | — |
| 9 | Одинаковые признаки при обучении и инференсе | `category.Featurizer`, `ner.build_view` | `tests/test_category.py`, `tests/test_ner_distant.py` |
| 10 | Таксономия проверяется при загрузке: тип, справочник enum, `keywords` у bool, уникальные `category_ids`, обязательная `generic` | `Taxonomy.__init__`, `_validate` | — |
| 11 | Без LLM, NER, модели категории, `data/processed`, калибровки — пайплайн работает (деградация, не падение) | [ORCHESTRATOR.md §9](ORCHESTRATOR.md) | `tests/conftest.py` выключает LLM и ML |
| 12 | Generative UI: LLM отдаёт только `UISpec` из фиксированного словаря, значения фронтенд берёт из `ExtractionResult`; невалидный ответ → сборка по ключевым словам | `genui/schema.py`, `composer.compose_llm`, `web/components.js` | `tests/test_llm.py::test_genui_rejects_invalid_spec` |
| 13 | Без pickle в артефактах: категория — npz + JSON (`allow_pickle=False`); `skus.pkl` читается unpickler-ом только со встроенными типами | `category.CategoryClassifier.load`, `data/yml.py` | — |
| 14 | Сгенерированное руками не правится: `brands.yaml`, `enum_map.json`, `model_styles.json`, `calibration.json`, `traffic_weights.yaml`, `models/*` | шапки файлов | — |
| 15 | «При конфликте принимать только с уверенностью ≥ 0.85» — одна константа `confidence.CONFLICT_ACCEPT` для `pipeline._finalize` и `eval/calibrate._accepted` | `confidence.py` | `tests/test_pipeline.py::test_conflict_accept_is_one_constant_for_pipeline_and_calibration` |
| 16 | `Extractor.extract_many` (пачки: один прогон NER, параллельные LLM-запросы) даёт те же значения, что `extract` по одной карточке, в порядке входа | `pipeline.Extractor._extract_chunk` | `tests/test_pipeline.py::test_extract_many_runs_ner_once_per_pack_with_same_results`, `tests/test_llm.py::test_extract_many_runs_llm_for_pack_in_parallel_with_same_results` |

## 6. Грабли (на 15.10)

- `NerExtractor.load` ленивый: без `ML_ENABLED` или без весов в `models/ner` возвращает None, не импортируя torch (`Extractor()` стартует за ~0,4 с).
- `/api/review/queue` читает сначала `outputs/` (последний `attrx extract` / `demo`), потом `submission/`; `/api/metrics` — наоборот,
  сначала `submission/` (`?source=outputs` — демо):
  после `attrx demo` в очереди 400 реальных карточек holdout из `data/sample`.
- `get_settings()` каждый раз создаёт новый `Settings`; LLM-клиент — синглтон `get_client()`; таксономия и калибратор —
  `lru_cache`: после правки YAML — перезапуск процесса.
- `LLM_ROUTE_THRESHOLD` маршрутизирует и в NER, и в LLM, и сравнивается с **сырой** (до калибровки) уверенностью.
- Docker-образ ставит только базовые зависимости и не копирует `models/` — см. [TECH_STACK.md](TECH_STACK.md).
