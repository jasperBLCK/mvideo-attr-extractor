# Оркестратор пайплайна — `src/attrx/pipeline.py:Extractor`

Одна карточка → `ExtractionResult`. Кратко для жюри — [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md); компоненты и
инварианты — [ARCHITECTURE.md](ARCHITECTURE.md). Здесь — каждый шаг, порог, переменная окружения и отказ. Сверено с кодом 15.10.

## 0. Инициализация `Extractor(taxonomy=None, settings=None, client=None, calibrated=True)`

| Что | Откуда | Когда грузится |
|---|---|---|
| `Settings` | env + `ROOT/.env` (`config.get_settings`; новый объект на каждый вызов) | конструктор |
| `Taxonomy` | `TAXONOMY_PATH` → `load_taxonomy` (lru_cache): категории, `config/dictionaries/*.yaml` + `learned.yaml`, `enum_map.json`, `model_styles.json`, валидация | конструктор |
| LLM-клиент | `llm.client.get_client()` — синглтон процесса; SQLite-кэш открывается сразу | конструктор |
| NER | `NerExtractor.load(ML_MODEL_PATH)`, если `ML_ENABLED`; `None` без torch/transformers или без `models/ner/config.json` | конструктор (torch импортируется, даже если модели нет) |
| Калибратор | `load_calibrator(CALIBRATION_PATH)` (lru_cache); `calibrated=False` — сырые оценки (так собирает точки `attrx calibrate`) | конструктор |
| Классификатор категории | `Taxonomy.classifier()` → `CategoryClassifier.load(taxonomy=…)`, один раз на таксономию | лениво: первая карточка без категории из таксономии |
| Дерево каталога, трафик-веса | `data/processed/categories.parquet`, `config/traffic_weights.yaml` (lru_cache, пути от `ROOT`, не из env) | лениво |

API держит один `Extractor` на процесс (`api/app.py:extractor()`).

## 1. Шаги `extract(card, use_llm=None, sources=None)`

`sources` ⊆ {seller, rules, ml, llm}, по умолчанию все. `use_llm=None` — по `client.enabled`; `False` — без LLM;
`True` — только если клиент включён. В скобках — ключ `timings_ms`.

| # | Шаг | Что делает | Код |
|---|---|---|---|
| 1 | категория (`category`) | `Taxonomy.classify(card)` → `CategoryDecision(name, confidence, source, title, path, review)`, лестница — §1.1 | `taxonomy.py:Taxonomy.classify` |
| 2 | продавец (`seller`) | ключ поля → атрибут (точный алиас / название атрибута / подстрока алиаса длиннее 3 символов) → `normalize_value`; span — «ключ: значение» в `full_text()` | `extractors/seller.py:extract` |
| 3 | правила (`rules`) | бренд (первое упоминание в названии вне «для / совместим», не платформа «Google TV», «Алиса»), модель (после бренда до хвоста характеристик или артикул по `model_styles.json`), память и пара «8/256», диагональ (+ размер ТВ из кода модели), батарея, габариты «ВхШхГ», величины с единицей рядом с подсказкой, enum (подпись > название > описание, `enum_map.json`), bool по `keywords`, процессор | `extractors/rules.py:extract` |
| 4 | агрегация | по каждому атрибуту схемы: `aggregate_full(кандидаты)` — §2.2 | `confidence.py:aggregate_full` |
| 5 | маршрутизация | «слабый» атрибут: нет значения, ИЛИ сырая уверенность < `LLM_ROUTE_THRESHOLD` (0.6), ИЛИ конфликт | `pipeline.py:weak_attrs` |
| 6 | NER (`ml`) | если `ml` в sources, модель загружена и есть слабые: `NerExtractor.extract` (в `extract_many` — один `extract_batch` на пачку) → только слабые атрибуты → `_checked()` (заземление; отброшенные молча) → повторная агрегация | `extractors/ner.py:NerExtractor.extract` |
| 7 | LLM (`llm`) | если LLM нужна: слабые пересчитываются после NER → `llm_extractor.extract` (в `extract_many` — `extract_batch` пачки параллельно): префильтр `relevant_attrs` → `build_request` → `client.complete_json` → `parse_response` (нормализация; вне справочника — отброс + заметка) → `_checked()` (незаземлённое — в `warnings`) → заметки в `warnings` → повторная агрегация | `extractors/llm.py:extract` |
| 8 | финализация (`aggregate`) | калибровка → статус (§3) → понижение при неуверенной категории → `AttributeValue`: `display`, `unit`, `evidence` и `evidence_span` лучшего кандидата (сначала со span, затем по score), `alternatives`, `required`, `traffic_weight` | `pipeline.py:_finalize` |
| 9 | итог (`total`) | `category_title` / `category_path`: лист классификатора, иначе дерево каталога по `category_id`, иначе название категории | `taxonomy.py:category_info` |

### 1.1 Категория: лестница `Taxonomy.classify`

| # | Условие | Категория (`source`) | confidence | review |
|---|---|---|---|---|
| 1 | `category_id` входит в `category_ids` таксономии | её категория (`catalog`) | 1.0 | нет |
| 1a | `category_id` — известный лист вне таксономии (есть в `categories.parquet` или в классах классификатора; либо классификатора нет) | `generic` (`catalog`) | 1.0 | нет |
| 2 | `card.category` совпадает с именем или названием категории | её категория (`title`) | 1.0 | нет |
| 3 | задан хук `Taxonomy.set_category_hook` | ответ хука (`hook`) | от хука | < 0.6 |
| 4 | загружен классификатор | категория листа или `generic` (`classifier`), + лист и путь для UI | калиброванная P(категория) | < `threshold` модели (0.9, `models/category/meta.json`) |
| 5 | нет классификатора, ключевое слово в `card.category` | `title` | 0.9 | нет |
| 6 | нет классификатора: ключевые слова в названии (×2) и тексте (×1) | лучшая или `generic` (`keywords`) | min(0.95, 0.6 + 0.15·score); без совпадений 0.3 | < 0.6 |

`review=True`: все `accepted` карточки → `review` + предупреждение «категория определена неуверенно …».

### 1.2 LLM-шаг подробнее (`extractors/llm.py`)

- **Префильтр** (`LLM_PREFILTER=true`): в запрос идут только атрибуты, о которых что-то сказано: строка без `kind` или `kind: model`
  — всегда; enum — если справочник нашёл значение в тексте; иначе слово-маркер (алиасы, название, слова вида величины,
  `keywords`, единица), для чисел — с цифрой в пределах 30 символов того же предложения.
- **Промпт**: категория, строка на атрибут (название, тип, единица, подсказка формата для model/cpu, справочник: весь,
  если ≤ `LLM_ENUM_VALUES_MAX` (60) значений, иначе найденные в тексте); карточка: название, описание, сокращённое
  до `LLM_MAX_INPUT_CHARS` (3500) дословными предложениями (`fit_text`), до 80 полей продавца.
- **Схема ответа**: `{attr: {evidence, value}}`, evidence первым; `LLM_SCHEMA_ENUM=false` — справочник в схему не зашивается.
- **Разбор**: null-подобные ответы пропускаются; `normalize_value` → канон; не нормализовалось — отброс и заметка
  `unmapped_note` → `ExtractionResult.warnings` (§5): enum с цитатой из текста — «кандидат в справочник …», без цитаты —
  как незаземлённое значение.
- `llm_used = True` — только если LLM действительно спросили (вызов или ответ из кэша, `extract(asked=…)`) и ошибки не было;
  префильтр отсеял все атрибуты — `False`.

## 2. Уверенность

### 2.1 Сырые score источников (важен порядок, абсолютные значения выравнивает калибровка)

| Источник | score | Где |
|---|---|---|
| seller | 0.8 × качество нормализации (enum: точное 1.0, найдено в строке 0.9, нечёткое ≥ 90 → ×0.85; процессор 0.95; строка 0.9; число и bool 1.0) | `seller.SELLER_BASE_SCORE`, `normalize.normalize_value` |
| rules | бренд 0.5 + 0.47·`_prior` (−0.05, если не первое упоминание); только «для X» — 0.35; в описании с подписью «Бренд:» — 0.85, без подписи — 0.45 + 0.2·`_prior`; enum: подпись или название 0.9 (перечисление в названии 0.6), одно значение в описании 0.75, перечисление 0.45 каждому; с `enum_map` — 0.3 + 0.67·точность (≤ 0.97); числа и величины 0.6–0.96 (подпись и название выше описания, +0.05…0.08 за название); модель — по форме строки и точности стиля бренда | `extractors/rules.py` |
| ml | калиброванная P(верно) × качество нормализации; без калибровки модели — p × 0.8; отсечка p < 0.35 и калиброванная < 0.5 | `extractors/ner.py:_candidates` (константы ведёт поток NER) |
| llm | 0.9 × качество (число или enum с цитатой), иначе 0.75 × качество | `extractors/llm.py:parse_response` |

### 2.2 Агрегация `aggregate_full`

1. Группы по значению: числа в пределах 1% (`NUM_TOL`) — одно значение; строки — без регистра.
2. Поддержка группы — noisy-OR лучших оценок разных источников: `1 − Π(1 − min(0.99, score × SOURCE_WEIGHT))`,
   веса: human, seller, rules — 1.0; ml — 0.9; llm — 0.85.
3. Побеждает группа с наибольшей поддержкой; при конфликте `conf = top − 0.5 × support(второй)`; `alternatives` — до трёх
   следующих групп `{value, confidence, sources}`.
4. `cross_conflict` — альтернативу предлагает источник, которого нет у победителя (продавец против текста).

### 2.3 Калибровка `Calibrator`

- Кривая — изотоническая регрессия (PAV) «сырая уверенность → P(значение верно)». Ключи от частного к общему:
  `категория/атрибут/источники[|conflict]` → `источники|тип[|conflict]` → `*|тип` (тип: brand, model, color, cpu, enum,
  number, bool, string). Кривой нет — сырая оценка.
- `attrx calibrate` (только dev, 2-fold кросс-фиттинг для порога) → `config/calibration.json`. Сейчас (финал 15.10): источники seller + rules + ml (NER),
  порог **0.69** (dev out-of-fold precision принятых 0.967, ECE 0.011; holdout 0.959, ECE 0.012); сочетания без своей кривой — монотонно
  (согласие источника не снижает уверенность); правило — нижняя 95% граница Уилсона точности принятых ≥ 0.95 и в каждой ячейке
  с ≥ 20 принятыми точность ≥ 0.75.
- После подключения NER и LLM: `attrx calibrate --sources seller,rules,ml,llm` — иначе сочетания с ml/llm получают
  кривую `*|тип`, выученную на правилах.

## 3. Статус (порядок проверок)

1. значения нет → `missing`, confidence 0.
2. источники ⊆ {llm} → `review` (инвариант доверия).
3. `cross_conflict` → `review`.
4. `conf ≥ CONFIDENCE_THRESHOLD` и не (`conflict` и `conf < CONFLICT_ACCEPT` = 0.85) → `accepted`, иначе `review`.
5. категория с `review=True` → `accepted` становится `review`.

## 4. Пороги и переменные окружения

| Порог / переменная | По умолчанию | Где читается | Влияние |
|---|---|---|---|
| `CONFIDENCE_THRESHOLD` | `threshold` из `config/calibration.json` (0.7), иначе 0.7 | `config._default_threshold` | порог `accepted`. В `.env.example` строка закомментирована: копия `.env` порог калибровки не перекрывает |
| `CALIBRATION_PATH` | `config/calibration.json` | `Settings` | кривые и порог |
| `LLM_ROUTE_THRESHOLD` | 0.6 | `Settings` | «слабый» атрибут для NER и LLM (по сырой уверенности) |
| конфликт | 0.85 | `confidence.CONFLICT_ACCEPT` (его же берут `pipeline._finalize` и `eval/calibrate._accepted`) | при конфликте `accepted` только с conf ≥ 0.85 |
| `NUM_TOL` / `SOURCE_WEIGHT` | 0.01 / см. §2.2 | `confidence.py` | |
| порог категории | 0.9 | `models/category/meta.json` (`attrx train-category --min-confidence --target-precision`) | ниже — карточка на проверку |
| ключевые слова, хук | review < 0.6 | `Taxonomy.classify` | |
| `ALIGN_MIN` | 0.85 | `verify.py` | доля «веса» цитаты, совпавшая с фрагментом; числа — точно, слова — общая основа (≥ 4 и ≥ 50% длины) или rapidfuzz ≥ 85 |
| NER `MIN_PROB` / `MIN_SCORE` | 0.35 / 0.5 | `extractors/ner.py` | отсечка сущностей |
| `ML_ENABLED` / `ML_MODEL_PATH` | true / `models/ner` | `Settings` | загрузка NER |
| `NER_DEVICE` / `NER_THREADS` | cpu / 2 | `NerExtractor.__init__` | |
| `CATEGORY_CLASSIFIER` / `CATEGORY_MODEL_PATH` | on / `models/category` | `CategoryClassifier.load` | off, 0, false, no — без классификатора |
| `LLM_ENABLED` | auto | `config._llm_mode` | auto — включена, если задан `LLM_*`, `OPENAI_*` или `LLM_LOCAL_URL` |
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | — / — / auto | `config` | явный провайдер; `OPENAI_API_KEY` не отправляется на чужой `LLM_BASE_URL` |
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL` | — | `config` | фолбэк: так ключ организаторов задан на ВМ |
| `LLM_LOCAL_URL` | — (в compose `http://host.docker.internal:11434/v1`) | `config`, `LLMClient.enabled` | «если запущена» — пинг `/models` |
| `LLM_MODEL_PREFERENCE` | `qwen3:4b-instruct,qwen2.5:7b-instruct,gpt-4.1-mini,…` | `config` | порядок выбора при `LLM_MODEL=auto` |
| `LLM_RESPONSE_FORMAT` | json_schema | клиент | начало цепочки json_schema → json_object → none |
| `LLM_TIMEOUT` / `LLM_MAX_RETRIES` / `LLM_RETRY_AFTER` | 60 с / 0 / 30 с | клиент | таймаут, повторы SDK, пауза предохранителя |
| `LLM_REASONING_EFFORT` | пусто | клиент | `none` отключает «размышления» qwen3.x |
| `LLM_CONCURRENCY` | 0 (авто: 1 для Ollama, 4 для API) | клиент | `client.map` / `extract_batch` — LLM-шаг пачки в `extract_many` |
| `LLM_MAX_TOKENS` / `LLM_MAX_INPUT_CHARS` / `LLM_ENUM_VALUES_MAX` | 700 / 3500 / 60 | клиент, промпт | |
| `LLM_FEW_SHOT` / `LLM_SCHEMA_ENUM` / `LLM_PREFILTER` / `LLM_CACHE` | false / false / true / true | промпт, клиент | |
| `LLM_MAX_CALLS` / `LLM_DAILY_MAX_CALLS` | 5000 / 0 | клиент | лимит на процесс / на сутки (0 — нет) |
| `LLM_PRICES_PATH` | `config/llm_prices.yaml` | `llm/cost.py` | тарифы |
| `TAXONOMY_PATH` | `config/taxonomy.yaml` | `Settings`, `category.py` | его папка — источник справочников, `enum_map.json`, `model_styles.json` |
| `DATASET_CONFIG` | `config/dataset.yaml` | `io.load_mapping` | маппинг колонок датасета |
| `CACHE_PATH` / `FEEDBACK_PATH` / `PREDICTIONS_PATH` | `outputs/llm_cache.sqlite` / `outputs/feedback.jsonl` / `outputs/predictions.jsonl` | `Settings` | |
| `ATTRX_ROOT` | — | `config._find_root` | корень проекта (config/, web/, data/) |

## 5. Предупреждения (`ExtractionResult.warnings`)

| Текст | Когда | Кто читает |
|---|---|---|
| `llm: отброшено незаземлённое значение {attr}={value!r}` | LLM-кандидат не прошёл `verify.ground` | UI «Спорные и отброшенные», `genui._has_rejected` |
| `llm: {причина}` | `LLMUnavailable`: нет соединения, таймаут, ключ, 429, лимиты, не JSON, ответ обрезан | `llm_used=False` |
| `кандидат в справочник {dictionary}: {attr} = "{value}" (LLM, цитата «…») — в справочнике нет, отброшено` | LLM назвала значение вне справочника, цитата есть в тексте | модератор (UI «Спорные и отброшенные»), пополнение справочника |
| `llm: {attr} = "{value}" не приводится к каноническому виду (…) — отброшено` | число / bool из ответа LLM не нормализовалось | UI «Спорные и отброшенные» |
| `{attr}: конфликт значений, альтернативы: {v1}, {v2}` | у значения есть альтернативы | `api._warning_alternatives`, `genui._disputed` |
| `категория определена неуверенно ({source}: {title}, {conf:.2f}) — карточка на проверку` | `decision.review` | UI |

## 6. Тайминги

`timings_ms`: `category`, `seller`, `rules`, `ml` и `llm` (если шаг выполнялся), `aggregate`, `total` — мс по `time.perf_counter`.
В пачках `extract_many` (с NER или LLM): `ml` — доля карточки во времени прогона NER пачки, `llm` — время её запроса,
`total` — сумма этапов карточки (не время ожидания пачки).
Первая карточка процесса медленнее: ленивая загрузка классификатора, дерева каталога, `traffic_weights.yaml`, компиляция регулярок.

## 7. Пакетные пути

| Путь | Как | Параллельность |
|---|---|---|
| `Extractor.extract_many` | без NER и LLM — по одной карточке; с ними — пачки по `pipeline.BATCH` (32): продавец и правила, один прогон NER на пачку, LLM-запросы пачки параллельно, финализация; порядок и значения — как у `extract` | NER — пачкой, LLM — `LLM_CONCURRENCY` |
| `attrx extract -i … [--llm on\|off]` | `extract_many` → `io.write_predictions` (jsonl + csv) | как `extract_many` |
| `attrx demo [--ml on] [--llm on]` | 400 карточек `data/sample/real_cards.jsonl` → метрики `eval.real.report` по `real_gold.jsonl` → `outputs/` | как `extract_many` |
| `POST /api/extract/batch` | до 1000 карточек, `extract_many` | как `extract_many` |
| `attrx eval-real` / `calibrate` / `report` | `extract_many` с выбранными `sources` | как `extract_many` |
| `extractors.llm.extract_batch` | `client.map` в `LLM_CONCURRENCY` потоках, порядок сохраняется; `info` — заметки, `asked`, время задачи | да (`extract_many`) |
| `NerExtractor.extract_batch` / `entities_batch` | один прогон модели на пачку | `extract_many`, калибровка NER (`ner_distant.calibration_points`) |
| `CategoryClassifier.predict_many` | пачки по 512 | `train_category`, не пайплайн |

## 8. Кэширование

| Что | Где | Ключ / сброс |
|---|---|---|
| Ответы LLM | `outputs/llm_cache.sqlite`, таблица `cache` (`CACHE_PATH`, `LLM_CACHE`) | sha256(модель, system, user, JSON-схема); бессрочно — удалить файл |
| Суточный счётчик LLM | та же БД, таблица `usage` (day, calls, токены) | для `LLM_DAILY_MAX_CALLS` |
| Таксономия и справочники | `load_taxonomy` lru_cache(4) по пути | перезапуск или `cache_clear()` (вызывают `attrx learn`, `build-dicts`) |
| Калибровка | `load_calibrator` lru_cache(4) | перезапуск |
| Классификатор, дерево каталога, трафик-веса | объект `Taxonomy`, lru_cache | перезапуск |
| Модель LLM при `auto` и доступность | клиент: проба `/models` (статус — 30 с, `enabled` для `LLM_LOCAL_URL` — 60 с, повторный выбор после фолбэка — 60 с) | — |
| Примеры для UI | `api._cards_cache` по mtime файла | автоматически |
| Регулярки справочников и NER | `Dictionary._patterns` (по размеру lookup), `ner._DICT_RES`, `ner._SCORERS` | — |

## 9. Отказы и деградация

| Ситуация | Поведение |
|---|---|
| LLM не настроена (`auto` без ключей) или `LLM_ENABLED=false` | шаг 7 пропускается; слабые атрибуты остаются `review` / `missing` |
| `LLM_LOCAL_URL`, сервер не запущен | `client.enabled` = False по пингу `/models` → без LLM |
| нет соединения | `LLMUnavailable` → `warnings`; предохранитель на `LLM_RETRY_AFTER` (30 с): карточки не ждут таймаут |
| ключ не принят / 429 | предохранитель ≥ 300 с / на `Retry-After` (нет заголовка — `LLM_RETRY_AFTER`) |
| таймаут | первый — ошибка карточки; второй подряд — предохранитель |
| формат не поддержан | откат json_schema → json_object → none (запоминается); неподдержанные параметры (`reasoning_effort`, `temperature`, `max_tokens`) снимаются |
| пустой, не-JSON или обрезанный ответ | `LLMUnavailable` с подсказкой (`LLM_REASONING_EFFORT=none`, `LLM_MAX_TOKENS`) |
| исчерпаны `LLM_MAX_CALLS` / `LLM_DAILY_MAX_CALLS` | `LLMUnavailable`; кэш продолжает отвечать |
| нет torch/transformers или `models/ner` | `ml = None`, шаг 6 пропускается; UI показывает «нет модели» |
| нет `models/category`, `CATEGORY_CLASSIFIER=off` или битый артефакт | предупреждение в лог, лестница категории идёт по ключевым словам |
| нет `data/processed` | путь категории — название категории таксономии; id вне таксономии сверяется с классами классификатора |
| нет `config/calibration.json` | уверенность = сырая оценка; порог = `CONFIDENCE_THRESHOLD` или 0.7 |
| нет `config/traffic_weights.yaml` | `traffic_weight` из таксономии |
| ошибка в таксономии | `ValueError` при загрузке (fail fast) |
