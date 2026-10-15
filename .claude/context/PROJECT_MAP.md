# Карта репозитория

Пометки: **[gen: …]** — файл создаёт команда, руками не править; **[не в git]** — локально или на ВМ; без пометки — пишется
руками. Не показаны `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `src/attrx.egg-info/`. Где лежат данные и
кто их читает — [STORAGE.md](STORAGE.md). Сверено 15.10.

```
mvideo-attr-extractor/
├── README.md                    карта для ревьюера: результат, требования кейса → где, артефакты, быстрый старт
├── CLAUDE.md                    вход для Claude: база знаний, дедлайны, команды
├── CONTRIBUTING.md              как работаем: CI зелёный, ветки и PR, без соавторов в коммитах
├── pyproject.toml               пакет attrx 0.1.0, зависимости, extras [dev] [ml] [eda], ruff, pytest
├── Makefile                     install, test, lint, demo, serve, data, prepare, traffic, eda-real, eda-nb (+ eda/ner/report на синтетике)
├── Dockerfile                   python:3.11-slim, базовые зависимости → attrx serve :8000
├── docker-compose.yml           app :8000 + профиль llm: ollama и ollama-pull (модель по умолчанию qwen3:4b-instruct)
├── .env.example                 переменные LLM / ML / порогов / путей с пояснениями (.env — [не в git])
├── .gitignore                   data/raw, data/processed, outputs/*, models/* кроме models/category, .env
├── skills-lock.json             источники и хэши навыков .claude/skills
├── .github/workflows/ci.yml     CI: pip install -e ".[dev]", ruff, pytest, attrx demo
├── .claude/
│   ├── context/                 база знаний команды (INDEX.md — оглавление)
│   │   └── sources/             тексты страниц кейса, хакатона и памятки (14.10)
│   └── skills/                  навыки: agent-browser, find-skills, frontend-design, grill-me, grilling, interface-design
├── src/attrx/
│   ├── __init__.py              версия
│   ├── schema.py                контракт: Card, Candidate, AttributeValue, ExtractionResult
│   ├── config.py                ROOT, .env, Settings из окружения, выбор источника LLM
│   ├── taxonomy.py              таксономия и справочники, BrandDictionary, лестница категории, трафик-вес, путь категории
│   ├── category.py              классификатор категории: TF-IDF + «раздел → лист», инференс (numpy) и обучение
│   ├── normalize.py             единицы, числа, enum, цвета, процессоры, model_core, display
│   ├── verify.py                заземление: выравнивание цитаты + «значение следует из фрагмента»
│   ├── confidence.py            агрегация кандидатов (noisy-OR) и калибровка (PAV)
│   ├── pipeline.py              оркестратор Extractor (подробно — ORCHESTRATOR.md)
│   ├── feedback.py              журнал решений модератора, attrx learn → learned.yaml
│   ├── dictgen.py               attrx build-dicts: бренды, enum_map, model_styles по train
│   ├── io.py                    чтение датасета и data/processed, запись и чтение предсказаний
│   ├── cli.py                   20 команд attrx (таблица ниже)
│   ├── extractors/
│   │   ├── seller.py            поля продавца → атрибуты по aliases
│   │   ├── rules.py             правила и справочники со смещениями в full_text()
│   │   ├── ner.py               NER на ruBERT-tiny2: вид карточки, обучение, инференс, калибровка сущностей
│   │   ├── ner_distant.py       distant supervision: эталон train → спаны и маски → обучение → models/ner
│   │   └── llm.py               промпт и схема из таксономии, префильтр, разбор ответа
│   ├── llm/
│   │   ├── client.py            OpenAI-совместимый клиент: откат формата, предохранитель, лимиты, учёт
│   │   ├── cache.py             SQLite-кэш ответов и суточный счётчик
│   │   ├── status.py            состояние LLM (attrx llm-status)
│   │   ├── cost.py              ₽ на 1000 карточек по config/llm_prices.yaml
│   │   └── bench.py             attrx llm-bench на реальных карточках
│   ├── eval/
│   │   ├── gold.py              эталон из каталога (attrx build-gold)
│   │   ├── metrics.py           сравнение (match_key), P/R/F1, покрытие, new_fills, traffic-weighted, сегменты, кривая порога
│   │   ├── real.py              attrx eval-real: стратифицированная выборка, прогон, отчёт
│   │   ├── calibrate.py         attrx calibrate: кривые и порог на dev
│   │   ├── report.py            attrx report → submission/
│   │   └── eda.py               attrx eda — старый EDA по списку Card
│   ├── data/
│   │   ├── __init__.py          split_of (md5 80/10/10), atomic_path
│   │   ├── prepare.py           attrx prepare: шаги yml, clicks, cards, taxonomy, sample
│   │   ├── yml.py               skus.pkl → categories / offers / params (unpickler только встроенных типов)
│   │   ├── text.py              HTML → текст, блок «Характеристики:» → seller_attrs
│   │   ├── clicks.py            агрегаты кликов через DuckDB
│   │   ├── taxo.py              taxonomy_candidates.json, param_stats
│   │   ├── traffic.py           attrx traffic: трафик-веса из поисковых запросов
│   │   └── eda.py               attrx eda-real и функции для ноутбука
│   ├── genui/
│   │   ├── schema.py            словарь компонентов, UISpec, strict JSON Schema
│   │   └── composer.py          экран: шаблон / по ключевым словам / LLM
│   └── api/app.py               FastAPI: /api/*, раздача web/
├── web/                         UI модератора без сборки (подробно — web/README.md)
│   ├── index.html               каркас: шапка, три раздела, палитра, справка
│   ├── app.js                   маршрутизация по #, горячие клавиши, тема, общие данные
│   ├── components.js            реестр компонентов generative UI
│   ├── tokens.css, styles.css   дизайн-токены (светлая и тёмная тема) и компоненты
│   ├── js/                      api, dom, copy (подписи), evidence (подсветка цитат), attrs, workbench («Карточка»),
│   │                            review («Очередь»), quality («Качество»), charts (SVG), palette (⌘K), help, ui
│   └── assets/                  шрифты Onest и Literata, иконки Phosphor, favicon, лицензии, CREDITS.md
├── config/
│   ├── taxonomy.yaml            12 категорий + generic: атрибуты, aliases, gold, keywords
│   ├── dataset.yaml             маппинг колонок датасета → Card
│   ├── brand_aliases.yaml       ручные синонимы брендов (вход build-dicts)
│   ├── llm_prices.yaml          тарифы LLM с источниками, on-prem ₽/ч
│   ├── enum_map.json            [gen: attrx build-dicts] находки справочника → значение каталога
│   ├── model_styles.json        [gen: attrx build-dicts] как бренд пишет модель
│   ├── calibration.json         [gen: attrx calibrate] кривые калибровки и порог
│   ├── traffic_weights.yaml     [gen: attrx traffic] веса атрибутов по поисковому спросу
│   └── dictionaries/
│       ├── brands.yaml          [gen: attrx build-dicts] 4 822 бренда каталога
│       ├── colors.yaml          17 базовых цветов, RU/EN
│       ├── chambers, connection_types, defrost, headphone_types, materials, resolutions, tv_os,
│       │   vacuum_types, washer_types .yaml    справочники enum-атрибутов
│       └── learned.yaml         [gen: attrx learn] синонимы из правок модераторов (создаётся при первом learn)
├── models/
│   ├── category/                [gen: attrx train-category] model.npz, vocab.json.gz, meta.json, metrics.json — в git
│   └── ner/                     [gen: attrx train-ner --distant] [не в git]
├── data/
│   ├── raw/                     [не в git] архив организаторов (make data)
│   ├── processed/               [не в git] [gen: prepare, build-gold, traffic, eval-real, eda-real]
│   └── sample/                  cards.jsonl, gold.jsonl (синтетика, 20) + real_{cards,params}.jsonl [gen: prepare]
│                                и real_gold.jsonl [gen: build-gold] — 400 карточек holdout
├── outputs/                     [не в git] predictions, metrics, feedback.jsonl, llm_cache.sqlite, llm_bench/
├── submission/                  [gen: attrx report] predictions.jsonl/.csv, metrics, compare, segments, errors.csv,
│                                REPORT.md; traffic.json [gen: attrx traffic]
├── docs/
│   ├── ARCHITECTURE.md          архитектура для жюри
│   ├── EVALUATION.md            протокол оценки и результаты правил (dev)
│   ├── CATEGORY.md              классификатор категории (holdout)
│   ├── LLM.md                   LLM-слой, бенчмарк, стоимость, стенд
│   ├── NER.md                   NER на distant supervision (пишет поток NER)
│   ├── EDA.md                   сводка EDA, «как EDA определила решение»
│   ├── PRD.md                   продукт: пользователи, требования, KPI, раскатка
│   ├── RISKS.md, DEPLOYMENT.md  риски кейса с мерами; внедрение в контур
│   ├── GENUI.md                 generative UI и словарь компонентов
│   ├── PLAN.md                  план хакатона
│   ├── PRESENTATION.md          структура финальной презентации
│   ├── SOLUTION_CARD.md         черновик карточки решения для платформы
│   ├── slides.html, slides.pdf  финальные слайды (старая версия на синтетике, пересобрать)
│   ├── llm_bench.json           сводка бенчмарка LLM
│   ├── img/                     eda_*.png [gen: attrx eda-real], ui_*.png (скриншоты)
│   └── checkpoint/              концепция к чекпоинту 15–16.10: concept.html/.pdf, SPEECH.md,
│                                demand_stats.py → demand_stats.json
├── notebooks/01_eda.ipynb       EDA реальных данных (исполнен; make eda-nb)
├── scripts/download_data.sh     загрузка и распаковка архива кейса
└── tests/                       pytest, без data/processed (LLM и ML выключены в conftest.py)
    ├── test_pipeline.py         сквозные случаи пайплайна, «нет в тексте — не выдумываем»
    ├── test_rules_real.py       правила на реальных паттернах каталога
    ├── test_normalize.py        числа с единицами, процессоры
    ├── test_grounding.py        выравнивание цитат, инвариант «только LLM → review»
    ├── test_llm.py              маршрутизация в LLM, отброс галлюцинаций, generative UI
    ├── test_llm_prompt.py       промпт со справочниками, разбор ответа, стоимость
    ├── test_llm_client.py, test_llm_client_mock.py   клиент: форматы, токены, кэш, лимиты, предохранитель
    ├── test_llm_config.py       настройки LLM из окружения (OPENAI_*, auto)
    ├── test_llm_bench.py        оценка ответа в бенчмарке
    ├── test_category.py         классификатор: признаки, отображение, инференс, обучение на синтетике
    ├── test_ner.py, test_ner_model.py   NER: BIO-метки, слабая разметка, модель (пропуск без [ml])
    ├── test_ner_distant.py      distant supervision: спаны, маски, вид карточки, калибровка
    ├── test_eval_gold.py        эталон, правила сравнения, метрики на неполном эталоне
    ├── test_report.py           attrx report: режимы, сегменты, ошибки
    ├── test_traffic.py          детектор атрибутов в поисковых запросах
    ├── test_data_text.py        очистка HTML, блок «Характеристики:», сплиты
    └── test_api.py, test_api_ui.py   API и эндпоинты интерфейса
```

## Команды `attrx`

| Команда | Что делает | Пишет |
|---|---|---|
| `extract -i <файл> [--llm on\|off]` | карточки (CSV/JSON/JSONL/Parquet по `config/dataset.yaml`) → предсказания | `outputs/predictions.jsonl` + `.csv` |
| `eval --gold <jsonl>` | метрики предсказаний по эталону + кривая порога | `outputs/metrics.json` |
| `demo [--llm …]` | `extract` + `eval` на `data/sample/cards.jsonl` (синтетика, 20 карточек) | `outputs/` |
| `serve [--host 127.0.0.1] [--port 8000] [--reload]` | API + UI (по умолчанию только 127.0.0.1; на ВМ — `--host 0.0.0.0`, `make serve-public`) | — |
| `learn` | правки модераторов → синонимы справочников | `config/dictionaries/learned.yaml` |
| `prepare [--steps yml,clicks,cards,taxonomy,sample]` | `data/raw` → `data/processed` | `data/processed/`, `data/sample/real_*` |
| `build-gold` | эталон из каталога | `gold.parquet`, `gold_{dev,holdout}.jsonl`, `data/sample/real_gold.jsonl` |
| `traffic [--refresh]` | трафик-веса атрибутов из поисковых кликов | `config/traffic_weights.yaml`, `submission/traffic.json` |
| `eda-real` | графики и цифры EDA реальных данных | `docs/img/eda_*.png`, `eda_stats.json`, `business_stats.json` |
| `build-dicts [--skip-brands] [--skip-enums]` | справочники из каталога (train) | `brands.yaml`, `enum_map.json`, `model_styles.json` |
| `train-category [--search]` | классификатор категории | `models/category/` |
| `train-ner --distant [--dry-run]` | NER на distant supervision (без `--distant` — слабая разметка от правил по `-i`) | `models/ner/` |
| `calibrate [--sources …]` | калибровка уверенности и порог (только dev) | `config/calibration.json` |
| `eval-real [--split dev] [--sources …] [--out …]` | метрики пайплайна на реальном эталоне | `data/processed/<имя>.json` |
| `report -i <карточки> --gold <эталон>` | сравнение режимов, сегменты, ошибки, предсказания | `submission/` |
| `llm-status [--no-probe]` | провайдер, модель, доступность LLM | stdout |
| `llm-bench --models … [--n 30]` | бенчмарк LLM: латентность, JSON, заземление, точность, ₽ | `outputs/llm_bench/` |
| `prelabel` | предсказания → черновик эталона для ручной проверки | `data/gold_draft.jsonl` |
| `split --gold <jsonl>` | разбиение произвольного эталона JSONL (старый путь; сплиты кейса — `split_of`) | `<имя>_{train,holdout}.jsonl` |
| `eda -i <файл>` | старый EDA по датасету в формате `Card` | `outputs/eda/` |
