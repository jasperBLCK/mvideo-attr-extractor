# attrx — автозаполнение характеристик товаров (кейс М.Видео №3, NLP и e-commerce)

Пайплайн принимает текстовую карточку товара (название, описание, данные продавца), определяет категорию,
извлекает обязательные характеристики, приводит их к единому формату и для каждого значения указывает
**уверенность, источник и цитату из текста**. Если уверенности не хватает, значение уходит модератору, выдумывать его модель не будет.

Сверху сделан **generative UI**: интерфейс модератора собирается под результат и под запрос
(«покажи только то, что надо проверить, с цитатами»). LLM выбирает компоненты из нашего фиксированного набора, HTML и JS она не пишет.

## Почему так

| Требование кейса | Как закрыто |
|---|---|
| Не придумывать атрибуты | Каждое значение LLM проверяется на «заземление»: цитата обязана быть в карточке, значение обязано из неё следовать (src/attrx/verify.py) |
| Нормализация | Единицы (ТБ→ГБ, см→дюймы, Ач→мА·ч), справочники синонимов с падежами/родами (config/dictionaries) |
| Уверенность | Агрегация источников (noisy-OR), штраф за конфликты; порог подбирается по кривой precision/coverage |
| Ручная проверка | Статусы accepted / review / missing, очередь проверки в UI, правки → feedback.jsonl → новые синонимы |
| Правила → ML → LLM | Сначала дешёвые правила, затем NER на ruBERT-tiny2 (BIO-теги, слабая разметка от правил), LLM только для того, что осталось непокрытым. Есть кэш и лимит вызовов |
| Честная оценка | attrx report: таблица «правила / +продавец / +ML / +LLM», качество по сегментам продавцов, разбор ошибок; attrx split — holdout из gold |
| Метрики | P/R/F1 по атрибутам, coverage, traffic-weighted F1, разрез по категориям, gold-набор |
| Внедрение в М.Видео | Любой OpenAI-совместимый endpoint (GigaChat, YandexGPT, vLLM/Ollama on-prem). Без LLM всё работает на правилах. VPN и прокси не нужны |

## Архитектура

~~~
Карточка ─► категория ─► продавец ─► правила/регулярки ─► агрегация уверенности
                                                          │  слабые/пустые/конфликтные атрибуты
                                                          ▼
                                       LLM (strict JSON + цитата) ─► проверка заземления
                                                          ▼
                     ExtractionResult {value, unit, confidence, status, sources, evidence}
                          │                     │                         │
                   predictions.jsonl/csv   FastAPI /api/extract   generative UI (UISpec)
                          │                                               │
                   attrx eval (метрики)                       модератор ─► feedback ─► словари
~~~

Подробнее: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), план: [docs/PLAN.md](docs/PLAN.md),
внедрение: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md), риски: [docs/RISKS.md](docs/RISKS.md), generative UI: [docs/GENUI.md](docs/GENUI.md).

## Быстрый старт

~~~bash
uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python -e ".[dev,ml]"   # или: make install
cp .env.example .env
.venv/bin/attrx demo            # прогон на data/sample + метрики в outputs/metrics.json
.venv/bin/attrx serve           # http://localhost:8000 — UI модератора + API (/docs — Swagger)
.venv/bin/pytest -q && .venv/bin/ruff check .
~~~

Docker: **docker compose up** (только правила) или **docker compose --profile llm up** (плюс локальная Ollama).

## Когда выдадут датасет

1. Положить файл в data/raw/ (в git не попадает).
2. Поправить имена колонок в config/dataset.yaml (код трогать не нужно).
3. Обновить обязательные атрибуты в config/taxonomy.yaml.
4. Прогнать EDA: **attrx eda -i data/raw/train.csv** или notebooks/01_eda.ipynb. Топ ключей продавцов из отчёта добавить в aliases.
5. **attrx extract -i data/raw/test.csv** → outputs/predictions.jsonl и .csv.
6. **attrx eval --gold data/raw/gold.jsonl** → метрики и кривая порога.

## ML-слой (NER) и отчёт для сдачи

~~~bash
uv pip install -p .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -p .venv/bin/python -e ".[ml]"
attrx split --gold data/raw/gold.jsonl                      # gold_train + gold_holdout (holdout не трогаем)
attrx train-ner -i data/raw/train.csv --gold data/raw/gold_train.jsonl   # -> models/ner (в git не идёт)
attrx report -i data/raw/holdout.csv --gold data/raw/gold_holdout.jsonl  # -> submission/
attrx prelabel                                              # черновик gold из предсказаний для ручной разметки
~~~

NER учится на **слабой разметке**: спаны значений, которые нашли правила, а при наличии gold остаются только те, что с ним совпали.
В пайплайне NER работает только на «хвосте»: атрибутах, которые продавец и правила не закрыли уверенно. Поэтому слабая модель не портит то, что правила уже сделали хорошо.

submission/ — артефакт для сдачи: predictions.jsonl/.csv, metrics.json, compare.json (сравнение подходов), segments.json (coverage по сегментам), errors.csv, REPORT.md.

## LLM-провайдер (.env)

| Провайдер | LLM_BASE_URL | Примечание |
|---|---|---|
| Ключ организаторов | выдадут | скорее всего OpenAI-совместимый |
| GigaChat | https://gigachat.devices.sberbank.ru/api/v1 (или через gpt2giga) | поддерживает json_schema |
| YandexGPT (AI Studio) | https://llm.api.cloud.yandex.net/v1 | OpenAI-совместимый |
| Локально (vLLM / Ollama) | http://localhost:11434/v1 | данные не уходят из контура |

LLM_RESPONSE_FORMAT: json_schema → json_object → none. Если провайдер не поддерживает формат, клиент сам повторит запрос без него.

## API

| Метод | Что делает |
|---|---|
| POST /api/extract | {card, use_llm?, intent?} → {result, ui} |
| POST /api/extract/batch | до 1000 карточек |
| POST /api/ui | пересобрать интерфейс под новый intent |
| GET /api/review | очередь проверки из predictions |
| POST /api/feedback | правка модератора |
| GET /api/metrics, /api/taxonomy, /api/health | служебное |

## Структура

~~~
config/            taxonomy.yaml, dataset.yaml, dictionaries/*.yaml (+ learned.yaml из фидбэка)
src/attrx/
  extractors/      seller.py, rules.py, ner.py (ML), llm.py
  normalize.py     единицы, числа, справочники, процессоры
  verify.py        проверка заземления (анти-галлюцинации)
  confidence.py    агрегация источников
  pipeline.py      оркестратор
  llm/             OpenAI-совместимый клиент + SQLite-кэш
  genui/           схема компонентов и композитор UI
  eval/            метрики, EDA, report.py (сравнение подходов, сегменты, ошибки)
  api/app.py       FastAPI
  cli.py           attrx extract | eval | eda | serve | learn | demo | train-ner | report | split | prelabel
web/               UI модератора (без сборки, vanilla JS, стиль М.Видео)
submission/        предсказания + отчёт (обязательный артефакт сдачи)
data/sample/       20 синтетических карточек + gold (smoke-тест, не настоящая оценка)
notebooks/         EDA
docs/              план, архитектура, внедрение, риски, презентация
~~~

## Команда

| Кто | Зона |
|---|---|
| jasperBLCK | правила, нормализация, LLM-промпты, метрики, EDA |
| Makhkets | UI и generative UI, API-интеграция, демо, презентация |

Как работаем: [CONTRIBUTING.md](CONTRIBUTING.md).

> Важно: data/sample написан вручную, и правила отлаживались на нём же. Поэтому метрики на sample просто проверяют, что всё работает, это не оценка качества. Реальные цифры будут только на gold-наборе организаторов.
