# Технологии

Версии — из `.venv` на 15.10 (`uv pip list -p .venv/bin/python`). Почему так устроено в целом — [ARCHITECTURE.md](ARCHITECTURE.md).

## Языки и инструменты

| Что | Версия | Почему |
|---|---|---|
| Python | 3.11 (venv 3.11.15; `requires-python >= 3.10`; Docker `python:3.11-slim`) | ML-экосистема, pydantic, один язык от данных до API |
| JavaScript (ES-модули) + CSS | без сборки, без npm | ревьюер запускает по README; работает в закрытом контуре |
| YAML / JSON | — | таксономия, справочники, калибровка — правятся без кода |
| uv | `~/.local/bin/uv` | быстрые `uv venv -p 3.11 .venv`, `uv pip install -p .venv/bin/python -e ".[dev]"` |
| ruff 0.17.0, pytest 9.1.1 | правила `E, F, I, UP, B`, строка 110 | то же гоняет CI |
| Bash + curl | `scripts/download_data.sh` | загрузка архива с докачкой, без авторизации |

## Python-зависимости

Базовые (`pip install -e .`, их ставят Docker и CI):

| Пакет | Версия | Зачем |
|---|---|---|
| pydantic | 2.14.1 | контракт данных (`schema.py`), валидация `UISpec` |
| pyyaml | 6.0.3 | таксономия, справочники, трафик-веса |
| rapidfuzz | 3.14.6 | нечёткий enum, опечатки при выравнивании цитат — быстрый C++ |
| openai | 3.31.0 | один клиент для любого OpenAI-совместимого API: Ollama, vLLM, GigaChat (gpt2giga), YandexGPT, шлюз организаторов |
| fastapi (starlette 1.8.0) | 0.143.1 | API + раздача `web/`; Swagger `/docs` из коробки |
| uvicorn[standard] | 0.54.0 | ASGI-сервер `attrx serve` |
| pandas | 3.0.6 | подготовка данных, отчёты |
| pyarrow | 26.0.0 | parquet (zstd), чтение с фильтрами — 21 млн строк параметров без загрузки целиком |
| duckdb | 1.5.6 | SQL по parquet: 31 млн кликов, EDA, трафик-веса (лимит 4 ГБ RAM) |
| numpy | 2.4.6 (транзитивно через pandas) | инференс классификатора категории без scikit-learn и pickle |

Extras (`pyproject.toml`):

| Extra | Пакеты (версия в venv) | Для чего |
|---|---|---|
| `[dev]` | pytest 9.1.1, ruff 0.17.0, httpx 0.28.1, scikit-learn 1.9.1 | тесты, `TestClient`, тест обучения классификатора |
| `[ml]` | torch 2.14.1+cu126, transformers 4.57.6 (`<5`; tokenizers 0.22.2, safetensors 0.8.0, huggingface-hub 0.36.2), scikit-learn 1.9.1 (scipy 1.17.1), matplotlib 3.11.2, jupyterlab (в venv не установлен) | NER, обучение классификатора категории |
| `[eda]` | matplotlib, nbformat 5.11.1, nbconvert 7.17.2, ipykernel 7.4.0 | `attrx eda-real`, `notebooks/01_eda.ipynb` |
| `[xlsx]` | openpyxl (в venv не установлен) | `attrx extract / eval` по таблице `.xlsx` (`io.read_table`; без extra — ошибка с командой установки) |

## Модели

| Модель | Где лежит | Обучение | Инференс | Почему |
|---|---|---|---|---|
| Классификатор категории: TF-IDF (title: слова 1–2 + символьные 2–5-граммы; 300 символов описания и поля продавца с весом 0.5) → линейные SVM (SGD, hinge, OvR) «раздел → лист», temperature scaling на dev | `models/category/` — в git, 7,8 МБ: npz int8 + JSON, без pickle | `attrx train-category`, ~6 мин на 4 ядрах, scikit-learn | numpy на CPU, ~1 мс на карточку; порог 0.9 | 75% карточек без категории; дёшево, без GPU, артефакт воспроизводим. [docs/CATEGORY.md](../../docs/CATEGORY.md) |
| NER: `cointegrated/rubert-tiny2`, дообученный токен-классификатор (BIO) на distant supervision из каталога (train) + изотоническая калибровка сущностей | `models/ner/` — не в git | `attrx train-ner --distant` (GPU, если свободно ≥ 2,5 ГБ, иначе CPU) | torch, по умолчанию CPU (`NER_DEVICE`, `NER_THREADS`) | маленький русскоязычный энкодер, тянет CPU; разметка без ручного труда. `extractors/ner.py`, `ner_distant.py` |
| LLM по умолчанию: `qwen3:4b-instruct` (Qwen3-4B-Instruct-2507, Q4_K_M), без «размышлений» | Ollama, `127.0.0.1:11434` | — (zero-shot по схеме категории) | 3,0 ГБ VRAM; 6,9 с на карточку (все атрибуты), ~3,1 с на вызов в гибриде — GTX 1060 | лучший баланс качества и скорости в бенчмарке, [docs/LLM.md §5](../../docs/LLM.md) |
| LLM — альтернативы | `qwen2.5:7b-instruct` (GPU ≥ 8 ГБ); vLLM в контуре; GigaChat, YandexGPT; ключ организаторов на ВМ (`LLM_MODEL=auto` — модель из `/models`) | — | — | провайдер меняется переменными окружения |

## Фронтенд (`web/`)

- Vanilla ES-модули (`app.js`, `components.js`, `js/*.js`), дизайн-токены `tokens.css` + `styles.css`; SVG-графики без
  библиотек (`js/charts.js`). Подробно — [web/README.md](../../web/README.md).
- Шрифты Onest (интерфейс) и Literata (текст карточки) — SIL OFL 1.1; иконки Phosphor — MIT; всё self-hosted, источники и
  лицензии — `web/assets/CREDITS.md` (правило хакатона о стороннем коде).
- Разделы по `#`: `#/card`, `#/review`, `#/quality`; палитра Ctrl/⌘K (generative UI); светлая и тёмная тема; WCAG AA.
- Раздаёт FastAPI: `/static` (`StaticFiles`), `/` → `index.html`; `Cache-Control: no-cache` для всего, кроме шрифтов.

## Инфраструктура

| Что | Как устроено | Почему |
|---|---|---|
| Docker | `Dockerfile`: python:3.11-slim, `pip install -e .`, копирует `src`, `config`, `web`, `data/sample`, `submission`, `models/category`; `attrx serve --host 0.0.0.0 --port 8000` | демо на ВМ :8000 одной командой |
| ⚠ ограничения образа | только базовые зависимости (нет torch → NER выключен); классификатор категории в образе есть (`models/category`), NER-модели нет | — |
| docker compose | `app`: порт 8000, тома `./outputs`, `./config`, `./data`; `.env` необязателен; пробрасывает `OPENAI_API_KEY` / `OPENAI_BASE_URL`; `LLM_LOCAL_URL=http://host.docker.internal:11434/v1` (`host-gateway`). Профиль `llm`: `ollama` (11434, том `ollama`, KEEP_ALIVE 10m, CONTEXT 4096, NUM_PARALLEL 1) + `ollama-pull` (качает `LLM_LOCAL_MODEL`, по умолчанию qwen3:4b-instruct); GPU — раскомментировать `deploy` | LLM в контуре, данные не уходят |
| CI | GitHub Actions `.github/workflows/ci.yml`: Python 3.11, `uv sync --frozen --extra dev` по `uv.lock`, `ruff`, `pytest`, `attrx demo`, `attrx verify`, `attrx monitor` | `main` всегда зелёный, цифры сдачи воспроизводимы |
| Репозитории | рабочий — GitHub `jasperBLCK/mvideo-attr-extractor`; сдача — `git push gitea main` | правило хакатона |
| Локальная LLM (эта машина) | `ollama-attrx` — **transient** user-юнит systemd (создан `systemd-run --user`, после перезагрузки запускать заново — команда в [docs/LLM.md §7](../../docs/LLM.md)): `~/.local/opt/ollama-attrx/bin/ollama serve` (Ollama 0.33.3 + Vulkan-бэкенд), `OLLAMA_HOST=127.0.0.1:11434`, `OLLAMA_VULKAN=1`, `OLLAMA_KEEP_ALIVE=10m`, `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_CONTEXT_LENGTH=4096`, `LLAMA_ARG_FIT_TARGET=512`. Модели в `~/.ollama/models`: qwen3:4b-instruct, qwen2.5:7b-instruct, qwen3.5:4b, mistral. `systemctl --user status ollama-attrx`, логи — `journalctl --user -u ollama-attrx` | CPU без AVX → только GPU; CUDA 13 не поддерживает Pascal, CUDA-12-сборка медленнее Vulkan (8,1 против 6,9 с на карточку) |
| Машина команды | Pentium Gold G5400 (4 потока, без AVX), 31 ГБ RAM, GTX 1060 6 ГБ; Docker только от root → запускаем нативно | — |
| QA интерфейса | `AGENT_BROWSER_EXECUTABLE_PATH=/usr/bin/chromium agent-browser …` | скриншоты `docs/img/ui_*.png` |
