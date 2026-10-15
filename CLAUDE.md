# CLAUDE.md — attrx (кейс М.Видео, хакатон «Точка сборки»)

Гибридный пайплайн автозаполнения характеристик товаров: карточка (title + description + атрибуты продавца) →
категория → извлечение (продавец → правила/словари → NER → LLM только для хвоста) → нормализация → проверка заземления →
уверенность → статус (accepted / review / missing) → очередь модератора → обратная связь в словари. Плюс FastAPI и UI модератора с generative UI.

## База знаний — читать отсюда, а не пересобирать заново
**[.claude/context/INDEX.md](.claude/context/INDEX.md)** — оглавление: правила хакатона и критерии оценки, требования кейса,
стратегия победы, данные, PRD, архитектура, технологии, хранилища, оркестратор, карта проекта, решения, статус.
Перед любой задачей открой INDEX.md и только нужные файлы из него. Обновляй соответствующий файл, когда меняешь то, что он описывает.

## Что нельзя забывать
- **Стоп-код для всех — 23.10** (зачётное LLM-ревью репозитория в Gitea, ветка `main`); 29.10 00:01 — только для финалистов; финал 31.10.
  Ближайшее: продуктовый чекпоинт 15.10 17:00 и 16.10 17:00. Подробно — `.claude/context/HACKATHON.md`.
- 5 из 8 критериев ставит **LLM, читающий репозиторий** по требованиям кейса → README и доки должны явно доказывать каждое требование
  (матрица «требование → где → метрика»). Никогда не выдавать синтетические метрики за реальные.
- Демо на ВМ слушает **порт 8000**. На ВМ заданы `OPENAI_API_KEY`/`OPENAI_BASE_URL` — LLM-клиент их подхватывает; без LLM всё работает.
- **Коммиты без `Co-Authored-By`** и прочих соавторов (правило команды, CONTRIBUTING.md). Финальная сдача — `git push gitea main`.
- Данные: `data/raw/` (1.1 ГБ архив организаторов, не в git), `data/processed/` (не в git), `data/sample/` (в git, маленький).
  **`data/raw/skus.pkl` при загрузке съедает 12 ГБ RAM** — работать с `data/processed/*.parquet`.
- Сторонний код/ассеты — только с указанием источника (правило хакатона) → `web/assets/CREDITS.md`, README.
- Язык: README, docs, UI, комментарии и docstring — по-русски. Коротко и по делу.

## Команды
~~~bash
export PATH="$HOME/.local/bin:$PATH"            # uv
uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python -e ".[dev]"   # [ml] — torch/transformers для NER
.venv/bin/pytest -q && .venv/bin/ruff check .   # то же гоняет CI (+ attrx demo)
.venv/bin/attrx prepare                         # data/raw -> data/processed
.venv/bin/attrx demo | serve | extract | eval | report | train-ner | split | learn
AGENT_BROWSER_EXECUTABLE_PATH=/usr/bin/chromium agent-browser ...   # скриншоты/QA UI
~~~
Docker на этой машине требует root (пользователь не в группе docker) — локально запускать нативно.

## Навыки проекта (.claude/skills)
`frontend-design` (Anthropic), `agent-browser` (браузер для QA), `find-skills` (поиск новых навыков), `grill-me` → `grilling`
(стресс-тест плана интервью). Источники и хэши — `skills-lock.json`.
