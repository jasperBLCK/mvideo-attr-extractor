# CLAUDE.md — attrx (кейс М.Видео, хакатон «Точка сборки»)

Гибридный пайплайн автозаполнения характеристик товаров: карточка (title + description + атрибуты продавца) →
категория → извлечение (продавец → правила/словари → NER → LLM только для хвоста) → нормализация → проверка заземления →
уверенность → статус (accepted / review / missing) → очередь модератора → обратная связь в словари. Плюс FastAPI и UI модератора с generative UI.

## Начало новой сессии
Сначала прочитай **последний** файл в `.claude/handoff/` (HANDOFF-NNN_дата.md, сейчас HANDOFF-001) — там состояние,
решения пользователя и открытые задачи. В конце крупной сессии создай следующий HANDOFF с номером +1.

## База знаний — читать отсюда, а не пересобирать заново
**[.claude/context/INDEX.md](.claude/context/INDEX.md)** — оглавление: правила хакатона и критерии оценки, требования кейса,
критерии → подтверждения, данные, PRD, архитектура, технологии, хранилища, оркестратор, карта проекта, решения, статус.
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
.venv/bin/attrx demo | serve | extract | eval-real | report-real | verify | monitor | train-ner --distant | calibrate | learn
make report                                     # финальный отчёт на holdout -> submission/ (только осознанно!)
AGENT_BROWSER_EXECUTABLE_PATH=/usr/bin/chromium agent-browser ...   # скриншоты/QA UI
~~~
Docker на этой машине требует root (пользователь не в группе docker) — локально запускать нативно.

Локальный стенд (user-юниты, не в автозапуске): `systemctl --user start|stop|restart attrx ollama-attrx` →
http://127.0.0.1:8000 (LLM qwen3:4b-instruct через Ollama/Vulkan, NER на GPU). Юниты: `~/.config/systemd/user/`.
После правок кода — `systemctl --user restart attrx`.

## Навыки проекта (.claude/skills)
`frontend-design` (Anthropic), `interface-design` (интерфейсы-инструменты), `agent-browser` (браузер для QA),
`find-skills` (поиск новых навыков), `grill-me` → `grilling` (стресс-тест плана интервью). Источники и хэши — `skills-lock.json`.
