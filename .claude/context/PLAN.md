# План работ и владение файлами (параллельные агенты)

Цель: к 23.10 (стоп-код, зачётное LLM-ревью) — полностью готовое решение на реальных данных + редизайн UI + документация.
Ближайший дедлайн: **продуктовый чекпоинт 15.10 17:00**.

## Потоки
| Поток | Агент | Владеет файлами (пишет только он) |
|---|---|---|
| A. Данные и EDA | data | `src/attrx/data/` (новый пакет), `src/attrx/io.py`, `config/dataset.yaml`, `scripts/`, `notebooks/`, `docs/EDA.md`, `docs/img/eda_*`, `data/sample/`, `data/processed/` (gitignored), раздел «Результаты EDA» в `.claude/context/DATA.md` |
| B. Качество извлечения | extraction | `src/attrx/extractors/{rules,seller,ner}.py`, `normalize.py`, `taxonomy.py`, `confidence.py`, `verify.py`, `pipeline.py`, `config/taxonomy.yaml`, `config/dictionaries/`, `src/attrx/eval/`, `submission/` |
| C. UI и API | ui | `web/`, `src/attrx/api/`, `src/attrx/genui/`, `docs/GENUI.md`, `docs/img/ui_*` |
| D. LLM и on-prem | llm | `src/attrx/llm/`, `src/attrx/extractors/llm.py`, LLM-часть `config.py`, `.env.example`, `docker-compose.yml` |
| E. Продукт и питч | product | `docs/PRD.md`, `docs/PRESENTATION.md`, `docs/slides.*`, `docs/checkpoint/`, `docs/SOLUTION_CARD.md`, `docs/DEPLOYMENT.md`, `docs/RISKS.md` |
| Оркестратор | главная сессия | `CLAUDE.md`, `.claude/context/`, `README.md`, `docs/PLAN.md`, `docs/ARCHITECTURE.md`, git (коммиты и push делает только оркестратор) |

Общие файлы (`src/attrx/cli.py`, `src/attrx/schema.py`, `pyproject.toml`, `Makefile`, `tests/`): правки только **аддитивные**
(новая команда, новое опциональное поле, новый тестовый файл), перечитать файл прямо перед правкой. Агенты **не коммитят**.

## Контракт данных (src/attrx/schema.py — уже расширен)
- `Card`: `id, title, description, seller_attrs, category, category_id?, seller?`; `full_text()` = title + "\n" + description + "\n" + "k: v; ...".
- `AttributeValue`: `attribute, title, value, unit, display, confidence, status(accepted|review|missing), sources[seller|rules|ml|llm|human], evidence`,
  новые: `evidence_span: (start, end) | None` — смещения в `Card.full_text()`, `alternatives: [{value, confidence, sources}]`, `required: bool`, `traffic_weight: float`.
- `ExtractionResult`: `card_id, category, category_confidence, attributes, llm_used, warnings`, новые: `category_title?, category_path[], timings_ms{}`.

## Файлы-артефакты, которые читает UI (пишет поток B, кроме отмеченных)
`submission/metrics.json` · `compare.json` (сравнение подходов) · `segments.json` · `errors.csv` · `predictions.jsonl/.csv` · `REPORT.md` ·
новые: `calibration.json` (бины уверенности → точность), `threshold.json` (кривая precision/coverage), `cost.json` (стоимость/латентность по режимам) ·
`traffic.json` (веса атрибутов по категориям из query_clicks, пишет поток A) · `data/sample/cards.jsonl` + `data/sample/gold.jsonl` (реальный сэмпл, пишет A).

Форматы, которые понимает UI (`web/js/quality.js`; имена режимов в cost.json = `mode` из compare.json):
- `threshold.json`: `{"target_precision": 0.95, "chosen": 0.7, "points": [{"threshold", "precision", "recall", "f1", "coverage"}]}`
- `calibration.json`: `{"ece": 0.03, "bins": [{"lo", "hi", "confidence", "accuracy", "n"}]}`
- `cost.json`: `{"note": "допущения…", "modes": [{"mode", "cost_per_1000", "ms_per_card", "llm_share", "llm_calls_per_1000", "cards_per_sec"}]}`

## Окружение
- venv: `.venv` (uv, Python 3.11). `uv` в `~/.local/bin`. Ставить пакеты: `uv pip install -p .venv/bin/python <pkg>`.
- Машина: 4 ядра, 31 ГБ RAM, GTX 1060 6 ГБ. **`skus.pkl` занимает 12 ГБ RAM — грузит только поток A**, остальные работают с `data/processed/*.parquet`.
- Браузер для скриншотов: `AGENT_BROWSER_EXECUTABLE_PATH=/usr/bin/chromium agent-browser ...`.
- Docker недоступен без root (пользователь не в группе docker) — проверять запуск нативно (`attrx serve`).
