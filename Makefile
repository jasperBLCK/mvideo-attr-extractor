PY ?= .venv/bin
.PHONY: install sync lock test lint demo serve serve-public verify monitor eda ner report report-sample data prepare
install: ; uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python -e ".[dev,ml]"
# установка по uv.lock (CI — то же без --inexact): базовые зависимости + dev; --inexact не удаляет уже
# поставленное вне lock (torch из индекса cu126 для GTX 10xx и прочий [ml])
sync: ; uv sync --frozen --extra dev --inexact
lock: ; uv lock
test: ; $(PY)/pytest -q
lint: ; $(PY)/ruff check .
demo: ; $(PY)/attrx demo
# локально: только 127.0.0.1; демо на ВМ и в сети — serve-public (Docker слушает 0.0.0.0 сам)
serve: ; $(PY)/attrx serve --reload
serve-public: ; $(PY)/attrx serve --host 0.0.0.0 --port 8000
# сверка: строки «правила» и «продавец + правила» submission/compare.json пересчитываются на закоммиченном holdout
# (data/sample/holdout_eval_*.jsonl) — офлайн, без data/processed, torch и LLM; расхождение > 0,002 — ошибка
verify: ; $(PY)/attrx verify
# дрейф: последние предсказания (attrx extract / demo) против итоговых на holdout -> outputs/monitor.json
monitor: ; $(PY)/attrx monitor -i outputs/predictions.jsonl --ref submission/predictions.jsonl
# EDA реальных данных (нужен data/processed): графики docs/img/eda_*.png + цифры для docs/EDA.md
eda: ; $(PY)/attrx eda-real
# NER на distant supervision (эталон каталога, train): нужны extra [ml] и data/processed (attrx prepare, build-gold)
ner: ; $(PY)/attrx train-ner --distant
# итоговый отчёт сдачи: holdout реального каталога -> submission/ (NER на GPU, LLM — локальная Ollama на подвыборке)
report: ; NER_DEVICE=$${NER_DEVICE:-cuda} $(PY)/attrx report-real $(ARGS)
# СИНТЕТИКА: smoke-тест генератора отчёта на 20 придуманных карточках -> outputs/report; не метрика решения
report-sample:
	@echo "report-sample: СИНТЕТИКА — 20 придуманных карточек data/sample/cards.jsonl, проверка генератора отчёта; цифры решения — make report (submission/) и attrx demo"
	$(PY)/attrx report -i data/sample/cards.jsonl --gold data/sample/gold.jsonl --out outputs/report --note "синтетика: 20 придуманных карточек, не метрика решения"
data: ; bash scripts/download_data.sh
prepare: ; $(PY)/attrx prepare
.PHONY: traffic eda-real eda-nb
traffic: ; $(PY)/attrx traffic
eda-real: ; $(PY)/attrx eda-real
eda-nb: ; $(PY)/jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

.PHONY: docker
docker: ; docker compose up --build   # на машине без группы docker: sudo make docker; порт: ATTRX_PORT=8001
