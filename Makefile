PY ?= .venv/bin
.PHONY: install test lint demo serve eda ner report report-sample data prepare
install: ; uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python -e ".[dev,ml]"
test: ; $(PY)/pytest -q
lint: ; $(PY)/ruff check .
demo: ; $(PY)/attrx demo
serve: ; $(PY)/attrx serve --reload
eda: ; $(PY)/attrx eda -i data/sample/cards.jsonl
ner: ; $(PY)/attrx train-ner -i data/sample/cards.jsonl
# итоговый отчёт сдачи: holdout реального каталога -> submission/ (NER на GPU, LLM — локальная Ollama на подвыборке)
report: ; NER_DEVICE=$${NER_DEVICE:-cuda} LLM_BASE_URL=$${LLM_BASE_URL:-http://127.0.0.1:11434/v1} LLM_MODEL=$${LLM_MODEL:-qwen3:4b-instruct} $(PY)/attrx report-real
# отчёт по маленькому синтетическому сэмплу — в outputs/report, submission/ не трогает
report-sample: ; $(PY)/attrx report -i data/sample/cards.jsonl --gold data/sample/gold.jsonl --out outputs/report
data: ; bash scripts/download_data.sh
prepare: ; $(PY)/attrx prepare
.PHONY: traffic eda-real eda-nb
traffic: ; $(PY)/attrx traffic
eda-real: ; $(PY)/attrx eda-real
eda-nb: ; $(PY)/jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
