PY ?= .venv/bin
.PHONY: install test lint demo serve eda ner report
install: ; uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python -e ".[dev,ml]"
test: ; $(PY)/pytest -q
lint: ; $(PY)/ruff check .
demo: ; $(PY)/attrx demo
serve: ; $(PY)/attrx serve --reload
eda: ; $(PY)/attrx eda -i data/sample/cards.jsonl
ner: ; $(PY)/attrx train-ner -i data/sample/cards.jsonl
report: ; $(PY)/attrx report -i data/sample/cards.jsonl --gold data/sample/gold.jsonl
