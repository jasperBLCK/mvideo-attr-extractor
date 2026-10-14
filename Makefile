PY ?= .venv/bin
.PHONY: install test lint demo serve eda
install: ; uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python -e ".[dev,ml]"
test: ; $(PY)/pytest -q
lint: ; $(PY)/ruff check .
demo: ; $(PY)/attrx demo
serve: ; $(PY)/attrx serve --reload
eda: ; $(PY)/attrx eda -i data/sample/cards.jsonl
