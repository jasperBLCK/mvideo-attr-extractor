FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ATTRX_ROOT=/app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .
COPY config ./config
COPY web ./web
COPY data/sample ./data/sample
COPY submission ./submission
# классификатор категории (7,8 МБ, npz без pickle) — нужен для карточек без категории; NER-модель в образ не входит
COPY models/category ./models/category
RUN mkdir -p outputs
EXPOSE 8000
# attrx serve по умолчанию слушает только 127.0.0.1; в контейнере — все интерфейсы, иначе порт 8000 не пробросить
CMD ["attrx", "serve", "--host", "0.0.0.0", "--port", "8000"]
