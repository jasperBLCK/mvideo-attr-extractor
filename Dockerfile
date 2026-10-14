FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .
COPY config ./config
COPY web ./web
COPY data/sample ./data/sample
COPY submission ./submission
RUN mkdir -p outputs
EXPOSE 8000
CMD ["attrx", "serve", "--port", "8000"]
