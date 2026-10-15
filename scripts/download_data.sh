#!/usr/bin/env bash
# Скачивает публичный архив кейса М.Видео (cu_ws.zip, ~1.16 ГБ) и бриф в data/raw/ и распаковывает.
# Докачка при обрыве (curl -C -), повторный запуск ничего не перекачивает. Авторизация не нужна.
# Дальше: `attrx prepare` (или `make prepare`) -> data/processed/*.parquet.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="${1:-$ROOT/data/raw}"
HUB="https://hub.gigaschool.ru/hub/events/c8e9ce58-7411-441b-bdcb-0c04a46fc365/page"
ARCHIVE_URL="$HUB/f346d28b-99f8-408a-8afa-b03a51c49d98"
BRIEF_URL="$HUB/864d1c63-3ff0-4a16-a2ac-496e54c3bfdd"

mkdir -p "$RAW"
fetch() {  # url file
  echo "-> $2"
  curl -fL --retry 5 --retry-delay 3 -C - -o "$2" "$1" || {
    # 416 = файл уже скачан полностью
    [ -s "$2" ] && echo "   уже скачан: $2" || exit 1
  }
}

fetch "$BRIEF_URL" "$RAW/case_brief.pdf"
fetch "$ARCHIVE_URL" "$RAW/cu_ws.zip"

if [ -f "$RAW/sku_desc.parquet" ] && [ -f "$RAW/query_clicks.parquet" ] && [ -f "$RAW/skus.pkl" ]; then
  echo "Уже распаковано (удалите файлы в $RAW, чтобы распаковать заново)"
else
  echo "Распаковка в $RAW ..."
  unzip -o -q "$RAW/cu_ws.zip" -d "$RAW"
fi
# в архиве файлы могут лежать во вложенной папке — поднимаем их наверх
for f in sku_desc.parquet query_clicks.parquet skus.pkl; do
  if [ ! -f "$RAW/$f" ]; then
    found="$(find "$RAW" -name "$f" -type f | head -n 1)"
    [ -n "$found" ] && mv "$found" "$RAW/$f"
  fi
  [ -f "$RAW/$f" ] || { echo "Нет $f после распаковки" >&2; exit 1; }
done
ls -lh "$RAW"
echo "Готово. Дальше: attrx prepare"
