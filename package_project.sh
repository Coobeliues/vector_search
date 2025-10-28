#!/bin/bash

##############################################################################
# Скрипт для упаковки проекта Vector Search для передачи коллеге
##############################################################################

echo "🚀 Упаковка проекта Vector Search Microservice..."
echo ""

# Определить директорию проекта
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$PROJECT_DIR")"
PROJECT_NAME="$(basename "$PROJECT_DIR")"
OUTPUT_FILE="$PARENT_DIR/vector_search_project.tar.gz"

echo "📁 Директория проекта: $PROJECT_DIR"
echo "📦 Файл архива: $OUTPUT_FILE"
echo ""

# Создать архив, исключая ненужные файлы
echo "⏳ Создание архива..."
tar -czf "$OUTPUT_FILE" \
  --exclude='vector_search_env' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='*.backup' \
  --exclude='benchmark_results*.json' \
  --exclude='hybrid_benchmark_results.json' \
  --exclude='bm25_benchmark_results.json' \
  --exclude='.git' \
  --exclude='.gitignore' \
  --exclude='*.log' \
  -C "$PARENT_DIR" "$PROJECT_NAME"

# Проверить результат
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Архив успешно создан!"
    echo ""
    echo "📊 Информация об архиве:"
    ls -lh "$OUTPUT_FILE"
    echo ""
    echo "📋 Содержимое архива:"
    tar -tzf "$OUTPUT_FILE" | head -20
    echo "   ..."
    echo ""
    echo "🎉 Готово!"
    echo ""
    echo "📤 Отправь файл коллеге:"
    echo "   $OUTPUT_FILE"
    echo ""
    echo "📝 Инструкция для коллеги:"
    echo "   1. Распаковать: tar -xzf vector_search_project.tar.gz"
    echo "   2. Перейти: cd Vector_search"
    echo "   3. Читать: cat БЫСТРЫЙ_СТАРТ.md"
else
    echo ""
    echo "❌ Ошибка при создании архива!"
    exit 1
fi
