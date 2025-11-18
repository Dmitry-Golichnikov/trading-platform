#!/bin/bash
# Полный пайплайн: от загрузки данных до бэктеста
# Использование: ./full_pipeline.sh TICKER TIMEFRAME

set -euo pipefail  # Остановить при ошибке

# Параметры
TICKER=${1:-SBER}
TIMEFRAME=${2:-1h}
START_DATE=${3:-2020-01-01}
END_DATE=${4:-2023-12-31}

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "🚀 Trading Platform - Full Pipeline"
echo "========================================="
echo "Ticker: $TICKER"
echo "Timeframe: $TIMEFRAME"
echo "Period: $START_DATE to $END_DATE"
echo "========================================="
echo ""

# Функция для вывода сообщений
log_step() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 1. Загрузка данных
log_step "📥 Шаг 1/7: Загрузка данных..."
if trading-cli data load \
    --ticker "$TICKER" \
    --from "$START_DATE" \
    --to "$END_DATE" \
    --timeframe "$TIMEFRAME"; then
    log_step "✅ Данные загружены"
else
    log_error "Ошибка загрузки данных"
    exit 1
fi

# 2. Проверка качества данных
log_step "🔍 Шаг 2/7: Проверка качества данных..."
QUALITY_REPORT="artifacts/reports/${TICKER}_${TIMEFRAME}_quality.html"
trading-cli data quality-report \
    --ticker "$TICKER" \
    --timeframe "$TIMEFRAME" \
    --format html \
    --output "$QUALITY_REPORT"
log_step "✅ Отчёт о качестве сохранён: $QUALITY_REPORT"

# 3. Генерация признаков
log_step "🔧 Шаг 3/7: Генерация признаков..."
FEATURES_OUTPUT="artifacts/features/${TICKER}_${TIMEFRAME}_features.parquet"
if trading-cli features generate \
    -c configs/features/default.yaml \
    -d "${TICKER}_${TIMEFRAME}" \
    -o "$FEATURES_OUTPUT" \
    --use-cache; then
    log_step "✅ Признаки сгенерированы: $FEATURES_OUTPUT"
else
    log_error "Ошибка генерации признаков"
    exit 1
fi

# 4. Разметка таргетов
log_step "🏷️  Шаг 4/7: Разметка таргетов..."
LABELS_CONFIG="configs/labeling/long_only.yaml"
if trading-cli labels label-dataset \
    --data-path "$FEATURES_OUTPUT" \
    --config "$LABELS_CONFIG" \
    --output-dir "artifacts/labels" \
    --dataset-id "${TICKER}_${TIMEFRAME}" \
    --visualize; then
    log_step "✅ Таргеты размечены"
else
    log_error "Ошибка разметки таргетов"
    exit 1
fi

LABELED_DATA="artifacts/labels/${TICKER}_${TIMEFRAME}_labeled.parquet"

# 5. Обучение модели
log_step "🎓 Шаг 5/7: Обучение модели..."
MODEL_CONFIG="configs/models/lightgbm.yaml"
EXPERIMENT_NAME="${TICKER}_${TIMEFRAME}_experiment"

# Извлечь Model ID из вывода
MODEL_OUTPUT=$(trading-cli model train \
    -c "$MODEL_CONFIG" \
    -d "$LABELED_DATA" \
    --experiment-name "$EXPERIMENT_NAME" \
    --run-name "${TICKER}_${TIMEFRAME}_$(date +'%Y%m%d_%H%M%S')" \
    --device auto \
    --verbose)

MODEL_ID=$(echo "$MODEL_OUTPUT" | grep "Model ID:" | awk '{print $3}')

if [ -z "$MODEL_ID" ]; then
    log_error "Не удалось получить Model ID"
    exit 1
fi

log_step "✅ Модель обучена: $MODEL_ID"

# 6. Оценка модели
log_step "🎯 Шаг 6/7: Оценка модели..."
TEST_DATA="artifacts/data/$TICKER/$TIMEFRAME/${TICKER}_${TIMEFRAME}.parquet"
EVAL_REPORT="artifacts/reports/${TICKER}_${TIMEFRAME}_evaluation.html"

if trading-cli model evaluate "$MODEL_ID" \
    -d "$TEST_DATA" \
    -t target \
    -o "$EVAL_REPORT"; then
    log_step "✅ Отчёт оценки сохранён: $EVAL_REPORT"
else
    log_warning "Оценка модели завершилась с ошибками"
fi

# 7. Бэктест
log_step "📈 Шаг 7/7: Запуск бэктеста..."
STRATEGY_CONFIG="configs/strategy.yaml"
INITIAL_CAPITAL=100000
COMMISSION=0.001
SLIPPAGE=0.0005

if trading-cli backtest run \
    -s "$STRATEGY_CONFIG" \
    -m "$MODEL_ID" \
    -d "$TEST_DATA" \
    --initial-capital "$INITIAL_CAPITAL" \
    --commission "$COMMISSION" \
    --slippage "$SLIPPAGE" \
    --plot; then
    log_step "✅ Бэктест завершён"
else
    log_error "Ошибка бэктеста"
    exit 1
fi

# Итоги
echo ""
echo "========================================="
echo "🎉 Пайплайн успешно завершён!"
echo "========================================="
echo "Ticker: $TICKER"
echo "Timeframe: $TIMEFRAME"
echo "Model ID: $MODEL_ID"
echo ""
echo "Отчёты:"
echo "  - Качество данных: $QUALITY_REPORT"
echo "  - Оценка модели: $EVAL_REPORT"
echo "  - Результаты в: artifacts/backtests/"
echo "========================================="
