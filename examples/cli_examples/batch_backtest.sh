#!/bin/bash
# Массовый бэктест для нескольких тикеров
# Использование: ./batch_backtest.sh MODEL_ID

set -euo pipefail

MODEL_ID=${1:-}
STRATEGY_CONFIG=${2:-configs/strategy.yaml}
TIMEFRAME=${3:-1h}

if [ -z "$MODEL_ID" ]; then
    echo "❌ Ошибка: Необходимо указать Model ID"
    echo "Использование: $0 MODEL_ID [STRATEGY_CONFIG] [TIMEFRAME]"
    exit 1
fi

# Список тикеров (можно также загрузить из файла)
TICKERS=(
    "SBER"
    "GAZP"
    "LKOH"
    "ROSN"
    "GMKN"
    "NVTK"
    "TATN"
    "MGNT"
)

# Параметры бэктеста
INITIAL_CAPITAL=100000
COMMISSION=0.001
SLIPPAGE=0.0005

echo "========================================="
echo "📊 Batch Backtesting"
echo "========================================="
echo "Model ID: $MODEL_ID"
echo "Strategy: $STRATEGY_CONFIG"
echo "Timeframe: $TIMEFRAME"
echo "Tickers: ${TICKERS[*]}"
echo "========================================="
echo ""

# Создать директорию для результатов
BATCH_DIR="artifacts/backtests/batch_$(date +'%Y%m%d_%H%M%S')"
mkdir -p "$BATCH_DIR"

# Счётчики
TOTAL=${#TICKERS[@]}
SUCCESS=0
FAILED=0

# Массив для хранения backtest IDs
declare -a BACKTEST_IDS

# Запустить бэктест для каждого тикера
for TICKER in "${TICKERS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📈 Обработка: $TICKER ($((SUCCESS + FAILED + 1))/$TOTAL)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    DATA_PATH="artifacts/data/$TICKER/$TIMEFRAME/${TICKER}_${TIMEFRAME}.parquet"

    # Проверить наличие данных
    if [ ! -f "$DATA_PATH" ]; then
        echo "⚠️  Данные не найдены: $DATA_PATH"
        echo "   Пропускаем $TICKER"
        ((FAILED++))
        continue
    fi

    # Запустить бэктест
    if BACKTEST_OUTPUT=$(trading-cli backtest run \
        -s "$STRATEGY_CONFIG" \
        -m "$MODEL_ID" \
        -d "$DATA_PATH" \
        --initial-capital "$INITIAL_CAPITAL" \
        --commission "$COMMISSION" \
        --slippage "$SLIPPAGE" \
        --plot 2>&1); then

        # Извлечь Backtest ID
        BACKTEST_ID=$(echo "$BACKTEST_OUTPUT" | grep "Backtest ID:" | awk '{print $3}')

        if [ -n "$BACKTEST_ID" ]; then
            BACKTEST_IDS+=("$BACKTEST_ID")
            echo "✅ Успешно: $TICKER (ID: $BACKTEST_ID)"
            ((SUCCESS++))
        else
            echo "⚠️  Предупреждение: не удалось получить Backtest ID для $TICKER"
            ((FAILED++))
        fi
    else
        echo "❌ Ошибка бэктеста для $TICKER"
        ((FAILED++))
    fi

    echo ""
done

# Итоговая статистика
echo "========================================="
echo "📊 Итоги массового бэктеста"
echo "========================================="
echo "Всего тикеров: $TOTAL"
echo "Успешно: $SUCCESS"
echo "Неудачно: $FAILED"
echo "========================================="

# Сравнить результаты если есть успешные бэктесты
if [ $SUCCESS -gt 1 ]; then
    echo ""
    echo "🔍 Сравнение результатов..."

    COMPARISON_REPORT="$BATCH_DIR/comparison.html"

    if trading-cli backtest compare "${BACKTEST_IDS[@]}" \
        --metric sharpe_ratio \
        -o "$COMPARISON_REPORT"; then
        echo "✅ Отчёт сравнения: $COMPARISON_REPORT"
    else
        echo "⚠️  Ошибка создания отчёта сравнения"
    fi
fi

# Создать сводный отчёт
SUMMARY_FILE="$BATCH_DIR/summary.txt"
{
    echo "Batch Backtest Summary"
    echo "======================="
    echo "Date: $(date)"
    echo "Model ID: $MODEL_ID"
    echo "Strategy: $STRATEGY_CONFIG"
    echo ""
    echo "Results:"
    echo "--------"
    for i in "${!TICKERS[@]}"; do
        TICKER="${TICKERS[$i]}"
        if [ $i -lt ${#BACKTEST_IDS[@]} ]; then
            echo "$TICKER: ${BACKTEST_IDS[$i]}"
        else
            echo "$TICKER: FAILED"
        fi
    done
    echo ""
    echo "Statistics:"
    echo "-----------"
    echo "Total: $TOTAL"
    echo "Success: $SUCCESS"
    echo "Failed: $FAILED"
} > "$SUMMARY_FILE"

echo ""
echo "📄 Сводный отчёт: $SUMMARY_FILE"
echo "🎉 Готово!"
