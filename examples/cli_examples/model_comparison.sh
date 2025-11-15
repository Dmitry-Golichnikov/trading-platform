#!/bin/bash
# Обучение и сравнение нескольких моделей
# Использование: ./model_comparison.sh TRAIN_DATA

set -euo pipefail

TRAIN_DATA=${1:-}
VAL_DATA=${2:-}
TEST_DATA=${3:-}

if [ -z "$TRAIN_DATA" ]; then
    echo "❌ Ошибка: Необходимо указать путь к обучающим данным"
    echo "Использование: $0 TRAIN_DATA [VAL_DATA] [TEST_DATA]"
    exit 1
fi

# Конфигурации моделей для сравнения
MODELS=(
    "configs/models/lightgbm.yaml"
    "configs/models/xgboost.yaml"
    "configs/models/catboost.yaml"
    "configs/models/random_forest.yaml"
)

EXPERIMENT_NAME="model_comparison_$(date +'%Y%m%d_%H%M%S')"

echo "========================================="
echo "🔬 Model Comparison Pipeline"
echo "========================================="
echo "Train data: $TRAIN_DATA"
[ -n "$VAL_DATA" ] && echo "Val data: $VAL_DATA"
[ -n "$TEST_DATA" ] && echo "Test data: $TEST_DATA"
echo "Experiment: $EXPERIMENT_NAME"
echo "Models: ${#MODELS[@]}"
echo "========================================="
echo ""

# Массивы для хранения результатов
declare -a MODEL_IDS
declare -a MODEL_NAMES
declare -a TRAIN_TIMES

# Обучить каждую модель
for MODEL_CONFIG in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL_CONFIG" .yaml)

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🎓 Обучение модели: $MODEL_NAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    START_TIME=$(date +%s)

    # Формируем команду
    CMD="trading-cli model train \
        -c $MODEL_CONFIG \
        -d $TRAIN_DATA \
        --experiment-name $EXPERIMENT_NAME \
        --run-name ${MODEL_NAME}_$(date +'%H%M%S') \
        --device auto \
        --verbose"

    # Добавляем валидационные данные если указаны
    [ -n "$VAL_DATA" ] && CMD="$CMD --val-data $VAL_DATA"

    # Запускаем обучение
    if MODEL_OUTPUT=$(eval "$CMD" 2>&1); then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        # Извлекаем Model ID
        MODEL_ID=$(echo "$MODEL_OUTPUT" | grep "Model ID:" | awk '{print $3}')

        if [ -n "$MODEL_ID" ]; then
            MODEL_IDS+=("$MODEL_ID")
            MODEL_NAMES+=("$MODEL_NAME")
            TRAIN_TIMES+=("$DURATION")

            echo "✅ Модель обучена: $MODEL_ID"
            echo "⏱️  Время обучения: ${DURATION}s"
        else
            echo "⚠️  Предупреждение: не удалось получить Model ID"
        fi
    else
        echo "❌ Ошибка обучения модели $MODEL_NAME"
    fi

    echo ""
done

# Проверка наличия обученных моделей
if [ ${#MODEL_IDS[@]} -eq 0 ]; then
    echo "❌ Ни одна модель не была успешно обучена"
    exit 1
fi

echo "========================================="
echo "📊 Результаты обучения"
echo "========================================="
for i in "${!MODEL_IDS[@]}"; do
    echo "${MODEL_NAMES[$i]}: ${MODEL_IDS[$i]} (${TRAIN_TIMES[$i]}s)"
done
echo "========================================="
echo ""

# Сравнение моделей
if [ ${#MODEL_IDS[@]} -gt 1 ]; then
    echo "🔍 Сравнение моделей..."

    # Сравнение по разным метрикам
    METRICS=("roc_auc" "accuracy" "f1_score" "precision" "recall")

    for METRIC in "${METRICS[@]}"; do
        echo ""
        echo "Метрика: $METRIC"
        echo "────────────────────────────────────────"

        if trading-cli model compare "${MODEL_IDS[@]}" --metric "$METRIC"; then
            echo "✅ Сравнение по $METRIC завершено"
        else
            echo "⚠️  Ошибка сравнения по $METRIC"
        fi
    done
else
    echo "⚠️  Недостаточно моделей для сравнения (нужно минимум 2)"
fi

# Оценка на тестовых данных если указаны
if [ -n "$TEST_DATA" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🎯 Оценка на тестовых данных"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    EVAL_DIR="artifacts/evaluation/$EXPERIMENT_NAME"
    mkdir -p "$EVAL_DIR"

    for i in "${!MODEL_IDS[@]}"; do
        MODEL_ID="${MODEL_IDS[$i]}"
        MODEL_NAME="${MODEL_NAMES[$i]}"

        echo ""
        echo "Оценка: $MODEL_NAME"

        EVAL_REPORT="$EVAL_DIR/${MODEL_NAME}_evaluation.html"

        if trading-cli model evaluate "$MODEL_ID" \
            -d "$TEST_DATA" \
            -t target \
            -o "$EVAL_REPORT"; then
            echo "✅ Отчёт: $EVAL_REPORT"
        else
            echo "⚠️  Ошибка оценки $MODEL_NAME"
        fi
    done
fi

# Создать сводный отчёт
SUMMARY_FILE="artifacts/reports/${EXPERIMENT_NAME}_summary.txt"
mkdir -p "$(dirname "$SUMMARY_FILE")"

{
    echo "Model Comparison Summary"
    echo "========================"
    echo "Date: $(date)"
    echo "Experiment: $EXPERIMENT_NAME"
    echo "Train data: $TRAIN_DATA"
    [ -n "$VAL_DATA" ] && echo "Val data: $VAL_DATA"
    [ -n "$TEST_DATA" ] && echo "Test data: $TEST_DATA"
    echo ""
    echo "Models:"
    echo "-------"
    for i in "${!MODEL_IDS[@]}"; do
        echo "${MODEL_NAMES[$i]}:"
        echo "  ID: ${MODEL_IDS[$i]}"
        echo "  Train time: ${TRAIN_TIMES[$i]}s"
    done
    echo ""
    echo "Total models trained: ${#MODEL_IDS[@]}"
} > "$SUMMARY_FILE"

echo ""
echo "========================================="
echo "🎉 Сравнение моделей завершено!"
echo "========================================="
echo "Обучено моделей: ${#MODEL_IDS[@]}"
echo "Эксперимент: $EXPERIMENT_NAME"
echo "Сводный отчёт: $SUMMARY_FILE"
[ -n "$TEST_DATA" ] && echo "Отчёты оценки: $EVAL_DIR"
echo "========================================="
