# CLI Examples - Примеры использования

Коллекция практических примеров использования Trading Platform CLI.

## Содержание

- [full_pipeline.sh](#full_pipelinesh) - Полный пайплайн от загрузки данных до бэктеста
- [batch_backtest.sh](#batch_backtestsh) - Массовый бэктест для нескольких тикеров
- [model_comparison.sh](#model_comparisonsh) - Сравнение разных моделей

## Подготовка

### Сделать скрипты исполняемыми

```bash
chmod +x examples/cli_examples/*.sh
```

### Проверить установку CLI

```bash
trading-cli --version
```

### Подготовить конфигурации

Убедитесь что существуют следующие файлы конфигураций:

- `configs/features/default.yaml` - конфигурация признаков
- `configs/labeling/long_only.yaml` - конфигурация разметки
- `configs/models/lightgbm.yaml` - конфигурация модели
- `configs/strategy.yaml` - конфигурация стратегии

## full_pipeline.sh

Полный автоматизированный пайплайн от загрузки данных до бэктеста.

### Что делает

1. **Загрузка данных** - загружает исторические данные для тикера
2. **Проверка качества** - создаёт отчёт о качестве данных
3. **Генерация признаков** - вычисляет технические индикаторы и признаки
4. **Разметка таргетов** - создаёт таргеты для обучения
5. **Обучение модели** - обучает модель машинного обучения
6. **Оценка модели** - оценивает качество модели
7. **Бэктест** - тестирует стратегию на исторических данных

### Использование

```bash
# Базовое использование (SBER, 1h, 2020-2023)
./examples/cli_examples/full_pipeline.sh

# С параметрами
./examples/cli_examples/full_pipeline.sh TICKER TIMEFRAME START_DATE END_DATE

# Примеры
./examples/cli_examples/full_pipeline.sh GAZP 1h 2021-01-01 2023-12-31
./examples/cli_examples/full_pipeline.sh LKOH 4h 2022-01-01 2023-06-30
```

### Параметры

- **TICKER** - тикер инструмента (по умолчанию: SBER)
- **TIMEFRAME** - таймфрейм (по умолчанию: 1h)
- **START_DATE** - дата начала (по умолчанию: 2020-01-01)
- **END_DATE** - дата окончания (по умолчанию: 2023-12-31)

### Выход

Скрипт создаёт:
- Загруженные данные в `artifacts/data/`
- Отчёт о качестве в `artifacts/reports/`
- Признаки в `artifacts/features/`
- Размеченные данные в `artifacts/labels/`
- Модель в MLflow
- Отчёт оценки в `artifacts/reports/`
- Результаты бэктеста в `artifacts/backtests/`

### Пример вывода

```
=========================================
🚀 Trading Platform - Full Pipeline
=========================================
Ticker: SBER
Timeframe: 1h
Period: 2020-01-01 to 2023-12-31
=========================================

[10:30:15] 📥 Шаг 1/7: Загрузка данных...
[10:32:45] ✅ Данные загружены
[10:32:45] 🔍 Шаг 2/7: Проверка качества данных...
[10:33:10] ✅ Отчёт о качестве сохранён: artifacts/reports/SBER_1h_quality.html
...
=========================================
🎉 Пайплайн успешно завершён!
=========================================
Ticker: SBER
Timeframe: 1h
Model ID: abc123def456
...
```

## batch_backtest.sh

Массовый запуск бэктестов для нескольких тикеров с одной моделью.

### Что делает

1. Принимает ID обученной модели
2. Запускает бэктест для каждого тикера из списка
3. Сравнивает результаты
4. Создаёт сводный отчёт

### Использование

```bash
# Базовое использование
./examples/cli_examples/batch_backtest.sh MODEL_ID

# С параметрами
./examples/cli_examples/batch_backtest.sh MODEL_ID STRATEGY_CONFIG TIMEFRAME

# Примеры
./examples/cli_examples/batch_backtest.sh abc123def
./examples/cli_examples/batch_backtest.sh abc123def configs/aggressive_strategy.yaml 4h
```

### Параметры

- **MODEL_ID** - ID модели из MLflow (обязательный)
- **STRATEGY_CONFIG** - путь к конфигурации стратегии (по умолчанию: configs/strategy.yaml)
- **TIMEFRAME** - таймфрейм данных (по умолчанию: 1h)

### Настройка тикеров

Редактируйте список тикеров в скрипте:

```bash
TICKERS=(
    "SBER"
    "GAZP"
    "LKOH"
    # Добавьте свои тикеры
)
```

### Выход

- Результаты бэктестов в `artifacts/backtests/batch_TIMESTAMP/`
- Отчёт сравнения в `artifacts/backtests/batch_TIMESTAMP/comparison.html`
- Сводный файл в `artifacts/backtests/batch_TIMESTAMP/summary.txt`

### Пример вывода

```
=========================================
📊 Batch Backtesting
=========================================
Model ID: abc123def
Strategy: configs/strategy.yaml
Timeframe: 1h
Tickers: SBER GAZP LKOH ROSN GMKN NVTK TATN MGNT
=========================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 Обработка: SBER (1/8)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Успешно: SBER (ID: backtest_20231215_120000)
...
=========================================
📊 Итоги массового бэктеста
=========================================
Всего тикеров: 8
Успешно: 7
Неудачно: 1
=========================================
```

## model_comparison.sh

Обучение и сравнение нескольких типов моделей.

### Что делает

1. Обучает несколько разных моделей (LightGBM, XGBoost, CatBoost, etc.)
2. Сравнивает их по различным метрикам
3. Оценивает на тестовых данных
4. Создаёт сводный отчёт

### Использование

```bash
# Только обучение
./examples/cli_examples/model_comparison.sh TRAIN_DATA

# С валидационными данными
./examples/cli_examples/model_comparison.sh TRAIN_DATA VAL_DATA

# С тестовыми данными
./examples/cli_examples/model_comparison.sh TRAIN_DATA VAL_DATA TEST_DATA

# Примеры
./examples/cli_examples/model_comparison.sh data/train.parquet
./examples/cli_examples/model_comparison.sh data/train.parquet data/val.parquet data/test.parquet
```

### Параметры

- **TRAIN_DATA** - путь к обучающим данным (обязательный)
- **VAL_DATA** - путь к валидационным данным (опционально)
- **TEST_DATA** - путь к тестовым данным (опционально)

### Настройка моделей

Редактируйте список конфигураций моделей:

```bash
MODELS=(
    "configs/models/lightgbm.yaml"
    "configs/models/xgboost.yaml"
    "configs/models/catboost.yaml"
    # Добавьте свои конфигурации
)
```

### Выход

- Модели в MLflow под общим экспериментом
- Отчёты оценки в `artifacts/evaluation/EXPERIMENT_NAME/`
- Сводный отчёт в `artifacts/reports/EXPERIMENT_NAME_summary.txt`

### Пример вывода

```
=========================================
🔬 Model Comparison Pipeline
=========================================
Train data: data/train.parquet
Val data: data/val.parquet
Test data: data/test.parquet
Experiment: model_comparison_20231215_120000
Models: 4
=========================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎓 Обучение модели: lightgbm
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Модель обучена: abc123
⏱️  Время обучения: 45s
...
=========================================
📊 Результаты обучения
=========================================
lightgbm: abc123 (45s)
xgboost: def456 (52s)
catboost: ghi789 (63s)
random_forest: jkl012 (38s)
=========================================
```

## Дополнительные примеры

### Создание своих скриптов

Используйте эти примеры как шаблоны для создания своих автоматизированных пайплайнов.

#### Пример: Еженедельный анализ

```bash
#!/bin/bash
# weekly_analysis.sh - Еженедельный анализ топ-10 тикеров

TOP_TICKERS=("SBER" "GAZP" "LKOH" "ROSN" "GMKN" "NVTK" "TATN" "MGNT" "YNDX" "FIVE")
WEEK_AGO=$(date -d "7 days ago" +%Y-%m-%d)
TODAY=$(date +%Y-%m-%d)

for ticker in "${TOP_TICKERS[@]}"; do
    echo "Analyzing $ticker..."

    # Загрузить последнюю неделю
    trading-cli data load --ticker "$ticker" --from "$WEEK_AGO" --to "$TODAY"

    # Создать отчёт
    trading-cli data quality-report --ticker "$ticker" --timeframe 1h
done

# Сравнить датасеты
DATASETS=$(printf "%s/1h," "${TOP_TICKERS[@]}")
trading-cli data compare-datasets --datasets "${DATASETS%,}"
```

#### Пример: Оптимизация и тестирование

```bash
#!/bin/bash
# optimize_and_test.sh - Оптимизация гиперпараметров и тестирование

# 1. Оптимизация
trading-cli hyperopt run \
    --config configs/hyperopt/search.yaml \
    --data-path data/train.parquet

# 2. Получить лучшие параметры
BEST_PARAMS=$(trading-cli hyperopt best --metric roc_auc)

# 3. Обучить финальную модель
MODEL_ID=$(trading-cli model train \
    -c configs/models/lightgbm.yaml \
    -d data/full_train.parquet \
    --params "$BEST_PARAMS")

# 4. Бэктест
trading-cli backtest run \
    -s configs/strategy.yaml \
    -m "$MODEL_ID" \
    -d data/test.parquet
```

## Лучшие практики

### 1. Логирование

Сохраняйте логи выполнения:

```bash
./examples/cli_examples/full_pipeline.sh SBER 1h 2>&1 | tee logs/pipeline_$(date +%Y%m%d).log
```

### 2. Обработка ошибок

Используйте `set -euo pipefail` в начале скриптов для остановки при ошибках.

### 3. Параллелизация

Для независимых задач используйте параллельное выполнение:

```bash
# Параллельная загрузка
for ticker in SBER GAZP LKOH; do
    trading-cli data load --ticker "$ticker" &
done
wait
```

### 4. Мониторинг

Отслеживайте использование ресурсов:

```bash
# С мониторингом памяти
/usr/bin/time -v ./full_pipeline.sh
```

### 5. Notifications

Добавьте уведомления о завершении:

```bash
# В конце скрипта
if [ $? -eq 0 ]; then
    echo "Success!" | mail -s "Pipeline completed" you@example.com
fi
```

## Troubleshooting

### Скрипт не запускается

```bash
chmod +x examples/cli_examples/*.sh
```

### Команда не найдена

```bash
# Убедитесь что CLI установлен
pip install -e .
which trading-cli
```

### Ошибки выполнения

Используйте debug режим:

```bash
bash -x ./full_pipeline.sh
```

## Ресурсы

- [CLI Guide](../../docs/CLI_GUIDE.md) - Полное руководство по CLI
- [Конфигурации](../../configs/) - Примеры конфигураций
- [Документация](../../docs/) - Техническая документация
