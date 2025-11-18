# Trading Platform CLI - Руководство пользователя

## Оглавление

- [Установка](#установка)
- [Быстрый старт](#быстрый-старт)
- [Команды](#команды)
  - [Data](#data---работа-с-данными)
  - [Features](#features---генерация-признаков)
  - [Labels](#labels---разметка-таргетов)
  - [Model](#model---работа-с-моделями)
  - [Backtest](#backtest---бэктестинг)
  - [Hyperopt](#hyperopt---оптимизация-гиперпараметров)
  - [Experiment](#experiment---управление-экспериментами)
  - [Pipeline](#pipeline---запуск-пайплайнов)
- [Автодополнение](#автодополнение)
- [Примеры использования](#примеры-использования)

## Установка

### Из исходного кода

```bash
# Клонировать репозиторий
git clone https://github.com/your-username/trading-platform.git
cd trading-platform

# Установить в режиме разработки
pip install -e .

# Проверить установку
trading-cli --help
```

### Через pip (когда опубликован)

```bash
pip install trading-platform
trading-cli --help
```

## Быстрый старт

```bash
# 1. Загрузить данные
trading-cli data load --ticker SBER --from 2020-01-01 --to 2023-12-31

# 2. Сгенерировать признаки
trading-cli features generate \
    -c configs/features/default.yaml \
    -d SBER_1m

# 3. Разметить таргеты
trading-cli labels label-dataset \
    --data-path artifacts/data/SBER/1m/SBER_1m.parquet \
    --config configs/labeling/long_only.yaml

# 4. Обучить модель
trading-cli model train \
    -c configs/models/lightgbm.yaml \
    -d artifacts/labels/SBER_1m_labeled.parquet

# 5. Запустить бэктест
trading-cli backtest run \
    -s configs/strategy.yaml \
    -m <model-id> \
    -d artifacts/data/SBER/1m/SBER_1m.parquet
```

## Команды

### Data - Работа с данными

#### `trading-cli data load`

Загрузить данные из источника.

```bash
# Загрузить один тикер
trading-cli data load --ticker SBER --from 2020-01-01 --to 2023-12-31

# Загрузить несколько тикеров
trading-cli data load \
    --ticker SBER --ticker GAZP --ticker LKOH \
    --from 2020-01-01

# Загрузить из файла со списком тикеров
trading-cli data load \
    --tickers-file tickers_all.txt \
    --from 2020-01-01
```

**Опции:**
- `--ticker` - тикер для загрузки (можно указать несколько раз)
- `--tickers-file` - файл со списком тикеров (один на строку)
- `--from-date` - дата начала (YYYY-MM-DD)
- `--to-date` - дата окончания (YYYY-MM-DD)
- `--timeframe` - таймфрейм (1m, 5m, 15m, 1h, 4h, 1d)
- `--source-type` - источник данных (local, api)

#### `trading-cli data list-datasets`

Показать список доступных датасетов.

```bash
# Все датасеты
trading-cli data list-datasets

# Фильтр по тикеру
trading-cli data list-datasets --ticker SBER

# Фильтр по таймфрейму
trading-cli data list-datasets --timeframe 1h
```

#### `trading-cli data dataset-info`

Информация о датасете.

```bash
# Информация по тикеру (все таймфреймы)
trading-cli data dataset-info --ticker SBER

# Информация по конкретному датасету
trading-cli data dataset-info --ticker SBER --timeframe 1h
```

#### `trading-cli data validate-dataset`

Валидация датасета.

```bash
trading-cli data validate-dataset --ticker SBER --timeframe 1h
```

#### `trading-cli data resample-dataset`

Ресэмплировать датасет в другой таймфрейм.

```bash
trading-cli data resample-dataset \
    --ticker SBER \
    --source-timeframe 1m \
    --target-timeframe 1h
```

#### `trading-cli data export-dataset`

Экспортировать датасет в файл.

```bash
# Экспорт в CSV
trading-cli data export-dataset \
    --ticker SBER \
    --timeframe 1h \
    --format csv \
    --output SBER_1h.csv

# Экспорт с фильтром по дате
trading-cli data export-dataset \
    --ticker SBER \
    --timeframe 1h \
    --from-date 2023-01-01 \
    --to-date 2023-12-31 \
    --format parquet \
    --compress
```

#### `trading-cli data filter-dataset`

Применить фильтры к датасету.

```bash
trading-cli data filter-dataset \
    --ticker SBER \
    --timeframe 1h \
    --price-anomaly \
    --volume-anomaly \
    --missing-data forward_fill
```

#### `trading-cli data quality-report`

Создать отчёт о качестве данных.

```bash
# HTML отчёт
trading-cli data quality-report \
    --ticker SBER \
    --timeframe 1h \
    --format html

# JSON отчёт
trading-cli data quality-report \
    --ticker SBER \
    --timeframe 1h \
    --format json \
    --output quality_report.json
```

### Features - Генерация признаков

#### `trading-cli features generate`

Сгенерировать признаки по конфигурации.

```bash
# Из датасета в хранилище
trading-cli features generate \
    -c configs/features/default.yaml \
    -d SBER_1m

# Из файла
trading-cli features generate \
    -c configs/features/minimal.yaml \
    -d data/SBER_1h.parquet \
    -o features/SBER_features.parquet

# С отключением кэша
trading-cli features generate \
    -c configs/features/full.yaml \
    -d SBER_1m \
    --no-cache
```

#### `trading-cli features list`

Список кэшированных признаков.

```bash
# Все кэшированные признаки
trading-cli features list

# Для конкретного датасета
trading-cli features list -d SBER_1m
```

#### `trading-cli features clear-cache`

Очистить кэш признаков.

```bash
# Очистить весь кэш
trading-cli features clear-cache

# Очистить для датасета
trading-cli features clear-cache -d SBER_1m
```

#### `trading-cli features validate-config`

Валидировать конфигурацию признаков.

```bash
trading-cli features validate-config configs/features/default.yaml
```

### Labels - Разметка таргетов

#### `trading-cli labels label-dataset`

Разметить датасет таргетами.

```bash
trading-cli labels label-dataset \
    --data-path data/SBER_1h.parquet \
    --config configs/labeling/long_only.yaml \
    --output-dir artifacts/labels

# С визуализацией
trading-cli labels label-dataset \
    --data-path data/SBER_1h.parquet \
    --config configs/labeling/triple_barrier.yaml \
    --visualize
```

### Model - Работа с моделями

#### `trading-cli model train`

Обучить модель.

```bash
# Базовое обучение
trading-cli model train \
    -c configs/models/lightgbm.yaml \
    -d data/train.parquet

# С валидационными данными
trading-cli model train \
    -c configs/models/lstm.yaml \
    -d data/train.parquet \
    --val-data data/val.parquet \
    --device cuda

# С указанием имени эксперимента
trading-cli model train \
    -c configs/models/xgboost.yaml \
    -d data/train.parquet \
    --experiment-name "sber-prediction" \
    --run-name "xgboost-v1"
```

#### `trading-cli model list`

Список обученных моделей.

```bash
# Все модели
trading-cli model list

# С фильтром по эксперименту
trading-cli model list -e model-training --limit 10

# Сортировка по метрике
trading-cli model list --sort-by accuracy
```

#### `trading-cli model info`

Информация о модели.

```bash
# Базовая информация
trading-cli model info abc123def

# С параметрами и метриками
trading-cli model info abc123def --show-params --show-metrics
```

#### `trading-cli model evaluate`

Оценить модель на тестовых данных.

```bash
trading-cli model evaluate abc123 \
    -d data/test.parquet \
    -o evaluation_report.html
```

#### `trading-cli model predict`

Сделать прогноз.

```bash
# Прогноз классов
trading-cli model predict abc123 \
    -d data/new.parquet \
    -o predictions.parquet

# Прогноз вероятностей
trading-cli model predict abc123 \
    -d data/new.parquet \
    -o predictions.parquet \
    --probabilities
```

#### `trading-cli model compare`

Сравнить модели.

```bash
trading-cli model compare abc123 def456 ghi789

# По конкретной метрике
trading-cli model compare abc123 def456 --metric f1_score
```

### Backtest - Бэктестинг

#### `trading-cli backtest run`

Запустить бэктест.

```bash
# С моделью из MLflow
trading-cli backtest run \
    -s configs/strategy.yaml \
    -m abc123 \
    -d data/test.parquet

# С моделью из файла
trading-cli backtest run \
    -s configs/strategy.yaml \
    --model-path models/model.pkl \
    -d data/test.parquet

# С параметрами
trading-cli backtest run \
    -s configs/strategy.yaml \
    -m abc123 \
    -d data/test.parquet \
    --initial-capital 100000 \
    --commission 0.001 \
    --slippage 0.0005
```

#### `trading-cli backtest list`

Список бэктестов.

```bash
# Последние бэктесты
trading-cli backtest list

# С сортировкой
trading-cli backtest list --sort-by sharpe --limit 10
```

#### `trading-cli backtest report`

Создать отчёт по бэктесту.

```bash
# HTML отчёт
trading-cli backtest report backtest_20231215_120000

# PDF отчёт
trading-cli backtest report backtest_20231215_120000 \
    --format pdf \
    -o report.pdf
```

#### `trading-cli backtest compare`

Сравнить бэктесты.

```bash
trading-cli backtest compare bt1 bt2 bt3

# С сохранением отчёта
trading-cli backtest compare bt1 bt2 bt3 \
    --metric sharpe_ratio \
    -o comparison.html
```

#### `trading-cli backtest analyze`

Углублённый анализ бэктеста.

```bash
trading-cli backtest analyze backtest_20231215_120000
```

### Hyperopt - Оптимизация гиперпараметров

#### `trading-cli hyperopt run`

Запустить оптимизацию.

```bash
trading-cli hyperopt run \
    --config configs/hyperopt/search.yaml \
    --data-path data/train.parquet \
    --experiment-name hyperopt-sber
```

### Experiment - Управление экспериментами

#### `trading-cli experiment create`

Создать эксперимент.

```bash
trading-cli experiment create \
    --name "SBER Prediction" \
    --config configs/experiments/full.yaml \
    --description "Full pipeline experiment"
```

#### `trading-cli experiment run`

Запустить эксперимент.

```bash
trading-cli experiment run exp001 --auto-register
```

#### `trading-cli experiment list`

Список экспериментов.

```bash
trading-cli experiment list

# С фильтром
trading-cli experiment list --status completed --limit 20
```

### Pipeline - Запуск пайплайнов

#### `trading-cli pipeline run`

Запустить пайплайн.

```bash
trading-cli pipeline run --config configs/pipelines/full.yaml
```

## Автодополнение

### Bash

```bash
# Добавить в ~/.bashrc
trading-cli autocomplete --shell bash >> ~/.bashrc
source ~/.bashrc
```

### Zsh

```bash
# Добавить в ~/.zshrc
trading-cli autocomplete --shell zsh >> ~/.zshrc
source ~/.zshrc
```

### Fish

```bash
# Создать файл автодополнения
trading-cli autocomplete --shell fish > ~/.config/fish/completions/trading-cli.fish
```

## Примеры использования

### Пример 1: Полный цикл от данных до бэктеста

```bash
#!/bin/bash

# 1. Загрузить данные
echo "📥 Загрузка данных..."
trading-cli data load \
    --ticker SBER \
    --from 2020-01-01 \
    --to 2023-12-31

# 2. Проверить качество
echo "✅ Проверка качества..."
trading-cli data quality-report \
    --ticker SBER \
    --timeframe 1h \
    -o reports/quality.html

# 3. Сгенерировать признаки
echo "🔧 Генерация признаков..."
trading-cli features generate \
    -c configs/features/default.yaml \
    -d SBER_1h \
    -o features/SBER_features.parquet

# 4. Разметить таргеты
echo "🏷️  Разметка таргетов..."
trading-cli labels label-dataset \
    --data-path features/SBER_features.parquet \
    --config configs/labeling/long_only.yaml

# 5. Обучить модель
echo "🎓 Обучение модели..."
MODEL_ID=$(trading-cli model train \
    -c configs/models/lightgbm.yaml \
    -d artifacts/labels/SBER_features_labeled.parquet \
    | grep "Model ID:" | cut -d: -f2 | tr -d ' ')

# 6. Запустить бэктест
echo "📈 Бэктест..."
trading-cli backtest run \
    -s configs/strategy.yaml \
    -m $MODEL_ID \
    -d artifacts/data/SBER/1h/SBER_1h.parquet

echo "✅ Готово!"
```

### Пример 2: Сравнение нескольких моделей

```bash
#!/bin/bash

# Обучить несколько моделей
MODELS=()

for config in configs/models/*.yaml; do
    echo "Training $(basename $config)..."
    MODEL_ID=$(trading-cli model train \
        -c $config \
        -d data/train.parquet \
        | grep "Model ID:" | cut -d: -f2 | tr -d ' ')
    MODELS+=($MODEL_ID)
done

# Сравнить
trading-cli model compare "${MODELS[@]}" --metric roc_auc
```

### Пример 3: Массовый бэктест

```bash
#!/bin/bash

# Список тикеров
TICKERS=("SBER" "GAZP" "LKOH" "ROSN")

# Для каждого тикера
for ticker in "${TICKERS[@]}"; do
    echo "Processing $ticker..."

    # Запустить бэктест
    trading-cli backtest run \
        -s configs/strategy.yaml \
        -m best_model_id \
        -d artifacts/data/$ticker/1h/${ticker}_1h.parquet
done

# Сравнить результаты
trading-cli backtest compare backtest_* --metric sharpe_ratio
```

## Опции глобального уровня

Доступны для всех команд:

- `--verbose, -v` - подробный вывод
- `--debug` - режим отладки
- `--help` - справка по команде

## Переменные окружения

- `TINKOFF_API_TOKEN` - токен для Tinkoff API
- `MLFLOW_TRACKING_URI` - URI для MLflow tracking server
- `ARTIFACTS_DIR` - директория для артефактов (по умолчанию: `artifacts/`)

## Troubleshooting

### Ошибка: "Модель не найдена"

Убедитесь что MLflow tracking server запущен и модель зарегистрирована:

```bash
mlflow ui
trading-cli model list
```

### Ошибка: "Датасет не найден"

Проверьте список доступных датасетов:

```bash
trading-cli data list-datasets
```

### Низкая производительность

Используйте кэширование для признаков и выключите визуализацию для больших датасетов:

```bash
trading-cli features generate \
    -c config.yaml \
    -d dataset \
    --use-cache

trading-cli backtest run ... --no-plot
```

## Дополнительные ресурсы

- [Техническая документация](../technical_spec.md)
- [План разработки](../plan/README.md)
- [Примеры конфигураций](../configs/)
- [API документация](../docs/)
