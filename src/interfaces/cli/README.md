# Trading Platform CLI

Полнофункциональный интерфейс командной строки для торговой платформы.

## Структура

```
cli/
├── __init__.py                   # Экспорты модуля
├── __main__.py                   # Точка входа CLI
├── utils.py                      # Утилиты (форматирование, прогресс-бары)
├── data_commands.py              # Команды работы с данными
├── feature_commands.py           # Команды генерации признаков
├── labeling_commands.py          # Команды разметки таргетов
├── model_commands.py             # Команды работы с моделями
├── backtest_commands.py          # Команды бэктестинга
├── hyperopt_commands.py          # Команды оптимизации гиперпараметров
├── experiment_commands.py        # Команды управления экспериментами
├── pipeline_commands.py          # Команды запуска пайплайнов
└── evaluation_commands.py        # Команды оценки моделей
```

## Использование

### Через модуль

```bash
python -m src.interfaces.cli --help
python -m src.interfaces.cli data load --ticker SBER
```

### Через установленный пакет

```bash
trading-cli --help
trading-cli data load --ticker SBER
```

## Группы команд

### 1. Data - Работа с данными

Загрузка, валидация, экспорт и обработка данных.

**Основные команды:**
- `load` - загрузить данные из источника
- `list-datasets` - список доступных датасетов
- `dataset-info` - информация о датасете
- `validate-dataset` - валидация датасета
- `resample-dataset` - ресэмплинг в другой таймфрейм
- `export-dataset` - экспорт в файл
- `filter-dataset` - применить фильтры
- `quality-report` - отчёт о качестве
- `compare-datasets` - сравнение датасетов

### 2. Features - Генерация признаков

Создание и управление признаками.

**Основные команды:**
- `generate` - генерация признаков по конфигурации
- `list` - список кэшированных признаков
- `clear-cache` - очистка кэша
- `validate-config` - валидация конфигурации

### 3. Labels - Разметка таргетов

Создание разметки для обучения моделей.

**Основные команды:**
- `label-dataset` - разметить датасет таргетами
- `analyze` - анализ разметки
- `validate` - валидация разметки

### 4. Model - Работа с моделями

Обучение, оценка и использование моделей.

**Основные команды:**
- `train` - обучить модель
- `list` - список обученных моделей
- `info` - информация о модели
- `evaluate` - оценить модель
- `predict` - сделать прогноз
- `compare` - сравнить модели

### 5. Backtest - Бэктестинг

Тестирование торговых стратегий на исторических данных.

**Основные команды:**
- `run` - запустить бэктест
- `list` - список бэктестов
- `report` - создать отчёт
- `compare` - сравнить бэктесты
- `analyze` - углублённый анализ

### 6. Hyperopt - Оптимизация гиперпараметров

Поиск оптимальных гиперпараметров моделей.

**Основные команды:**
- `run` - запустить оптимизацию
- `status` - статус оптимизации
- `best` - лучшие параметры

### 7. Experiment - Управление экспериментами

Создание и отслеживание экспериментов.

**Основные команды:**
- `create` - создать эксперимент
- `run` - запустить эксперимент
- `list` - список экспериментов
- `compare` - сравнить эксперименты

### 8. Pipeline - Запуск пайплайнов

Выполнение end-to-end пайплайнов.

**Основные команды:**
- `run` - запустить пайплайн
- `resume` - продолжить с определённого шага
- `status` - статус выполнения

## Утилиты

### Форматирование

Модуль `utils.py` содержит функции для красивого вывода:

- `format_number()` - форматирование чисел с разделителями
- `format_percentage()` - форматирование процентов
- `format_bytes()` - форматирование размеров
- `format_duration()` - форматирование времени
- `format_timestamp()` - форматирование дат

### Вывод сообщений

- `print_success()` - зелёное сообщение об успехе
- `print_info()` - информационное сообщение
- `print_warning()` - жёлтое предупреждение
- `print_error()` - красное сообщение об ошибке
- `print_header()` - заголовок в рамке

### Таблицы и прогресс

- `create_table()` - создание таблицы Rich
- `create_progress_bar()` - создание прогресс-бара
- `ProgressTracker` - контекстный менеджер для отслеживания прогресса

### Валидация

- `validate_ticker()` - валидация тикера
- `validate_timeframe()` - валидация таймфрейма
- `validate_date()` - валидация даты
- `validate_path()` - валидация пути

### Декораторы

- `@handle_errors` - обработка ошибок с красивым выводом
- `@handle_keyboard_interrupt` - обработка Ctrl+C
- `@timed_operation` - замер времени выполнения

## Примеры

### Базовый workflow

```bash
# 1. Загрузить данные
trading-cli data load --ticker SBER --from 2020-01-01

# 2. Проверить качество
trading-cli data quality-report --ticker SBER --timeframe 1h

# 3. Сгенерировать признаки
trading-cli features generate -c configs/features/default.yaml -d SBER_1h

# 4. Обучить модель
trading-cli model train -c configs/models/lightgbm.yaml -d labeled_data.parquet

# 5. Бэктест
trading-cli backtest run -s strategy.yaml -m <model-id> -d test_data.parquet
```

### Пайплайн через скрипт

```bash
#!/bin/bash
# full_pipeline.sh

set -e  # Остановить при ошибке

TICKER="SBER"
TIMEFRAME="1h"

echo "🚀 Starting full pipeline for $TICKER..."

# Load data
trading-cli data load --ticker $TICKER --from 2020-01-01
echo "✅ Data loaded"

# Generate features
trading-cli features generate \
    -c configs/features/default.yaml \
    -d ${TICKER}_${TIMEFRAME}
echo "✅ Features generated"

# Label dataset
trading-cli labels label-dataset \
    --data-path artifacts/features/${TICKER}_features.parquet \
    --config configs/labeling/long_only.yaml
echo "✅ Labels created"

# Train model
MODEL_ID=$(trading-cli model train \
    -c configs/models/lightgbm.yaml \
    -d artifacts/labels/${TICKER}_labeled.parquet \
    | grep "Model ID:" | awk '{print $3}')
echo "✅ Model trained: $MODEL_ID"

# Backtest
trading-cli backtest run \
    -s configs/strategy.yaml \
    -m $MODEL_ID \
    -d artifacts/data/$TICKER/$TIMEFRAME/${TICKER}_${TIMEFRAME}.parquet
echo "✅ Backtest completed"

echo "🎉 Pipeline finished successfully!"
```

## Автодополнение

### Установка для Bash

```bash
trading-cli autocomplete --shell bash >> ~/.bashrc
source ~/.bashrc
```

### Установка для Zsh

```bash
trading-cli autocomplete --shell zsh >> ~/.zshrc
source ~/.zshrc
```

### Установка для Fish

```bash
trading-cli autocomplete --shell fish > ~/.config/fish/completions/trading-cli.fish
```

## Конфигурация

### Переменные окружения

```bash
# ~/.bashrc или ~/.zshrc

# Tinkoff API
export TINKOFF_API_TOKEN="your-token-here"

# MLflow
export MLFLOW_TRACKING_URI="http://localhost:5000"

# Артефакты
export ARTIFACTS_DIR="./artifacts"

# Логирование
export LOG_LEVEL="INFO"
```

### Файл конфигурации

Создайте `~/.trading-cli/config.yaml`:

```yaml
# Значения по умолчанию
defaults:
  timeframe: "1h"
  initial_capital: 100000
  commission: 0.001

# Пути
paths:
  artifacts: "./artifacts"
  configs: "./configs"
  data: "./data"

# MLflow
mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "default"

# API
api:
  tinkoff:
    rate_limit: 120  # requests per minute
    timeout: 30
```

## Производительность

### Кэширование

Используйте кэш для ускорения генерации признаков:

```bash
# Первый запуск (медленный)
trading-cli features generate -c config.yaml -d dataset --use-cache

# Последующие запуски (быстро)
trading-cli features generate -c config.yaml -d dataset --use-cache
```

### Параллелизм

Загрузка данных поддерживает параллелизм:

```bash
# 8 параллельных загрузок
trading-cli data load \
    --tickers-file tickers.txt \
    --concurrency 8
```

### Оптимизация памяти

Для больших датасетов используйте batch обработку:

```bash
# Отключить визуализацию
trading-cli backtest run ... --no-plot

# Ограничить период
trading-cli data load \
    --from 2023-01-01 \
    --to 2023-12-31
```

## Отладка

### Verbose режим

```bash
trading-cli --verbose data load --ticker SBER
```

### Debug режим

```bash
trading-cli --debug model train -c config.yaml -d data.parquet
```

### Логи

Логи сохраняются в `logs/cli.log`:

```bash
tail -f logs/cli.log
```

## Troubleshooting

### Проблема: Команда не найдена

**Решение:** Убедитесь что пакет установлен:

```bash
pip install -e .
which trading-cli
```

### Проблема: Модуль не найден

**Решение:** Проверьте PYTHONPATH:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Проблема: Ошибка импорта

**Решение:** Переустановите зависимости:

```bash
pip install -r requirements.txt
```

### Проблема: Медленная работа

**Решение:**
1. Используйте кэширование
2. Уменьшите размер данных
3. Увеличьте concurrency
4. Отключите визуализацию

## Тестирование

Запуск тестов CLI:

```bash
# Все тесты
pytest tests/unit/test_cli.py -v

# Конкретная группа
pytest tests/unit/test_cli.py::TestDataCommands -v

# С покрытием
pytest tests/unit/test_cli.py --cov=src.interfaces.cli
```

## Разработка

### Добавление новой команды

1. Создайте функцию с декоратором `@click.command()`:

```python
@data.command("my-command")
@click.option("--param", help="Parameter")
def my_command(param: str) -> None:
    """My command description."""
    print_info(f"Running with param: {param}")
```

2. Добавьте в соответствующую группу в `__main__.py`

3. Добавьте тесты в `tests/unit/test_cli.py`

4. Обновите документацию

### Стиль кода

- Используйте `click` для определения команд
- Используйте `rich` для красивого вывода
- Используйте утилиты из `utils.py`
- Добавляйте docstrings и type hints
- Обрабатывайте ошибки с помощью `@handle_errors`

## Дополнительные ресурсы

- [Полное руководство по CLI](../../../docs/CLI_GUIDE.md)
- [Техническая документация](../../../technical_spec.md)
- [Примеры конфигураций](../../../configs/)
- [API документация](../../../docs/)
