# Система пайплайнов

Модульная система пайплайнов для выполнения end-to-end процессов от подготовки данных до бэктестинга.

## Содержание

1. [Обзор](#обзор)
2. [Базовый пайплайн](#базовый-пайплайн)
3. [Доступные пайплайны](#доступные-пайплайны)
4. [Конфигурация](#конфигурация)
5. [CLI команды](#cli-команды)
6. [Чекпоинты и идемпотентность](#чекпоинты-и-идемпотентность)
7. [Примеры использования](#примеры-использования)

## Обзор

Система пайплайнов предоставляет унифицированный способ выполнения комплексных задач обработки данных, обучения моделей и тестирования стратегий.

### Основные возможности

- **Модульность**: каждый пайплайн выполняет конкретную задачу
- **Композиция**: пайплайны могут комбинироваться в end-to-end процессы
- **Чекпоинты**: автоматическое сохранение промежуточных результатов
- **Идемпотентность**: повторный запуск не изменяет результат
- **Возобновление**: продолжение с последнего чекпоинта после сбоя
- **Конфигурируемость**: все параметры задаются через YAML

## Базовый пайплайн

Все пайплайны наследуются от `BasePipeline`:

```python
from src.pipelines.base import BasePipeline

class MyPipeline(BasePipeline):
    @property
    def name(self) -> str:
        return "my_pipeline"

    def _get_steps(self) -> list[str]:
        return ["step1", "step2", "step3"]

    def _execute_step(self, step_name: str, input_data: Any) -> Any:
        if step_name == "step1":
            return self._step1(input_data)
        # ...
```

## Доступные пайплайны

### 1. DataPreparationPipeline

Подготовка исторических данных: загрузка, валидация, ресэмплинг.

**Шаги:**
- Загрузка данных из источника
- Валидация качества
- Ресэмплинг в целевые таймфреймы
- Сохранение в Parquet

### 2. FeatureEngineeringPipeline

Генерация признаков из сырых данных.

**Шаги:**
- Загрузка данных
- Генерация признаков по конфигурации
- Сохранение результатов

### 3. NormalizationPipeline

Нормализация числовых признаков.

**Шаги:**
- Загрузка данных
- Определение числовых признаков
- Нормализация (Standard, Robust, MinMax)
- Сохранение скейлера и данных

### 4. FeatureSelectionPipeline

Отбор наиболее важных признаков.

**Шаги:**
- Загрузка данных
- Отбор признаков (по важности, корреляции, дисперсии)
- Сохранение отобранных признаков

### 5. LabelingPipeline

Разметка таргетов для обучения.

**Шаги:**
- Загрузка данных
- Генерация меток (Horizon, Triple Barrier)
- Постобработка и балансировка
- Сохранение размеченных данных

### 6. TrainingPipeline

Обучение моделей машинного обучения.

**Шаги:**
- Загрузка данных
- Разделение на train/val/test
- Обучение модели
- Сохранение модели

### 7. ValidationPipeline

Валидация обученных моделей.

**Шаги:**
- Загрузка модели и данных
- Валидация (простая, walk-forward, CV)
- Расчет метрик
- Сохранение результатов

### 8. BacktestPipeline

Бэктестинг торговых стратегий.

**Шаги:**
- Загрузка модели и данных
- Создание стратегии
- Запуск бэктеста
- Расчет метрик
- Генерация отчета

### 9. FullPipeline

Полный end-to-end процесс.

**Объединяет все предыдущие пайплайны в единый процесс:**
1. Data Preparation
2. Feature Engineering
3. Normalization
4. Feature Selection
5. Labeling
6. Training
7. Validation
8. Backtest
9. Reporting

## Конфигурация

Пайплайны конфигурируются через YAML файлы.

### Пример: Full Pipeline

```yaml
name: full_training_backtest_pipeline
description: End-to-end pipeline

checkpoints:
  enabled: true
  dir: artifacts/checkpoints/full_pipeline

# Включение/выключение шагов
run_data_preparation: true
run_feature_engineering: true
run_normalization: true
run_feature_selection: true
run_labeling: true
run_training: true
run_validation: true
run_backtest: true

# Конфигурация каждого шага
data_preparation:
  ticker: SBER
  timeframe: 5m
  from_date: 2020-01-01
  to_date: 2023-12-31

feature_engineering:
  feature_config: configs/features/default.yaml

training:
  model_type: lightgbm
  model_config:
    n_estimators: 100
    max_depth: 5

# ... другие шаги
```

### Готовые конфигурации

В директории `configs/pipelines/` доступны:

- `full_pipeline_example.yaml` - полный end-to-end процесс
- `training_only.yaml` - только обучение
- `backtest_only.yaml` - только бэктест
- `feature_pipeline.yaml` - работа с признаками

## CLI команды

### Запуск пайплайна

```bash
python -m src.interfaces.cli pipeline run configs/pipelines/full_pipeline_example.yaml
```

### Запуск с форсированным пересчетом

```bash
python -m src.interfaces.cli pipeline run configs/pipelines/full_pipeline_example.yaml --force
```

### Возобновление с чекпоинта

```bash
python -m src.interfaces.cli pipeline resume configs/pipelines/full_pipeline_example.yaml
```

### Статус выполнения

```bash
python -m src.interfaces.cli pipeline status artifacts/checkpoints/full_pipeline
```

### Список доступных конфигураций

```bash
python -m src.interfaces.cli pipeline list
```

### Очистка чекпоинтов

```bash
python -m src.interfaces.cli pipeline clear artifacts/checkpoints/full_pipeline
```

## Чекпоинты и идемпотентность

### Чекпоинты

Каждый шаг пайплайна автоматически сохраняет результат:

```
artifacts/checkpoints/
└── full_pipeline/
    ├── load_data_a1b2c3d4.parquet
    ├── generate_features_a1b2c3d4.parquet
    ├── train_model_a1b2c3d4.pkl
    ├── state_a1b2c3d4.json
    └── result_20231201_123456.json
```

Формат имени: `{step_name}_{config_hash}.{ext}`

### Идемпотентность

Пайплайны вычисляют хэш конфигурации и используют его для:

1. Определения, нужно ли пересчитывать шаг
2. Загрузки правильного чекпоинта
3. Гарантии воспроизводимости

Если конфигурация не изменилась, повторный запуск использует кэш.

### Возобновление после сбоя

При сбое пайплайна можно продолжить с последнего успешного шага:

```bash
python -m src.interfaces.cli pipeline resume configs/pipelines/full_pipeline_example.yaml
```

## Примеры использования

### Пример 1: Полный цикл от данных до бэктеста

```bash
# 1. Создать конфигурацию
cat > my_pipeline.yaml << EOF
name: my_experiment
data_preparation:
  ticker: GAZP
  timeframe: 15m
  from_date: 2022-01-01
  to_date: 2023-12-31
training:
  model_type: xgboost
backtest:
  initial_capital: 100000
EOF

# 2. Запустить
python -m src.interfaces.cli pipeline run my_pipeline.yaml

# 3. Посмотреть результаты
ls artifacts/backtests/
```

### Пример 2: Только обучение на готовых данных

```yaml
# training_pipeline.yaml
name: train_only
run_data_preparation: false
run_feature_engineering: false
run_normalization: false
run_feature_selection: false
run_labeling: false
run_training: true
run_validation: true
run_backtest: false

training:
  data_path: artifacts/features/labeled_data.parquet
  model_type: lightgbm
```

```bash
python -m src.interfaces.cli pipeline run training_pipeline.yaml
```

### Пример 3: Программное использование

```python
from pathlib import Path
from src.pipelines.full_pipeline import FullPipeline

config = {
    "name": "my_pipeline",
    "data_preparation": {
        "ticker": "SBER",
        "timeframe": "5m",
        # ...
    },
    "training": {
        "model_type": "lightgbm",
        # ...
    },
}

pipeline = FullPipeline(
    config=config,
    checkpoint_dir=Path("artifacts/checkpoints/my_pipeline"),
    enable_checkpoints=True,
)

result = pipeline.run()

print(f"Status: {result.status}")
print(f"Duration: {result.duration:.2f}s")
print(f"Artifacts: {result.artifacts}")
```

### Пример 4: Мониторинг прогресса

```python
import time
from src.pipelines.full_pipeline import FullPipeline

def progress_callback(step_name, status):
    print(f"[{time.strftime('%H:%M:%S')}] {step_name}: {status}")

pipeline = FullPipeline(config=config)
# Добавить callback если поддерживается
result = pipeline.run()
```

## Расширение системы

### Создание собственного пайплайна

```python
from src.pipelines.base import BasePipeline
from typing import Any

class CustomPipeline(BasePipeline):
    @property
    def name(self) -> str:
        return "custom_pipeline"

    def _get_steps(self) -> list[str]:
        return ["load", "process", "save"]

    def _execute_step(self, step_name: str, input_data: Any) -> Any:
        if step_name == "load":
            # Ваша логика загрузки
            return self._load_data()
        elif step_name == "process":
            # Ваша логика обработки
            return self._process_data(input_data)
        elif step_name == "save":
            # Ваша логика сохранения
            return self._save_data(input_data)
```

### Интеграция в FullPipeline

Добавьте свой пайплайн в `full_pipeline.py`:

```python
def _execute_step(self, step_name: str, input_data: Any) -> Any:
    # ...
    elif step_name == "custom":
        return self._run_custom(input_data)

def _run_custom(self, input_data: dict[str, Any]) -> dict[str, Any]:
    config = self.config.get("custom", {})
    pipeline = CustomPipeline(config=config)
    result = pipeline.run()
    return {"result": result}
```

## Best Practices

1. **Используйте чекпоинты** для длительных операций
2. **Разбивайте сложные задачи** на отдельные пайплайны
3. **Версионируйте конфигурации** в Git
4. **Документируйте изменения** в конфигах
5. **Тестируйте на маленьких данных** перед полным запуском
6. **Мониторьте логи** для отладки
7. **Очищайте старые чекпоинты** для экономии места

## Troubleshooting

### Пайплайн зависает на шаге

```bash
# Проверьте статус
python -m src.interfaces.cli pipeline status artifacts/checkpoints/my_pipeline

# Посмотрите логи
tail -f logs/pipeline.log
```

### Ошибка "Checkpoint not found"

```bash
# Запустите с force для пересчета
python -m src.interfaces.cli pipeline run config.yaml --force
```

### Пайплайн использует старые данные

```bash
# Очистите чекпоинты
python -m src.interfaces.cli pipeline clear artifacts/checkpoints/my_pipeline --yes

# Запустите заново
python -m src.interfaces.cli pipeline run config.yaml
```

## См. также

- [Подготовка данных](../pipelines/data_preparation.md)
- [Генерация признаков](../features/README.md)
- [Обучение моделей](../models/README.md)
- [Бэктестинг](../backtesting/README.md)
