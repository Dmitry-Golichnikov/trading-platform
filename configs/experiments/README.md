# Конфигурации экспериментов

Эта директория содержит конфигурационные файлы для различных типов экспериментов.

## Типы экспериментов

### 1. Простое обучение
**Файл:** `simple_training_experiment.yaml`

Базовый эксперимент для обучения одной модели на одном тикере.

**Использование:**
```bash
python -m src.interfaces.cli experiment create \
  --name "my_experiment" \
  --config configs/experiments/simple_training_experiment.yaml \
  --description "Мой первый эксперимент"
```

### 2. Гиперпараметрическая оптимизация
**Файл:** `hyperopt_experiment.yaml`

Эксперимент с автоматическим поиском оптимальных гиперпараметров.

**Особенности:**
- Использует Optuna для поиска
- Поддерживает различные sampler'ы (TPE, Random, Grid)
- Автоматически логирует все попытки в MLflow

### 3. Полный пайплайн
**Файл:** `full_pipeline_experiment.yaml`

End-to-end эксперимент: обучение → оценка → бэктест.

**Включает:**
- Обучение модели
- Оценку на тестовом наборе
- Полный бэктест с учетом комиссий
- Визуализацию результатов

### 4. Ансамбль моделей
**Файл:** `ensemble_experiment.yaml`

Эксперимент с несколькими моделями для повышения стабильности предсказаний.

**Поддерживаемые методы:**
- Weighted Average
- Voting (soft/hard)
- Stacking

### 5. Нейросетевая модель
**Файл:** `neural_network_experiment.yaml`

Эксперимент с LSTM/GRU для работы с последовательностями.

**Особенности:**
- Поддержка GPU
- Learning rate scheduling
- Early stopping
- TensorBoard integration

## Структура конфигурации

Каждая конфигурация эксперимента содержит следующие секции:

### experiment
Метаданные эксперимента:
- `name` - имя эксперимента
- `description` - описание
- `tags` - теги для фильтрации и поиска

### data
Параметры данных:
- `ticker/tickers` - тикер(ы) для обучения
- `timeframe` - таймфрейм
- `train_start/train_end` - период обучения
- `test_start/test_end` - период тестирования

### features
Конфигурация признаков:
- `config` - путь к конфигурации признаков
- `selection` - настройки отбора признаков
- `normalization` - настройки нормализации

### labeling
Конфигурация таргетов:
- `config` - путь к конфигурации разметки

### model / models
Конфигурация модели(ей):
- `type` - тип модели
- `config` - путь к конфигурации модели
- `hyperopt` - настройки оптимизации (опционально)

### training
Параметры обучения:
- `validation_split` - размер валидационной выборки
- `random_seed` - seed для воспроизводимости
- `early_stopping_rounds` - ранняя остановка
- `use_gpu` - использовать GPU

### evaluation
Параметры оценки:
- `metrics` - список метрик для расчета
- `calibration` - настройки калибровки
- `visualization` - настройки визуализации

### backtest (опционально)
Параметры бэктеста:
- `initial_capital` - начальный капитал
- `commission` - комиссия
- `entry/exit` - правила входа/выхода
- `metrics` - метрики стратегии

## Примеры использования

### Создание эксперимента
```bash
python -m src.interfaces.cli experiment create \
  --name "sber_lightgbm" \
  --config configs/experiments/simple_training_experiment.yaml \
  --tag model=lightgbm \
  --tag ticker=SBER
```

### Запуск эксперимента
```bash
python -m src.interfaces.cli experiment run sber_lightgbm_20241115_120000
```

### Просмотр результатов
```bash
# Список всех экспериментов
python -m src.interfaces.cli experiment list

# Статус конкретного эксперимента
python -m src.interfaces.cli experiment status sber_lightgbm_20241115_120000

# Лучший эксперимент по метрике
python -m src.interfaces.cli experiment best --metric f1_score
```

### Сравнение экспериментов
```bash
python -m src.interfaces.cli experiment compare \
  exp_id_1 exp_id_2 exp_id_3 \
  --metric accuracy \
  --metric f1_score \
  --output comparison.csv
```

## Создание собственной конфигурации

1. Скопируйте один из примеров
2. Измените параметры под свою задачу
3. Сохраните с уникальным именем
4. Используйте в CLI

## Интеграция с MLflow

Все эксперименты автоматически логируются в MLflow:
- Параметры конфигурации
- Метрики обучения и оценки
- Артефакты (модели, графики, отчеты)
- Теги для фильтрации

Просмотр в MLflow UI:
```bash
mlflow ui --backend-store-uri file:./artifacts/mlruns
```

## Best Practices

1. **Используйте описательные имена** для экспериментов
2. **Добавляйте теги** для упрощения поиска
3. **Фиксируйте random_seed** для воспроизводимости
4. **Документируйте** изменения в конфигурациях
5. **Сравнивайте** результаты нескольких экспериментов
6. **Сохраняйте** успешные конфигурации

## Troubleshooting

### Ошибка "Config not found"
Проверьте, что пути к вложенным конфигурациям корректны.

### Ошибка "MLflow connection failed"
MLflow недоступен, но эксперимент будет продолжен с локальным логированием.

### Ошибка "GPU not available"
Установите `use_gpu: false` в секции training.

## Дополнительно

Для более сложных сценариев используйте Python API:

```python
from src.orchestration import ExperimentManager

manager = ExperimentManager()

# Создание эксперимента программно
exp_id = manager.create_experiment(
    name="custom_experiment",
    config={...},
    tags={"custom": "true"}
)

# Запуск
manager.run_experiment(exp_id)
```
