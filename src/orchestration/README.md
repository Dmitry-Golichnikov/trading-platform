# Orchestration Module

Модуль оркестрации для управления экспериментами, планирования задач и мониторинга.

## Компоненты

### 1. ExperimentManager
Управление ML экспериментами с интеграцией MLflow.

```python
from src.orchestration import ExperimentManager

manager = ExperimentManager()

# Создание эксперимента
exp_id = manager.create_experiment(
    name="my_experiment",
    config={"model": "lightgbm"},
    tags={"version": "1.0"}
)

# Запуск
manager.run_experiment(exp_id)

# Сравнение
df = manager.compare_experiments([exp_id_1, exp_id_2])
```

### 2. MLflowManager
Полная обёртка над MLflow для трекинга и Model Registry.

```python
from src.orchestration import MLflowManager

mlflow = MLflowManager(experiment_name="my_experiments")

# Запуск run
run_id = mlflow.start_run(run_name="experiment_1")

# Логирование
mlflow.log_params({"learning_rate": 0.01})
mlflow.log_metrics({"accuracy": 0.95})
mlflow.log_model(model, "model")

# Завершение
mlflow.end_run()
```

### 3. TaskScheduler
Cron-like планировщик задач.

```python
from src.orchestration import TaskScheduler

scheduler = TaskScheduler()

# Планирование задачи
scheduler.schedule_task(
    task_id="daily_training",
    name="Daily Model Training",
    func_or_name=train_model,
    schedule="0 9 * * *"  # Каждый день в 9:00
)

# Запуск планировщика
scheduler.start()
```

**Формат cron:** `minute hour day month day_of_week`

Примеры:
- `* * * * *` - каждую минуту
- `0 * * * *` - каждый час
- `0 9 * * *` - каждый день в 9:00
- `0 9 * * 1` - каждый понедельник в 9:00
- `0,30 * * * *` - каждые 30 минут
- `0 9-17 * * *` - каждый час с 9 до 17

### 4. TaskExecutor
Асинхронное выполнение задач (thread/process-based).

```python
from src.orchestration import TaskExecutor

executor = TaskExecutor(max_workers=4, execution_mode="thread")

def process_data(data):
    # Обработка данных
    return result

# Отправка задачи
task_id = executor.submit(process_data, data=my_data)

# Получение результата
result = executor.get_result(task_id, timeout=60)

# Пакетная отправка
tasks = [
    {"func": process_data, "args": (data1,)},
    {"func": process_data, "args": (data2,)},
]
task_ids = executor.submit_batch(tasks)
```

### 5. TaskQueue
Очередь задач (локальная или Redis-based).

```python
from src.orchestration import TaskQueueManager

# Локальная очередь
queue = TaskQueueManager(backend="local")

# Redis очередь (для распределенных систем)
# queue = TaskQueueManager(backend="redis", redis_url="redis://localhost:6379")

# Добавление задачи
queue.enqueue(
    task_id="task_1",
    task_type="training",
    payload={"model": "lightgbm"},
    priority=1
)

# Извлечение задачи
message = queue.dequeue(timeout=5)
if message:
    process(message.payload)
```

### 6. MonitoringHooks
Система событий и мониторинга.

```python
from src.orchestration.monitoring import (
    get_monitoring_manager,
    FileHook,
    MetricsHook,
)

manager = get_monitoring_manager()

# Добавление хуков
manager.add_hook(FileHook("logs/events.json", format="json"))
manager.add_hook(MetricsHook())

# Эмиссия событий
manager.on_training_start("model_1", {"lr": 0.001})
manager.on_training_epoch("model_1", 1, {"loss": 0.5})
manager.on_training_complete("model_1", {"accuracy": 0.9}, duration=120)

# Пользовательские события
manager.emit(
    event_type="custom_event",
    source="my_component",
    data={"key": "value"},
    severity="info"
)
```

Доступные хуки:
- **LoggingHook** - логирование в logger
- **FileHook** - запись в файл (JSON/text)
- **MetricsHook** - сбор метрик
- **CallbackHook** - пользовательский callback
- **MLflowHook** - интеграция с MLflow

## CLI Команды

```bash
# Создание эксперимента
python -m src.interfaces.cli experiment create \
  --name "my_experiment" \
  --config configs/experiments/simple_training_experiment.yaml \
  --tag model=lightgbm

# Запуск эксперимента
python -m src.interfaces.cli experiment run exp_id

# Список экспериментов
python -m src.interfaces.cli experiment list --status completed

# Статус эксперимента
python -m src.interfaces.cli experiment status exp_id

# Сравнение экспериментов
python -m src.interfaces.cli experiment compare exp_1 exp_2 exp_3 \
  --metric accuracy --metric f1_score

# Лучший эксперимент
python -m src.interfaces.cli experiment best --metric f1_score

# Экспорт
python -m src.interfaces.cli experiment export \
  --output experiments.csv --format csv
```

## Конфигурации

Примеры конфигураций экспериментов в `configs/experiments/`:

- **simple_training_experiment.yaml** - базовое обучение
- **hyperopt_experiment.yaml** - оптимизация гиперпараметров
- **full_pipeline_experiment.yaml** - полный пайплайн
- **ensemble_experiment.yaml** - ансамбль моделей
- **neural_network_experiment.yaml** - нейросети

Структура конфигурации:
```yaml
experiment:
  name: experiment_name
  description: Description
  tags:
    model_type: lightgbm

data:
  ticker: SBER
  timeframe: 1h
  train_start: "2020-01-01"
  train_end: "2023-12-31"

features:
  config: configs/features/default.yaml

labeling:
  config: configs/labeling/long_only.yaml

model:
  type: lightgbm
  config: configs/models/lightgbm_default.yaml

training:
  validation_split: 0.15
  random_seed: 42

evaluation:
  metrics:
    - accuracy
    - f1_score
```

## Интеграция

### С Pipeline
```python
from src.orchestration import ExperimentManager
from src.pipelines import TrainingPipeline

manager = ExperimentManager()
pipeline = TrainingPipeline()

manager.run_experiment(
    experiment_id,
    pipeline=pipeline
)
```

### С MLflow
Все эксперименты автоматически логируются в MLflow:
```bash
mlflow ui --backend-store-uri file:./artifacts/mlruns
```

### С Monitoring
```python
from src.orchestration.monitoring import emit_event

# Из любого модуля
emit_event(
    event_type="model_trained",
    source="training_module",
    data={"accuracy": 0.95}
)
```

## Архитектура

```
ExperimentManager
    ↓
MLflowManager → MLflow Tracking Server
    ↓
TaskScheduler → TaskExecutor → TaskQueue
    ↓                              ↓
MonitoringHooks              LocalQueue / RedisQueue
```

## Best Practices

1. **Используйте теги** для организации экспериментов
2. **Фиксируйте random_seed** для воспроизводимости
3. **Логируйте все параметры** в MLflow
4. **Сравнивайте эксперименты** перед выбором модели
5. **Мониторьте события** через hooks
6. **Планируйте долгие задачи** через scheduler
7. **Используйте TaskQueue** для распределенных вычислений

## Тесты

```bash
# Запуск тестов модуля
pytest tests/unit/test_orchestration_*

# С покрытием
pytest tests/unit/test_orchestration_* --cov=src.orchestration
```

## Примеры использования

### Автоматическое обучение моделей
```python
from src.orchestration import TaskScheduler, ExperimentManager

scheduler = TaskScheduler()
manager = ExperimentManager()

def daily_training():
    exp_id = manager.create_experiment(
        name="daily_model",
        config=load_config("training.yaml")
    )
    manager.run_experiment(exp_id)

# Запускать каждый день в 2:00
scheduler.schedule_task(
    task_id="daily_training",
    name="Daily Model Training",
    func_or_name=daily_training,
    schedule="0 2 * * *"
)

scheduler.start()
```

### Параллельное обучение на разных тикерах
```python
from src.orchestration import TaskExecutor

executor = TaskExecutor(max_workers=4)

tickers = ["SBER", "GAZP", "LKOH", "YNDX"]

def train_on_ticker(ticker):
    # Обучение модели
    return model

# Параллельное обучение
task_ids = [
    executor.submit(train_on_ticker, ticker)
    for ticker in tickers
]

# Ожидание всех
executor.wait_for_all(task_ids)

# Получение результатов
models = {
    ticker: executor.get_result(task_id)
    for ticker, task_id in zip(tickers, task_ids)
}
```

### Мониторинг обучения в реальном времени
```python
from src.orchestration.monitoring import get_monitoring_manager, MetricsHook

manager = get_monitoring_manager()
metrics_hook = MetricsHook()
manager.add_hook(metrics_hook)

# В процессе обучения
for epoch in range(100):
    loss = train_epoch()

    manager.on_training_epoch(
        "model_name",
        epoch,
        {"loss": loss, "lr": current_lr}
    )

# Получение статистики
all_losses = metrics_hook.get_metric("model_name_loss")
print(f"Average loss: {sum(all_losses) / len(all_losses)}")
```

## Troubleshooting

### MLflow не подключается
```python
# Используйте локальный tracking
manager = ExperimentManager(
    mlflow_tracking_uri="file:./artifacts/mlruns"
)
```

### Redis недоступен
```python
# Автоматический fallback на локальную очередь
queue = TaskQueueManager(backend="redis")  # Упадет на local если Redis недоступен
```

### Задачи не выполняются
```bash
# Проверьте scheduler
scheduler = TaskScheduler()
task = scheduler.get_task("task_id")
print(task.enabled)  # Должно быть True

next_run = scheduler.get_next_run_time("task_id")
print(next_run)  # Время следующего запуска
```

## Дополнительная информация

- Полная документация: `docs/system/orchestration.md`
- Примеры конфигураций: `configs/experiments/`
- План этапа: `plan/Этап_13_Оркестрация.md`
- Отчет о выполнении: `plan/Этап_13_ВЫПОЛНЕН.md`
