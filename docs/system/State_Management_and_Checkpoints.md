# Управление состоянием и чекпоинты

## 1. Общие принципы

### 1.1 Цели системы чекпоинтов
- **Idempotency**: Повторный запуск даёт тот же результат
- **Resumability**: Возможность продолжить с любой точки останова
- **Reproducibility**: Фиксация всех параметров для воспроизведения
- **Efficiency**: Минимальные накладные расходы на сохранение

### 1.2 Типы состояний
- **Persistent state**: Данные, модели, конфигурации (долгосрочное хранение)
- **Transient state**: Промежуточные результаты, кэши (можно пересчитать)
- **Execution state**: Текущий прогресс операций (для resume)
- **System state**: Окружение, версии зависимостей (для воспроизводимости)

## 2. Форматы чекпоинтов

### 2.1 Структура чекпоинта

```yaml
# checkpoint_metadata.yaml
checkpoint:
  id: "ckpt_20250126_143022_a7b3c4d"
  timestamp: "2025-01-26T14:30:22Z"
  type: "model_training"  # data_ingestion, feature_engineering, training, backtesting
  status: "in_progress"   # started, in_progress, completed, failed, cancelled

operation:
  pipeline: "training_pipeline_v1"
  stage: "hyperparameter_search"
  substage: "trial_15"
  progress:
    completed: 15
    total: 50
    percentage: 30.0

state:
  # Специфично для типа операции
  model_state_path: "models/weights/epoch_10.pt"
  optimizer_state_path: "models/optimizer/epoch_10.pt"
  scheduler_state_path: "models/scheduler/epoch_10.pt"
  random_state:
    python_seed: 42
    numpy_seed: 42
    torch_seed: 42
  current_epoch: 10
  best_metric: 0.876
  epochs_without_improvement: 2

data:
  dataset_version: "v1.2.3"
  dataset_hash: "sha256:abcd1234..."
  train_split: [0, 50000]
  val_split: [50000, 60000]

config:
  config_path: "configs/experiments/exp_042.yaml"
  config_hash: "sha256:efgh5678..."

environment:
  python_version: "3.10.12"
  torch_version: "2.1.0"
  cuda_version: "12.1"
  platform: "linux-x86_64"

resources:
  gpu_used: true
  device: "cuda:0"
  memory_peak_mb: 8192

parent_checkpoint: "ckpt_20250126_120000_x1y2z3"
children_checkpoints: []
```

### 2.2 Именование
```
checkpoint_{timestamp}_{operation}_{short_hash}
└─ checkpoint_20250126_143022_train_a7b3c4d
   ├── metadata.yaml          # Основная информация
   ├── state.pkl              # Состояние Python объектов
   ├── model_weights.pt       # Веса модели (если применимо)
   ├── optimizer.pt           # Состояние оптимизатора
   └── random_states.pkl      # RNG состояния для воспроизводимости
```

## 3. Стратегии чекпоинтов по модулям

### 3.1 Data Ingestion

#### Что сохраняется
- Список загруженных файлов/API запросов
- Timestamp последней успешной записи
- Статистика обработанных данных (rows, tickers, date ranges)
- Хеши сырых данных

#### Частота сохранения
- После каждого успешного батча (1000 записей или 1 тикер)
- Каждые 5 минут при длительной загрузке
- При завершении работы (graceful shutdown)

#### Формат
```json
{
  "ingestion_id": "ing_20250126_140000",
  "source": "tinkoff_api",
  "tickers_completed": ["SBER", "GAZP", "LKOH"],
  "tickers_in_progress": "YNDX",
  "current_ticker_progress": {
    "ticker": "YNDX",
    "date_from": "2023-01-01",
    "date_to": "2024-12-31",
    "last_fetched_date": "2024-06-15",
    "rows_fetched": 125000
  },
  "tickers_pending": ["ROSN", "MGNT"],
  "total_rows": 1250000,
  "errors": []
}
```

#### Восстановление
1. Загрузка чекпоинта
2. Skip завершённых тикеров
3. Продолжение с `last_fetched_date` для текущего тикера
4. Обработка оставшихся тикеров

### 3.2 Feature Engineering

#### Что сохраняется
- Граф зависимостей признаков
- Вычисленные фичи (группами)
- Параметры индикаторов
- Кэш промежуточных расчётов

#### Частота сохранения
- После вычисления каждой группы признаков (по типу индикатора)
- Каждые 10 минут для долгих расчётов
- При изменении конфигурации

#### Инкрементальные вычисления
```python
# Пример структуры кэша
{
  "feature_group": "technical_indicators_current_tf",
  "features": ["SMA_20", "RSI_14", "MACD"],
  "last_computed_timestamp": "2024-12-31T23:59:00",
  "cache_path": "cache/features/tech_indicators_v1.parquet",
  "dependency_hashes": {
    "data": "sha256:abc123",
    "config": "sha256:def456"
  },
  "status": "valid"
}
```

#### Инвалидация кэша
При изменении:
- Исходных данных (проверка по хешу)
- Параметров индикаторов
- Версии кода вычисления (semantic versioning)

### 3.3 Model Training

#### Что сохраняется
- **Model checkpoint**: веса, архитектура, конфигурация
- **Optimizer state**: моментумы, адаптивные lr
- **Scheduler state**: текущий lr, счётчики
- **Training metrics**: loss, validation metrics по эпохам
- **Random states**: torch, numpy, python для воспроизводимости
- **Data samplers**: порядок батчей (для точного воспроизведения)

#### Частота сохранения
- **Периодические**: Каждые N эпох (по умолчанию N=5)
- **Best model**: При улучшении валидационной метрики
- **Last model**: В конце обучения
- **Recovery**: Каждый час для защиты от сбоев

#### Стратегии ротации
```python
# Политика хранения чекпоинтов
CHECKPOINT_POLICY = {
    "keep_best": 3,           # Топ-3 по метрике
    "keep_last": 1,           # Последний
    "keep_periodic": 5,       # Каждые 5 эпох
    "keep_recovery": 2,       # Последние 2 recovery
    "max_total": 10,          # Максимум чекпоинтов
    "cleanup_older_than_days": 30
}
```

#### Формат
```python
# PyTorch checkpoint
checkpoint = {
    'epoch': 42,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': 0.234,
    'metrics': {
        'train_loss': [0.5, 0.4, 0.3, ...],
        'val_loss': [0.6, 0.5, 0.4, ...],
        'val_accuracy': [0.7, 0.75, 0.8, ...]
    },
    'random_state': {
        'torch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(),
        'cuda_rng_state': torch.cuda.get_rng_state_all()
    },
    'config': config_dict,
    'metadata': {
        'timestamp': '2025-01-26T14:30:00Z',
        'duration_seconds': 3600,
        'device': 'cuda:0'
    }
}
```

#### Восстановление
```python
def resume_training(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    # Восстановление модели
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Восстановление RNG
    torch.set_rng_state(checkpoint['random_state']['torch_rng_state'])
    np.random.set_state(checkpoint['random_state']['numpy_rng_state'])
    random.setstate(checkpoint['random_state']['python_rng_state'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint['random_state']['cuda_rng_state'])

    # Продолжение с следующей эпохи
    start_epoch = checkpoint['epoch'] + 1
    return start_epoch
```

### 3.4 Hyperparameter Search

#### Что сохраняется
- **Study database**: Optuna SQLite/Postgres
- **Trial results**: Параметры + метрики всех trials
- **Best trials**: Топ-N конфигураций
- **Search state**: Sampler state для адаптивных алгоритмов

#### Частота сохранения
- После каждого завершённого trial
- Автоматическое персистирование в БД

#### Формат
```python
# Optuna study
study = optuna.create_study(
    study_name='experiment_042',
    storage='sqlite:///artifacts/optuna/studies.db',
    load_if_exists=True  # Автоматическое resume
)

# Метаданные trial
{
    "trial_id": 15,
    "params": {
        "learning_rate": 0.001,
        "batch_size": 64,
        "hidden_size": 128
    },
    "value": 0.876,  # Целевая метрика
    "state": "COMPLETE",  # RUNNING, COMPLETE, PRUNED, FAIL
    "datetime_start": "2025-01-26T14:00:00Z",
    "datetime_complete": "2025-01-26T14:30:00Z",
    "duration_seconds": 1800
}
```

#### Восстановление
- Автоматическое при использовании persistent storage
- Trials в состоянии RUNNING помечаются как FAIL при перезапуске
- Продолжение с того же sampler state

### 3.5 Backtesting

#### Что сохраняется
- Результаты по каждой сделке
- Агрегированные метрики
- Состояние портфеля (balance, positions) на каждом шаге
- События (entries, exits, alerts)

#### Частота сохранения
- После обработки каждого года данных
- При завершении бэктеста
- Опционально: каждые N сделок для очень длинных симуляций

#### Формат
```json
{
  "backtest_id": "bt_20250126_150000",
  "strategy_config": "configs/strategies/strategy_001.yaml",
  "data_range": {"start": "2020-01-01", "end": "2024-12-31"},
  "current_date": "2023-06-15",
  "trades_completed": 1523,
  "trades_results_path": "artifacts/backtests/bt_20250126/trades.parquet",
  "portfolio_state": {
    "balance": 105342.67,
    "positions": [
      {"ticker": "SBER", "quantity": 100, "entry_price": 250.5}
    ],
    "equity": 125342.67
  },
  "metrics_snapshot": {
    "total_return": 0.2534,
    "sharpe_ratio": 1.45,
    "max_drawdown": -0.12
  }
}
```

#### Восстановление
1. Загрузка завершённых сделок
2. Восстановление состояния портфеля
3. Продолжение с `current_date`

## 4. Механизм чекпоинтов

### 4.1 Checkpoint Manager

```python
class CheckpointManager:
    """Управление чекпоинтами для всех операций"""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.active_checkpoints = {}

    def create(self, operation_type: str, metadata: dict) -> str:
        """Создать новый чекпоинт"""
        checkpoint_id = self._generate_id(operation_type)
        checkpoint_path = self.base_path / checkpoint_id
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Сохранить метаданные
        with open(checkpoint_path / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f)

        self.active_checkpoints[checkpoint_id] = checkpoint_path
        return checkpoint_id

    def save(self, checkpoint_id: str, state: dict):
        """Сохранить состояние"""
        checkpoint_path = self.active_checkpoints[checkpoint_id]

        # Атомарная запись через temp file
        temp_path = checkpoint_path / "state.pkl.tmp"
        with open(temp_path, "wb") as f:
            pickle.dump(state, f)
        temp_path.rename(checkpoint_path / "state.pkl")

        # Обновить timestamp
        self._update_metadata(checkpoint_id, {"last_updated": datetime.now()})

    def load(self, checkpoint_id: str) -> dict:
        """Загрузить состояние"""
        checkpoint_path = self.base_path / checkpoint_id
        with open(checkpoint_path / "state.pkl", "rb") as f:
            return pickle.load(f)

    def verify_integrity(self, checkpoint_id: str) -> bool:
        """Проверить целостность чекпоинта"""
        try:
            state = self.load(checkpoint_id)
            metadata = self._load_metadata(checkpoint_id)
            # Проверка хешей, версий, etc
            return True
        except Exception as e:
            logger.error(f"Checkpoint {checkpoint_id} corrupted: {e}")
            return False

    def cleanup(self, policy: dict):
        """Очистка старых чекпоинтов согласно политике"""
        # Реализация CHECKPOINT_POLICY
        pass
```

### 4.2 Автоматическое сохранение

```python
class AutoCheckpoint:
    """Контекстный менеджер для автоматических чекпоинтов"""

    def __init__(self, operation_type: str, interval_seconds: int = 300):
        self.operation_type = operation_type
        self.interval = interval_seconds
        self.last_save = time.time()
        self.checkpoint_manager = CheckpointManager(CHECKPOINT_BASE_PATH)
        self.checkpoint_id = None

    def __enter__(self):
        self.checkpoint_id = self.checkpoint_manager.create(
            self.operation_type,
            metadata={"status": "started"}
        )
        return self

    def maybe_save(self, state: dict):
        """Сохранить если прошло достаточно времени"""
        if time.time() - self.last_save >= self.interval:
            self.checkpoint_manager.save(self.checkpoint_id, state)
            self.last_save = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Успешное завершение
            self.checkpoint_manager.save(self.checkpoint_id, {"status": "completed"})
        else:
            # Ошибка - сохранить для отладки
            self.checkpoint_manager.save(
                self.checkpoint_id,
                {"status": "failed", "error": str(exc_val)}
            )

# Использование
with AutoCheckpoint("model_training", interval_seconds=300) as ckpt:
    for epoch in range(100):
        train_step()
        state = get_current_state()
        ckpt.maybe_save(state)
```

## 5. Параллельные операции

### 5.1 Изоляция состояния
- Каждая параллельная задача имеет уникальный checkpoint_id
- Shared state минимизирован
- Lock-free чтение общих артефактов (immutable)

### 5.2 Координация
```python
# Синхронизация через state file
{
  "orchestrator_id": "orch_main",
  "running_tasks": [
    {"task_id": "task_001", "type": "training", "device": "gpu:0", "status": "running"},
    {"task_id": "task_002", "type": "backtest", "device": "cpu", "status": "running"}
  ],
  "completed_tasks": ["task_000"],
  "failed_tasks": []
}
```

### 5.3 Dependency management
- Граф зависимостей между задачами
- Задача не стартует, пока зависимости не завершены
- Чекпоинт содержит ссылки на зависимости

## 6. Версионирование состояний

### 6.1 Schema Evolution
При изменении формата чекпоинта:

```python
CHECKPOINT_SCHEMA_VERSION = 3

def migrate_checkpoint(checkpoint, from_version, to_version):
    """Миграция между версиями схем"""
    if from_version == 1 and to_version >= 2:
        # Добавлены random_states
        checkpoint['random_state'] = get_default_random_state()

    if from_version <= 2 and to_version >= 3:
        # Изменён формат метрик
        checkpoint['metrics'] = {
            'train': checkpoint.pop('train_metrics'),
            'val': checkpoint.pop('val_metrics')
        }

    return checkpoint
```

### 6.2 Backward Compatibility
- Новый код должен читать старые чекпоинты
- Warning при загрузке устаревшего формата
- Автоматическая миграция при сохранении

## 7. Распределённые чекпоинты

### 7.1 Синхронизация между устройствами
- Чекпоинты сохраняются в shared storage (S3/MinIO)
- Ноутбук и стационарный ПК работают с одним checkpoint manager
- При resume автоматический выбор последнего актуального чекпоинта

### 7.2 Conflict Resolution
- Timestamp-based: выбор более свежего чекпоинта
- Lock при создании нового чекпоинта (опционально)

## 8. Monitoring чекпоинтов

### 8.1 Метрики
- Размер чекпоинтов (disk usage)
- Частота сохранений
- Время сохранения/загрузки
- Количество failed checkpoints

### 8.2 Алерты
- Диск заполнен > 90%
- Чекпоинт не сохранялся > 1 час при длительной операции
- Повреждённый чекпоинт обнаружен

## 9. Best Practices

### 9.1 Что включать в чекпоинт
✅ **Да:**
- Состояние, которое дорого пересчитывать
- Параметры и конфигурации
- Метрики для анализа прогресса
- Random states для воспроизводимости

❌ **Нет:**
- Данные, которые можно быстро загрузить из источника
- Огромные объекты (лучше сохранить отдельно и хранить ссылку)
- Transient кэши

### 9.2 Частота сохранения
- Не слишком часто: overhead на I/O
- Не слишком редко: риск потери прогресса
- Адаптивная: чаще в начале, реже при стабильной работе

### 9.3 Тестирование
- Обязательные тесты resume для всех длительных операций
- Проверка воспроизводимости результата при загрузке чекпоинта
- Stress-тесты с искусственными прерываниями
