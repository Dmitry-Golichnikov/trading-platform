# Интеграция между модулями

## 1. Принципы интеграции

### 1.1 Loose Coupling
- Модули взаимодействуют через чётко определённые интерфейсы
- Минимальная зависимость между модулями
- Изменения в одном модуле не должны ломать другие

### 1.2 High Cohesion
- Каждый модуль решает конкретную задачу
- Связанные функции группируются вместе
- Чёткая ответственность каждого модуля

### 1.3 Dependency Injection
- Зависимости передаются извне, а не создаются внутри
- Упрощает тестирование и замену компонентов
- Централизованное управление зависимостями

## 2. Архитектура интеграции

### 2.1 Слоистая архитектура

```
┌────────────────────────────────────────┐
│         Presentation Layer             │  GUI/CLI
├────────────────────────────────────────┤
│         Application Layer              │  Orchestration, Pipelines
├────────────────────────────────────────┤
│          Domain Layer                  │  Business Logic
├────────────────────────────────────────┤
│      Infrastructure Layer              │  Data Access, External APIs
└────────────────────────────────────────┘
```

### 2.2 Dependency Graph

```python
# Граф зависимостей (пример)
"""
GUI/CLI
  ↓
Orchestrator
  ↓
Pipelines
  ↓
┌───────┬──────────┬──────────┐
│ Data  │ Features │ Modeling │
└───────┴──────────┴──────────┘
  ↓         ↓          ↓
Common (utils, config, logging)
"""
```

## 3. Контракты между модулями

### 3.1 Интерфейсы (Protocols)

```python
# src/common/interfaces.py
from typing import Protocol, Any
import pandas as pd

class DataLoader(Protocol):
    """Интерфейс для загрузчиков данных"""

    def load(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Загрузить данные

        Args:
            ticker: Тикер инструмента
            from_date: Начальная дата
            to_date: Конечная дата

        Returns:
            DataFrame с OHLCV данными
        """
        ...

class FeatureCalculator(Protocol):
    """Интерфейс для расчёта признаков"""

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать признаки

        Args:
            data: Входные OHLCV данные

        Returns:
            DataFrame с признаками
        """
        ...

class Model(Protocol):
    """Интерфейс для моделей"""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Обучить модель"""
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Сделать предсказания"""
        ...

    def save(self, path: Path) -> None:
        """Сохранить модель"""
        ...

    @classmethod
    def load(cls, path: Path) -> 'Model':
        """Загрузить модель"""
        ...
```

### 3.2 Data Transfer Objects (DTOs)

```python
# src/common/dto.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass(frozen=True)
class OHLCVBar:
    """DTO для одного OHLCV бара"""
    timestamp: datetime
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: int

    def validate(self) -> None:
        """Валидация данных"""
        if self.high < self.low:
            raise ValueError(f"High ({self.high}) < Low ({self.low})")
        if self.high < self.open or self.high < self.close:
            raise ValueError("High must be >= open and close")
        if self.low > self.open or self.low > self.close:
            raise ValueError("Low must be <= open and close")
        if self.volume < 0:
            raise ValueError("Volume must be non-negative")

@dataclass
class TrainingConfig:
    """DTO для конфигурации обучения"""
    model_type: str
    hyperparameters: dict
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42
    use_gpu: bool = True

@dataclass
class BacktestResult:
    """DTO для результатов бэктеста"""
    strategy_name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    profit_factor: float
    equity_curve: list[float]
    trades: list[dict]
```

## 4. Service Layer

### 4.1 Сервисы как посредники

```python
# src/services/data_service.py
class DataService:
    """Сервис для работы с данными"""

    def __init__(
        self,
        loader: DataLoader,
        validator: DataValidator,
        cache: CacheManager
    ):
        self.loader = loader
        self.validator = validator
        self.cache = cache

    def get_data(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Получить данные с валидацией и кэшированием

        Args:
            ticker: Тикер
            from_date: Начальная дата
            to_date: Конечная дата
            use_cache: Использовать кэш

        Returns:
            Валидированные данные
        """
        cache_key = f"{ticker}_{from_date}_{to_date}"

        # Проверить кэш
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        # Загрузить данные
        data = self.loader.load(ticker, from_date, to_date)

        # Валидировать
        self.validator.validate(data)

        # Сохранить в кэш
        if use_cache:
            self.cache.set(cache_key, data)

        return data

# src/services/model_service.py
class ModelService:
    """Сервис для работы с моделями"""

    def __init__(
        self,
        model_registry: ModelRegistry,
        feature_service: FeatureService
    ):
        self.registry = model_registry
        self.feature_service = feature_service

    def train_model(
        self,
        config: TrainingConfig,
        data: pd.DataFrame
    ) -> Model:
        """Обучить модель"""
        # Получить модель из registry
        model_class = self.registry.get_model_class(config.model_type)
        model = model_class(**config.hyperparameters)

        # Подготовить признаки
        features = self.feature_service.calculate_features(data)

        # Разделить данные
        X_train, X_val, y_train, y_val = self._split_data(
            features,
            config
        )

        # Обучить
        model.fit(X_train, y_train)

        # Оценить
        val_score = model.score(X_val, y_val)
        logger.info(f"Validation score: {val_score}")

        return model
```

### 4.2 Dependency Injection Container

```python
# src/common/container.py
from dependency_injector import containers, providers

class Container(containers.DeclarativeContainer):
    """IoC Container для управления зависимостями"""

    # Config
    config = providers.Configuration()

    # Infrastructure
    cache = providers.Singleton(
        CacheManager,
        cache_dir=config.cache_dir
    )

    database = providers.Singleton(
        DatabaseConnection,
        url=config.database_url
    )

    # Data
    tinkoff_client = providers.Factory(
        TinkoffAPIClient,
        token=config.tinkoff_token
    )

    data_loader = providers.Factory(
        TinkoffDataLoader,
        client=tinkoff_client
    )

    data_validator = providers.Factory(
        DataValidator,
        config=config.validation
    )

    # Services
    data_service = providers.Factory(
        DataService,
        loader=data_loader,
        validator=data_validator,
        cache=cache
    )

    feature_service = providers.Factory(
        FeatureService,
        config=config.features
    )

    model_service = providers.Factory(
        ModelService,
        model_registry=providers.Singleton(ModelRegistry),
        feature_service=feature_service
    )

# Использование
container = Container()
container.config.from_yaml('config.yaml')

# Получить сервис
data_service = container.data_service()
data = data_service.get_data('SBER', '2023-01-01', '2023-12-31')
```

## 5. Event-Driven Integration

### 5.1 Event Bus

```python
# src/common/events.py
from typing import Callable, Dict, List
from dataclasses import dataclass
from enum import Enum

class EventType(Enum):
    DATA_LOADED = "data_loaded"
    FEATURES_CALCULATED = "features_calculated"
    MODEL_TRAINED = "model_trained"
    BACKTEST_COMPLETED = "backtest_completed"
    ERROR_OCCURRED = "error_occurred"

@dataclass
class Event:
    """Базовый класс события"""
    event_type: EventType
    data: dict
    timestamp: datetime = field(default_factory=datetime.now)

class EventBus:
    """Шина событий для асинхронной интеграции"""

    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}

    def subscribe(self, event_type: EventType, handler: Callable):
        """Подписаться на событие"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def publish(self, event: Event):
        """Опубликовать событие"""
        handlers = self._subscribers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")

# Использование
event_bus = EventBus()

# Подписка
def on_data_loaded(event: Event):
    logger.info(f"Data loaded: {event.data}")
    # Автоматически запустить расчёт признаков
    feature_service.calculate_features(event.data['dataframe'])

event_bus.subscribe(EventType.DATA_LOADED, on_data_loaded)

# Публикация
data = data_service.get_data('SBER', '2023-01-01', '2023-12-31')
event_bus.publish(Event(
    event_type=EventType.DATA_LOADED,
    data={'ticker': 'SBER', 'dataframe': data}
))
```

### 5.2 Message Queue (для распределённых вычислений)

```python
# src/common/queue.py
import redis
import json
from typing import Any

class TaskQueue:
    """Очередь задач для распределённых вычислений"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)

    def enqueue(self, queue_name: str, task: dict):
        """Добавить задачу в очередь"""
        self.redis.rpush(queue_name, json.dumps(task))

    def dequeue(self, queue_name: str, timeout: int = 0) -> dict:
        """Получить задачу из очереди"""
        result = self.redis.blpop(queue_name, timeout=timeout)
        if result:
            _, task_json = result
            return json.loads(task_json)
        return None

# Worker на стационарном ПК
class Worker:
    """Worker для выполнения задач"""

    def __init__(self, queue: TaskQueue):
        self.queue = queue

    def run(self):
        """Запустить worker"""
        while True:
            task = self.queue.dequeue('training_tasks')
            if task:
                self.execute_task(task)

    def execute_task(self, task: dict):
        """Выполнить задачу"""
        task_type = task['type']

        if task_type == 'train_model':
            # Обучить модель
            result = train_model(**task['params'])

            # Отправить результат
            self.queue.enqueue('results', {
                'task_id': task['id'],
                'result': result
            })

# Клиент (ноутбук)
task_queue = TaskQueue()

# Отправить задачу на обучение
task_queue.enqueue('training_tasks', {
    'id': 'task_001',
    'type': 'train_model',
    'params': {
        'config': 'configs/model.yaml',
        'data': 'data/train.parquet'
    }
})

# Ожидать результат
result = task_queue.dequeue('results')
```

## 6. API Gateway Pattern

### 6.1 Unified API

```python
# src/interfaces/api/gateway.py
from fastapi import FastAPI, Depends

app = FastAPI(title="Trading Platform API Gateway")

class APIGateway:
    """Единая точка входа для всех сервисов"""

    def __init__(
        self,
        data_service: DataService,
        model_service: ModelService,
        backtest_service: BacktestService
    ):
        self.data_service = data_service
        self.model_service = model_service
        self.backtest_service = backtest_service

# Dependency
def get_gateway() -> APIGateway:
    container = Container()
    return APIGateway(
        data_service=container.data_service(),
        model_service=container.model_service(),
        backtest_service=container.backtest_service()
    )

# Endpoints
@app.get("/data/{ticker}")
async def get_data(
    ticker: str,
    from_date: str,
    to_date: str,
    gateway: APIGateway = Depends(get_gateway)
):
    """Получить данные"""
    data = gateway.data_service.get_data(ticker, from_date, to_date)
    return data.to_dict()

@app.post("/models/train")
async def train_model(
    config: TrainingConfig,
    gateway: APIGateway = Depends(get_gateway)
):
    """Обучить модель"""
    model = gateway.model_service.train_model(config)
    return {"model_id": model.id, "status": "success"}

@app.post("/backtest/run")
async def run_backtest(
    model_id: str,
    data_params: dict,
    gateway: APIGateway = Depends(get_gateway)
):
    """Запустить бэктест"""
    result = gateway.backtest_service.run(model_id, data_params)
    return result
```

## 7. Адаптеры для внешних API

### 7.1 Adapter Pattern

```python
# src/adapters/tinkoff_adapter.py
class TinkoffAdapter:
    """Адаптер для Tinkoff Investments API"""

    def __init__(self, client: InvestAPI):
        self.client = client

    def get_candles(
        self,
        ticker: str,
        from_: datetime,
        to: datetime,
        interval: str
    ) -> pd.DataFrame:
        """
        Адаптировать Tinkoff API к нашему интерфейсу

        Returns:
            DataFrame в стандартном формате
        """
        # Получить FIGI по тикеру
        figi = self._get_figi(ticker)

        # Запросить свечи
        candles = self.client.market_data.get_candles(
            figi=figi,
            from_=from_,
            to=to,
            interval=self._map_interval(interval)
        )

        # Преобразовать в наш формат
        data = []
        for candle in candles.candles:
            data.append({
                'timestamp': candle.time,
                'ticker': ticker,
                'open': self._quotation_to_float(candle.open),
                'high': self._quotation_to_float(candle.high),
                'low': self._quotation_to_float(candle.low),
                'close': self._quotation_to_float(candle.close),
                'volume': candle.volume
            })

        return pd.DataFrame(data)

    def _map_interval(self, interval: str) -> CandleInterval:
        """Маппинг интервалов"""
        mapping = {
            '1m': CandleInterval.CANDLE_INTERVAL_1_MIN,
            '5m': CandleInterval.CANDLE_INTERVAL_5_MIN,
            '1h': CandleInterval.CANDLE_INTERVAL_HOUR,
            '1d': CandleInterval.CANDLE_INTERVAL_DAY,
        }
        return mapping[interval]
```

## 8. Cross-Cutting Concerns

### 8.1 Logging Interceptor

```python
# src/common/interceptors.py
import functools
import logging

def log_execution(logger: logging.Logger = None):
    """Декоратор для логирования выполнения"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or logging.getLogger(func.__module__)

            _logger.info(
                f"Calling {func.__name__}",
                extra={
                    'function': func.__name__,
                    'args': args,
                    'kwargs': kwargs
                }
            )

            try:
                result = func(*args, **kwargs)
                _logger.info(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                _logger.error(
                    f"{func.__name__} failed: {e}",
                    exc_info=True
                )
                raise

        return wrapper
    return decorator
```

### 8.2 Performance Monitoring

```python
# src/common/monitoring.py
import time
from contextlib import contextmanager

class PerformanceMonitor:
    """Мониторинг производительности интеграционных точек"""

    @contextmanager
    def measure(self, operation: str):
        """Измерить время выполнения"""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.record_metric(operation, duration)

    def record_metric(self, operation: str, duration: float):
        """Записать метрику"""
        # Отправить в систему мониторинга
        pass

# Использование
monitor = PerformanceMonitor()

with monitor.measure('data_loading'):
    data = data_service.get_data('SBER', '2023-01-01', '2023-12-31')
```

## 9. Testing Integration

### 9.1 Integration Tests

```python
# tests/integration/test_data_to_features.py
def test_data_loading_and_feature_calculation():
    """Тест интеграции data -> features"""
    # Arrange
    data_service = DataService(...)
    feature_service = FeatureService(...)

    # Act
    data = data_service.get_data('TEST', '2023-01-01', '2023-12-31')
    features = feature_service.calculate_features(data)

    # Assert
    assert not features.empty
    assert 'SMA_20' in features.columns
    assert len(features) == len(data)
```

### 9.2 Contract Tests

```python
# tests/contracts/test_data_loader_contract.py
def test_data_loader_contract():
    """Проверить что все DataLoader реализации соответствуют контракту"""
    loaders = [
        TinkoffDataLoader(...),
        LocalFileLoader(...),
        CachedDataLoader(...)
    ]

    for loader in loaders:
        # Проверить интерфейс
        assert hasattr(loader, 'load')

        # Проверить результат
        result = loader.load('TEST', '2023-01-01', '2023-12-31')
        assert isinstance(result, pd.DataFrame)
        assert all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume'])
```

## 10. Best Practices

### 10.1 Checklist
- [ ] Чёткие интерфейсы между модулями
- [ ] Dependency Injection для управления зависимостями
- [ ] DTOs для передачи данных
- [ ] Service Layer для бизнес-логики
- [ ] Адаптеры для внешних API
- [ ] Event Bus для асинхронной интеграции
- [ ] Логирование на границах модулей
- [ ] Integration tests для проверки взаимодействия

### 10.2 Антипаттерны
- ❌ Прямое обращение к внутренностям других модулей
- ❌ Circular dependencies
- ❌ God objects (один класс делает все)
- ❌ Tight coupling
- ❌ Отсутствие error handling на границах
- ❌ Нет версионирования API между модулями
- ❌ Shared mutable state между модулями
