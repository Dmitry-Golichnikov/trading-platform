# Требования к производительности

## 1. Целевые метрики

### 1.1 Data Ingestion
- **Throughput**: ≥ 50,000 баров/сек (локальные файлы)
- **Throughput API**: ≥ 1,000 баров/сек (Tinkoff API с учетом rate limits)
- **Latency**: < 100ms для загрузки одного дня минутных данных

### 1.2 Feature Engineering
- **Throughput**: ≥ 10,000 баров/сек (расчёт всех технических индикаторов)
- **Memory**: < 2GB RAM для 1M баров
- **Latency**: < 1 секунда для расчёта одного индикатора на 10K барах

### 1.3 Model Training
- **GPU Utilization**: ≥ 80% во время обучения
- **Training Speed**: < 5 минут на 100K samples (LSTM, GPU)
- **Training Speed**: < 2 минуты на 100K samples (LightGBM, CPU)
- **Memory**: < 8GB GPU memory для обучения

### 1.4 Model Inference
- **Latency P50**: < 10ms (single prediction)
- **Latency P95**: < 20ms
- **Latency P99**: < 50ms
- **Throughput**: ≥ 1,000 predictions/sec (batch)

### 1.5 Backtesting
- **Throughput**: ≥ 5,000 баров/сек (с учетом расчёта метрик)
- **Memory**: < 4GB RAM для симуляции 1M баров
- **Full Backtest**: < 5 минут для 2 лет минутных данных

### 1.6 GUI/Dashboard
- **Page Load**: < 2 секунды
- **Chart Rendering**: < 500ms для 10K точек
- **API Response**: < 100ms (P95)
- **Interactive Updates**: < 100ms

## 2. Benchmark Suite

### 2.1 Структура бенчмарков

```python
# tests/benchmarks/conftest.py
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def benchmark_data_small():
    """10K баров для быстрых тестов"""
    return generate_ohlcv_data(n=10_000)

@pytest.fixture
def benchmark_data_medium():
    """100K баров для средних тестов"""
    return generate_ohlcv_data(n=100_000)

@pytest.fixture
def benchmark_data_large():
    """1M баров для стресс-тестов"""
    return generate_ohlcv_data(n=1_000_000)

def generate_ohlcv_data(n: int) -> pd.DataFrame:
    """Генерировать синтетические OHLCV данные"""
    np.random.seed(42)

    close = 100 + np.cumsum(np.random.randn(n) * 0.1)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(1000, 100000, n)

    return pd.DataFrame({
        'timestamp': pd.date_range('2020-01-01', periods=n, freq='1min'),
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
```

### 2.2 Feature Engineering Benchmarks

```python
# tests/benchmarks/test_feature_performance.py
import pytest
from src.features.indicators import SMA, RSI, MACD, BollingerBands

class TestFeaturePerformance:
    """Бенчмарки производительности признаков"""

    def test_sma_performance_small(self, benchmark, benchmark_data_small):
        """SMA на 10K данных"""
        sma = SMA(window=20)

        # pytest-benchmark автоматически запустит несколько итераций
        result = benchmark(sma.calculate, benchmark_data_small)

        # Проверить throughput
        bars_per_sec = len(benchmark_data_small) / benchmark.stats['mean']
        assert bars_per_sec > 50_000, f"Too slow: {bars_per_sec:.0f} bars/sec"

    def test_all_indicators_medium(self, benchmark, benchmark_data_medium):
        """Все технические индикаторы на 100K данных"""

        def calculate_all_features(data):
            features = pd.DataFrame(index=data.index)
            features['SMA_20'] = SMA(20).calculate(data)
            features['RSI_14'] = RSI(14).calculate(data)
            features['MACD'] = MACD().calculate(data)
            features['BB_upper'], features['BB_lower'] = BollingerBands(20).calculate(data)
            return features

        result = benchmark(calculate_all_features, benchmark_data_medium)

        bars_per_sec = len(benchmark_data_medium) / benchmark.stats['mean']
        assert bars_per_sec > 10_000, f"Too slow: {bars_per_sec:.0f} bars/sec"

    @pytest.mark.slow
    def test_feature_memory_usage(self, benchmark_data_large):
        """Проверка потребления памяти"""
        import tracemalloc

        tracemalloc.start()

        # Рассчитать признаки
        features = calculate_all_features(benchmark_data_large)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)

        # Для 1M баров не должно быть больше 2GB
        assert peak_mb < 2048, f"Too much memory: {peak_mb:.0f} MB"
```

### 2.3 Model Training Benchmarks

```python
# tests/benchmarks/test_training_performance.py
class TestTrainingPerformance:
    """Бенчмарки обучения моделей"""

    def test_lightgbm_training_speed(self, benchmark):
        """Скорость обучения LightGBM"""
        X, y = generate_classification_dataset(n_samples=100_000, n_features=50)

        def train_model():
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(n_estimators=100)
            model.fit(X, y)
            return model

        model = benchmark(train_model)

        # Должно быть меньше 2 минут
        assert benchmark.stats['mean'] < 120, \
            f"Training too slow: {benchmark.stats['mean']:.1f}s"

    @pytest.mark.gpu
    def test_lstm_training_speed_gpu(self, benchmark):
        """Скорость обучения LSTM на GPU"""
        import torch

        X, y = generate_sequence_dataset(n_samples=100_000, seq_len=50, n_features=10)

        def train_lstm():
            model = LSTMModel(input_size=10, hidden_size=128, num_layers=2)
            model.cuda()

            # Обучение 10 эпох
            for epoch in range(10):
                # ... training loop
                pass

            return model

        model = benchmark(train_lstm)

        # Должно быть меньше 5 минут
        assert benchmark.stats['mean'] < 300, \
            f"LSTM training too slow: {benchmark.stats['mean']:.1f}s"

    @pytest.mark.gpu
    def test_gpu_utilization(self):
        """Проверка утилизации GPU"""
        import torch
        import GPUtil

        # Запустить обучение
        model = train_model_gpu()

        # Проверить утилизацию
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            assert gpu.load > 0.8, \
                f"Low GPU utilization: {gpu.load*100:.0f}%"
```

### 2.4 Inference Benchmarks

```python
# tests/benchmarks/test_inference_performance.py
class TestInferencePerformance:
    """Бенчмарки inference"""

    def test_single_prediction_latency(self, benchmark, trained_model):
        """Latency одного предсказания"""
        X_single = generate_single_input()

        result = benchmark(trained_model.predict, X_single)

        latency_ms = benchmark.stats['mean'] * 1000

        assert latency_ms < 10, f"P50 latency too high: {latency_ms:.2f}ms"
        assert benchmark.stats['stddev'] * 1000 < 5, \
            f"High variance: {benchmark.stats['stddev']*1000:.2f}ms"

    def test_batch_prediction_throughput(self, benchmark, trained_model):
        """Throughput батчевых предсказаний"""
        X_batch = generate_batch_input(batch_size=1000)

        result = benchmark(trained_model.predict, X_batch)

        predictions_per_sec = len(X_batch) / benchmark.stats['mean']

        assert predictions_per_sec > 1000, \
            f"Throughput too low: {predictions_per_sec:.0f} pred/sec"

    def test_inference_percentiles(self, trained_model):
        """P50, P95, P99 latencies"""
        X_single = generate_single_input()

        latencies = []
        for _ in range(1000):
            start = time.perf_counter()
            trained_model.predict(X_single)
            latencies.append(time.perf_counter() - start)

        p50 = np.percentile(latencies, 50) * 1000
        p95 = np.percentile(latencies, 95) * 1000
        p99 = np.percentile(latencies, 99) * 1000

        assert p50 < 10, f"P50: {p50:.2f}ms"
        assert p95 < 20, f"P95: {p95:.2f}ms"
        assert p99 < 50, f"P99: {p99:.2f}ms"
```

### 2.5 Backtesting Benchmarks

```python
# tests/benchmarks/test_backtest_performance.py
class TestBacktestPerformance:
    """Бенчмарки бэктестинга"""

    def test_backtest_throughput(self, benchmark, benchmark_data_large):
        """Throughput бэктеста"""
        strategy = SimpleStrategy()

        def run_backtest():
            return backtest(benchmark_data_large, strategy)

        result = benchmark(run_backtest)

        bars_per_sec = len(benchmark_data_large) / benchmark.stats['mean']

        assert bars_per_sec > 5000, \
            f"Backtest too slow: {bars_per_sec:.0f} bars/sec"

    @pytest.mark.slow
    def test_full_backtest_time(self):
        """Время полного бэктеста (2 года минутных данных)"""
        # 2 года * 365 дней * 24 часа * 60 минут = ~1M баров
        data = generate_ohlcv_data(n=1_000_000)
        strategy = ComplexStrategy()

        start = time.time()
        result = backtest(data, strategy)
        duration = time.time() - start

        # Должно быть меньше 5 минут
        assert duration < 300, f"Full backtest too slow: {duration:.0f}s"
```

## 3. Profiling

### 3.1 CPU Profiling

```python
# scripts/profile_cpu.py
import cProfile
import pstats
from pstats import SortKey

def profile_function(func, *args, **kwargs):
    """Profile функции"""
    profiler = cProfile.Profile()
    profiler.enable()

    result = func(*args, **kwargs)

    profiler.disable()

    # Вывести статистику
    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)  # Топ-20 функций

    # Сохранить в файл для визуализации
    stats.dump_stats('profile_results.prof')

    return result

# Использование
profile_function(calculate_all_features, data)

# Визуализация с snakeviz
# pip install snakeviz
# snakeviz profile_results.prof
```

### 3.2 Line Profiling

```python
# Установка: pip install line_profiler

# Добавить @profile декоратор к функции
@profile
def calculate_features(data):
    # ... код
    pass

# Запуск
# kernprof -l -v script.py
```

### 3.3 Memory Profiling

```python
# Установка: pip install memory_profiler

from memory_profiler import profile

@profile
def memory_intensive_function():
    # ... код
    pass

# Запуск
# python -m memory_profiler script.py

# Или построчно
# mprof run script.py
# mprof plot
```

### 3.4 GPU Profiling

```python
# PyTorch Profiler
import torch
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    model(input_tensor)

# Вывести результаты
print(prof.key_averages().table(sort_by="cuda_time_total"))

# Экспорт для Chrome Trace Viewer
prof.export_chrome_trace("trace.json")
```

## 4. Optimization Strategies

### 4.1 Vectorization

```python
# ❌ Медленно: Loops
def calculate_returns_slow(prices):
    returns = []
    for i in range(1, len(prices)):
        ret = (prices[i] - prices[i-1]) / prices[i-1]
        returns.append(ret)
    return returns

# ✅ Быстро: Vectorized
def calculate_returns_fast(prices):
    return prices.pct_change()

# Benchmark показывает ~100x ускорение
```

### 4.2 Numba JIT Compilation

```python
from numba import jit, prange

@jit(nopython=True, parallel=True)
def calculate_sma_numba(prices, window):
    """SMA с Numba JIT"""
    n = len(prices)
    sma = np.empty(n)
    sma[:window-1] = np.nan

    for i in prange(window-1, n):
        sma[i] = np.mean(prices[i-window+1:i+1])

    return sma

# Может дать 10-100x ускорение для numerical operations
```

### 4.3 Caching

```python
from functools import lru_cache
import diskcache

# Memory cache
@lru_cache(maxsize=128)
def expensive_calculation(param):
    # ... долгие вычисления
    return result

# Disk cache
cache = diskcache.Cache('cache_directory')

@cache.memoize(expire=3600)  # 1 час
def calculate_features(ticker, date):
    # ... вычисления
    return features
```

### 4.4 Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def process_ticker(ticker):
    """Обработать один тикер"""
    data = load_data(ticker)
    features = calculate_features(data)
    return ticker, features

def process_all_tickers_parallel(tickers):
    """Обработать все тикеры параллельно"""
    n_workers = multiprocessing.cpu_count()

    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_ticker, ticker): ticker
            for ticker in tickers
        }

        for future in as_completed(futures):
            ticker, features = future.result()
            results[ticker] = features

    return results

# Линейное масштабирование с количеством ядер
```

### 4.5 GPU Acceleration

```python
import cupy as cp  # GPU-версия NumPy

# CPU
def calculate_on_cpu(data):
    import numpy as np
    return np.mean(data ** 2)

# GPU
def calculate_on_gpu(data):
    data_gpu = cp.array(data)
    result = cp.mean(data_gpu ** 2)
    return cp.asnumpy(result)

# Для больших массивов GPU может быть 10-100x быстрее
```

## 5. Monitoring Performance

### 5.1 Real-time Metrics

```python
class PerformanceMonitor:
    """Мониторинг производительности в реальном времени"""

    def __init__(self):
        self.metrics = defaultdict(list)

    def record_operation(self, operation_name: str, duration: float):
        """Записать длительность операции"""
        self.metrics[operation_name].append(duration)

    def get_statistics(self, operation_name: str) -> dict:
        """Получить статистику операции"""
        durations = self.metrics[operation_name]

        if not durations:
            return {}

        return {
            'count': len(durations),
            'mean': np.mean(durations),
            'median': np.median(durations),
            'p95': np.percentile(durations, 95),
            'p99': np.percentile(durations, 99),
            'min': min(durations),
            'max': max(durations)
        }

# Использование с context manager
from contextlib import contextmanager

@contextmanager
def measure_time(monitor, operation_name):
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        monitor.record_operation(operation_name, duration)

# Применение
monitor = PerformanceMonitor()

with measure_time(monitor, 'feature_calculation'):
    features = calculate_features(data)

print(monitor.get_statistics('feature_calculation'))
```

### 5.2 Performance Alerts

```python
class PerformanceAlerter:
    """Алерты при деградации производительности"""

    def __init__(self, thresholds: dict):
        self.thresholds = thresholds
        self.monitor = PerformanceMonitor()

    def check_and_alert(self, operation_name: str):
        """Проверить и отправить алерт если нужно"""
        stats = self.monitor.get_statistics(operation_name)

        if not stats:
            return

        threshold = self.thresholds.get(operation_name)
        if not threshold:
            return

        if stats['p95'] > threshold:
            self._send_alert(
                f"Performance degradation: {operation_name} "
                f"P95={stats['p95']:.3f}s (threshold={threshold}s)"
            )

# Конфигурация
thresholds = {
    'feature_calculation': 1.0,  # 1 секунда
    'model_inference': 0.01,      # 10ms
    'backtest_bar': 0.001         # 1ms на бар
}
```

## 6. Best Practices

### 6.1 Optimization Guidelines
- ✅ Профилировать перед оптимизацией
- ✅ Фокус на горячих участках (90/10 rule)
- ✅ Измерять результат оптимизации
- ✅ Документировать причины оптимизации
- ✅ Использовать векторизацию где возможно
- ✅ Кэшировать дорогие вычисления
- ✅ Мониторить производительность в production

### 6.2 Антипаттерны
- ❌ Преждевременная оптимизация
- ❌ Оптимизация без измерения
- ❌ Жертвовать читаемостью без значительного выигрыша
- ❌ Игнорировать memory usage
- ❌ Отсутствие бенчмарков
- ❌ Не тестировать на больших данных
