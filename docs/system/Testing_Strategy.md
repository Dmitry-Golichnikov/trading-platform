# Стратегия тестирования

## 1. Типы тестов

### 1.1 Test Pyramid

```
           /\
          /E2E\         10% - End-to-End тесты
         /______\
        /Integration\   30% - Интеграционные тесты
       /____________\
      /    Unit      \  60% - Модульные тесты
     /________________\
```

### 1.2 Уровни тестирования
- **Unit Tests**: Отдельные функции и классы
- **Integration Tests**: Взаимодействие между модулями
- **End-to-End Tests**: Полные пайплайны от начала до конца
- **Performance Tests**: Нагрузочное и стресс-тестирование
- **Regression Tests**: Проверка на регрессию после изменений

## 2. Unit Tests

### 2.1 Структура

```python
# tests/unit/test_indicators.py
import pytest
import pandas as pd
import numpy as np
from src.features.indicators import SMA, RSI, MACD

class TestSMA:
    """Тесты для Simple Moving Average"""

    @pytest.fixture
    def sample_data(self):
        """Подготовка тестовых данных"""
        return pd.DataFrame({
            'close': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
        })

    def test_sma_calculation(self, sample_data):
        """Проверка корректности расчёта SMA"""
        sma = SMA(window=3)
        result = sma.calculate(sample_data)

        # Первые 2 значения должны быть NaN
        assert np.isnan(result['SMA_3'].iloc[0])
        assert np.isnan(result['SMA_3'].iloc[1])

        # Третье значение = (100 + 102 + 101) / 3 = 101
        assert result['SMA_3'].iloc[2] == pytest.approx(101.0)

    def test_sma_with_nan_input(self, sample_data):
        """Обработка NaN во входных данных"""
        sample_data.loc[2, 'close'] = np.nan
        sma = SMA(window=3)
        result = sma.calculate(sample_data)

        # Должен корректно обработать NaN
        assert not result['SMA_3'].isna().all()

    def test_sma_invalid_window(self):
        """Валидация параметров"""
        with pytest.raises(ValueError, match="Window must be positive"):
            SMA(window=0)

    def test_sma_window_larger_than_data(self, sample_data):
        """Окно больше данных"""
        sma = SMA(window=20)
        result = sma.calculate(sample_data)

        # Все значения должны быть NaN
        assert result['SMA_20'].isna().all()

class TestRSI:
    """Тесты для Relative Strength Index"""

    def test_rsi_range(self, sample_data):
        """RSI должен быть в диапазоне [0, 100]"""
        rsi = RSI(period=14)
        result = rsi.calculate(sample_data)

        valid_values = result['RSI_14'].dropna()
        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()

    def test_rsi_known_values(self):
        """Проверка на известных значениях"""
        # Генерируем данные с известным RSI
        prices = [44, 44.34, 44.09, 43.61, 44.33, 44.83,
                  45.10, 45.42, 45.84, 46.08, 45.89, 46.03,
                  45.61, 46.28, 46.28, 46.00, 46.03, 46.41,
                  46.22, 45.64]

        df = pd.DataFrame({'close': prices})
        rsi = RSI(period=14)
        result = rsi.calculate(df)

        # Известное значение RSI для этих данных ≈ 70.46
        assert result['RSI_14'].iloc[-1] == pytest.approx(70.46, abs=0.5)
```

### 2.2 Покрытие кода

```yaml
# configs/testing/coverage.yaml
coverage:
  minimum_percentage: 80

  exclude:
    - "*/tests/*"
    - "*/migrations/*"
    - "*/__init__.py"

  report:
    html: true
    xml: true
    console: true

  fail_under: 80  # Fail если покрытие ниже
```

```bash
# Запуск с покрытием
pytest --cov=src --cov-report=html --cov-report=term --cov-fail-under=80
```

## 3. Integration Tests

### 3.1 Тестирование пайплайнов

```python
# tests/integration/test_feature_pipeline.py
import pytest
from src.pipelines import FeatureEngineeringPipeline
from src.data import DataLoader

class TestFeaturePipeline:
    """Интеграционные тесты feature engineering pipeline"""

    @pytest.fixture
    def pipeline_config(self):
        """Конфигурация для тестирования"""
        return {
            'features': ['SMA_20', 'RSI_14', 'MACD'],
            'data_source': 'test_data.parquet',
            'output_path': 'test_features.parquet'
        }

    def test_full_pipeline(self, pipeline_config, tmp_path):
        """Тест полного цикла пайплайна"""
        # Подготовка тестовых данных
        test_data = self._create_test_data()
        data_path = tmp_path / "test_data.parquet"
        test_data.to_parquet(data_path)

        pipeline_config['data_source'] = str(data_path)
        pipeline_config['output_path'] = str(tmp_path / "features.parquet")

        # Выполнение пайплайна
        pipeline = FeatureEngineeringPipeline(pipeline_config)
        result = pipeline.run()

        # Проверки
        assert result['status'] == 'success'
        assert (tmp_path / "features.parquet").exists()

        # Загрузить результат
        features = pd.read_parquet(pipeline_config['output_path'])

        # Проверить наличие всех признаков
        assert 'SMA_20' in features.columns
        assert 'RSI_14' in features.columns
        assert 'MACD' in features.columns

        # Проверить корректность
        assert not features['SMA_20'].isna().all()
        assert features['RSI_14'].between(0, 100).all()

    def test_pipeline_with_missing_data(self, pipeline_config, tmp_path):
        """Обработка пропущенных данных"""
        test_data = self._create_test_data()
        test_data.loc[10:20, 'close'] = np.nan

        data_path = tmp_path / "test_data.parquet"
        test_data.to_parquet(data_path)

        pipeline_config['data_source'] = str(data_path)
        pipeline_config['output_path'] = str(tmp_path / "features.parquet")

        pipeline = FeatureEngineeringPipeline(pipeline_config)
        result = pipeline.run()

        # Пайплайн должен обработать missing data
        assert result['status'] == 'success'
        assert result['warnings']  # Должно быть предупреждение
```

### 3.2 Тестирование API интеграции

```python
# tests/integration/test_tinkoff_api.py
import pytest
from unittest.mock import Mock, patch
from src.data.tinkoff_client import TinkoffAPIClient
from datetime import datetime, timedelta

class TestTinkoffAPIIntegration:
    """Тесты интеграции с Tinkoff API"""

    @pytest.fixture
    def api_client(self):
        """Мок API клиент для тестирования"""
        with patch('src.data.tinkoff_client.InvestAPI') as mock_api:
            client = TinkoffAPIClient(token="test_token")
            return client

    def test_fetch_candles_success(self, api_client):
        """Успешная загрузка свечей"""
        from_date = datetime.now() - timedelta(days=7)
        to_date = datetime.now()

        candles = api_client.get_candles(
            ticker="SBER",
            from_=from_date,
            to=to_date,
            interval="1h"
        )

        # Проверки
        assert isinstance(candles, pd.DataFrame)
        assert not candles.empty
        assert all(col in candles.columns for col in ['open', 'high', 'low', 'close', 'volume'])

    def test_rate_limiting(self, api_client):
        """Проверка соблюдения rate limits"""
        # Выполнить много запросов
        request_count = 0
        start_time = time.time()

        for _ in range(10):
            api_client.get_trading_status("SBER")
            request_count += 1

        elapsed = time.time() - start_time
        rate = request_count / elapsed

        # Не должно превышать лимит (300 req/min = 5 req/sec)
        assert rate <= 5.5  # Небольшой запас
```

## 4. End-to-End Tests

### 4.1 Полный цикл

```python
# tests/e2e/test_training_backtest_workflow.py
class TestTrainingBacktestWorkflow:
    """E2E тест: обучение модели → бэктест → анализ"""

    def test_complete_workflow(self, tmp_path):
        """Полный цикл от данных до результатов"""
        # 1. Подготовка данных
        data_pipeline = DataPreparationPipeline(
            source='test_data.parquet',
            output=tmp_path / 'prepared_data.parquet'
        )
        data_pipeline.run()

        # 2. Генерация признаков
        feature_pipeline = FeatureEngineeringPipeline(
            data=tmp_path / 'prepared_data.parquet',
            features=['SMA_20', 'RSI_14'],
            output=tmp_path / 'features.parquet'
        )
        feature_pipeline.run()

        # 3. Разметка
        labeling_pipeline = LabelingPipeline(
            data=tmp_path / 'features.parquet',
            target_type='triple_barrier',
            horizon=20,
            output=tmp_path / 'labeled_data.parquet'
        )
        labeling_pipeline.run()

        # 4. Обучение модели
        training_pipeline = ModelTrainingPipeline(
            data=tmp_path / 'labeled_data.parquet',
            model_type='lightgbm',
            output=tmp_path / 'model.pkl'
        )
        training_result = training_pipeline.run()

        assert training_result['status'] == 'success'
        assert (tmp_path / 'model.pkl').exists()

        # 5. Бэктест
        backtest_pipeline = BacktestPipeline(
            model=tmp_path / 'model.pkl',
            data=tmp_path / 'features.parquet',
            output=tmp_path / 'backtest_results.parquet'
        )
        backtest_result = backtest_pipeline.run()

        assert backtest_result['status'] == 'success'
        assert backtest_result['metrics']['total_trades'] > 0

        # 6. Проверка воспроизводимости
        # Повторный запуск должен дать те же результаты
        backtest_result_2 = backtest_pipeline.run()

        assert backtest_result['metrics']['total_pnl'] == pytest.approx(
            backtest_result_2['metrics']['total_pnl'],
            rel=1e-6
        )
```

## 5. Regression Tests

### 5.1 Эталонные датасеты

```python
# tests/regression/test_known_results.py
class TestRegressionKnownResults:
    """Тесты на воспроизводимость известных результатов"""

    @pytest.fixture
    def reference_datasets(self):
        """Эталонные датасеты с известными результатами"""
        return {
            'dataset_v1': {
                'data': 'tests/data/reference/dataset_v1.parquet',
                'expected_metrics': {
                    'accuracy': 0.856,
                    'precision': 0.832,
                    'recall': 0.845
                }
            }
        }

    def test_model_performance_regression(self, reference_datasets):
        """Проверка отсутствия регрессии производительности модели"""
        for name, dataset in reference_datasets.items():
            # Обучить модель на эталонных данных
            model = train_model(dataset['data'], seed=42)

            # Оценить
            metrics = evaluate_model(model, dataset['data'])

            # Сравнить с ожидаемыми метриками
            for metric_name, expected_value in dataset['expected_metrics'].items():
                actual_value = metrics[metric_name]

                # Допускаем небольшое отклонение (±0.01)
                assert actual_value == pytest.approx(expected_value, abs=0.01), \
                    f"Regression detected in {metric_name} for {name}"
```

## 6. Performance Tests

### 6.1 Load Testing

```python
# tests/performance/test_throughput.py
import pytest
from time import time

class TestPerformance:
    """Тесты производительности"""

    def test_feature_calculation_throughput(self):
        """Пропускная способность расчёта признаков"""
        data = generate_large_dataset(rows=100000)

        start = time()
        features = calculate_all_features(data)
        duration = time() - start

        rows_per_second = len(data) / duration

        # Должно обрабатывать хотя бы 10k строк в секунду
        assert rows_per_second > 10000, \
            f"Too slow: {rows_per_second:.0f} rows/sec"

    def test_model_inference_latency(self, trained_model):
        """Latency предсказаний модели"""
        test_input = generate_test_batch(size=1)

        latencies = []
        for _ in range(100):
            start = time()
            prediction = trained_model.predict(test_input)
            latencies.append(time() - start)

        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)

        # Требования к latency
        assert p50 < 0.010, f"P50 latency too high: {p50*1000:.2f}ms"
        assert p95 < 0.020, f"P95 latency too high: {p95*1000:.2f}ms"
        assert p99 < 0.050, f"P99 latency too high: {p99*1000:.2f}ms"

    @pytest.mark.slow
    def test_backtest_large_dataset(self):
        """Бэктест на больших данных (performance)"""
        # 1M баров
        data = generate_large_dataset(rows=1000000)

        start = time()
        backtest_result = run_backtest(data)
        duration = time() - start

        # Должно завершиться за разумное время
        assert duration < 300, f"Backtest too slow: {duration:.0f} seconds"

        bars_per_second = len(data) / duration
        assert bars_per_second > 3000, \
            f"Backtest throughput too low: {bars_per_second:.0f} bars/sec"
```

### 6.2 Memory Profiling

```python
# tests/performance/test_memory.py
import tracemalloc

class TestMemoryUsage:
    """Тесты потребления памяти"""

    def test_feature_pipeline_memory(self):
        """Проверка утечек памяти в pipeline"""
        tracemalloc.start()

        for i in range(10):
            data = generate_large_dataset(rows=50000)
            features = calculate_all_features(data)
            del data, features

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Peak memory не должна расти линейно с итерациями
        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 500, f"Memory leak detected: {peak_mb:.0f} MB peak"
```

## 7. Property-Based Testing

```python
# tests/property/test_indicators_properties.py
from hypothesis import given, strategies as st
import hypothesis.extra.pandas as pdst

class TestIndicatorProperties:
    """Property-based тесты для индикаторов"""

    @given(
        prices=st.lists(
            st.floats(min_value=1, max_value=1000),
            min_size=30,
            max_size=1000
        )
    )
    def test_sma_properties(self, prices):
        """Свойства SMA для любых входных данных"""
        df = pd.DataFrame({'close': prices})
        sma = SMA(window=20)
        result = sma.calculate(df)

        # 1. SMA не должна содержать inf
        assert not np.isinf(result['SMA_20']).any()

        # 2. SMA должна быть между min и max входных данных
        valid_sma = result['SMA_20'].dropna()
        if len(valid_sma) > 0:
            assert valid_sma.min() >= min(prices)
            assert valid_sma.max() <= max(prices)

    @given(
        data=pdst.data_frames([
            pdst.column('close', dtype=float, elements=st.floats(min_value=1, max_value=1000)),
        ], index=pdst.range_indexes(min_size=50, max_size=500))
    )
    def test_rsi_always_in_range(self, data):
        """RSI всегда в [0, 100] для любых данных"""
        rsi = RSI(period=14)
        result = rsi.calculate(data)

        valid_rsi = result['RSI_14'].dropna()
        if len(valid_rsi) > 0:
            assert (valid_rsi >= 0).all()
            assert (valid_rsi <= 100).all()
```

## 8. CI/CD Testing

### 8.1 GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Lint with flake8
        run: |
          flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
          flake8 src/ --count --max-complexity=10 --max-line-length=127 --statistics

      - name: Type check with mypy
        run: |
          mypy src/

      - name: Run unit tests
        run: |
          pytest tests/unit/ -v --cov=src --cov-report=xml

      - name: Run integration tests
        run: |
          pytest tests/integration/ -v

      - name: Run regression tests
        run: |
          pytest tests/regression/ -v

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          fail_ci_if_error: true

  performance:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Run performance tests
        run: |
          pytest tests/performance/ -v --benchmark-only

      - name: Performance regression check
        run: |
          python scripts/check_performance_regression.py
```

## 9. Test Fixtures и Helpers

### 9.1 Общие fixtures

```python
# tests/conftest.py
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_ohlcv_data():
    """Базовые OHLCV данные для тестов"""
    np.random.seed(42)
    n = 1000

    close = np.cumsum(np.random.randn(n)) + 100
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 100000, n)

    return pd.DataFrame({
        'timestamp': pd.date_range('2020-01-01', periods=n, freq='1H'),
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'ticker': 'TEST'
    })

@pytest.fixture
def trained_test_model(sample_ohlcv_data):
    """Обученная модель для тестов"""
    # Подготовить данные
    features, labels = prepare_training_data(sample_ohlcv_data)

    # Обучить простую модель
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(features, labels)

    return model

@pytest.fixture(scope="session")
def test_database():
    """Временная БД для тестов"""
    import tempfile
    import sqlite3

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name

    conn = sqlite3.connect(db_path)
    # Создать схему
    conn.execute("""
        CREATE TABLE experiments (
            id INTEGER PRIMARY KEY,
            name TEXT,
            metrics TEXT
        )
    """)
    conn.commit()

    yield db_path

    # Cleanup
    conn.close()
    os.unlink(db_path)
```

## 10. Best Practices

### 10.1 Рекомендации
- ✅ Тестировать граничные случаи (edge cases)
- ✅ Использовать параметризованные тесты
- ✅ Изолировать тесты (не зависят друг от друга)
- ✅ Моки для внешних зависимостей
- ✅ Стремиться к высокому покрытию (>80%)
- ✅ Быстрые unit тесты, медленные - отдельно
- ✅ Тестировать на разных версиях Python
- ✅ Автоматические тесты в CI/CD

### 10.2 Антипаттерны
- ❌ Тесты зависят от порядка выполнения
- ❌ Тесты используют production данные
- ❌ Долгие unit тесты (>1 сек)
- ❌ Пропуск тестов вместо исправления
- ❌ Отсутствие тестов для критичного кода
- ❌ Тесты не запускаются автоматически

### 10.3 Coverage Goals
- Unit tests: **80%+** покрытие
- Integration tests: Все критичные пайплайны
- E2E tests: Основные пользовательские сценарии
- Regression tests: Все найденные баги
