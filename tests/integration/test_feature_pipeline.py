"""Интеграционные тесты для пайплайна генерации признаков."""

import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import FeatureCache, FeatureGenerator


@pytest.fixture
def large_ohlc_data():
    """Создать большой датасет для тестирования производительности."""
    dates = pd.date_range("2020-01-01", periods=10000, freq="5T")
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "open": 100 + np.cumsum(np.random.randn(10000) * 0.1),
            "high": 100 + np.cumsum(np.random.randn(10000) * 0.1) + 0.5,
            "low": 100 + np.cumsum(np.random.randn(10000) * 0.1) - 0.5,
            "close": 100 + np.cumsum(np.random.randn(10000) * 0.1),
            "volume": np.random.randint(1000, 10000, 10000),
        },
        index=dates,
    )

    # Обеспечиваем high >= open, close и low <= open, close
    data["high"] = data[["open", "high", "close"]].max(axis=1)
    data["low"] = data[["open", "low", "close"]].min(axis=1)

    return data


class TestFeatureGenerationPipeline:
    """Интеграционные тесты для полного пайплайна."""

    def test_minimal_config_pipeline(self):
        """Тест с минимальной конфигурацией."""
        config_path = Path("configs/features/minimal.yaml")

        if not config_path.exists():
            pytest.skip("Минимальная конфигурация не найдена")

        # Создаём небольшой датасет
        dates = pd.date_range("2024-01-01", periods=100, freq="1H")
        data = pd.DataFrame(
            {
                "open": 100 + np.cumsum(np.random.randn(100) * 0.1),
                "high": 101 + np.cumsum(np.random.randn(100) * 0.1),
                "low": 99 + np.cumsum(np.random.randn(100) * 0.1),
                "close": 100 + np.cumsum(np.random.randn(100) * 0.1),
                "volume": np.random.randint(1000, 10000, 100),
            },
            index=dates,
        )

        generator = FeatureGenerator(config_path, cache_enabled=False)
        features = generator.generate(data, use_cache=False)

        # Проверяем что признаки сгенерированы
        assert not features.empty
        assert len(features) == len(data)
        assert features.shape[1] > 0

    def test_default_config_pipeline(self):
        """Тест с дефолтной конфигурацией."""
        config_path = Path("configs/features/default.yaml")

        if not config_path.exists():
            pytest.skip("Дефолтная конфигурация не найдена")

        # Создаём средний датасет
        dates = pd.date_range("2024-01-01", periods=500, freq="1H")
        data = pd.DataFrame(
            {
                "open": 100 + np.cumsum(np.random.randn(500) * 0.1),
                "high": 101 + np.cumsum(np.random.randn(500) * 0.1),
                "low": 99 + np.cumsum(np.random.randn(500) * 0.1),
                "close": 100 + np.cumsum(np.random.randn(500) * 0.1),
                "volume": np.random.randint(1000, 10000, 500),
            },
            index=dates,
        )

        generator = FeatureGenerator(config_path, cache_enabled=False)
        features = generator.generate(data, use_cache=False)

        # Проверяем что признаки сгенерированы
        assert not features.empty
        assert len(features) == len(data)
        assert features.shape[1] >= 10  # Должно быть много признаков

    def test_full_pipeline_with_caching(self):
        """Тест полного пайплайна с кэшированием."""
        config_path = Path("configs/features/minimal.yaml")

        if not config_path.exists():
            pytest.skip("Конфигурация не найдена")

        dates = pd.date_range("2024-01-01", periods=200, freq="1H")
        data = pd.DataFrame(
            {
                "open": 100 + np.cumsum(np.random.randn(200) * 0.1),
                "high": 101 + np.cumsum(np.random.randn(200) * 0.1),
                "low": 99 + np.cumsum(np.random.randn(200) * 0.1),
                "close": 100 + np.cumsum(np.random.randn(200) * 0.1),
                "volume": np.random.randint(1000, 10000, 200),
            },
            index=dates,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "features"

            # Первый прогон - создание кэша
            generator = FeatureGenerator(config_path, cache_enabled=True, cache_dir=cache_dir)

            start_time = time.time()
            features1 = generator.generate(data, dataset_id="test", use_cache=True)
            time_without_cache = time.time() - start_time

            # Второй прогон - использование кэша
            start_time = time.time()
            features2 = generator.generate(data, dataset_id="test", use_cache=True)
            time_with_cache = time.time() - start_time

            # Результаты должны быть идентичны
            pd.testing.assert_frame_equal(features1, features2)

            # С кэшем должно быть быстрее
            assert time_with_cache < time_without_cache

            # Проверяем статистику кэша
            cache = FeatureCache(cache_dir)
            stats = cache.get_stats()
            assert stats["total_entries"] >= 1

    def test_performance_10k_bars(self, large_ohlc_data):
        """Тест производительности на 10К баров."""
        config = {
            "version": "1.0",
            "cache_enabled": False,
            "features": [
                {"type": "price", "features": ["returns", "high_low_ratio"]},
                {"type": "volume", "features": ["volume_change"]},
                {"type": "calendar", "features": ["hour", "day_of_week"]},
                {
                    "type": "indicator",
                    "name": "SMA",
                    "params": {"window": 20},
                },
                {
                    "type": "indicator",
                    "name": "RSI",
                    "params": {"period": 14},
                },
                {
                    "type": "rolling",
                    "window": 20,
                    "functions": ["mean", "std"],
                    "columns": ["close"],
                },
                {
                    "type": "lags",
                    "lags": [1, 2, 5],
                    "columns": ["close"],
                },
            ],
        }

        generator = FeatureGenerator(config, cache_enabled=False)

        start_time = time.time()
        features = generator.generate(large_ohlc_data, use_cache=False)
        elapsed_time = time.time() - start_time

        # Проверяем что признаки сгенерированы
        assert not features.empty
        assert len(features) == len(large_ohlc_data)

        # Требование: 10K баров < 10 сек для всех признаков
        error_msg = "Генерация заняла " f"{elapsed_time:.2f} сек (требуется < 10 сек)"
        assert elapsed_time < 10.0, error_msg

        processed_bars = len(large_ohlc_data)
        print("\nПроизводительность: " f"{processed_bars} баров за {elapsed_time:.2f} сек")
        print(f"Скорость: {processed_bars / elapsed_time:.0f} баров/сек")
        print(f"Признаков: {features.shape[1]}")

    def test_nan_handling(self):
        """Тест обработки NaN значений."""
        # Создаём данные с пропусками
        dates = pd.date_range("2024-01-01", periods=100, freq="1H")
        data = pd.DataFrame(
            {
                "open": 100 + np.cumsum(np.random.randn(100) * 0.1),
                "high": 101 + np.cumsum(np.random.randn(100) * 0.1),
                "low": 99 + np.cumsum(np.random.randn(100) * 0.1),
                "close": 100 + np.cumsum(np.random.randn(100) * 0.1),
                "volume": np.random.randint(1000, 10000, 100),
            },
            index=dates,
        )

        # Добавляем пропуски
        data.loc[data.index[10:15], "close"] = np.nan

        config = {
            "version": "1.0",
            "cache_enabled": False,
            "features": [
                {"type": "price", "features": ["returns"]},
                {
                    "type": "rolling",
                    "window": 10,
                    "functions": ["mean"],
                    "columns": ["close"],
                },
            ],
        }

        generator = FeatureGenerator(config, cache_enabled=False)
        features = generator.generate(data, use_cache=False)

        # Признаки должны быть сгенерированы (с NaN где нужно)
        assert not features.empty
        assert len(features) == len(data)

    def test_causality_check(self):
        """Проверка каузальности признаков (no look-ahead)."""
        dates = pd.date_range("2024-01-01", periods=100, freq="1H")
        data = pd.DataFrame(
            {
                "open": 100 + np.cumsum(np.random.randn(100) * 0.1),
                "high": 101 + np.cumsum(np.random.randn(100) * 0.1),
                "low": 99 + np.cumsum(np.random.randn(100) * 0.1),
                "close": 100 + np.cumsum(np.random.randn(100) * 0.1),
                "volume": np.random.randint(1000, 10000, 100),
            },
            index=dates,
        )

        config = {
            "version": "1.0",
            "cache_enabled": False,
            "features": [
                {"type": "price", "features": ["returns"]},
                {
                    "type": "rolling",
                    "window": 10,
                    "functions": ["mean"],
                    "columns": ["close"],
                },
                {
                    "type": "lags",
                    "lags": [1],
                    "columns": ["close"],
                },
            ],
        }

        generator = FeatureGenerator(config, cache_enabled=False)
        features = generator.generate(data, use_cache=False)

        # Проверяем что первые значения NaN где ожидается
        # (признаки не используют будущую информацию)
        assert pd.isna(features["returns"].iloc[0])
        assert pd.isna(features["close_lag_1"].iloc[0])
        # Rolling с min_periods=1 не должен быть NaN для первой строки
        assert not pd.isna(features["close_rolling_mean_10"].iloc[0])
