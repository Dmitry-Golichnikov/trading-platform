"""Unit тесты для FeatureGenerator."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import FeatureCache, FeatureGenerator


@pytest.fixture
def sample_ohlc_data():
    """Создать тестовые OHLC данные."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1H")
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "open": 100 + np.cumsum(np.random.randn(100) * 0.1),
            "high": 100 + np.cumsum(np.random.randn(100) * 0.1) + 0.5,
            "low": 100 + np.cumsum(np.random.randn(100) * 0.1) - 0.5,
            "close": 100 + np.cumsum(np.random.randn(100) * 0.1),
            "volume": np.random.randint(1000, 10000, 100),
        },
        index=dates,
    )

    # Обеспечиваем high >= open, close и low <= open, close
    data["high"] = data[["open", "high", "close"]].max(axis=1)
    data["low"] = data[["open", "low", "close"]].min(axis=1)

    return data


@pytest.fixture
def minimal_config():
    """Минимальная конфигурация признаков."""
    return {
        "version": "1.0",
        "cache_enabled": True,
        "features": [
            {"type": "price", "features": ["returns", "high_low_ratio"]},
            {
                "type": "rolling",
                "window": 10,
                "functions": ["mean"],
                "columns": ["close"],
            },
        ],
    }


class TestFeatureGenerator:
    """Тесты для FeatureGenerator."""

    def test_initialization_with_dict(self, minimal_config):
        """Тест инициализации с dict конфигурацией."""
        generator = FeatureGenerator(minimal_config, cache_enabled=False)

        assert generator.config is not None
        assert len(generator.config.features) == 2

    def test_initialization_with_yaml(self):
        """Тест инициализации с YAML файлом."""
        config_path = Path("configs/features/minimal.yaml")
        if config_path.exists():
            generator = FeatureGenerator(config_path, cache_enabled=False)
            assert generator.config is not None

    def test_generate_price_features(self, sample_ohlc_data, minimal_config):
        """Тест генерации ценовых признаков."""
        generator = FeatureGenerator(minimal_config, cache_enabled=False)
        features = generator.generate(sample_ohlc_data, use_cache=False)

        # Должны быть ценовые признаки и rolling
        assert "returns" in features.columns
        assert "high_low_ratio" in features.columns
        assert "close_rolling_mean_10" in features.columns
        assert len(features) == len(sample_ohlc_data)

    def test_generate_with_indicator(self, sample_ohlc_data):
        """Тест генерации с индикатором."""
        config = {
            "version": "1.0",
            "cache_enabled": False,
            "features": [
                {
                    "type": "indicator",
                    "name": "SMA",
                    "params": {"window": 20},
                }
            ],
        }

        generator = FeatureGenerator(config, cache_enabled=False)
        features = generator.generate(sample_ohlc_data, use_cache=False)

        assert "SMA" in features.columns or "SMA_20" in features.columns
        assert len(features) == len(sample_ohlc_data)

    def test_generate_multiple_feature_types(self, sample_ohlc_data):
        """Тест генерации нескольких типов признаков."""
        config = {
            "version": "1.0",
            "cache_enabled": False,
            "features": [
                {"type": "price", "features": ["returns"]},
                {"type": "volume", "features": ["volume_change"]},
                {"type": "calendar", "features": ["hour"]},
                {
                    "type": "lags",
                    "lags": [1, 2],
                    "columns": ["close"],
                },
            ],
        }

        generator = FeatureGenerator(config, cache_enabled=False)
        features = generator.generate(sample_ohlc_data, use_cache=False)

        # Проверяем наличие всех типов
        assert "returns" in features.columns
        assert "volume_change" in features.columns
        assert "hour" in features.columns
        assert "close_lag_1" in features.columns
        assert "close_lag_2" in features.columns

    def test_caching(self, sample_ohlc_data, minimal_config):
        """Тест кэширования."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "features"
            generator = FeatureGenerator(
                minimal_config, cache_enabled=True, cache_dir=cache_dir
            )

            # Первая генерация - кэш пустой
            features1 = generator.generate(
                sample_ohlc_data, dataset_id="test", use_cache=True
            )

            # Вторая генерация - должна использовать кэш
            features2 = generator.generate(
                sample_ohlc_data, dataset_id="test", use_cache=True
            )

            # Результаты должны быть идентичны
            pd.testing.assert_frame_equal(features1, features2)

            # Проверяем что кэш создан
            cache = FeatureCache(cache_dir)
            stats = cache.get_stats()
            assert stats["total_entries"] == 1

    def test_feature_selection(self, sample_ohlc_data):
        """Тест feature selection."""
        # Создаём простой таргет
        target = pd.Series(
            np.random.randint(0, 2, len(sample_ohlc_data)),
            index=sample_ohlc_data.index,
        )

        config = {
            "version": "1.0",
            "cache_enabled": False,
            "features": [
                {"type": "price", "features": ["returns", "high_low_ratio"]},
                {
                    "type": "rolling",
                    "window": 10,
                    "functions": ["mean", "std"],
                    "columns": ["close", "volume"],
                },
            ],
            "selection": {
                "enabled": True,
                "method": "variance_threshold",
                "params": {"threshold": 0.0},
                "top_k": 3,
            },
        }

        generator = FeatureGenerator(config, cache_enabled=False)
        features = generator.generate(sample_ohlc_data, target=target, use_cache=False)

        # После selection должно остаться не больше 3 признаков
        assert len(features.columns) <= 3


class TestFeatureCache:
    """Тесты для FeatureCache."""

    def test_cache_save_and_get(self):
        """Тест сохранения и получения из кэша."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(Path(tmpdir))

            # Создаём тестовые признаки
            features = pd.DataFrame(
                {
                    "feature1": np.random.randn(100),
                    "feature2": np.random.randn(100),
                }
            )

            config = {"test": "config"}
            dataset_id = "test_dataset"

            # Сохраняем
            cache.save(dataset_id, config, features)

            # Получаем
            retrieved = cache.get(dataset_id, config)

            assert retrieved is not None
            pd.testing.assert_frame_equal(features, retrieved)

    def test_cache_different_configs(self):
        """Тест с разными конфигурациями."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(Path(tmpdir))

            features1 = pd.DataFrame({"feature1": [1, 2, 3]})
            features2 = pd.DataFrame({"feature2": [4, 5, 6]})

            config1 = {"version": "1.0"}
            config2 = {"version": "2.0"}

            dataset_id = "test"

            # Сохраняем с разными конфигами
            cache.save(dataset_id, config1, features1)
            cache.save(dataset_id, config2, features2)

            # Получаем правильные данные для каждого конфига
            retrieved1 = cache.get(dataset_id, config1)
            retrieved2 = cache.get(dataset_id, config2)

            pd.testing.assert_frame_equal(features1, retrieved1)
            pd.testing.assert_frame_equal(features2, retrieved2)

    def test_cache_invalidate(self):
        """Тест инвалидации кэша."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(Path(tmpdir))

            features = pd.DataFrame({"feature1": [1, 2, 3]})
            config = {"test": "config"}
            dataset_id = "test"

            # Сохраняем
            cache.save(dataset_id, config, features)
            assert cache.get(dataset_id, config) is not None

            # Инвалидируем
            cache.invalidate(dataset_id)
            assert cache.get(dataset_id, config) is None

    def test_cache_list_cached(self):
        """Тест листинга кэшированных признаков."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(Path(tmpdir))

            features = pd.DataFrame({"feature1": [1, 2, 3]})
            config = {"test": "config"}

            # Сохраняем несколько датасетов
            cache.save("dataset1", config, features)
            cache.save("dataset2", config, features)

            # Получаем список
            cached = cache.list_cached()
            assert len(cached) == 2

            # Фильтруем по датасету
            cached_filtered = cache.list_cached(dataset_id="dataset1")
            assert len(cached_filtered) == 1
            assert cached_filtered.iloc[0]["dataset_id"] == "dataset1"

    def test_cache_stats(self):
        """Тест статистики кэша."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(Path(tmpdir))

            features = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
            config = {"test": "config"}

            cache.save("dataset1", config, features)

            stats = cache.get_stats()
            assert stats["total_entries"] == 1
            assert stats["total_datasets"] == 1
            assert stats["size_mb"] > 0
