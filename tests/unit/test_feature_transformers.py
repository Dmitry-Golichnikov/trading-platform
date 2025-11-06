"""Unit тесты для feature transformers."""

import numpy as np
import pandas as pd
import pytest

from src.features.transformers.differences import DifferencesTransformer
from src.features.transformers.lags import LagsTransformer
from src.features.transformers.ratios import RatiosTransformer
from src.features.transformers.rolling import RollingTransformer


@pytest.fixture
def sample_data():
    """Создать тестовые данные."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1D")
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "close": 100 + np.cumsum(np.random.randn(100) * 0.5),
            "volume": np.random.randint(1000, 10000, 100),
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
        },
        index=dates,
    )

    return data


class TestRollingTransformer:
    """Тесты для RollingTransformer."""

    def test_rolling_mean(self, sample_data):
        """Тест rolling mean."""
        transformer = RollingTransformer(window=10, functions=["mean"], columns=["close"])
        features = transformer.transform(sample_data)

        assert "close_rolling_mean_10" in features.columns
        # Проверяем что вычисления корректны
        expected = sample_data["close"].rolling(window=10, min_periods=1).mean()
        pd.testing.assert_series_equal(features["close_rolling_mean_10"], expected, check_names=False)

    def test_multiple_functions(self, sample_data):
        """Тест нескольких функций."""
        transformer = RollingTransformer(window=20, functions=["mean", "std", "min", "max"], columns=["close"])
        features = transformer.transform(sample_data)

        assert len(features.columns) == 4
        assert "close_rolling_mean_20" in features.columns
        assert "close_rolling_std_20" in features.columns
        assert "close_rolling_min_20" in features.columns
        assert "close_rolling_max_20" in features.columns

    def test_multiple_columns(self, sample_data):
        """Тест нескольких колонок."""
        transformer = RollingTransformer(window=10, functions=["mean"], columns=["close", "volume"])
        features = transformer.transform(sample_data)

        assert len(features.columns) == 2
        assert "close_rolling_mean_10" in features.columns
        assert "volume_rolling_mean_10" in features.columns

    def test_invalid_function(self):
        """Тест с невалидной функцией."""
        with pytest.raises(ValueError):
            RollingTransformer(window=10, functions=["invalid"], columns=["close"])

    def test_missing_column(self, sample_data):
        """Тест с отсутствующей колонкой."""
        transformer = RollingTransformer(window=10, functions=["mean"], columns=["nonexistent"])

        with pytest.raises(ValueError):
            transformer.transform(sample_data)


class TestLagsTransformer:
    """Тесты для LagsTransformer."""

    def test_single_lag(self, sample_data):
        """Тест одного лага."""
        transformer = LagsTransformer(lags=[1], columns=["close"])
        features = transformer.transform(sample_data)

        assert "close_lag_1" in features.columns
        # Проверяем корректность
        expected = sample_data["close"].shift(1)
        pd.testing.assert_series_equal(features["close_lag_1"], expected, check_names=False)

    def test_multiple_lags(self, sample_data):
        """Тест нескольких лагов."""
        transformer = LagsTransformer(lags=[1, 5, 10], columns=["close"])
        features = transformer.transform(sample_data)

        assert len(features.columns) == 3
        assert "close_lag_1" in features.columns
        assert "close_lag_5" in features.columns
        assert "close_lag_10" in features.columns

    def test_invalid_lag(self):
        """Тест с невалидным лагом."""
        with pytest.raises(ValueError):
            LagsTransformer(lags=[0], columns=["close"])

        with pytest.raises(ValueError):
            LagsTransformer(lags=[-1], columns=["close"])

    def test_missing_column(self, sample_data):
        """Тест с отсутствующей колонкой."""
        transformer = LagsTransformer(lags=[1], columns=["nonexistent"])

        with pytest.raises(ValueError):
            transformer.transform(sample_data)


class TestDifferencesTransformer:
    """Тесты для DifferencesTransformer."""

    def test_diff_method(self, sample_data):
        """Тест метода diff."""
        transformer = DifferencesTransformer(periods=[1], columns=["close"], method="diff")
        features = transformer.transform(sample_data)

        assert "close_diff_1" in features.columns
        # Проверяем корректность
        expected = sample_data["close"].diff(1)
        pd.testing.assert_series_equal(features["close_diff_1"], expected, check_names=False)

    def test_pct_change_method(self, sample_data):
        """Тест метода pct_change."""
        transformer = DifferencesTransformer(periods=[1], columns=["close"], method="pct_change")
        features = transformer.transform(sample_data)

        assert "close_pct_change_1" in features.columns
        # Проверяем корректность
        expected = sample_data["close"].pct_change(1)
        pd.testing.assert_series_equal(features["close_pct_change_1"], expected, check_names=False)

    def test_multiple_periods(self, sample_data):
        """Тест нескольких периодов."""
        transformer = DifferencesTransformer(periods=[1, 5], columns=["close"], method="diff")
        features = transformer.transform(sample_data)

        assert len(features.columns) == 2
        assert "close_diff_1" in features.columns
        assert "close_diff_5" in features.columns

    def test_invalid_period(self):
        """Тест с невалидным периодом."""
        with pytest.raises(ValueError):
            DifferencesTransformer(periods=[0], columns=["close"], method="diff")

    def test_missing_column(self, sample_data):
        """Тест с отсутствующей колонкой."""
        transformer = DifferencesTransformer(periods=[1], columns=["nonexistent"], method="diff")

        with pytest.raises(ValueError):
            transformer.transform(sample_data)


class TestRatiosTransformer:
    """Тесты для RatiosTransformer."""

    def test_single_ratio(self, sample_data):
        """Тест одного соотношения."""
        transformer = RatiosTransformer(pairs=[("feature1", "feature2")])
        features = transformer.transform(sample_data)

        assert "feature1_div_feature2" in features.columns
        # Проверяем корректность (игнорируя NaN)
        expected = sample_data["feature1"] / sample_data["feature2"]
        # Сравниваем только не-NaN значения
        mask = ~features["feature1_div_feature2"].isna()
        pd.testing.assert_series_equal(
            features["feature1_div_feature2"][mask],
            expected[mask],
            check_names=False,
        )

    def test_multiple_ratios(self, sample_data):
        """Тест нескольких соотношений."""
        transformer = RatiosTransformer(pairs=[("feature1", "feature2"), ("close", "volume")])
        features = transformer.transform(sample_data)

        assert len(features.columns) == 2
        assert "feature1_div_feature2" in features.columns
        assert "close_div_volume" in features.columns

    def test_division_by_zero(self):
        """Тест деления на ноль."""
        data = pd.DataFrame({"a": [1, 2, 3], "b": [1, 0, 2]})  # b содержит 0

        transformer = RatiosTransformer(pairs=[("a", "b")])
        features = transformer.transform(data)

        # Должен быть NaN где b=0
        assert pd.isna(features["a_div_b"].iloc[1])
        # Остальные значения должны быть корректными
        assert features["a_div_b"].iloc[0] == 1.0
        assert features["a_div_b"].iloc[2] == 1.5

    def test_missing_column(self, sample_data):
        """Тест с отсутствующей колонкой."""
        transformer = RatiosTransformer(pairs=[("close", "nonexistent")])

        with pytest.raises(ValueError):
            transformer.transform(sample_data)
