"""
Тесты для Data Splitting.
"""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.modeling.splitting import DataSplitter


class TestDataSplitter:
    """Тесты для DataSplitter."""

    @pytest.fixture
    def sample_data(self):
        """Создаём тестовые данные."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        data = pd.DataFrame(
            {
                "timestamp": dates,
                "value": np.random.randn(100),
                "feature": np.arange(100),
            }
        )
        return data

    def test_split_sequential(self, sample_data):
        """Тест последовательного разделения."""

        train, val, test = DataSplitter.split_sequential(
            sample_data, train_size=0.7, val_size=0.15, test_size=0.15
        )

        # Проверяем размеры
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

        # Проверяем порядок
        assert train["timestamp"].max() < val["timestamp"].min()
        assert val["timestamp"].max() < test["timestamp"].min()

    def test_split_sequential_invalid_sizes(self, sample_data):
        """Тест с неправильными размерами."""

        with pytest.raises(ValueError):
            DataSplitter.split_sequential(
                sample_data, train_size=0.5, val_size=0.3, test_size=0.3  # Сумма > 1.0
            )

    def test_walk_forward(self, sample_data):
        """Тест walk-forward splitting."""

        splits = DataSplitter.split_walk_forward(
            sample_data, n_splits=3, test_size=10, min_train_size=30
        )

        # Должно быть 3 сплита
        assert len(splits) == 3

        # Проверяем каждый сплит
        for i, (train, test) in enumerate(splits):
            # Train растёт
            assert len(train) == 30 + i * 10
            # Test фиксирован
            assert len(test) == 10
            # Нет overlap
            assert train["timestamp"].max() < test["timestamp"].min()

    def test_split_purged(self, sample_data):
        """Тест purged split с embargo."""

        train, val, test = DataSplitter.split_purged(
            sample_data,
            train_size=0.7,
            val_size=0.15,
            test_size=0.15,
            embargo_td=timedelta(days=2),
        )

        # Общее количество должно быть меньше из-за embargo
        total = len(train) + len(val) + len(test)
        assert total < len(sample_data)

        # Проверяем embargo между train и val
        train_end = train["timestamp"].max()
        val_start = val["timestamp"].min()
        embargo_actual = (val_start - train_end).days
        assert embargo_actual >= 2

    def test_split_by_date(self, sample_data):
        """Тест разделения по датам."""

        train, val, test = DataSplitter.split_by_date(
            sample_data, train_end_date="2020-02-01", val_end_date="2020-03-01"
        )

        # Проверяем границы
        assert train["timestamp"].max() < pd.Timestamp("2020-02-01")
        assert val["timestamp"].min() >= pd.Timestamp("2020-02-01")
        assert val["timestamp"].max() < pd.Timestamp("2020-03-01")
        assert test["timestamp"].min() >= pd.Timestamp("2020-03-01")

    def test_validate_split_valid(self, sample_data):
        """Тест валидации правильного сплита."""

        train, val, test = DataSplitter.split_sequential(sample_data)

        validation = DataSplitter.validate_split(train, val, test)

        assert validation["valid"] is True
        assert len(validation["issues"]) == 0
        assert validation["train_size"] == len(train)
        assert validation["val_size"] == len(val)
        assert validation["test_size"] == len(test)

    def test_validate_split_overlap(self, sample_data):
        """Тест валидации с перекрытием."""

        # Создаём перекрывающиеся сплиты
        train = sample_data.iloc[:60].copy()
        val = sample_data.iloc[50:80].copy()  # Overlap с train
        test = sample_data.iloc[70:].copy()

        validation = DataSplitter.validate_split(train, val, test)

        assert validation["valid"] is False
        assert len(validation["issues"]) > 0

    def test_validate_split_empty(self):
        """Тест валидации пустых сплитов."""

        empty_df = pd.DataFrame()

        validation = DataSplitter.validate_split(empty_df, empty_df, empty_df)

        assert validation["valid"] is False
        assert (
            "пустая" in " ".join(validation["issues"]).lower()
            or "empty" in " ".join(validation["issues"]).lower()
        )
