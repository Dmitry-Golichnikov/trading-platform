"""Тесты для фильтров разметки."""

import numpy as np
import pandas as pd
import pytest

from src.labeling.filters.danger_zones import DangerZonesFilter
from src.labeling.filters.majority_vote import MajorityVoteFilter
from src.labeling.filters.sequence_filter import SequenceFilter
from src.labeling.filters.smoothing import SmoothingFilter


@pytest.fixture
def sample_labels():
    """Создание тестовых меток."""
    # Последовательность с шумом
    labels = pd.Series([1, 1, -1, 1, 1, 1, 0, 1, -1, -1, -1, 0, 1, 1, 0, 0])
    return labels


@pytest.fixture
def sample_data():
    """Создание тестовых данных."""
    dates = pd.date_range("2023-01-01", periods=50, freq="1H")
    np.random.seed(42)

    close_prices = 100 + np.cumsum(np.random.randn(50) * 2.0)

    data = pd.DataFrame(
        {
            "open": close_prices - np.random.rand(50) * 0.5,
            "high": close_prices + np.random.rand(50) * 2.0,
            "low": close_prices - np.random.rand(50) * 2.0,
            "close": close_prices,
            "volume": np.random.randint(1000, 10000, 50),
        },
        index=dates,
    )

    return data


# === SmoothingFilter Tests ===


def test_smoothing_filter_median(sample_labels):
    """Тест медианного сглаживания."""
    filter_obj = SmoothingFilter(method="median", window=3)
    smoothed = filter_obj.apply(sample_labels)

    assert len(smoothed) == len(sample_labels)
    assert smoothed.dtype == int


def test_smoothing_filter_moving_average(sample_labels):
    """Тест сглаживания скользящим средним."""
    filter_obj = SmoothingFilter(method="moving_average", window=3)
    smoothed = filter_obj.apply(sample_labels)

    assert len(smoothed) == len(sample_labels)
    assert smoothed.dtype == int


def test_smoothing_filter_exponential(sample_labels):
    """Тест экспоненциального сглаживания."""
    filter_obj = SmoothingFilter(method="exponential", alpha=0.3)
    smoothed = filter_obj.apply(sample_labels)

    assert len(smoothed) == len(sample_labels)
    assert smoothed.dtype == int


def test_smoothing_filter_invalid_params():
    """Тест валидации параметров сглаживания."""
    with pytest.raises(ValueError):
        SmoothingFilter(window=0)

    with pytest.raises(ValueError):
        SmoothingFilter(method="invalid")


# === SequenceFilter Tests ===


def test_sequence_filter_basic(sample_labels):
    """Тест базовой фильтрации последовательностей."""
    filter_obj = SequenceFilter(min_length=2)
    filtered = filter_obj.apply(sample_labels)

    assert len(filtered) == len(sample_labels)

    # Одиночные метки должны быть заменены на neutral
    # Например, sample_labels[2] = -1 одиночная
    assert filtered.iloc[2] == 0


def test_sequence_filter_min_length_3(sample_labels):
    """Тест с минимальной длиной 3."""
    filter_obj = SequenceFilter(min_length=3)
    filtered = filter_obj.apply(sample_labels)

    # Последовательности длиной < 3 должны быть отфильтрованы
    assert len(filtered) == len(sample_labels)


def test_sequence_filter_no_change():
    """Тест без изменений при min_length=1."""
    labels = pd.Series([1, 1, -1, -1, 0, 0])
    filter_obj = SequenceFilter(min_length=1)
    filtered = filter_obj.apply(labels)

    assert (filtered == labels).all()


# === MajorityVoteFilter Tests ===


def test_majority_vote_filter_basic(sample_labels):
    """Тест мажоритарного голосования."""
    filter_obj = MajorityVoteFilter(window=3, weighted=False)
    filtered = filter_obj.apply(sample_labels)

    assert len(filtered) == len(sample_labels)
    assert filtered.dtype == int


def test_majority_vote_filter_weighted(sample_labels):
    """Тест взвешенного голосования."""
    filter_obj = MajorityVoteFilter(window=5, weighted=True)
    filtered = filter_obj.apply(sample_labels)

    assert len(filtered) == len(sample_labels)


def test_majority_vote_filter_window_1(sample_labels):
    """Тест с окном размера 1."""
    filter_obj = MajorityVoteFilter(window=1)
    filtered = filter_obj.apply(sample_labels)

    # С окном 1 не должно быть изменений
    assert (filtered == sample_labels).all()


# === DangerZonesFilter Tests ===


def test_danger_zones_filter_basic(sample_data):
    """Тест фильтра опасных зон."""
    labels = pd.Series(1, index=sample_data.index)

    filter_obj = DangerZonesFilter(
        high_volatility_threshold=2.0, gap_threshold_pct=0.05
    )

    filtered = filter_obj.apply(labels, sample_data)

    assert len(filtered) == len(labels)

    # Некоторые метки должны быть отфильтрованы
    # (заменены на neutral)


def test_danger_zones_get_mask(sample_data):
    """Тест получения маски опасных зон."""
    filter_obj = DangerZonesFilter(
        high_volatility_threshold=2.0, gap_threshold_pct=0.05
    )

    danger_mask = filter_obj.get_danger_zones(sample_data)

    assert len(danger_mask) == len(sample_data)
    assert danger_mask.dtype == bool


def test_danger_zones_mark_only(sample_data):
    """Тест режима только маркировки."""
    labels = pd.Series(1, index=sample_data.index)

    filter_obj = DangerZonesFilter(high_volatility_threshold=2.0, mark_only=True)

    filtered = filter_obj.apply(labels, sample_data)

    # В режиме mark_only метки не должны измениться
    assert (filtered == labels).all()


def test_danger_zones_with_volume_threshold(sample_data):
    """Тест с порогом ликвидности."""
    labels = pd.Series(1, index=sample_data.index)

    filter_obj = DangerZonesFilter(
        low_liquidity_threshold=5000  # Половина среднего объёма
    )

    filtered = filter_obj.apply(labels, sample_data)

    # Некоторые метки с низким объёмом должны быть отфильтрованы
    assert (filtered == 0).any()
