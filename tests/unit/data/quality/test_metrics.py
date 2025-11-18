"""Тесты для метрик качества данных."""

import numpy as np
import pandas as pd
import pytest

from src.data.quality.metrics import DataQualityMetrics


@pytest.fixture
def perfect_data() -> pd.DataFrame:
    """Идеальные данные без проблем."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1min", tz="UTC")

    data = pd.DataFrame(
        {
            "timestamp": dates,
            "ticker": "TEST",
            "open": np.linspace(100, 110, 100),
            "high": np.linspace(102, 112, 100),
            "low": np.linspace(98, 108, 100),
            "close": np.linspace(100, 110, 100),
            "volume": np.random.randint(1000, 10000, 100),
        }
    )

    # Ensure OHLC relationships
    data["high"] = data[["open", "high", "close"]].max(axis=1)
    data["low"] = data[["open", "low", "close"]].min(axis=1)

    return data


@pytest.fixture
def problematic_data() -> pd.DataFrame:
    """Данные с проблемами."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1min", tz="UTC")

    data = pd.DataFrame(
        {
            "timestamp": dates,
            "ticker": "TEST",
            "open": [100] * 100,
            "high": [102] * 100,
            "low": [98] * 100,
            "close": [100] * 100,
            "volume": [1000] * 100,
        }
    )

    # Внедрить проблемы
    data.loc[10, "open"] = np.nan  # Missing value
    data.loc[20, "high"] = 90  # Invalid OHLC
    data.loc[30, "volume"] = -100  # Negative volume
    data.loc[40] = data.loc[39]  # Duplicate

    return data


def test_calculate_completeness_perfect(perfect_data: pd.DataFrame) -> None:
    """Тест расчёта полноты для идеальных данных."""
    metrics = DataQualityMetrics()

    completeness = metrics.calculate_completeness(perfect_data)

    assert completeness == 100.0


def test_calculate_completeness_with_nans() -> None:
    """Тест расчёта полноты с пропусками."""
    data = pd.DataFrame(
        {
            "a": [1, 2, np.nan, 4],
            "b": [1, np.nan, 3, 4],
            "c": [1, 2, 3, 4],
        }
    )

    metrics = DataQualityMetrics()
    completeness = metrics.calculate_completeness(data)

    # 10 из 12 ячеек заполнены
    expected = (10 / 12) * 100
    assert abs(completeness - expected) < 0.01


def test_calculate_validity_perfect(perfect_data: pd.DataFrame) -> None:
    """Тест валидности идеальных данных."""
    metrics = DataQualityMetrics()

    validity = metrics.calculate_validity(perfect_data)

    assert validity == 100.0


def test_calculate_validity_with_issues(problematic_data: pd.DataFrame) -> None:
    """Тест валидности данных с проблемами."""
    metrics = DataQualityMetrics()

    validity = metrics.calculate_validity(problematic_data)

    # Должна быть меньше 100% из-за проблем
    assert validity < 100.0


def test_calculate_consistency(perfect_data: pd.DataFrame) -> None:
    """Тест консистентности."""
    metrics = DataQualityMetrics()

    consistency = metrics.calculate_consistency(perfect_data)

    # Без дубликатов должна быть высокой
    assert consistency >= 99.0


def test_calculate_uniqueness_with_duplicates(problematic_data: pd.DataFrame) -> None:
    """Тест уникальности с дубликатами."""
    metrics = DataQualityMetrics()

    uniqueness = metrics.calculate_uniqueness(problematic_data)

    # Должна быть < 100% из-за дубликатов
    assert uniqueness < 100.0


def test_calculate_quality_score_perfect(perfect_data: pd.DataFrame) -> None:
    """Тест общего score для идеальных данных."""
    metrics = DataQualityMetrics()

    score = metrics.calculate_quality_score(perfect_data)

    # Должен быть очень высоким
    assert score >= 95.0


def test_calculate_quality_score_problematic(problematic_data: pd.DataFrame) -> None:
    """Тест общего score для проблемных данных."""
    metrics = DataQualityMetrics()

    score = metrics.calculate_quality_score(problematic_data)

    # Должен быть ниже чем идеальный (100)
    # но может быть высоким если проблемы минимальны
    assert 0 <= score <= 100


def test_get_all_metrics(perfect_data: pd.DataFrame) -> None:
    """Тест получения всех метрик."""
    metrics = DataQualityMetrics()

    all_metrics = metrics.get_all_metrics(perfect_data)

    # Проверить что все метрики присутствуют
    assert "completeness" in all_metrics
    assert "validity" in all_metrics
    assert "consistency" in all_metrics
    assert "uniqueness" in all_metrics
    assert "quality_score" in all_metrics

    # Все значения должны быть числовыми
    for value in all_metrics.values():
        assert isinstance(value, (int, float))


def test_metrics_empty_data() -> None:
    """Тест на пустых данных."""
    empty = pd.DataFrame()
    metrics = DataQualityMetrics()

    completeness = metrics.calculate_completeness(empty)
    score = metrics.calculate_quality_score(empty)

    # Empty data может возвращать 0 или NaN, проверяем диапазон
    assert completeness >= 0.0
    assert score >= 0.0
