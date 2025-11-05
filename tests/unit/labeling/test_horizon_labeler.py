"""Тесты для HorizonLabeler."""

import numpy as np
import pandas as pd
import pytest

from src.labeling.methods.horizon import HorizonLabeler


@pytest.fixture
def sample_data():
    """Создание тестовых данных."""
    dates = pd.date_range("2023-01-01", periods=100, freq="1H")
    np.random.seed(42)

    # Генерируем синтетические данные с трендом
    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

    data = pd.DataFrame(
        {
            "open": close_prices - np.random.rand(100) * 0.5,
            "high": close_prices + np.random.rand(100) * 1.0,
            "low": close_prices - np.random.rand(100) * 1.0,
            "close": close_prices,
            "volume": np.random.randint(1000, 10000, 100),
        },
        index=dates,
    )

    return data


def test_horizon_labeler_initialization():
    """Тест инициализации HorizonLabeler."""
    labeler = HorizonLabeler(horizon=20, direction="long+short", threshold_pct=0.01)

    assert labeler.params["horizon"] == 20
    assert labeler.params["direction"] == "long+short"
    assert labeler.params["threshold_pct"] == 0.01


def test_horizon_labeler_invalid_params():
    """Тест валидации параметров."""
    with pytest.raises(ValueError):
        HorizonLabeler(horizon=-10)

    with pytest.raises(ValueError):
        HorizonLabeler(direction="invalid")

    with pytest.raises(ValueError):
        HorizonLabeler(threshold_pct=-0.01)


def test_horizon_labeler_fixed_horizon(sample_data):
    """Тест фиксированного горизонта."""
    labeler = HorizonLabeler(horizon=10, direction="long+short", threshold_pct=0.02)

    result = labeler.label(sample_data)

    # Проверяем наличие нужных колонок
    assert "label" in result.columns
    assert "horizon_used" in result.columns
    assert "future_return" in result.columns

    # Проверяем что все горизонты = 10
    assert (result["horizon_used"] == 10).all()

    # Проверяем типы меток
    assert result["label"].dtype == int
    assert set(result["label"].unique()).issubset({-1, 0, 1})


def test_horizon_labeler_adaptive_horizon(sample_data):
    """Тест адаптивного горизонта."""
    labeler = HorizonLabeler(
        horizon="adaptive",
        adaptive_method="atr",
        min_horizon=5,
        max_horizon=20,
        direction="long+short",
        threshold_pct=0.01,
    )

    result = labeler.label(sample_data)

    # Проверяем наличие колонок
    assert "label" in result.columns
    assert "horizon_used" in result.columns

    # Проверяем что горизонты в допустимом диапазоне
    assert result["horizon_used"].min() >= 5
    assert result["horizon_used"].max() <= 20

    # Проверяем что горизонты адаптивны (не все одинаковые)
    assert len(result["horizon_used"].unique()) > 1


def test_horizon_labeler_long_only(sample_data):
    """Тест long-only режима."""
    labeler = HorizonLabeler(horizon=10, direction="long", threshold_pct=0.01)

    result = labeler.label(sample_data)

    # В long-only не должно быть short меток (-1)
    assert -1 not in result["label"].values
    assert set(result["label"].unique()).issubset({0, 1})


def test_horizon_labeler_short_only(sample_data):
    """Тест short-only режима."""
    labeler = HorizonLabeler(horizon=10, direction="short", threshold_pct=0.01)

    result = labeler.label(sample_data)

    # В short-only метки 1 означают short сигналы
    assert set(result["label"].unique()).issubset({0, 1})


def test_horizon_labeler_empty_data():
    """Тест с пустыми данными."""
    labeler = HorizonLabeler(horizon=10)

    empty_data = pd.DataFrame()

    with pytest.raises(ValueError):
        labeler.label(empty_data)


def test_horizon_labeler_missing_columns():
    """Тест с отсутствующими колонками."""
    labeler = HorizonLabeler(horizon=10)

    dates = pd.date_range("2023-01-01", periods=10, freq="1H")
    data = pd.DataFrame({"close": [100] * 10}, index=dates)

    with pytest.raises(ValueError):
        labeler.label(data)
