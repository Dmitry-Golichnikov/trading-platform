"""
Тесты для SMA индикатора.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.indicators.trend.sma import SMA


def test_sma_known_values():
    """Тест SMA на известных значениях."""
    # Простые данные
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=10, freq="1D"),
            "close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "open": [1] * 10,
            "high": [1] * 10,
            "low": [1] * 10,
            "volume": [1000] * 10,
        }
    )

    sma = SMA(window=3)
    result = sma.calculate(data)

    # SMA(3) для [1,2,3] = 2.0
    assert result["SMA_3"].iloc[2] == pytest.approx(2.0)
    # SMA(3) для [4,5,6] = 5.0
    assert result["SMA_3"].iloc[5] == pytest.approx(5.0)
    # SMA(3) для [8,9,10] = 9.0
    assert result["SMA_3"].iloc[9] == pytest.approx(9.0)

    # Первые 2 значения должны быть NaN
    assert pd.isna(result["SMA_3"].iloc[0])
    assert pd.isna(result["SMA_3"].iloc[1])


def test_sma_params_validation():
    """Тест валидации параметров."""
    with pytest.raises(ValueError):
        SMA(window=0)

    with pytest.raises(ValueError):
        SMA(window=-5)

    with pytest.raises(ValueError):
        SMA(window="invalid")


def test_sma_lookback_period():
    """Тест получения периода разогрева."""
    sma = SMA(window=20)
    assert sma.get_lookback_period() == 20


def test_sma_required_columns():
    """Тест получения необходимых колонок."""
    sma = SMA(window=20, column="close")
    assert sma.get_required_columns() == ["close"]


def test_sma_uptrend(simple_uptrend_data):
    """Тест SMA на восходящем тренде."""
    sma = SMA(window=10)
    result = sma.calculate(simple_uptrend_data)

    # SMA должна быть возрастающей на восходящем тренде
    sma_values = result["SMA_10"].dropna()
    assert all(sma_values.diff().dropna() > 0)


def test_sma_constant_price(constant_price_data):
    """Тест SMA на константной цене."""
    sma = SMA(window=10)
    result = sma.calculate(constant_price_data)

    # SMA должна быть константной
    sma_values = result["SMA_10"].dropna()
    assert np.allclose(sma_values, 100.0)
