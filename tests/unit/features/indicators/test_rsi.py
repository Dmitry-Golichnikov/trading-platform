"""
Тесты для RSI индикатора.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.indicators.momentum.rsi import RSI


def test_rsi_range(sample_ohlcv_data):
    """Тест что RSI в диапазоне [0, 100]."""
    rsi = RSI(window=14)
    result = rsi.calculate(sample_ohlcv_data)

    rsi_values = result["RSI_14"].dropna()

    assert all(rsi_values >= 0)
    assert all(rsi_values <= 100)


def test_rsi_uptrend(simple_uptrend_data):
    """Тест RSI на восходящем тренде."""
    rsi = RSI(window=14)
    result = rsi.calculate(simple_uptrend_data)

    rsi_values = result["RSI_14"].dropna()

    # На сильном восходящем тренде RSI должен быть > 50
    assert rsi_values.mean() > 50


def test_rsi_downtrend(simple_downtrend_data):
    """Тест RSI на нисходящем тренде."""
    rsi = RSI(window=14)
    result = rsi.calculate(simple_downtrend_data)

    rsi_values = result["RSI_14"].dropna()

    # На сильном нисходящем тренде RSI должен быть < 50
    assert rsi_values.mean() < 50


def test_rsi_constant_price(constant_price_data):
    """Тест RSI на константной цене."""
    rsi = RSI(window=14)
    result = rsi.calculate(constant_price_data)

    rsi_values = result["RSI_14"].dropna()

    # При константной цене RSI должен быть около 50 или NaN
    # (деление на ноль даёт NaN)
    assert all(pd.isna(rsi_values) | (np.isclose(rsi_values, 50.0, atol=5.0)))


def test_rsi_params_validation():
    """Тест валидации параметров."""
    with pytest.raises(ValueError):
        RSI(window=0)

    with pytest.raises(ValueError):
        RSI(window=-5)


def test_rsi_lookback_period():
    """Тест периода разогрева."""
    rsi = RSI(window=14)
    assert rsi.get_lookback_period() == 15  # window + 1 для diff
