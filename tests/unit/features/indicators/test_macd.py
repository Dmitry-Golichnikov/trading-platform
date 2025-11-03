"""
Тесты для MACD индикатора.
"""

import pytest

from src.features.indicators.trend.macd import MACD


def test_macd_output_columns(sample_ohlcv_data):
    """Тест что MACD возвращает правильные колонки."""
    macd = MACD(fast=12, slow=26, signal=9)
    result = macd.calculate(sample_ohlcv_data)

    assert "MACD" in result.columns
    assert "MACD_signal" in result.columns
    assert "MACD_hist" in result.columns


def test_macd_histogram(sample_ohlcv_data):
    """Тест что гистограмма = MACD - сигнальная линия."""
    macd = MACD(fast=12, slow=26, signal=9)
    result = macd.calculate(sample_ohlcv_data)

    # Убираем NaN
    result = result.dropna()

    # Проверяем что histogram = MACD - signal
    expected_hist = result["MACD"] - result["MACD_signal"]

    assert all(abs(result["MACD_hist"] - expected_hist) < 1e-10)


def test_macd_uptrend(simple_uptrend_data):
    """Тест MACD на восходящем тренде."""
    macd = MACD(fast=12, slow=26, signal=9)
    result = macd.calculate(simple_uptrend_data)

    macd_values = result["MACD"].dropna()

    # На восходящем тренде MACD должен быть в основном положительным
    assert macd_values.mean() > 0


def test_macd_params_validation():
    """Тест валидации параметров."""
    # fast >= slow - недопустимо
    with pytest.raises(ValueError):
        MACD(fast=26, slow=12, signal=9)

    with pytest.raises(ValueError):
        MACD(fast=26, slow=26, signal=9)

    with pytest.raises(ValueError):
        MACD(fast=0, slow=26, signal=9)


def test_macd_lookback_period():
    """Тест периода разогрева."""
    macd = MACD(fast=12, slow=26, signal=9)
    # slow + signal
    assert macd.get_lookback_period() == 26 + 9
