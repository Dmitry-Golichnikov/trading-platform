"""
Тесты для Bollinger Bands индикатора.
"""

import pytest

from src.features.indicators.volatility.bollinger_bands import BollingerBands


def test_bollinger_bands_output_columns(sample_ohlcv_data):
    """Тест что Bollinger Bands возвращает правильные колонки."""
    bb = BollingerBands(window=20, std_dev=2.0)
    result = bb.calculate(sample_ohlcv_data)

    assert "BB_upper" in result.columns
    assert "BB_middle" in result.columns
    assert "BB_lower" in result.columns
    assert "BB_width" in result.columns
    assert "BB_pct" in result.columns


def test_bollinger_bands_order(sample_ohlcv_data):
    """Тест что upper > middle > lower."""
    bb = BollingerBands(window=20, std_dev=2.0)
    result = bb.calculate(sample_ohlcv_data)

    result = result.dropna()

    assert all(result["BB_upper"] >= result["BB_middle"])
    assert all(result["BB_middle"] >= result["BB_lower"])


def test_bollinger_bands_width(sample_ohlcv_data):
    """Тест что ширина = upper - lower."""
    bb = BollingerBands(window=20, std_dev=2.0)
    result = bb.calculate(sample_ohlcv_data)

    result = result.dropna()

    expected_width = result["BB_upper"] - result["BB_lower"]

    assert all(abs(result["BB_width"] - expected_width) < 1e-10)


def test_bollinger_bands_percent_range(sample_ohlcv_data):
    """Тест что %B обычно в диапазоне [0, 1]."""
    bb = BollingerBands(window=20, std_dev=2.0)
    result = bb.calculate(sample_ohlcv_data)

    pct_values = result["BB_pct"].dropna()

    # Большинство значений должны быть в [0, 1]
    # (могут быть выбросы за пределы)
    values_in_range = ((pct_values >= 0) & (pct_values <= 1)).sum()
    assert values_in_range / len(pct_values) > 0.7


def test_bollinger_bands_params_validation():
    """Тест валидации параметров."""
    with pytest.raises(ValueError):
        BollingerBands(window=0)

    with pytest.raises(ValueError):
        BollingerBands(std_dev=-2.0)


def test_bollinger_bands_lookback_period():
    """Тест периода разогрева."""
    bb = BollingerBands(window=20)
    assert bb.get_lookback_period() == 20
