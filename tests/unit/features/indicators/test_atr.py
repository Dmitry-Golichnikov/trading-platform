"""
Тесты для ATR индикатора.
"""

import pytest

from src.features.indicators.volatility.atr import ATR


def test_atr_positive_values(sample_ohlcv_data):
    """Тест что ATR всегда положительный."""
    atr = ATR(window=14)
    result = atr.calculate(sample_ohlcv_data)

    atr_values = result["ATR_14"].dropna()

    assert all(atr_values >= 0)


def test_atr_high_volatility():
    """Тест что ATR выше при высокой волатильности."""
    import numpy as np
    import pandas as pd

    np.random.seed(42)

    # Низкая волатильность
    n = 50
    low_vol_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=n, freq="1D"),
            "open": [100] * n,
            "high": [100.5] * n,
            "low": [99.5] * n,
            "close": [100] * n,
            "volume": [1000] * n,
        }
    )

    # Высокая волатильность
    high_vol_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=n, freq="1D"),
            "open": [100] * n,
            "high": [105] * n,
            "low": [95] * n,
            "close": [100] * n,
            "volume": [1000] * n,
        }
    )

    atr = ATR(window=14)

    low_vol_atr = atr.calculate(low_vol_data)["ATR_14"].dropna().mean()
    high_vol_atr = atr.calculate(high_vol_data)["ATR_14"].dropna().mean()

    assert high_vol_atr > low_vol_atr


def test_atr_params_validation():
    """Тест валидации параметров."""
    with pytest.raises(ValueError):
        ATR(window=0)

    with pytest.raises(ValueError):
        ATR(window=-5)


def test_atr_lookback_period():
    """Тест периода разогрева."""
    atr = ATR(window=14)
    assert atr.get_lookback_period() == 15  # window + 1 для diff
