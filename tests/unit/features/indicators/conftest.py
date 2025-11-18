"""
Fixtures для тестирования индикаторов.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv_data():
    """Генерировать простые OHLCV данные для тестирования."""
    np.random.seed(42)
    n = 100

    timestamps = pd.date_range(start="2023-01-01", periods=n, freq="1D")

    # Простые данные с небольшой волатильностью
    close_prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

    data = []
    for i, close in enumerate(close_prices):
        high = close * (1 + abs(np.random.randn() * 0.01))
        low = close * (1 - abs(np.random.randn() * 0.01))

        if i == 0:
            open_price = 100
        else:
            open_price = close_prices[i - 1]

        high = max(high, open_price, close)
        low = min(low, open_price, close)

        volume = np.random.uniform(100000, 1000000)

        data.append(
            {
                "timestamp": timestamps[i],
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    return pd.DataFrame(data)


@pytest.fixture
def simple_uptrend_data():
    """Простой восходящий тренд для тестирования."""
    n = 50
    timestamps = pd.date_range(start="2023-01-01", periods=n, freq="1D")

    # Линейный восходящий тренд
    close_prices = np.linspace(100, 150, n)

    data = []
    for i, close in enumerate(close_prices):
        high = close * 1.01
        low = close * 0.99

        if i == 0:
            open_price = 100
        else:
            open_price = close_prices[i - 1]

        volume = 1000000

        data.append(
            {
                "timestamp": timestamps[i],
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    return pd.DataFrame(data)


@pytest.fixture
def simple_downtrend_data():
    """Простой нисходящий тренд для тестирования."""
    n = 50
    timestamps = pd.date_range(start="2023-01-01", periods=n, freq="1D")

    # Линейный нисходящий тренд
    close_prices = np.linspace(150, 100, n)

    data = []
    for i, close in enumerate(close_prices):
        high = close * 1.01
        low = close * 0.99

        if i == 0:
            open_price = 150
        else:
            open_price = close_prices[i - 1]

        volume = 1000000

        data.append(
            {
                "timestamp": timestamps[i],
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    return pd.DataFrame(data)


@pytest.fixture
def constant_price_data():
    """Константная цена для граничного тестирования."""
    n = 50
    timestamps = pd.date_range(start="2023-01-01", periods=n, freq="1D")

    data = []
    for i in range(n):
        data.append(
            {
                "timestamp": timestamps[i],
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 1000000,
            }
        )

    return pd.DataFrame(data)
