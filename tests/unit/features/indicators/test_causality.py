"""
Тесты каузальности индикаторов.
"""

import pytest

from src.features.indicators.registry import IndicatorRegistry
from src.features.indicators.validation import (
    generate_random_ohlcv,
    validate_causality,
)

# Список всех индикаторов для тестирования
INDICATOR_CONFIGS = [
    ("sma", {"window": 20}),
    ("ema", {"window": 20}),
    ("wma", {"window": 20}),
    ("macd", {"fast": 12, "slow": 26, "signal": 9}),
    ("parabolic_sar", {}),
    ("ichimoku", {"tenkan": 9, "kijun": 26, "senkou": 52}),
    ("rsi", {"window": 14}),
    ("stochastic", {"k": 14, "smooth_k": 3, "smooth_d": 3}),
    ("stochastic_rsi", {"rsi_period": 14, "stoch_period": 14}),
    ("cci", {"window": 20}),
    ("williams_r", {"window": 14}),
    ("atr", {"window": 14}),
    ("bollinger_bands", {"window": 20, "std_dev": 2.0}),
    ("keltner_channels", {"window": 20, "atr_window": 10}),
    ("donchian_channels", {"window": 20}),
    ("obv", {}),
    ("vwap", {"window": 20}),
    ("mfi", {"window": 14}),
    ("chaikin_mf", {"window": 20}),
    ("accumulation_distribution", {}),
    # Volume Profile пропускаем (долгий расчёт)
    ("adx", {"window": 14}),
    ("elder_force", {"window": 13}),
    ("trix", {"window": 15, "signal": 9}),
    ("dpo", {"window": 20}),
    # FDI пропускаем (долгий расчёт)
    # Nadaraya-Watson пропускаем (долгий расчёт)
    ("heikin_ashi", {}),
    ("pivot_points", {"method": "classic"}),
]


@pytest.mark.parametrize("indicator_name,params", INDICATOR_CONFIGS)
def test_indicator_causality(indicator_name, params):
    """Тест каузальности для каждого индикатора."""
    # Генерируем тестовые данные
    test_data = generate_random_ohlcv(n_bars=200)

    # Создаём индикатор
    indicator = IndicatorRegistry.get(indicator_name, **params)

    # Проверяем каузальность
    is_causal = validate_causality(indicator, test_data, n_future_bars=10)

    assert is_causal, f"Индикатор {indicator_name} не является каузальным!"


def test_validate_causality_with_sma():
    """Детальный тест каузальности с SMA."""
    data = generate_random_ohlcv(n_bars=100)
    sma = IndicatorRegistry.get("sma", window=20)

    # Рассчитываем на исходных данных
    result1 = sma.calculate(data)

    # Добавляем будущие данные
    from src.features.indicators.validation import append_future_bars

    extended_data = append_future_bars(data, n=10)
    result2 = sma.calculate(extended_data)

    # Проверяем что старые значения не изменились
    import numpy as np

    overlap = result2.iloc[: len(result1)]
    assert np.allclose(
        result1.values, overlap.values, rtol=1e-5, atol=1e-8, equal_nan=True
    )


def test_generate_random_ohlcv():
    """Тест генерации случайных OHLCV данных."""
    data = generate_random_ohlcv(n_bars=100, start_price=100.0)

    assert len(data) == 100
    assert all(
        col in data.columns
        for col in ["timestamp", "open", "high", "low", "close", "volume"]
    )

    # Проверяем что high >= max(open, close) и low <= min(open, close)
    for i in range(len(data)):
        assert data["high"].iloc[i] >= data["open"].iloc[i]
        assert data["high"].iloc[i] >= data["close"].iloc[i]
        assert data["low"].iloc[i] <= data["open"].iloc[i]
        assert data["low"].iloc[i] <= data["close"].iloc[i]


def test_append_future_bars():
    """Тест добавления будущих баров."""
    from src.features.indicators.validation import append_future_bars

    data = generate_random_ohlcv(n_bars=50)
    extended = append_future_bars(data, n=10)

    assert len(extended) == 60

    # Проверяем что первые 50 баров идентичны
    import numpy as np

    for col in ["open", "high", "low", "close", "volume"]:
        assert np.allclose(data[col].values, extended[col].iloc[:50].values)
