"""
Тесты для реестра индикаторов.
"""

import pytest

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


def test_registry_register_decorator():
    """Тест регистрации индикатора через декоратор."""

    @IndicatorRegistry.register("test_indicator")
    class TestIndicator(Indicator):
        def calculate(self, data):
            return data[["close"]].copy()

        def get_required_columns(self):
            return ["close"]

        def get_lookback_period(self):
            return 1

    assert IndicatorRegistry.is_registered("test_indicator")
    assert "test_indicator" in IndicatorRegistry.list_all()


def test_registry_get_indicator():
    """Тест получения индикатора из реестра."""
    indicator = IndicatorRegistry.get("sma", window=20)

    assert indicator is not None
    assert indicator.params["window"] == 20


def test_registry_get_unknown_indicator():
    """Тест получения несуществующего индикатора."""
    with pytest.raises(ValueError, match="Неизвестный индикатор"):
        IndicatorRegistry.get("unknown_indicator_xyz")


def test_registry_list_all():
    """Тест получения списка всех индикаторов."""
    all_indicators = IndicatorRegistry.list_all()

    # Проверяем что основные индикаторы зарегистрированы
    assert "sma" in all_indicators
    assert "ema" in all_indicators
    assert "rsi" in all_indicators
    assert "macd" in all_indicators
    assert "atr" in all_indicators


def test_registry_case_insensitive() -> None:
    """Тест что регистр не важен."""
    indicator1 = IndicatorRegistry.get("SMA", window=20)
    indicator2 = IndicatorRegistry.get("sma", window=20)

    assert isinstance(indicator1, type(indicator2))
