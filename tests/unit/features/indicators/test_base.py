"""
Тесты для базового класса Indicator.
"""

import pytest

from src.features.indicators.base import Indicator


class DummyIndicator(Indicator):
    """Dummy индикатор для тестирования."""

    def calculate(self, data):
        return data[["close"]].copy()

    def get_required_columns(self):
        return ["close"]

    def get_lookback_period(self):
        return 1


def test_indicator_initialization():
    """Тест инициализации индикатора."""
    indicator = DummyIndicator(window=20, column="close")

    assert indicator.params["window"] == 20
    assert indicator.params["column"] == "close"


def test_indicator_name():
    """Тест получения имени индикатора."""
    indicator = DummyIndicator()

    assert indicator.name == "DummyIndicator"


def test_indicator_repr():
    """Тест строкового представления."""
    indicator = DummyIndicator(window=20, column="close")

    repr_str = repr(indicator)
    assert "DummyIndicator" in repr_str
    assert "window=20" in repr_str
    assert "column=close" in repr_str


def test_indicator_validate_data_success(sample_ohlcv_data):
    """Тест успешной валидации данных."""
    indicator = DummyIndicator()

    # Не должно вызывать исключение
    indicator._validate_data(sample_ohlcv_data)


def test_indicator_validate_data_missing_columns(sample_ohlcv_data):
    """Тест валидации с отсутствующими колонками."""
    indicator = DummyIndicator()

    # Удаляем необходимую колонку
    data = sample_ohlcv_data.drop(columns=["close"])

    with pytest.raises(ValueError, match="требует колонки"):
        indicator._validate_data(data)
