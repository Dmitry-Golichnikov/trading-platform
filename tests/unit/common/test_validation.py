"""Тесты для модуля validation."""

from datetime import datetime

import pandas as pd
import pytest
from pydantic import ValidationError

from src.common.validation import (
    OHLCVData,
    check_look_ahead,
    validate_ohlcv_dataframe,
)


def test_ohlcv_data_valid() -> None:
    """Тест валидных OHLCV данных."""
    data = OHLCVData(
        timestamp=datetime.fromisoformat("2024-01-01 00:00:00"),
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=1000.0,
    )

    assert data.open == 100.0
    assert data.high == 105.0
    assert data.low == 95.0
    assert data.close == 102.0
    assert data.volume == 1000.0


def test_ohlcv_data_invalid_high_low() -> None:
    """Тест невалидных OHLCV данных (high < low)."""
    with pytest.raises(ValidationError):
        OHLCVData(
            timestamp=datetime.fromisoformat("2024-01-01 00:00:00"),
            open=100.0,
            high=90.0,  # меньше low
            low=95.0,
            close=102.0,
            volume=1000.0,
        )


@pytest.mark.parametrize(
    "high,low,open,close,error_message",
    [
        (90.0, 85.0, 95.0, 92.0, "high должен быть >= open"),
        (95.0, 85.0, 90.0, 96.0, "high должен быть >= close"),
        (120.0, 110.0, 100.0, 102.0, "low должен быть <= open"),
        (150.0, 120.0, 130.0, 100.0, "low должен быть <= close"),
    ],
)
def test_ohlcv_data_relationships_invalid(
    high: float,
    low: float,
    open: float,
    close: float,
    error_message: str,
) -> None:
    """Проверяет ошибки взаимосвязи ценовых полей."""

    with pytest.raises(ValidationError, match=error_message):
        OHLCVData(
            timestamp=datetime.fromisoformat("2024-01-01 00:00:00"),
            open=open,
            high=high,
            low=low,
            close=close,
            volume=500.0,
        )


def test_validate_ohlcv_dataframe_valid(sample_ohlcv_data: dict) -> None:
    """Тест валидации корректного DataFrame."""
    df = pd.DataFrame(sample_ohlcv_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Не должно быть исключений
    validate_ohlcv_dataframe(df)


def test_validate_ohlcv_dataframe_missing_timestamp(sample_ohlcv_data: dict) -> None:
    """Проверяет что пустые значения timestamp вызывают ошибку."""

    df = pd.DataFrame(sample_ohlcv_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.loc[0, "timestamp"] = pd.NaT

    with pytest.raises(
        ValueError,
        match="Колонка timestamp содержит пустые значения",
    ):
        validate_ohlcv_dataframe(df)


def test_validate_ohlcv_dataframe_timestamp_wrong_dtype(
    sample_ohlcv_data: dict,
) -> None:
    """Проверяет что строковый timestamp вызывает ошибку."""

    df = pd.DataFrame(sample_ohlcv_data)

    with pytest.raises(
        ValueError,
        match="timestamp должна иметь тип datetime",
    ):
        validate_ohlcv_dataframe(df)


@pytest.mark.parametrize("column", ["open", "high", "low", "close"])
def test_validate_ohlcv_dataframe_missing_price_values(
    sample_ohlcv_data: dict, column: str
) -> None:
    """Проверяет что пустые значения ценовых колонок вызывают ошибку."""

    df = pd.DataFrame(sample_ohlcv_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.loc[0, column] = None

    with pytest.raises(
        ValueError,
        match=f"{column} содержит пустые значения",
    ):
        validate_ohlcv_dataframe(df)


def test_validate_ohlcv_dataframe_missing_volume(sample_ohlcv_data: dict) -> None:
    """Проверяет что пустые значения объема вызывают ошибку."""

    df = pd.DataFrame(sample_ohlcv_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.loc[0, "volume"] = None

    with pytest.raises(
        ValueError,
        match="Колонка volume содержит пустые значения",
    ):
        validate_ohlcv_dataframe(df)

    if (df["volume"] < 0).any():
        raise ValueError("Колонка volume содержит отрицательные значения")

    # Проверка соотношений цен
    if (df["high"] < df["low"]).any():
        raise ValueError("high не может быть меньше low")


def test_validate_ohlcv_dataframe_missing_columns() -> None:
    """Тест валидации DataFrame с недостающими колонками."""
    df = pd.DataFrame({"timestamp": ["2024-01-01"], "open": [100.0]})
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    with pytest.raises(ValueError, match="Отсутствуют обязательные колонки"):
        validate_ohlcv_dataframe(df)


def test_validate_ohlcv_dataframe_negative_values() -> None:
    """Тест валидации DataFrame с отрицательными значениями."""
    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01"],
            "open": [-100.0],
            "high": [105.0],
            "low": [95.0],
            "close": [102.0],
            "volume": [1000.0],
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    with pytest.raises(ValueError, match="неположительные значения"):
        validate_ohlcv_dataframe(df)


def test_validate_ohlcv_dataframe_negative_volume() -> None:
    """Проверяет что отрицательный объём вызывает ошибку."""

    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01"],
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [105.0],
            "volume": [-1.0],
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    with pytest.raises(
        ValueError,
        match="volume содержит отрицательные значения",
    ):
        validate_ohlcv_dataframe(df)


def test_validate_ohlcv_dataframe_high_low_violation() -> None:
    """Тест валидации DataFrame с нарушением high < low."""
    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01"],
            "open": [100.0],
            "high": [90.0],  # меньше low
            "low": [95.0],
            "close": [102.0],
            "volume": [1000.0],
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    with pytest.raises(ValueError, match="high не может быть меньше low"):
        validate_ohlcv_dataframe(df)


def test_validate_ohlcv_dataframe_high_less_than_close() -> None:
    """Проверяет ошибку high < close."""

    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01"],
            "open": [110.0],
            "high": [112.0],
            "low": [100.0],
            "close": [115.0],
            "volume": [1000.0],
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    with pytest.raises(ValueError, match="high не может быть меньше close"):
        validate_ohlcv_dataframe(df)


@pytest.mark.parametrize(
    "column,values,regex",
    [
        (
            "high",
            {
                "open": [105.0],
                "high": [90.0],
                "low": [80.0],
                "close": [95.0],
            },
            "high не может быть меньше open",
        ),
        (
            "high",
            {
                "open": [95.0],
                "high": [94.0],
                "low": [80.0],
                "close": [100.0],
            },
            "high не может быть меньше open",
        ),
        (
            "low",
            {
                "open": [120.0],
                "high": [130.0],
                "low": [110.0],
                "close": [100.0],
            },
            "low не может быть больше close",
        ),
        (
            "low",
            {
                "open": [85.0],
                "high": [110.0],
                "low": [90.0],
                "close": [88.0],
            },
            "low не может быть больше open",
        ),
    ],
)
def test_validate_ohlcv_dataframe_relationship_violations(
    column: str, values: dict, regex: str
) -> None:
    """Проверяет ошибки соотношений цен в DataFrame."""

    base = {
        "timestamp": ["2024-01-01"],
        "volume": [1000.0],
    }
    base.update(values)
    df = pd.DataFrame(base)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    with pytest.raises(ValueError, match=regex):
        validate_ohlcv_dataframe(df)


def test_validate_ohlcv_dataframe_low_greater_than_close() -> None:
    """Проверяет ошибку low > close."""

    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01"],
            "open": [65.0],
            "high": [80.0],
            "low": [60.0],
            "close": [55.0],
            "volume": [1000.0],
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    with pytest.raises(ValueError, match="low не может быть больше close"):
        validate_ohlcv_dataframe(df)


def test_check_look_ahead_returns_true(sample_ohlcv_data: dict) -> None:
    """Проверяет корректное отсутствие look-ahead bias."""

    df = pd.DataFrame(sample_ohlcv_data)
    assert check_look_ahead(df, feature_cols=["feat1", "feat2"], target_col="target")


def test_check_look_ahead_detects_bias(sample_ohlcv_data: dict) -> None:
    """Проверяет что look-ahead bias обнаруживается."""

    df = pd.DataFrame(sample_ohlcv_data)
    assert not check_look_ahead(
        df, feature_cols=["target", "feat"], target_col="target"
    )
