"""Базовые валидаторы данных."""

from datetime import datetime
from typing import List

import pandas as pd
from pydantic import BaseModel, Field, model_validator


class OHLCVData(BaseModel):
    """Схема для OHLCV данных."""

    timestamp: datetime
    open: float = Field(gt=0, description="Цена открытия")
    high: float = Field(gt=0, description="Максимальная цена")
    low: float = Field(gt=0, description="Минимальная цена")
    close: float = Field(gt=0, description="Цена закрытия")
    volume: float = Field(ge=0, description="Объем торгов")

    @model_validator(mode="after")
    def validate_price_relationships(self) -> "OHLCVData":
        """Проверяет взаимосвязь ценовых полей."""
        values = self
        if values.high < values.low:
            raise ValueError("high не может быть меньше low")
        if values.high < values.open:
            raise ValueError("high должен быть >= open")
        if values.high < values.close:
            raise ValueError("high должен быть >= close")
        if values.low > values.open:
            raise ValueError("low должен быть <= open")
        if values.low > values.close:
            raise ValueError("low должен быть <= close")
        return values


def validate_ohlcv_dataframe(df: pd.DataFrame) -> None:
    """
    Валидирует DataFrame с OHLCV данными.

    Args:
        df: DataFrame с колонками timestamp, open, high, low, close, volume

    Raises:
        ValueError: Если данные невалидны
    """
    required_columns = ["timestamp", "open", "high", "low", "close", "volume"]

    # Проверка наличия обязательных колонок
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing_columns}")

    # Проверка типов данных
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise ValueError("Колонка timestamp должна иметь тип datetime")

    if df["timestamp"].isnull().any():
        raise ValueError("Колонка timestamp содержит пустые значения")

    # Проверка на отрицательные значения
    for col in ["open", "high", "low", "close"]:
        if df[col].isnull().any():
            raise ValueError(f"Колонка {col} содержит пустые значения")
        if (df[col] <= 0).any():
            raise ValueError(f"Колонка {col} содержит неположительные значения")

    if df["volume"].isnull().any():
        raise ValueError("Колонка volume содержит пустые значения")
    if (df["volume"] < 0).any():
        raise ValueError("Колонка volume содержит отрицательные значения")

    # Проверка соотношений цен
    if (df["high"] < df["low"]).any():
        raise ValueError("high не может быть меньше low")

    if (df["high"] < df["open"]).any():
        raise ValueError("high не может быть меньше open")

    if (df["high"] < df["close"]).any():
        raise ValueError("high не может быть меньше close")

    if (df["low"] > df["open"]).any():
        raise ValueError("low не может быть больше open")

    if (df["low"] > df["close"]).any():
        raise ValueError("low не может быть больше close")


def check_look_ahead(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> bool:
    """
    Проверяет наличие look-ahead bias.

    Args:
        df: DataFrame с данными
        feature_cols: Список колонок с признаками
        target_col: Колонка с таргетом

    Returns:
        True если look-ahead отсутствует, False в противном случае
    """
    # TODO: Реализовать более сложную проверку
    # Пока просто проверяем что таргет не входит в признаки
    return target_col not in feature_cols
