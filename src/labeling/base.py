"""Базовый класс для методов разметки таргетов."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd


class BaseLabeler(ABC):
    """Абстрактный базовый класс для всех методов разметки."""

    def __init__(self, **kwargs):
        """
        Инициализация базового лабелера.

        Args:
            **kwargs: Параметры метода разметки
        """
        self.params = kwargs
        self._validate_params()

    @abstractmethod
    def _validate_params(self) -> None:
        """Валидация параметров метода."""
        pass

    @abstractmethod
    def label(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Основной метод разметки данных.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с добавленными колонками разметки

        Raises:
            ValueError: Если данные не соответствуют требованиям
        """
        pass

    def validate_data(self, data: pd.DataFrame) -> None:
        """
        Проверка входных данных.

        Args:
            data: DataFrame для проверки

        Raises:
            ValueError: Если данные не валидны
        """
        required_columns = ["open", "high", "low", "close"]
        missing = [col for col in required_columns if col not in data.columns]

        if missing:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing}")

        if data.empty:
            raise ValueError("DataFrame не должен быть пустым")

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Индекс должен быть DatetimeIndex")

        # Проверка на NaN в ценовых данных
        price_cols = ["open", "high", "low", "close"]
        if data[price_cols].isna().any().any():
            raise ValueError("Ценовые данные содержат NaN значения")

    def get_params(self) -> Dict[str, Any]:
        """
        Получить параметры лабелера.

        Returns:
            Словарь с параметрами
        """
        return self.params.copy()

    def __repr__(self) -> str:
        """Строковое представление."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})"
