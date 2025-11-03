"""
Базовый класс для технических индикаторов.
"""

from abc import ABC, abstractmethod
from typing import Any, List

import pandas as pd


class Indicator(ABC):
    """
    Базовый класс для технических индикаторов.

    Все индикаторы должны быть каузальными (не использовать будущие данные).
    """

    def __init__(self, **params: Any) -> None:
        """
        Инициализация индикатора.

        Args:
            **params: Параметры индикатора
        """
        self.params = params
        self.validate_params()

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать индикатор. Должен быть каузальным!

        Args:
            data: DataFrame с OHLCV данными
                Ожидаемые колонки: timestamp, open, high, low, close, volume

        Returns:
            DataFrame с колонками индикатора
        """
        pass

    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """
        Необходимые колонки для расчёта индикатора.

        Returns:
            Список названий колонок
        """
        pass

    @abstractmethod
    def get_lookback_period(self) -> int:
        """
        Количество баров для warm-up (период разогрева).

        Returns:
            Количество баров
        """
        pass

    def validate_params(self) -> None:
        """
        Валидация параметров индикатора.

        Raises:
            ValueError: Если параметры невалидны
        """
        pass

    @property
    def name(self) -> str:
        """
        Имя индикатора для колонок.

        Returns:
            Название индикатора
        """
        return self.__class__.__name__

    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Проверить что данные содержат необходимые колонки.

        Args:
            data: DataFrame для проверки

        Raises:
            ValueError: Если отсутствуют необходимые колонки
        """
        required = self.get_required_columns()
        missing = set(required) - set(data.columns)
        if missing:
            raise ValueError(
                f"Индикатор {self.name} требует колонки {required}, "
                f"но отсутствуют: {missing}"
            )

    def __repr__(self) -> str:
        """Строковое представление индикатора."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({params_str})"
