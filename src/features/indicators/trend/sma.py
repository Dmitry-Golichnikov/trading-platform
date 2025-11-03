"""
Simple Moving Average (SMA) - Простая скользящая средняя.
"""

from typing import List

import pandas as pd

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register("sma")
class SMA(Indicator):
    """
    Simple Moving Average (SMA) - Простая скользящая средняя.

    Формула: SMA = (P1 + P2 + ... + Pn) / n
    где P - цена, n - период окна

    Параметры:
        window (int): Период окна (по умолчанию 20)
        column (str): Колонка для расчёта (по умолчанию 'close')

    Выходные колонки:
        SMA_{window}: Значение скользящей средней

    Применение:
        - Определение тренда (цена выше SMA - восходящий тренд)
        - Уровни поддержки/сопротивления
        - Генерация торговых сигналов (пересечение цены с SMA)

    Example:
        >>> sma = SMA(window=20)
        >>> result = sma.calculate(data)
        >>> # result содержит колонку 'SMA_20'
    """

    def __init__(self, window: int = 20, column: str = "close"):
        """
        Инициализация SMA.

        Args:
            window: Период окна
            column: Колонка для расчёта
        """
        super().__init__(window=window, column=column)

    def validate_params(self) -> None:
        """Валидация параметров."""
        window = self.params.get("window", 20)
        if not isinstance(window, int) or window < 1:
            raise ValueError(
                f"window должен быть положительным целым числом, получено: {window}"
            )

    def get_required_columns(self) -> List[str]:
        """Необходимые колонки."""
        return [self.params.get("column", "close")]

    def get_lookback_period(self) -> int:
        """Период разогрева."""
        return self.params.get("window", 20)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать SMA.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с колонкой SMA_{window}
        """
        self._validate_data(data)

        window = self.params["window"]
        column = self.params["column"]

        result = data[[column]].copy()
        col_name = f"SMA_{window}"

        # Каузальный расчёт: rolling с min_periods=window
        result[col_name] = (
            data[column].rolling(window=window, min_periods=window).mean()
        )

        return result[[col_name]]
