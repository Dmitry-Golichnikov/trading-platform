"""
Weighted Moving Average (WMA) - Взвешенная скользящая средняя.
"""

from typing import List

import numpy as np
import pandas as pd

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register("wma")
class WMA(Indicator):
    """
    Weighted Moving Average (WMA) - Взвешенная скользящая средняя.

    Формула:
        WMA = (n*P1 + (n-1)*P2 + ... + 1*Pn) / (n + (n-1) + ... + 1)
        где n - период окна, P - цена

    Более новые цены имеют больший вес.

    Параметры:
        window (int): Период окна (по умолчанию 20)
        column (str): Колонка для расчёта (по умолчанию 'close')

    Выходные колонки:
        WMA_{window}: Значение взвешенной скользящей средней

    Применение:
        - Больший вес для последних цен
        - Быстрее реагирует на изменения чем SMA
        - Медленнее чем EMA

    Example:
        >>> wma = WMA(window=20)
        >>> result = wma.calculate(data)
        >>> # result содержит колонку 'WMA_20'
    """

    def __init__(self, window: int = 20, column: str = "close"):
        """
        Инициализация WMA.

        Args:
            window: Период окна
            column: Колонка для расчёта
        """
        super().__init__(window=window, column=column)

    def validate_params(self) -> None:
        """Валидация параметров."""
        window = self.params.get("window", 20)
        if not isinstance(window, int) or window < 1:
            raise ValueError(f"window должен быть положительным целым числом, получено: {window}")

    def get_required_columns(self) -> List[str]:
        """Необходимые колонки."""
        return [self.params.get("column", "close")]

    def get_lookback_period(self) -> int:
        """Период разогрева."""
        return self.params.get("window", 20)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать WMA.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с колонкой WMA_{window}
        """
        self._validate_data(data)

        window = self.params["window"]
        column = self.params["column"]

        result = data[[column]].copy()
        col_name = f"WMA_{window}"

        # Создаём веса: [1, 2, 3, ..., window]
        weights = np.arange(1, window + 1)

        # Каузальный расчёт с помощью rolling и apply
        result[col_name] = (
            data[column]
            .rolling(window=window, min_periods=window)
            .apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        )

        return result[[col_name]]
