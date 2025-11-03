"""
Exponential Moving Average (EMA) - Экспоненциальная скользящая средняя.
"""

from typing import List

import pandas as pd

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register("ema")
class EMA(Indicator):
    """
    Exponential Moving Average (EMA) - Экспоненциальная скользящая средняя.

    Формула:
        EMA_t = α * P_t + (1 - α) * EMA_{t-1}
        где α = 2 / (window + 1) - сглаживающий фактор
        P_t - текущая цена

    Параметры:
        window (int): Период окна (по умолчанию 20)
        column (str): Колонка для расчёта (по умолчанию 'close')
        adjust (bool): Корректировка для начальных значений (по умолчанию True)

    Выходные колонки:
        EMA_{window}: Значение экспоненциальной скользящей средней

    Применение:
        - Более чувствительна к недавним изменениям цены чем SMA
        - Быстрее реагирует на изменения тренда
        - Часто используется в торговых системах

    Example:
        >>> ema = EMA(window=12)
        >>> result = ema.calculate(data)
        >>> # result содержит колонку 'EMA_12'
    """

    def __init__(self, window: int = 20, column: str = "close", adjust: bool = True):
        """
        Инициализация EMA.

        Args:
            window: Период окна
            column: Колонка для расчёта
            adjust: Корректировка для начальных значений
        """
        super().__init__(window=window, column=column, adjust=adjust)

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
        Рассчитать EMA.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с колонкой EMA_{window}
        """
        self._validate_data(data)

        window = self.params["window"]
        column = self.params["column"]
        adjust = self.params.get("adjust", True)

        result = data[[column]].copy()
        col_name = f"EMA_{window}"

        # Каузальный расчёт с помощью pandas.ewm
        # min_periods=window гарантирует что первые window-1 значений будут NaN
        result[col_name] = (
            data[column].ewm(span=window, adjust=adjust, min_periods=window).mean()
        )

        return result[[col_name]]
