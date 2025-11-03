"""
Williams %R - Индикатор Уильямса.
"""

from typing import List

import pandas as pd

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register("williams_r")
class WilliamsR(Indicator):
    """
    Williams %R - Индикатор Уильямса.

    Формула:
        %R = -100 * (High_n - Close) / (High_n - Low_n)

    где High_n и Low_n - максимум и минимум за период.

    Параметры:
        window (int): Период окна (по умолчанию 14)

    Выходные колонки:
        Williams_R_{window}: Значение Williams %R (от -100 до 0)

    Применение:
        - %R > -20 - перекупленность
        - %R < -80 - перепроданность
        - Обратная версия Stochastic Oscillator

    Example:
        >>> williams_r = WilliamsR(window=14)
        >>> result = williams_r.calculate(data)
        >>> # result содержит колонку 'Williams_R_14'
    """

    def __init__(self, window: int = 14):
        """
        Инициализация Williams %R.

        Args:
            window: Период окна
        """
        super().__init__(window=window)

    def validate_params(self) -> None:
        """Валидация параметров."""
        window = self.params.get("window", 14)
        if not isinstance(window, int) or window < 1:
            raise ValueError(
                f"window должен быть положительным целым числом, получено: {window}"
            )

    def get_required_columns(self) -> List[str]:
        """Необходимые колонки."""
        return ["high", "low", "close"]

    def get_lookback_period(self) -> int:
        """Период разогрева."""
        return self.params.get("window", 14)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать Williams %R.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с колонкой Williams_R_{window}
        """
        self._validate_data(data)

        window = self.params["window"]

        # Рассчитываем max и min за период
        high_max = data["high"].rolling(window=window, min_periods=window).max()
        low_min = data["low"].rolling(window=window, min_periods=window).min()

        # Williams %R
        williams_r = -100 * (high_max - data["close"]) / (high_max - low_min)

        col_name = f"Williams_R_{window}"
        result = pd.DataFrame({col_name: williams_r}, index=data.index)

        return result
