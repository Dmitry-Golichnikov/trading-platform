"""
OBV (On Balance Volume) - Балансовый объём.
"""

from typing import List

import numpy as np
import pandas as pd

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register("obv")
class OBV(Indicator):
    """
    OBV (On Balance Volume) - Балансовый объём.

    Формула:
        Если Close_t > Close_{t-1}: OBV_t = OBV_{t-1} + Volume_t
        Если Close_t < Close_{t-1}: OBV_t = OBV_{t-1} - Volume_t
        Если Close_t == Close_{t-1}: OBV_t = OBV_{t-1}

    Параметры:
        Нет параметров

    Выходные колонки:
        OBV: Значение балансового объёма

    Применение:
        - Подтверждение тренда (OBV растёт вместе с ценой)
        - Дивергенции (цена растёт, OBV падает - слабость тренда)
        - Измерение давления покупок/продаж

    Example:
        >>> obv = OBV()
        >>> result = obv.calculate(data)
        >>> # result содержит колонку 'OBV'
    """

    def __init__(self):
        """Инициализация OBV."""
        super().__init__()

    def get_required_columns(self) -> List[str]:
        """Необходимые колонки."""
        return ["close", "volume"]

    def get_lookback_period(self) -> int:
        """Период разогрева."""
        return 1  # Нужен только 1 бар для начала

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать OBV.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с колонкой OBV
        """
        self._validate_data(data)

        close = data["close"]
        volume = data["volume"]

        # Определяем направление (1, 0, -1)
        direction = np.sign(close.diff())

        # OBV = кумулятивная сумма (direction * volume)
        obv = (direction * volume).cumsum()

        result = pd.DataFrame({"OBV": obv}, index=data.index)

        return result
