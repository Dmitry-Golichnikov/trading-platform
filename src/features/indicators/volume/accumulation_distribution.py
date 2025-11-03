"""
Accumulation/Distribution Line - Линия аккумуляции/распределения.
"""

from typing import List

import pandas as pd

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register("accumulation_distribution")
class AccumulationDistribution(Indicator):
    """
    Accumulation/Distribution Line - Линия аккумуляции/распределения.

    Формулы:
        Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
        Money Flow Volume = Money Flow Multiplier * Volume
        A/D = cumsum(Money Flow Volume)

    Параметры:
        Нет параметров

    Выходные колонки:
        AD: Значение A/D линии

    Применение:
        - Подтверждение тренда
        - Дивергенции с ценой (цена растёт, A/D падает - слабость)
        - Показывает накопление (покупки) или распределение (продажи)

    Example:
        >>> ad = AccumulationDistribution()
        >>> result = ad.calculate(data)
        >>> # result содержит колонку 'AD'
    """

    def __init__(self):
        """Инициализация Accumulation/Distribution."""
        super().__init__()

    def get_required_columns(self) -> List[str]:
        """Необходимые колонки."""
        return ["high", "low", "close", "volume"]

    def get_lookback_period(self) -> int:
        """Период разогрева."""
        return 1

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать Accumulation/Distribution Line.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с колонкой AD
        """
        self._validate_data(data)

        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        # Money Flow Multiplier
        mf_multiplier = ((close - low) - (high - close)) / (high - low)

        # Обработка случая когда high == low
        mf_multiplier = mf_multiplier.fillna(0)

        # Money Flow Volume
        mf_volume = mf_multiplier * volume

        # A/D Line (кумулятивная сумма)
        ad = mf_volume.cumsum()

        result = pd.DataFrame({"AD": ad}, index=data.index)

        return result
