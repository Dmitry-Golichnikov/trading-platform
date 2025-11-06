"""
Chaikin Money Flow - Денежный поток Чайкина.
"""

from typing import List

import pandas as pd

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register("chaikin_mf")
class ChaikinMF(Indicator):
    """
    Chaikin Money Flow - Денежный поток Чайкина.

    Формулы:
        Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
        Money Flow Volume = Money Flow Multiplier * Volume

        Chaikin MF = sum(Money Flow Volume, window) / sum(Volume, window)

    Параметры:
        window (int): Период окна (по умолчанию 20)

    Выходные колонки:
        CMF_{window}: Значение Chaikin Money Flow (-1 до +1)

    Применение:
        - CMF > 0 - давление покупок
        - CMF < 0 - давление продаж
        - Подтверждение тренда
        - Дивергенции с ценой

    Example:
        >>> cmf = ChaikinMF(window=20)
        >>> result = cmf.calculate(data)
        >>> # result содержит колонку 'CMF_20'
    """

    def __init__(self, window: int = 20):
        """
        Инициализация Chaikin Money Flow.

        Args:
            window: Период окна
        """
        super().__init__(window=window)

    def validate_params(self) -> None:
        """Валидация параметров."""
        window = self.params.get("window", 20)
        if not isinstance(window, int) or window < 1:
            raise ValueError(f"window должен быть положительным целым числом, получено: {window}")

    def get_required_columns(self) -> List[str]:
        """Необходимые колонки."""
        return ["high", "low", "close", "volume"]

    def get_lookback_period(self) -> int:
        """Период разогрева."""
        return self.params.get("window", 20)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать Chaikin Money Flow.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с колонкой CMF_{window}
        """
        self._validate_data(data)

        window = self.params["window"]

        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        # Money Flow Multiplier
        mf_multiplier = ((close - low) - (high - close)) / (high - low)

        # Обработка случая когда high == low (избегаем деления на ноль)
        mf_multiplier = mf_multiplier.fillna(0)

        # Money Flow Volume
        mf_volume = mf_multiplier * volume

        # Chaikin MF
        mf_volume_sum = mf_volume.rolling(window=window, min_periods=window).sum()
        volume_sum = volume.rolling(window=window, min_periods=window).sum()

        cmf = mf_volume_sum / volume_sum

        col_name = f"CMF_{window}"
        result = pd.DataFrame({col_name: cmf}, index=data.index)

        return result
