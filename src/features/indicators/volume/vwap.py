"""
VWAP (Volume Weighted Average Price) - Средневзвешенная по объёму цена.
"""

from typing import List

import pandas as pd

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register("vwap")
class VWAP(Indicator):
    """
    VWAP (Volume Weighted Average Price) - Средневзвешенная по объёму цена.

    Формула:
        Typical Price = (High + Low + Close) / 3
        VWAP = sum(Typical Price * Volume) / sum(Volume)

    Параметры:
        window (int): Период окна (по умолчанию None - накопительный VWAP)
                      Если указан - rolling VWAP за период

    Выходные колонки:
        VWAP: Значение VWAP

    Применение:
        - Определение справедливой цены
        - Цена выше VWAP - восходящий тренд
        - Цена ниже VWAP - нисходящий тренд
        - Уровень поддержки/сопротивления

    Example:
        >>> vwap = VWAP()  # Накопительный VWAP
        >>> result = vwap.calculate(data)
        >>>
        >>> vwap_rolling = VWAP(window=20)  # Rolling VWAP
        >>> result = vwap_rolling.calculate(data)
    """

    def __init__(self, window: int | None = None):
        """
        Инициализация VWAP.

        Args:
            window: Период окна (None для накопительного VWAP)
        """
        super().__init__(window=window)

    def validate_params(self) -> None:
        """Валидация параметров."""
        window = self.params.get("window")
        if window is not None and (not isinstance(window, int) or window < 1):
            raise ValueError(
                f"window должен быть положительным целым числом или "
                f"None, получено: {window}"
            )

    def get_required_columns(self) -> List[str]:
        """Необходимые колонки."""
        return ["high", "low", "close", "volume"]

    def get_lookback_period(self) -> int:
        """Период разогрева."""
        window = self.params.get("window")
        return window if window is not None else 1

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать VWAP.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с колонкой VWAP
        """
        self._validate_data(data)

        window = self.params.get("window")

        # Typical Price
        tp = (data["high"] + data["low"] + data["close"]) / 3

        # TP * Volume
        tp_volume = tp * data["volume"]

        if window is None:
            # Накопительный VWAP
            vwap = tp_volume.cumsum() / data["volume"].cumsum()
        else:
            # Rolling VWAP
            tp_volume_sum = tp_volume.rolling(window=window, min_periods=window).sum()
            volume_sum = data["volume"].rolling(window=window, min_periods=window).sum()
            vwap = tp_volume_sum / volume_sum

        result = pd.DataFrame({"VWAP": vwap}, index=data.index)

        return result
