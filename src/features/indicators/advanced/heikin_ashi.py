"""
Heikin-Ashi - Японские свечи Хейкен-Аши.
"""

from typing import List

import pandas as pd

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register("heikin_ashi")
class HeikinAshi(Indicator):
    """
    Heikin-Ashi - Японские свечи Хейкен-Аши.

    Формулы:
        HA_Close = (Open + High + Low + Close) / 4
        HA_Open = (HA_Open_prev + HA_Close_prev) / 2
        HA_High = max(High, HA_Open, HA_Close)
        HA_Low = min(Low, HA_Open, HA_Close)

    Параметры:
        Нет параметров

    Выходные колонки:
        HA_open: Heikin-Ashi open
        HA_high: Heikin-Ashi high
        HA_low: Heikin-Ashi low
        HA_close: Heikin-Ashi close

    Применение:
        - Сглаживает ценовые данные
        - Яснее показывает тренд
        - Зелёные свечи без нижних теней - сильный восходящий тренд
        - Красные свечи без верхних теней - сильный нисходящий тренд

    Example:
        >>> ha = HeikinAshi()
        >>> result = ha.calculate(data)
    """

    def __init__(self):
        """Инициализация Heikin-Ashi."""
        super().__init__()

    def get_required_columns(self) -> List[str]:
        """Необходимые колонки."""
        return ["open", "high", "low", "close"]

    def get_lookback_period(self) -> int:
        """Период разогрева."""
        return 1

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать Heikin-Ashi.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с колонками HA_open, HA_high, HA_low, HA_close
        """
        self._validate_data(data)

        n = len(data)
        ha_open = [0.0] * n
        ha_high = [0.0] * n
        ha_low = [0.0] * n
        ha_close = [0.0] * n

        # Первый бар
        ha_close[0] = (data["open"].iloc[0] + data["high"].iloc[0] + data["low"].iloc[0] + data["close"].iloc[0]) / 4
        ha_open[0] = (data["open"].iloc[0] + data["close"].iloc[0]) / 2
        ha_high[0] = max(data["high"].iloc[0], ha_open[0], ha_close[0])
        ha_low[0] = min(data["low"].iloc[0], ha_open[0], ha_close[0])

        # Остальные бары
        for i in range(1, n):
            # HA Close
            ha_close[i] = (
                data["open"].iloc[i] + data["high"].iloc[i] + data["low"].iloc[i] + data["close"].iloc[i]
            ) / 4

            # HA Open (зависит от предыдущего)
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2

            # HA High и Low
            ha_high[i] = max(data["high"].iloc[i], ha_open[i], ha_close[i])
            ha_low[i] = min(data["low"].iloc[i], ha_open[i], ha_close[i])

        result = pd.DataFrame(
            {
                "HA_open": ha_open,
                "HA_high": ha_high,
                "HA_low": ha_low,
                "HA_close": ha_close,
            },
            index=data.index,
        )

        return result
