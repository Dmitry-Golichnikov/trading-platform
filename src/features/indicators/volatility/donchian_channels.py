"""
Donchian Channels - Каналы Дончиана.
"""

from typing import List

import pandas as pd

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register("donchian_channels")
class DonchianChannels(Indicator):
    """
    Donchian Channels - Каналы Дончиана.

    Формулы:
        Upper Channel = max(High, window)
        Lower Channel = min(Low, window)
        Middle Channel = (Upper + Lower) / 2

    Параметры:
        window (int): Период окна (по умолчанию 20)

    Выходные колонки:
        DC_upper: Верхний канал (максимум за период)
        DC_middle: Средний канал
        DC_lower: Нижний канал (минимум за период)

    Применение:
        - Прорыв верхнего канала - сигнал к покупке
        - Прорыв нижнего канала - сигнал к продаже
        - Используется в трендовых стратегиях прорыва

    Example:
        >>> dc = DonchianChannels(window=20)
        >>> result = dc.calculate(data)
    """

    def __init__(self, window: int = 20):
        """
        Инициализация Donchian Channels.

        Args:
            window: Период окна
        """
        super().__init__(window=window)

    def validate_params(self) -> None:
        """Валидация параметров."""
        window = self.params.get("window", 20)
        if not isinstance(window, int) or window < 1:
            raise ValueError(
                f"window должен быть положительным целым числом, получено: {window}"
            )

    def get_required_columns(self) -> List[str]:
        """Необходимые колонки."""
        return ["high", "low"]

    def get_lookback_period(self) -> int:
        """Период разогрева."""
        return self.params.get("window", 20)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать Donchian Channels.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с колонками DC_upper, DC_middle, DC_lower
        """
        self._validate_data(data)

        window = self.params["window"]

        # Upper Channel (максимум High за период)
        upper = data["high"].rolling(window=window, min_periods=window).max()

        # Lower Channel (минимум Low за период)
        lower = data["low"].rolling(window=window, min_periods=window).min()

        # Middle Channel
        middle = (upper + lower) / 2

        result = pd.DataFrame(
            {"DC_upper": upper, "DC_middle": middle, "DC_lower": lower},
            index=data.index,
        )

        return result
