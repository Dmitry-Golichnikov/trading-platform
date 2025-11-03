"""
ATR (Average True Range) - Средний истинный диапазон.
"""

from typing import List

import pandas as pd

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register("atr")
class ATR(Indicator):
    """
    ATR (Average True Range) - Средний истинный диапазон.

    Формула:
        True Range = max(High - Low, |High - Close_prev|, |Low - Close_prev|)
        ATR = EMA(True Range, window)

    Параметры:
        window (int): Период окна (по умолчанию 14)

    Выходные колонки:
        ATR_{window}: Значение ATR

    Применение:
        - Измерение волатильности
        - Определение размера стоп-лосса
        - Фильтрация сигналов в периоды высокой/низкой волатильности

    Example:
        >>> atr = ATR(window=14)
        >>> result = atr.calculate(data)
        >>> # result содержит колонку 'ATR_14'
    """

    def __init__(self, window: int = 14):
        """
        Инициализация ATR.

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
        return self.params.get("window", 14) + 1  # +1 для расчёта diff

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать ATR.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с колонкой ATR_{window}
        """
        self._validate_data(data)

        window = self.params["window"]

        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Предыдущее закрытие
        close_prev = close.shift(1)

        # True Range
        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR = EMA(TR, window)
        atr = tr.ewm(span=window, adjust=False, min_periods=window).mean()

        col_name = f"ATR_{window}"
        result = pd.DataFrame({col_name: atr}, index=data.index)

        return result
