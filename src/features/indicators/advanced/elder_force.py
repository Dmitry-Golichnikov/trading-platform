"""
Elder Force Index - Индекс силы Элдера.
"""

from typing import List

import pandas as pd

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register("elder_force")
class ElderForce(Indicator):
    """
    Elder Force Index - Индекс силы Элдера.

    Формула:
        Force Index = (Close - Close_prev) * Volume
        Smoothed Force = EMA(Force Index, window)

    Параметры:
        window (int): Период EMA (по умолчанию 13)

    Выходные колонки:
        Elder_Force_{window}: Значение индекса силы

    Применение:
        - Положительное значение - давление покупок
        - Отрицательное значение - давление продаж
        - Подтверждение тренда
        - Дивергенции с ценой

    Example:
        >>> elder = ElderForce(window=13)
        >>> result = elder.calculate(data)
    """

    def __init__(self, window: int = 13):
        """
        Инициализация Elder Force Index.

        Args:
            window: Период EMA
        """
        super().__init__(window=window)

    def validate_params(self) -> None:
        """Валидация параметров."""
        window = self.params.get("window", 13)
        if not isinstance(window, int) or window < 1:
            raise ValueError(f"window должен быть положительным целым числом, получено: {window}")

    def get_required_columns(self) -> List[str]:
        """Необходимые колонки."""
        return ["close", "volume"]

    def get_lookback_period(self) -> int:
        """Период разогрева."""
        return self.params.get("window", 13) + 1

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать Elder Force Index.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с колонкой Elder_Force_{window}
        """
        self._validate_data(data)

        window = self.params["window"]

        # Force Index (raw)
        price_change = data["close"].diff()
        force_index = price_change * data["volume"]

        # Сглаживание
        force_smoothed = force_index.ewm(span=window, adjust=False, min_periods=window).mean()

        col_name = f"Elder_Force_{window}"
        result = pd.DataFrame({col_name: force_smoothed}, index=data.index)

        return result
