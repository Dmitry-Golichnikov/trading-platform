"""
DPO (Detrended Price Oscillator) - Детрендированный ценовой осциллятор.
"""

from typing import List

import pandas as pd

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register("dpo")
class DPO(Indicator):
    """
    DPO (Detrended Price Oscillator) - Детрендированный ценовой осциллятор.

    Формула:
        DPO = Close - SMA(Close, window) сдвинутая на (window/2 + 1) назад

    Параметры:
        window (int): Период SMA (по умолчанию 20)
        column (str): Колонка для расчёта (по умолчанию 'close')

    Выходные колонки:
        DPO_{window}: Значение DPO

    Применение:
        - Устраняет тренд, показывает циклы
        - Положительное значение - цена выше среднего
        - Отрицательное значение - цена ниже среднего
        - НЕ используется для определения тренда

    Example:
        >>> dpo = DPO(window=20)
        >>> result = dpo.calculate(data)
    """

    def __init__(self, window: int = 20, column: str = "close"):
        """
        Инициализация DPO.

        Args:
            window: Период SMA
            column: Колонка для расчёта
        """
        super().__init__(window=window, column=column)

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
        window = self.params.get("window", 20)
        return window + (window // 2) + 1

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать DPO.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с колонкой DPO_{window}
        """
        self._validate_data(data)

        window = self.params["window"]
        column = self.params["column"]

        # SMA
        sma = data[column].rolling(window=window, min_periods=window).mean()

        # Сдвиг SMA назад
        shift_period = window // 2 + 1
        sma_shifted = sma.shift(shift_period)

        # DPO
        dpo = data[column] - sma_shifted

        col_name = f"DPO_{window}"
        result = pd.DataFrame({col_name: dpo}, index=data.index)

        return result
