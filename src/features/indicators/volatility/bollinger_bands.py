"""
Bollinger Bands - Полосы Боллинджера.
"""

from typing import List

import pandas as pd

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register("bollinger_bands")
class BollingerBands(Indicator):
    """
    Bollinger Bands - Полосы Боллинджера.

    Формулы:
        Middle Band = SMA(Close, window)
        Upper Band = Middle Band + (std_dev * std)
        Lower Band = Middle Band - (std_dev * std)

    где std = стандартное отклонение за период window.

    Параметры:
        window (int): Период окна (по умолчанию 20)
        std_dev (float): Количество стандартных отклонений (по умолчанию 2.0)
        column (str): Колонка для расчёта (по умолчанию 'close')

    Выходные колонки:
        BB_upper: Верхняя полоса
        BB_middle: Средняя полоса (SMA)
        BB_lower: Нижняя полоса
        BB_width: Ширина полос (upper - lower)
        BB_pct: Процентное положение цены в полосе

    Применение:
        - Цена у верхней полосы - возможная перекупленность
        - Цена у нижней полосы - возможная перепроданность
        - Сужение полос - низкая волатильность, возможен прорыв
        - Расширение полос - высокая волатильность

    Example:
        >>> bb = BollingerBands(window=20, std_dev=2.0)
        >>> result = bb.calculate(data)
    """

    def __init__(self, window: int = 20, std_dev: float = 2.0, column: str = "close"):
        """
        Инициализация Bollinger Bands.

        Args:
            window: Период окна
            std_dev: Количество стандартных отклонений
            column: Колонка для расчёта
        """
        super().__init__(window=window, std_dev=std_dev, column=column)

    def validate_params(self) -> None:
        """Валидация параметров."""
        window = self.params.get("window", 20)
        std_dev = self.params.get("std_dev", 2.0)

        if not isinstance(window, int) or window < 1:
            raise ValueError(f"window должен быть положительным целым числом, получено: {window}")
        if std_dev <= 0:
            raise ValueError(f"std_dev должен быть положительным, получено: {std_dev}")

    def get_required_columns(self) -> List[str]:
        """Необходимые колонки."""
        return [self.params.get("column", "close")]

    def get_lookback_period(self) -> int:
        """Период разогрева."""
        return self.params.get("window", 20)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать Bollinger Bands.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с колонками BB_upper, BB_middle, BB_lower, BB_width, BB_pct
        """
        self._validate_data(data)

        window = self.params["window"]
        std_dev = self.params["std_dev"]
        column = self.params["column"]

        # Middle Band (SMA)
        middle = data[column].rolling(window=window, min_periods=window).mean()

        # Стандартное отклонение
        std = data[column].rolling(window=window, min_periods=window).std()

        # Upper и Lower bands
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)

        # Ширина полос
        width = upper - lower

        # Процентное положение цены в полосе
        # %B = (Close - Lower) / (Upper - Lower)
        pct = (data[column] - lower) / (upper - lower)

        result = pd.DataFrame(
            {
                "BB_upper": upper,
                "BB_middle": middle,
                "BB_lower": lower,
                "BB_width": width,
                "BB_pct": pct,
            },
            index=data.index,
        )

        return result
