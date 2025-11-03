"""
CCI (Commodity Channel Index) - Индекс товарного канала.
"""

from typing import List

import pandas as pd

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register("cci")
class CCI(Indicator):
    """
    CCI (Commodity Channel Index) - Индекс товарного канала.

    Формула:
        Typical Price = (High + Low + Close) / 3
        CCI = (Typical Price - SMA(Typical Price)) / (0.015 * Mean Deviation)

    где Mean Deviation = среднее абсолютное отклонение от SMA.

    Параметры:
        window (int): Период окна (по умолчанию 20)
        constant (float): Константа для масштабирования (по умолчанию 0.015)

    Выходные колонки:
        CCI_{window}: Значение CCI

    Применение:
        - CCI > 100 - перекупленность
        - CCI < -100 - перепроданность
        - Пересечение нулевой линии - изменение тренда

    Example:
        >>> cci = CCI(window=20)
        >>> result = cci.calculate(data)
        >>> # result содержит колонку 'CCI_20'
    """

    def __init__(self, window: int = 20, constant: float = 0.015):
        """
        Инициализация CCI.

        Args:
            window: Период окна
            constant: Константа для масштабирования
        """
        super().__init__(window=window, constant=constant)

    def validate_params(self) -> None:
        """Валидация параметров."""
        window = self.params.get("window", 20)
        constant = self.params.get("constant", 0.015)

        if not isinstance(window, int) or window < 1:
            raise ValueError(
                f"window должен быть положительным целым числом, получено: {window}"
            )
        if constant <= 0:
            raise ValueError(
                f"constant должна быть положительной, получено: {constant}"
            )

    def get_required_columns(self) -> List[str]:
        """Необходимые колонки."""
        return ["high", "low", "close"]

    def get_lookback_period(self) -> int:
        """Период разогрева."""
        return self.params.get("window", 20)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать CCI.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с колонкой CCI_{window}
        """
        self._validate_data(data)

        window = self.params["window"]
        constant = self.params["constant"]

        # Typical Price
        tp = (data["high"] + data["low"] + data["close"]) / 3

        # SMA от Typical Price
        tp_sma = tp.rolling(window=window, min_periods=window).mean()

        # Mean Deviation
        mean_dev = tp.rolling(window=window, min_periods=window).apply(
            lambda x: abs(x - x.mean()).mean(), raw=False
        )

        # CCI
        cci = (tp - tp_sma) / (constant * mean_dev)

        col_name = f"CCI_{window}"
        result = pd.DataFrame({col_name: cci}, index=data.index)

        return result
