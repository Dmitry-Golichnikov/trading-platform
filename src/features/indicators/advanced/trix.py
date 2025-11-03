"""
TRIX (Triple Exponential Average) - Тройная экспоненциальная средняя.
"""

from typing import List

import pandas as pd

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register("trix")
class TRIX(Indicator):
    """
    TRIX (Triple Exponential Average) - Тройная экспоненциальная средняя.

    Формула:
        EMA1 = EMA(Close, window)
        EMA2 = EMA(EMA1, window)
        EMA3 = EMA(EMA2, window)

        TRIX = 100 * (EMA3 - EMA3_prev) / EMA3_prev

    Параметры:
        window (int): Период EMA (по умолчанию 15)
        signal (int): Период сигнальной линии (по умолчанию 9)
        column (str): Колонка для расчёта (по умолчанию 'close')

    Выходные колонки:
        TRIX: Значение TRIX
        TRIX_signal: Сигнальная линия (EMA от TRIX)

    Применение:
        - Фильтрует рыночный шум
        - Пересечение нулевой линии - изменение тренда
        - Пересечение TRIX и сигнальной линии - торговые сигналы

    Example:
        >>> trix = TRIX(window=15, signal=9)
        >>> result = trix.calculate(data)
    """

    def __init__(self, window: int = 15, signal: int = 9, column: str = "close"):
        """
        Инициализация TRIX.

        Args:
            window: Период EMA
            signal: Период сигнальной линии
            column: Колонка для расчёта
        """
        super().__init__(window=window, signal=signal, column=column)

    def validate_params(self) -> None:
        """Валидация параметров."""
        window = self.params.get("window", 15)
        signal = self.params.get("signal", 9)

        if not isinstance(window, int) or window < 1:
            raise ValueError(
                f"window должен быть положительным целым числом, получено: {window}"
            )
        if not isinstance(signal, int) or signal < 1:
            raise ValueError(
                f"signal должен быть положительным целым числом, получено: {signal}"
            )

    def get_required_columns(self) -> List[str]:
        """Необходимые колонки."""
        return [self.params.get("column", "close")]

    def get_lookback_period(self) -> int:
        """Период разогрева."""
        window = self.params.get("window", 15)
        signal = self.params.get("signal", 9)
        return 3 * window + signal + 1

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать TRIX.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с колонками TRIX, TRIX_signal
        """
        self._validate_data(data)

        window = self.params["window"]
        signal = self.params["signal"]
        column = self.params["column"]

        # Тройная EMA
        ema1 = data[column].ewm(span=window, adjust=False, min_periods=window).mean()
        ema2 = ema1.ewm(span=window, adjust=False, min_periods=window).mean()
        ema3 = ema2.ewm(span=window, adjust=False, min_periods=window).mean()

        # TRIX (процентное изменение EMA3)
        trix = 100 * ema3.pct_change()

        # Сигнальная линия
        trix_signal = trix.ewm(span=signal, adjust=False, min_periods=signal).mean()

        result = pd.DataFrame(
            {"TRIX": trix, "TRIX_signal": trix_signal}, index=data.index
        )

        return result
