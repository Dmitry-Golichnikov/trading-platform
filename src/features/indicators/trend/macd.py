"""
MACD (Moving Average Convergence Divergence) - Схождение-расхождение скользящих средних.
"""

from typing import List

import pandas as pd

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register("macd")
class MACD(Indicator):
    """
    MACD (Moving Average Convergence Divergence).

    Формула:
        MACD Line = EMA(fast) - EMA(slow)
        Signal Line = EMA(MACD Line, signal)
        Histogram = MACD Line - Signal Line

    Параметры:
        fast (int): Период быстрой EMA (по умолчанию 12)
        slow (int): Период медленной EMA (по умолчанию 26)
        signal (int): Период сигнальной линии (по умолчанию 9)
        column (str): Колонка для расчёта (по умолчанию 'close')

    Выходные колонки:
        MACD: Основная линия MACD
        MACD_signal: Сигнальная линия
        MACD_hist: Гистограмма (разница между MACD и сигнальной линией)

    Применение:
        - Пересечение MACD и сигнальной линии - сигнал к покупке/продаже
        - Дивергенции с ценой - потенциальный разворот тренда
        - Гистограмма показывает силу тренда

    Example:
        >>> macd = MACD(fast=12, slow=26, signal=9)
        >>> result = macd.calculate(data)
        >>> # result содержит колонки 'MACD', 'MACD_signal', 'MACD_hist'
    """

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        column: str = "close",
    ):
        """
        Инициализация MACD.

        Args:
            fast: Период быстрой EMA
            slow: Период медленной EMA
            signal: Период сигнальной линии
            column: Колонка для расчёта
        """
        super().__init__(fast=fast, slow=slow, signal=signal, column=column)

    def validate_params(self) -> None:
        """Валидация параметров."""
        fast = self.params.get("fast", 12)
        slow = self.params.get("slow", 26)
        signal = self.params.get("signal", 9)

        if not isinstance(fast, int) or fast < 1:
            raise ValueError(f"fast должен быть положительным целым числом, получено: {fast}")
        if not isinstance(slow, int) or slow < 1:
            raise ValueError(f"slow должен быть положительным целым числом, получено: {slow}")
        if not isinstance(signal, int) or signal < 1:
            raise ValueError(f"signal должен быть положительным целым числом, получено: {signal}")
        if fast >= slow:
            raise ValueError(f"fast ({fast}) должен быть меньше slow ({slow})")

    def get_required_columns(self) -> List[str]:
        """Необходимые колонки."""
        return [self.params.get("column", "close")]

    def get_lookback_period(self) -> int:
        """Период разогрева."""
        # Максимальный период + signal для расчёта сигнальной линии
        slow = self.params.get("slow", 26)
        signal = self.params.get("signal", 9)
        return slow + signal

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать MACD.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с колонками MACD, MACD_signal, MACD_hist
        """
        self._validate_data(data)

        fast = self.params["fast"]
        slow = self.params["slow"]
        signal = self.params["signal"]
        column = self.params["column"]

        # Рассчитываем быструю и медленную EMA
        ema_fast = data[column].ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_slow = data[column].ewm(span=slow, adjust=False, min_periods=slow).mean()

        # MACD Line
        macd_line = ema_fast - ema_slow

        # Signal Line (EMA от MACD Line)
        signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()

        # Histogram
        histogram = macd_line - signal_line

        result = pd.DataFrame(
            {
                "MACD": macd_line,
                "MACD_signal": signal_line,
                "MACD_hist": histogram,
            },
            index=data.index,
        )

        return result
