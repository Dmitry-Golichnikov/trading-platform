"""
MFI (Money Flow Index) - Индекс денежного потока.
"""

from typing import List

import pandas as pd

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register("mfi")
class MFI(Indicator):
    """
    MFI (Money Flow Index) - Индекс денежного потока.

    Формулы:
        Typical Price = (High + Low + Close) / 3
        Raw Money Flow = Typical Price * Volume

        Если TP_t > TP_{t-1}: Positive Money Flow
        Если TP_t < TP_{t-1}: Negative Money Flow

        Money Flow Ratio = sum(Positive MF, window) / sum(Negative MF, window)
        MFI = 100 - (100 / (1 + Money Flow Ratio))

    Параметры:
        window (int): Период окна (по умолчанию 14)

    Выходные колонки:
        MFI_{window}: Значение MFI (0-100)

    Применение:
        - Аналог RSI с учётом объёма
        - MFI > 80 - перекупленность
        - MFI < 20 - перепроданность
        - Дивергенции с ценой

    Example:
        >>> mfi = MFI(window=14)
        >>> result = mfi.calculate(data)
        >>> # result содержит колонку 'MFI_14'
    """

    def __init__(self, window: int = 14):
        """
        Инициализация MFI.

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
        return ["high", "low", "close", "volume"]

    def get_lookback_period(self) -> int:
        """Период разогрева."""
        return self.params.get("window", 14) + 1

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать MFI.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с колонкой MFI_{window}
        """
        self._validate_data(data)

        window = self.params["window"]

        # Typical Price
        tp = (data["high"] + data["low"] + data["close"]) / 3

        # Raw Money Flow
        rmf = tp * data["volume"]

        # Определяем направление
        tp_diff = tp.diff()

        # Positive и Negative Money Flow
        positive_mf = rmf.where(tp_diff > 0, 0.0)  # type: ignore[operator]
        negative_mf = rmf.where(tp_diff < 0, 0.0)  # type: ignore[operator]

        # Суммы за период
        positive_mf_sum = positive_mf.rolling(window=window, min_periods=window).sum()
        negative_mf_sum = negative_mf.rolling(window=window, min_periods=window).sum()

        # Money Flow Ratio
        mf_ratio = positive_mf_sum / negative_mf_sum

        # MFI
        mfi = 100 - (100 / (1 + mf_ratio))

        col_name = f"MFI_{window}"
        result = pd.DataFrame({col_name: mfi}, index=data.index)

        return result
