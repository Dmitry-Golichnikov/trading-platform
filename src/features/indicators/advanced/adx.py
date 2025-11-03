"""
ADX (Average Directional Index) - Средний индекс направленности.
"""

from typing import List

import pandas as pd

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register("adx")
class ADX(Indicator):
    """
    ADX (Average Directional Index) - Средний индекс направленности.

    Формулы:
        +DM = High_t - High_{t-1} если > 0 и > (Low_{t-1} - Low_t), иначе 0
        -DM = Low_{t-1} - Low_t если > 0 и > (High_t - High_{t-1}), иначе 0

        +DI = 100 * EMA(+DM, window) / ATR
        -DI = 100 * EMA(-DM, window) / ATR

        DX = 100 * |+DI - -DI| / (+DI + -DI)
        ADX = EMA(DX, window)

    Параметры:
        window (int): Период окна (по умолчанию 14)

    Выходные колонки:
        ADX_{window}: Значение ADX
        DI_plus: Положительный индикатор направленности
        DI_minus: Отрицательный индикатор направленности

    Применение:
        - ADX > 25 - сильный тренд
        - ADX < 20 - слабый тренд, флэт
        - +DI > -DI - восходящий тренд
        - +DI < -DI - нисходящий тренд

    Example:
        >>> adx = ADX(window=14)
        >>> result = adx.calculate(data)
    """

    def __init__(self, window: int = 14):
        """
        Инициализация ADX.

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
        # Нужно 2*window для расчёта EMA(DX)
        return self.params.get("window", 14) * 2 + 1

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать ADX.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с колонками ADX_{window}, DI_plus, DI_minus
        """
        self._validate_data(data)

        window = self.params["window"]

        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Рассчитываем ATR для знаменателя
        close_prev = close.shift(1)
        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=window, adjust=False, min_periods=window).mean()

        # Directional Movement
        high_diff = high.diff()
        low_diff = -low.diff()

        # +DM и -DM
        plus_dm = ((high_diff > low_diff) & (high_diff > 0)) * high_diff
        minus_dm = ((low_diff > high_diff) & (low_diff > 0)) * low_diff

        # Сглаживание DM
        plus_dm_smooth = plus_dm.ewm(
            span=window, adjust=False, min_periods=window
        ).mean()
        minus_dm_smooth = minus_dm.ewm(
            span=window, adjust=False, min_periods=window
        ).mean()

        # Directional Indicators
        di_plus = 100 * plus_dm_smooth / atr
        di_minus = 100 * minus_dm_smooth / atr

        # DX
        dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)

        # ADX (EMA от DX)
        adx = dx.ewm(span=window, adjust=False, min_periods=window).mean()

        col_name = f"ADX_{window}"
        result = pd.DataFrame(
            {col_name: adx, "DI_plus": di_plus, "DI_minus": di_minus},
            index=data.index,
        )

        return result
