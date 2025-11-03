"""
Keltner Channels - Каналы Кельтнера.
"""

from typing import List

import pandas as pd

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register("keltner_channels")
class KeltnerChannels(Indicator):
    """
    Keltner Channels - Каналы Кельтнера.

    Формулы:
        Middle Line = EMA(Close, window)
        Upper Channel = Middle Line + (multiplier * ATR)
        Lower Channel = Middle Line - (multiplier * ATR)

    Параметры:
        window (int): Период EMA (по умолчанию 20)
        atr_window (int): Период ATR (по умолчанию 10)
        multiplier (float): Множитель для ATR (по умолчанию 2.0)
        column (str): Колонка для расчёта (по умолчанию 'close')

    Выходные колонки:
        KC_upper: Верхний канал
        KC_middle: Средняя линия (EMA)
        KC_lower: Нижний канал

    Применение:
        - Похожи на Bollinger Bands, но используют ATR вместо стандартного отклонения
        - Прорыв верхнего канала - сильный восходящий тренд
        - Прорыв нижнего канала - сильный нисходящий тренд

    Example:
        >>> kc = KeltnerChannels(window=20, atr_window=10, multiplier=2.0)
        >>> result = kc.calculate(data)
    """

    def __init__(
        self,
        window: int = 20,
        atr_window: int = 10,
        multiplier: float = 2.0,
        column: str = "close",
    ):
        """
        Инициализация Keltner Channels.

        Args:
            window: Период EMA
            atr_window: Период ATR
            multiplier: Множитель для ATR
            column: Колонка для расчёта
        """
        super().__init__(
            window=window,
            atr_window=atr_window,
            multiplier=multiplier,
            column=column,
        )

    def validate_params(self) -> None:
        """Валидация параметров."""
        window = self.params.get("window", 20)
        atr_window = self.params.get("atr_window", 10)
        multiplier = self.params.get("multiplier", 2.0)

        if not isinstance(window, int) or window < 1:
            raise ValueError(
                f"window должен быть положительным целым числом, получено: {window}"
            )
        if not isinstance(atr_window, int) or atr_window < 1:
            raise ValueError(
                f"atr_window должен быть положительным целым числом, "
                f"получено: {atr_window}"
            )
        if multiplier <= 0:
            raise ValueError(
                f"multiplier должен быть положительным, получено: {multiplier}"
            )

    def get_required_columns(self) -> List[str]:
        """Необходимые колонки."""
        return ["high", "low", self.params.get("column", "close")]

    def get_lookback_period(self) -> int:
        """Период разогрева."""
        window = self.params.get("window", 20)
        atr_window = self.params.get("atr_window", 10)
        return max(window, atr_window) + 1

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать Keltner Channels.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с колонками KC_upper, KC_middle, KC_lower
        """
        self._validate_data(data)

        window = self.params["window"]
        atr_window = self.params["atr_window"]
        multiplier = self.params["multiplier"]
        column = self.params["column"]

        # Middle Line (EMA)
        middle = data[column].ewm(span=window, adjust=False, min_periods=window).mean()

        # ATR
        high = data["high"]
        low = data["low"]
        close = data["close"]
        close_prev = close.shift(1)

        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.ewm(span=atr_window, adjust=False, min_periods=atr_window).mean()

        # Upper и Lower channels
        upper = middle + (multiplier * atr)
        lower = middle - (multiplier * atr)

        result = pd.DataFrame(
            {"KC_upper": upper, "KC_middle": middle, "KC_lower": lower},
            index=data.index,
        )

        return result
