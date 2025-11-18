"""
Ichimoku Cloud - Облако Ишимоку.
"""

from typing import List

import pandas as pd

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register("ichimoku")
class Ichimoku(Indicator):
    """
    Ichimoku Cloud (Облако Ишимоку) - система технического анализа.

    Формулы:
        Tenkan-sen (Conversion Line) =
            (max(high, tenkan) + min(low, tenkan)) / 2
        Kijun-sen (Base Line) =
            (max(high, kijun) + min(low, kijun)) / 2
        Senkou Span A (Leading Span A) =
            (Tenkan + Kijun) / 2, сдвиг на kijun вперёд
        Senkou Span B (Leading Span B) =
            (max(high, senkou) + min(low, senkou)) / 2
        Chikou Span (Lagging Span) = Close, сдвиг на kijun назад

    Параметры:
        tenkan (int): Период Tenkan-sen (по умолчанию 9)
        kijun (int): Период Kijun-sen (по умолчанию 26)
        senkou (int): Период Senkou Span B (по умолчанию 52)

    Выходные колонки:
        Ichimoku_tenkan: Линия конверсии
        Ichimoku_kijun: Базовая линия
        Ichimoku_senkou_a: Ведущая линия A (облако)
        Ichimoku_senkou_b: Ведущая линия B (облако)
        Ichimoku_chikou: Запаздывающая линия

    Применение:
        - Цена выше облака - восходящий тренд
        - Цена ниже облака - нисходящий тренд
        - Пересечение Tenkan и Kijun - торговые сигналы
        - Толщина облака - сила тренда

    Example:
        >>> ichimoku = Ichimoku(tenkan=9, kijun=26, senkou=52)
        >>> result = ichimoku.calculate(data)
    """

    def __init__(self, tenkan: int = 9, kijun: int = 26, senkou: int = 52):
        """
        Инициализация Ichimoku.

        Args:
            tenkan: Период Tenkan-sen
            kijun: Период Kijun-sen
            senkou: Период Senkou Span B
        """
        super().__init__(tenkan=tenkan, kijun=kijun, senkou=senkou)

    def validate_params(self) -> None:
        """Валидация параметров."""
        tenkan = self.params.get("tenkan", 9)
        kijun = self.params.get("kijun", 26)
        senkou = self.params.get("senkou", 52)

        for name, value in [("tenkan", tenkan), ("kijun", kijun), ("senkou", senkou)]:
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} должен быть положительным целым числом, получено: {value}")

    def get_required_columns(self) -> List[str]:
        """Необходимые колонки."""
        return ["high", "low", "close"]

    def get_lookback_period(self) -> int:
        """Период разогрева."""
        # Максимальный период
        return self.params.get("senkou", 52)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать Ichimoku Cloud.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с компонентами Ichimoku
        """
        self._validate_data(data)

        tenkan = self.params["tenkan"]
        kijun = self.params["kijun"]
        senkou = self.params["senkou"]

        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(window=tenkan, min_periods=tenkan).max()
        tenkan_low = low.rolling(window=tenkan, min_periods=tenkan).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2

        # Kijun-sen (Base Line)
        kijun_high = high.rolling(window=kijun, min_periods=kijun).max()
        kijun_low = low.rolling(window=kijun, min_periods=kijun).min()
        kijun_sen = (kijun_high + kijun_low) / 2

        # Senkou Span A (Leading Span A)
        # ВАЖНО: НЕ сдвигаем вперёд для каузальности!
        # Пользователь может сдвинуть сам если нужно
        senkou_span_a = (tenkan_sen + kijun_sen) / 2

        # Senkou Span B (Leading Span B)
        senkou_high = high.rolling(window=senkou, min_periods=senkou).max()
        senkou_low = low.rolling(window=senkou, min_periods=senkou).min()
        senkou_span_b = (senkou_high + senkou_low) / 2

        # Chikou Span (Lagging Span)
        # ВАЖНО: НЕ сдвигаем назад для каузальности!
        # Это просто close, сдвиг пользователь может сделать сам
        chikou_span = close

        result = pd.DataFrame(
            {
                "Ichimoku_tenkan": tenkan_sen,
                "Ichimoku_kijun": kijun_sen,
                "Ichimoku_senkou_a": senkou_span_a,
                "Ichimoku_senkou_b": senkou_span_b,
                "Ichimoku_chikou": chikou_span,
            },
            index=data.index,
        )

        return result
