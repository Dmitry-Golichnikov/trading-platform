"""
Stochastic Oscillator - Стохастический осциллятор.
"""

from typing import List

import pandas as pd

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register("stochastic")
class Stochastic(Indicator):
    """
    Stochastic Oscillator - Стохастический осциллятор.

    Формулы:
        %K = 100 * (Close - Low_n) / (High_n - Low_n)
        %D = SMA(%K, smooth_d)

    где High_n и Low_n - максимум и минимум за период k.

    Параметры:
        k (int): Период %K (по умолчанию 14)
        smooth_k (int): Период сглаживания %K (по умолчанию 3)
        smooth_d (int): Период %D (по умолчанию 3)

    Выходные колонки:
        Stoch_k: Быстрая линия %K
        Stoch_d: Медленная линия %D

    Применение:
        - %K > 80 - перекупленность
        - %K < 20 - перепроданность
        - Пересечение %K и %D - торговые сигналы

    Example:
        >>> stoch = Stochastic(k=14, smooth_k=3, smooth_d=3)
        >>> result = stoch.calculate(data)
        >>> # result содержит колонки 'Stoch_k', 'Stoch_d'
    """

    def __init__(self, k: int = 14, smooth_k: int = 3, smooth_d: int = 3):
        """
        Инициализация Stochastic.

        Args:
            k: Период %K
            smooth_k: Период сглаживания %K
            smooth_d: Период %D
        """
        super().__init__(k=k, smooth_k=smooth_k, smooth_d=smooth_d)

    def validate_params(self) -> None:
        """Валидация параметров."""
        k = self.params.get("k", 14)
        smooth_k = self.params.get("smooth_k", 3)
        smooth_d = self.params.get("smooth_d", 3)

        for name, value in [("k", k), ("smooth_k", smooth_k), ("smooth_d", smooth_d)]:
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} должен быть положительным целым числом, получено: {value}")

    def get_required_columns(self) -> List[str]:
        """Необходимые колонки."""
        return ["high", "low", "close"]

    def get_lookback_period(self) -> int:
        """Период разогрева."""
        k = self.params.get("k", 14)
        smooth_k = self.params.get("smooth_k", 3)
        smooth_d = self.params.get("smooth_d", 3)
        return k + smooth_k + smooth_d

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать Stochastic Oscillator.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с колонками Stoch_k, Stoch_d
        """
        self._validate_data(data)

        k = self.params["k"]
        smooth_k = self.params["smooth_k"]
        smooth_d = self.params["smooth_d"]

        # Рассчитываем max и min за период k
        high_max = data["high"].rolling(window=k, min_periods=k).max()
        low_min = data["low"].rolling(window=k, min_periods=k).min()

        # Рассчитываем %K (raw)
        stoch_k_raw = 100 * (data["close"] - low_min) / (high_max - low_min)

        # Сглаживаем %K
        stoch_k = stoch_k_raw.rolling(window=smooth_k, min_periods=smooth_k).mean()

        # Рассчитываем %D (SMA от %K)
        stoch_d = stoch_k.rolling(window=smooth_d, min_periods=smooth_d).mean()

        result = pd.DataFrame({"Stoch_k": stoch_k, "Stoch_d": stoch_d}, index=data.index)

        return result
