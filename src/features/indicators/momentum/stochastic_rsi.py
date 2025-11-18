"""
Stochastic RSI - Стохастический RSI.
"""

from typing import List

import pandas as pd

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register("stochastic_rsi")
class StochasticRSI(Indicator):
    """
    Stochastic RSI - Стохастический RSI.

    Применяет формулу стохастического осциллятора к значениям RSI.

    Формулы:
        RSI = RSI(close, rsi_period)
        StochRSI = (RSI - Min(RSI, stoch_period)) /
                   (Max(RSI, stoch_period) - Min(RSI, stoch_period))
        %K = SMA(StochRSI, smooth_k) * 100
        %D = SMA(%K, smooth_d)

    Параметры:
        rsi_period (int): Период RSI (по умолчанию 14)
        stoch_period (int): Период Stochastic (по умолчанию 14)
        smooth_k (int): Период сглаживания %K (по умолчанию 3)
        smooth_d (int): Период %D (по умолчанию 3)
        column (str): Колонка для расчёта (по умолчанию 'close')

    Выходные колонки:
        StochRSI_k: Быстрая линия %K
        StochRSI_d: Медленная линия %D

    Применение:
        - Более чувствителен чем обычный Stochastic
        - %K > 80 - перекупленность
        - %K < 20 - перепроданность

    Example:
        >>> stoch_rsi = StochasticRSI(rsi_period=14, stoch_period=14)
        >>> result = stoch_rsi.calculate(data)
    """

    def __init__(
        self,
        rsi_period: int = 14,
        stoch_period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3,
        column: str = "close",
    ):
        """
        Инициализация Stochastic RSI.

        Args:
            rsi_period: Период RSI
            stoch_period: Период Stochastic
            smooth_k: Период сглаживания %K
            smooth_d: Период %D
            column: Колонка для расчёта
        """
        super().__init__(
            rsi_period=rsi_period,
            stoch_period=stoch_period,
            smooth_k=smooth_k,
            smooth_d=smooth_d,
            column=column,
        )

    def validate_params(self) -> None:
        """Валидация параметров."""
        for param_name in ["rsi_period", "stoch_period", "smooth_k", "smooth_d"]:
            value = self.params.get(param_name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{param_name} должен быть положительным целым " f"числом, получено: {value}")

    def get_required_columns(self) -> List[str]:
        """Необходимые колонки."""
        return [self.params.get("column", "close")]

    def get_lookback_period(self) -> int:
        """Период разогрева."""
        rsi_period = self.params.get("rsi_period", 14)
        stoch_period = self.params.get("stoch_period", 14)
        smooth_k = self.params.get("smooth_k", 3)
        smooth_d = self.params.get("smooth_d", 3)
        return rsi_period + stoch_period + smooth_k + smooth_d

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать Stochastic RSI.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с колонками StochRSI_k, StochRSI_d
        """
        self._validate_data(data)

        rsi_period = self.params["rsi_period"]
        stoch_period = self.params["stoch_period"]
        smooth_k = self.params["smooth_k"]
        smooth_d = self.params["smooth_d"]
        column = self.params["column"]

        # Рассчитываем RSI
        delta = data[column].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(span=rsi_period, adjust=False, min_periods=rsi_period).mean()
        avg_loss = loss.ewm(span=rsi_period, adjust=False, min_periods=rsi_period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Применяем Stochastic к RSI
        rsi_max = rsi.rolling(window=stoch_period, min_periods=stoch_period).max()
        rsi_min = rsi.rolling(window=stoch_period, min_periods=stoch_period).min()

        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min)

        # Сглаживаем %K
        stoch_rsi_k = stoch_rsi.rolling(window=smooth_k, min_periods=smooth_k).mean() * 100

        # Рассчитываем %D
        stoch_rsi_d = stoch_rsi_k.rolling(window=smooth_d, min_periods=smooth_d).mean()

        result = pd.DataFrame(
            {"StochRSI_k": stoch_rsi_k, "StochRSI_d": stoch_rsi_d},
            index=data.index,
        )

        return result
