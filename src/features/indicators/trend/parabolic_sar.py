"""
Parabolic SAR (Stop and Reverse) - Параболическая система.
"""

from typing import List

import numpy as np
import pandas as pd

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register("parabolic_sar")
class ParabolicSAR(Indicator):
    """
    Parabolic SAR (Stop and Reverse) - Параболическая система.

    Формула:
        SAR_t = SAR_{t-1} + AF * (EP - SAR_{t-1})
        где:
        - AF (Acceleration Factor) - фактор ускорения
        - EP (Extreme Point) - экстремальная точка

    Параметры:
        af_start (float): Начальное значение AF (по умолчанию 0.02)
        af_increment (float): Приращение AF (по умолчанию 0.02)
        af_max (float): Максимальное значение AF (по умолчанию 0.20)

    Выходные колонки:
        PSAR: Значение Parabolic SAR
        PSAR_trend: Направление тренда (1 - восходящий, -1 - нисходящий)

    Применение:
        - Определение стоп-лосса и точек разворота
        - Когда цена пересекает SAR - сигнал к развороту позиции
        - SAR ниже цены - восходящий тренд, выше - нисходящий

    Example:
        >>> psar = ParabolicSAR(af_start=0.02, af_max=0.20)
        >>> result = psar.calculate(data)
        >>> # result содержит колонки 'PSAR', 'PSAR_trend'
    """

    def __init__(
        self,
        af_start: float = 0.02,
        af_increment: float = 0.02,
        af_max: float = 0.20,
    ):
        """
        Инициализация Parabolic SAR.

        Args:
            af_start: Начальное значение фактора ускорения
            af_increment: Приращение фактора ускорения
            af_max: Максимальное значение фактора ускорения
        """
        super().__init__(af_start=af_start, af_increment=af_increment, af_max=af_max)

    def validate_params(self) -> None:
        """Валидация параметров."""
        af_start = self.params.get("af_start", 0.02)
        af_increment = self.params.get("af_increment", 0.02)
        af_max = self.params.get("af_max", 0.20)

        if af_start <= 0 or af_start > 1:
            raise ValueError(f"af_start должен быть в (0, 1], получено: {af_start}")
        if af_increment <= 0 or af_increment > 1:
            raise ValueError(f"af_increment должен быть в (0, 1], получено: {af_increment}")
        if af_max <= 0 or af_max > 1:
            raise ValueError(f"af_max должен быть в (0, 1], получено: {af_max}")
        if af_start > af_max:
            raise ValueError(f"af_start ({af_start}) не может быть больше af_max ({af_max})")

    def get_required_columns(self) -> List[str]:
        """Необходимые колонки."""
        return ["high", "low", "close"]

    def get_lookback_period(self) -> int:
        """Период разогрева."""
        return 2  # Минимум нужно 2 бара для начала

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать Parabolic SAR.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с колонками PSAR, PSAR_trend
        """
        self._validate_data(data)

        af_start = self.params["af_start"]
        af_increment = self.params["af_increment"]
        af_max = self.params["af_max"]

        high = data["high"].values
        low = data["low"].values
        close = data["close"].values
        n = len(data)

        # Инициализация массивов
        psar = np.full(n, np.nan)
        trend = np.zeros(n)  # 1 = восходящий, -1 = нисходящий
        af = np.full(n, np.nan)
        ep = np.full(n, np.nan)

        # Начальные значения (первый бар)
        if n < 2:
            return pd.DataFrame({"PSAR": psar, "PSAR_trend": trend}, index=data.index)

        # Определяем начальный тренд (на основе первых двух баров)
        if close[1] > close[0]:
            trend[0] = 1  # Восходящий
            psar[0] = low[0]
            ep[0] = high[0]
        else:
            trend[0] = -1  # Нисходящий
            psar[0] = high[0]
            ep[0] = low[0]

        af[0] = af_start

        # Расчёт для каждого бара
        for i in range(1, n):
            # Копируем значения с предыдущего бара
            psar[i] = psar[i - 1] + af[i - 1] * (ep[i - 1] - psar[i - 1])
            trend[i] = trend[i - 1]
            af[i] = af[i - 1]
            ep[i] = ep[i - 1]

            # Восходящий тренд
            if trend[i] == 1:
                # Обновляем PSAR (не выше минимумов последних двух баров)
                psar[i] = min(psar[i], low[i - 1])
                if i > 1:
                    psar[i] = min(psar[i], low[i - 2])

                # Проверяем разворот (цена опустилась ниже SAR)
                if low[i] < psar[i]:
                    trend[i] = -1  # Переход в нисходящий тренд
                    psar[i] = ep[i]  # SAR = последняя EP
                    ep[i] = low[i]  # Новая EP = текущий low
                    af[i] = af_start  # Сброс AF
                else:
                    # Обновляем EP если новый максимум
                    if high[i] > ep[i]:
                        ep[i] = high[i]
                        af[i] = min(af[i] + af_increment, af_max)

            # Нисходящий тренд
            else:
                # Обновляем PSAR (не ниже максимумов последних двух баров)
                psar[i] = max(psar[i], high[i - 1])
                if i > 1:
                    psar[i] = max(psar[i], high[i - 2])

                # Проверяем разворот (цена поднялась выше SAR)
                if high[i] > psar[i]:
                    trend[i] = 1  # Переход в восходящий тренд
                    psar[i] = ep[i]  # SAR = последняя EP
                    ep[i] = high[i]  # Новая EP = текущий high
                    af[i] = af_start  # Сброс AF
                else:
                    # Обновляем EP если новый минимум
                    if low[i] < ep[i]:
                        ep[i] = low[i]
                        af[i] = min(af[i] + af_increment, af_max)

        result = pd.DataFrame({"PSAR": psar, "PSAR_trend": trend}, index=data.index)

        return result
