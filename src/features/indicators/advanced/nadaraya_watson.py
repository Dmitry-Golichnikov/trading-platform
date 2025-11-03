"""
Nadaraya-Watson Envelope - Конверт Надарая-Ватсона.
"""

from typing import List

import numpy as np
import pandas as pd

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register("nadaraya_watson")
class NadarayaWatson(Indicator):
    """
    Nadaraya-Watson Envelope - Конверт Надарая-Ватсона.

    Непараметрическая регрессия с ядерным сглаживанием.

    Формула:
        NW(x) = sum(K(x - x_i) * y_i) / sum(K(x - x_i))
        где K - ядро (обычно Gaussian)

    Параметры:
        window (int): Период окна (по умолчанию 50)
        bandwidth (float): Ширина ядра (по умолчанию 8.0)
        column (str): Колонка для расчёта (по умолчанию 'close')

    Выходные колонки:
        NW_smooth: Сглаженная линия

    Применение:
        - Адаптивное сглаживание (лучше чем простая MA)
        - Поддержка/сопротивление
        - Определение тренда

    Example:
        >>> nw = NadarayaWatson(window=50, bandwidth=8.0)
        >>> result = nw.calculate(data)
    """

    def __init__(self, window: int = 50, bandwidth: float = 8.0, column: str = "close"):
        """
        Инициализация Nadaraya-Watson.

        Args:
            window: Период окна
            bandwidth: Ширина ядра
            column: Колонка для расчёта
        """
        super().__init__(window=window, bandwidth=bandwidth, column=column)

    def validate_params(self) -> None:
        """Валидация параметров."""
        window = self.params.get("window", 50)
        bandwidth = self.params.get("bandwidth", 8.0)

        if not isinstance(window, int) or window < 2:
            raise ValueError(f"window должен быть >= 2, получено: {window}")
        if bandwidth <= 0:
            raise ValueError(
                f"bandwidth должен быть положительным, получено: {bandwidth}"
            )

    def get_required_columns(self) -> List[str]:
        """Необходимые колонки."""
        return [self.params.get("column", "close")]

    def get_lookback_period(self) -> int:
        """Период разогрева."""
        return self.params.get("window", 50)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать Nadaraya-Watson Envelope.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с колонкой NW_smooth
        """
        self._validate_data(data)

        window = self.params["window"]
        bandwidth = self.params["bandwidth"]
        column = self.params["column"]

        def gaussian_kernel(x, bandwidth):
            """Гауссово ядро."""
            return np.exp(-0.5 * (x / bandwidth) ** 2)

        def nadaraya_watson(prices):
            """Рассчитать NW для окна."""
            n = len(prices)
            if n < 2:
                return np.nan

            # Текущая точка - последняя в окне
            # Рассчитываем веса для всех точек
            weights = np.zeros(n)
            for i in range(n):
                distance = n - 1 - i  # Расстояние до текущей точки
                weights[i] = gaussian_kernel(distance, bandwidth)

            # Нормализация весов
            weights_sum = weights.sum()
            if weights_sum > 0:
                weights /= weights_sum

            # Взвешенная сумма
            nw_value = (weights * prices).sum()

            return nw_value

        # Применяем rolling
        nw_smooth = (
            data[column]
            .rolling(window=window, min_periods=window)
            .apply(nadaraya_watson, raw=True)
        )

        result = pd.DataFrame({"NW_smooth": nw_smooth}, index=data.index)

        return result
