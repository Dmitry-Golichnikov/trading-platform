"""
FDI (Fractal Dimension Index) - Индекс фрактальной размерности.
"""

from typing import List

import numpy as np
import pandas as pd

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register("fdi")
class FDI(Indicator):
    """
    FDI (Fractal Dimension Index) - Индекс фрактальной размерности.

    Измеряет сложность (фрактальность) ценового движения.

    Формула (упрощённая):
        - Рассчитываем длину пути цены за период
        - Нормализуем относительно прямой линии
        - FDI близко к 1 = тренд (прямая линия)
        - FDI близко к 2 = флэт (хаотичное движение)

    Параметры:
        window (int): Период окна (по умолчанию 20)
        column (str): Колонка для расчёта (по умолчанию 'close')

    Выходные колонки:
        FDI_{window}: Значение FDI (от 1 до 2)

    Применение:
        - FDI < 1.5 - сильный тренд
        - FDI > 1.5 - флэт, хаотичное движение
        - Фильтр для трендовых стратегий

    Example:
        >>> fdi = FDI(window=20)
        >>> result = fdi.calculate(data)
    """

    def __init__(self, window: int = 20, column: str = "close"):
        """
        Инициализация FDI.

        Args:
            window: Период окна
            column: Колонка для расчёта
        """
        super().__init__(window=window, column=column)

    def validate_params(self) -> None:
        """Валидация параметров."""
        window = self.params.get("window", 20)
        if not isinstance(window, int) or window < 2:
            raise ValueError(f"window должен быть >= 2, получено: {window}")

    def get_required_columns(self) -> List[str]:
        """Необходимые колонки."""
        return [self.params.get("column", "close")]

    def get_lookback_period(self) -> int:
        """Период разогрева."""
        return self.params.get("window", 20)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать FDI.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с колонкой FDI_{window}
        """
        self._validate_data(data)

        window = self.params["window"]
        column = self.params["column"]

        def calc_fdi(prices):
            """Рассчитать FDI для окна."""
            n = len(prices)
            if n < 2:
                return np.nan

            # Нормализуем цены к [0, 1]
            prices_norm = (prices - prices.min()) / (prices.max() - prices.min() + 1e-10)

            # Длина пути (сумма евклидовых расстояний между точками)
            path_length = 0
            for i in range(1, n):
                dx = 1  # Шаг по оси X (всегда 1)
                dy = prices_norm[i] - prices_norm[i - 1]
                path_length += np.sqrt(dx**2 + dy**2)

            # Прямая линия от начала до конца
            straight_line = np.sqrt((n - 1) ** 2 + (prices_norm[-1] - prices_norm[0]) ** 2)

            # FDI
            if straight_line > 0:
                fdi = 1 + (np.log(path_length) - np.log(straight_line)) / np.log(2 * (n - 1))
            else:
                fdi = 1.0

            # Ограничиваем диапазон [1, 2]
            fdi = np.clip(fdi, 1.0, 2.0)

            return fdi

        # Применяем rolling
        fdi_values = data[column].rolling(window=window, min_periods=window).apply(calc_fdi, raw=True)

        col_name = f"FDI_{window}"
        result = pd.DataFrame({col_name: fdi_values}, index=data.index)

        return result
