"""
Pivot Points - Точки разворота.
"""

from typing import List

import pandas as pd

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register("pivot_points")
class PivotPoints(Indicator):
    """
    Pivot Points - Точки разворота.

    Формулы (Classic):
        Pivot = (High + Low + Close) / 3
        R1 = 2*Pivot - Low
        R2 = Pivot + (High - Low)
        R3 = High + 2*(Pivot - Low)
        S1 = 2*Pivot - High
        S2 = Pivot - (High - Low)
        S3 = Low - 2*(High - Pivot)

    Fibonacci:
        R1 = Pivot + 0.382 * (High - Low)
        R2 = Pivot + 0.618 * (High - Low)
        R3 = Pivot + 1.000 * (High - Low)
        S1 = Pivot - 0.382 * (High - Low)
        S2 = Pivot - 0.618 * (High - Low)
        S3 = Pivot - 1.000 * (High - Low)

    Camarilla:
        R1 = Close + (High - Low) * 1.1/12
        R2 = Close + (High - Low) * 1.1/6
        R3 = Close + (High - Low) * 1.1/4
        R4 = Close + (High - Low) * 1.1/2
        S1 = Close - (High - Low) * 1.1/12
        S2 = Close - (High - Low) * 1.1/6
        S3 = Close - (High - Low) * 1.1/4
        S4 = Close - (High - Low) * 1.1/2

    Параметры:
        method (str): Метод расчёта ('classic', 'fibonacci', 'camarilla')
        window (int): Период для расчёта (по умолчанию 1 - предыдущий бар)

    Выходные колонки:
        PP: Pivot Point
        R1, R2, R3: Уровни сопротивления
        S1, S2, S3: Уровни поддержки
        (R4, S4 для Camarilla)

    Применение:
        - Уровни поддержки/сопротивления
        - Цели для входа/выхода
        - Определение диапазона торговли

    Example:
        >>> pivot = PivotPoints(method='classic')
        >>> result = pivot.calculate(data)
    """

    def __init__(self, method: str = "classic", window: int = 1):
        """
        Инициализация Pivot Points.

        Args:
            method: Метод расчёта ('classic', 'fibonacci', 'camarilla')
            window: Период для расчёта
        """
        super().__init__(method=method, window=window)

    def validate_params(self) -> None:
        """Валидация параметров."""
        method = self.params.get("method", "classic")
        window = self.params.get("window", 1)

        if method not in ["classic", "fibonacci", "camarilla"]:
            raise ValueError(
                f"method должен быть 'classic', 'fibonacci' или "
                f"'camarilla', получено: {method}"
            )
        if not isinstance(window, int) or window < 1:
            raise ValueError(
                f"window должен быть положительным целым числом, получено: {window}"
            )

    def get_required_columns(self) -> List[str]:
        """Необходимые колонки."""
        return ["high", "low", "close"]

    def get_lookback_period(self) -> int:
        """Период разогрева."""
        return self.params.get("window", 1)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать Pivot Points.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с колонками PP, R1, R2, R3, S1, S2, S3 (и R4, S4 для Camarilla)
        """
        self._validate_data(data)

        method = self.params["method"]
        window = self.params["window"]

        # Используем данные за предыдущий период
        high = data["high"].shift(window)
        low = data["low"].shift(window)
        close = data["close"].shift(window)

        if method == "classic":
            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            r3 = high + 2 * (pivot - low)
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            s3 = low - 2 * (high - pivot)

            result = pd.DataFrame(
                {
                    "PP": pivot,
                    "R1": r1,
                    "R2": r2,
                    "R3": r3,
                    "S1": s1,
                    "S2": s2,
                    "S3": s3,
                },
                index=data.index,
            )

        elif method == "fibonacci":
            pivot = (high + low + close) / 3
            range_hl = high - low

            r1 = pivot + 0.382 * range_hl
            r2 = pivot + 0.618 * range_hl
            r3 = pivot + 1.000 * range_hl
            s1 = pivot - 0.382 * range_hl
            s2 = pivot - 0.618 * range_hl
            s3 = pivot - 1.000 * range_hl

            result = pd.DataFrame(
                {
                    "PP": pivot,
                    "R1": r1,
                    "R2": r2,
                    "R3": r3,
                    "S1": s1,
                    "S2": s2,
                    "S3": s3,
                },
                index=data.index,
            )

        elif method == "camarilla":
            range_hl = high - low

            r1 = close + range_hl * 1.1 / 12
            r2 = close + range_hl * 1.1 / 6
            r3 = close + range_hl * 1.1 / 4
            r4 = close + range_hl * 1.1 / 2

            s1 = close - range_hl * 1.1 / 12
            s2 = close - range_hl * 1.1 / 6
            s3 = close - range_hl * 1.1 / 4
            s4 = close - range_hl * 1.1 / 2

            # Pivot для Camarilla - просто close
            pivot = close

            result = pd.DataFrame(
                {
                    "PP": pivot,
                    "R1": r1,
                    "R2": r2,
                    "R3": r3,
                    "R4": r4,
                    "S1": s1,
                    "S2": s2,
                    "S3": s3,
                    "S4": s4,
                },
                index=data.index,
            )

        return result
