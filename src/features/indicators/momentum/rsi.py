"""
RSI (Relative Strength Index) - Индекс относительной силы.
"""

from typing import List

import pandas as pd

from src.features.indicators.base import Indicator
from src.features.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register("rsi")
class RSI(Indicator):
    """
    RSI (Relative Strength Index) - Индекс относительной силы.

    Формула:
        RS = Average Gain / Average Loss
        RSI = 100 - (100 / (1 + RS))

    где Average Gain и Average Loss - экспоненциальные средние приростов и убытков.

    Параметры:
        window (int): Период окна (по умолчанию 14)
        column (str): Колонка для расчёта (по умолчанию 'close')

    Выходные колонки:
        RSI_{window}: Значение RSI (0-100)

    Применение:
        - RSI > 70 - перекупленность (возможна коррекция)
        - RSI < 30 - перепроданность (возможен отскок)
        - Дивергенции с ценой - сигнал разворота

    Example:
        >>> rsi = RSI(window=14)
        >>> result = rsi.calculate(data)
        >>> # result содержит колонку 'RSI_14'
    """

    def __init__(self, window: int = 14, column: str = "close"):
        """
        Инициализация RSI.

        Args:
            window: Период окна
            column: Колонка для расчёта
        """
        super().__init__(window=window, column=column)

    def validate_params(self) -> None:
        """Валидация параметров."""
        window = self.params.get("window", 14)
        if not isinstance(window, int) or window < 1:
            raise ValueError(f"window должен быть положительным целым числом, получено: {window}")

    def get_required_columns(self) -> List[str]:
        """Необходимые колонки."""
        return [self.params.get("column", "close")]

    def get_lookback_period(self) -> int:
        """Период разогрева."""
        # Нужен window + 1 для расчёта изменений
        return self.params.get("window", 14) + 1

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать RSI.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с колонкой RSI_{window}
        """
        self._validate_data(data)

        window = self.params["window"]
        column = self.params["column"]

        # Рассчитываем изменения цены
        delta = data[column].diff()

        # Разделяем на приросты и убытки
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # Экспоненциальные средние приростов и убытков
        avg_gain = gain.ewm(span=window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(span=window, adjust=False, min_periods=window).mean()

        # Относительная сила
        rs = avg_gain / avg_loss

        # RSI
        rsi = 100 - (100 / (1 + rs))

        col_name = f"RSI_{window}"
        result = pd.DataFrame({col_name: rsi}, index=data.index)

        return result
