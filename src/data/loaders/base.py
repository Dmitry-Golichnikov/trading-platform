"""
Базовый интерфейс для загрузчиков данных.
"""

from datetime import datetime
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DataLoader(Protocol):
    """
    Интерфейс для загрузчиков OHLCV данных.

    Все загрузчики должны реализовывать этот протокол для обеспечения
    единообразного API при работе с различными источниками данных.
    """

    def load(
        self,
        ticker: str,
        from_date: datetime,
        to_date: datetime,
        timeframe: str = "1m",
        **kwargs: object,
    ) -> pd.DataFrame:
        """
        Загрузить OHLCV данные для указанного инструмента и периода.

        Args:
            ticker: Тикер инструмента (или FIGI)
            from_date: Начальная дата загрузки
            to_date: Конечная дата загрузки
            timeframe: Временной интервал ('1m', '5m', '15m', '1h', '4h', '1d')
            **kwargs: Дополнительные параметры, специфичные для загрузчика

        Returns:
            DataFrame с колонками:
                - timestamp: pd.DatetimeIndex с timezone
                - ticker: str
                - open: float
                - high: float
                - low: float
                - close: float
                - volume: int

        Raises:
            DataLoadError: При ошибке загрузки данных
            ValidationError: При невалидных данных
        """
        ...

    def get_available_tickers(self) -> list[str]:
        """
        Получить список доступных тикеров.

        Returns:
            Список тикеров/FIGI, доступных в данном источнике
        """
        ...
