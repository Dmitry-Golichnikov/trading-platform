"""
Фильтры ликвидности и торговых сессий.
"""

import logging
from datetime import time
from typing import Any, Literal

import pandas as pd

from src.data.filters.base import DataFilter

logger = logging.getLogger(__name__)


class LiquidityFilter(DataFilter):
    """
    Фильтр периодов с низкой ликвидностью.

    Удаляет или маркирует бары с недостаточным объёмом торгов,
    слишком большим spread или низким turnover.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Инициализировать фильтр.

        Args:
            config: Конфигурация:
                - min_volume: минимальный объём за бар
                - min_turnover: минимальный turnover (volume * price)
                - max_spread_pct: максимальный spread в %
                - action: 'remove' | 'mark'
        """
        super().__init__(config)
        self.min_volume: int = self.config.get("min_volume", 1000)
        self.min_turnover: float = self.config.get("min_turnover", 0)
        self.max_spread_pct: float = self.config.get("max_spread_pct", 5.0)
        self.action: Literal["remove", "mark"] = self.config.get("action", "remove")

    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """Применить фильтр ликвидности."""
        if data.empty:
            return data

        data_before = data.copy()
        mask = self.get_filter_mask(data)

        if self.action == "remove":
            data = data[mask].reset_index(drop=True)
        elif self.action == "mark":
            data = data.copy()
            data["low_liquidity"] = ~mask

        self.stats.calculate_stats(data_before, data)
        self.stats.reasons["low_liquidity"] = (~mask).sum()
        self.log_statistics()

        return data

    def get_filter_mask(self, data: pd.DataFrame) -> pd.Series:
        """Получить маску ликвидности."""
        # Проверка минимального объёма
        mask_volume = data["volume"] >= self.min_volume

        # Проверка turnover
        if self.min_turnover > 0:
            turnover = data["volume"] * data["close"]
            mask_turnover = turnover >= self.min_turnover
        else:
            mask_turnover = pd.Series([True] * len(data), index=data.index)

        # Проверка spread
        spread_pct = ((data["high"] - data["low"]) / data["close"]) * 100
        mask_spread = spread_pct <= self.max_spread_pct

        return mask_volume & mask_turnover & mask_spread


class TradingHoursFilter(DataFilter):
    """
    Фильтр данных вне торговых часов.

    Удаляет или маркирует бары, которые находятся вне основной
    торговой сессии биржи.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Инициализировать фильтр.

        Args:
            config: Конфигурация:
                - start_time: начало торговой сессии (формат "HH:MM")
                - end_time: конец торговой сессии (формат "HH:MM")
                - include_premarket: включить pre-market
                - include_afterhours: включить after-hours
                - action: 'remove' | 'mark'
        """
        super().__init__(config)
        self.start_time = self._parse_time(self.config.get("start_time", "10:00"))
        self.end_time = self._parse_time(self.config.get("end_time", "18:40"))
        self.include_premarket: bool = self.config.get("include_premarket", False)
        self.include_afterhours: bool = self.config.get("include_afterhours", False)
        self.action: Literal["remove", "mark"] = self.config.get("action", "remove")

    def _parse_time(self, time_str: str) -> time:
        """Парсинг времени из строки."""
        parts = time_str.split(":")
        return time(hour=int(parts[0]), minute=int(parts[1]))

    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """Применить фильтр торговых часов."""
        if data.empty:
            return data

        # Проверка наличия timestamp
        if "timestamp" not in data.columns:
            logger.warning("No timestamp column, skipping trading hours filter")
            return data

        data_before = data.copy()
        mask = self.get_filter_mask(data)

        if self.action == "remove":
            data = data[mask].reset_index(drop=True)
        elif self.action == "mark":
            data = data.copy()
            data["outside_trading_hours"] = ~mask

        self.stats.calculate_stats(data_before, data)
        self.stats.reasons["outside_trading_hours"] = (~mask).sum()
        self.log_statistics()

        return data

    def get_filter_mask(self, data: pd.DataFrame) -> pd.Series:
        """Получить маску торговых часов."""
        if "timestamp" not in data.columns:
            return pd.Series([True] * len(data), index=data.index)

        # Преобразовать timestamp к datetime если нужно
        timestamps = pd.to_datetime(data["timestamp"])

        # Извлечь время дня
        time_of_day = timestamps.dt.time

        # Основная торговая сессия
        mask = (time_of_day >= self.start_time) & (time_of_day <= self.end_time)

        # TODO: Добавить поддержку pre-market и after-hours когда потребуется
        if self.include_premarket or self.include_afterhours:
            logger.debug("Pre-market/after-hours filtering not yet implemented")

        return mask
