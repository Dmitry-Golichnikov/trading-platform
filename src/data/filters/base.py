"""
Базовые классы для фильтров данных.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FilterStatistics:
    """Статистика применения фильтра."""

    rows_before: int = 0
    rows_after: int = 0
    rows_filtered: int = 0
    filter_percentage: float = 0.0
    reasons: dict[str, int] = field(default_factory=dict)

    def calculate_stats(
        self, data_before: pd.DataFrame, data_after: pd.DataFrame
    ) -> None:
        """
        Вычислить статистику фильтрации.

        Args:
            data_before: Данные до фильтрации
            data_after: Данные после фильтрации
        """
        self.rows_before = len(data_before)
        self.rows_after = len(data_after)
        self.rows_filtered = self.rows_before - self.rows_after

        if self.rows_before > 0:
            self.filter_percentage = (self.rows_filtered / self.rows_before) * 100
        else:
            self.filter_percentage = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Преобразовать в словарь."""
        return {
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "rows_filtered": self.rows_filtered,
            "filter_percentage": round(self.filter_percentage, 2),
            "reasons": self.reasons,
        }


class DataFilter(ABC):
    """
    Базовый класс для фильтров данных.

    Все фильтры должны наследоваться от этого класса и реализовывать
    методы filter() и get_filter_mask().
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Инициализировать фильтр.

        Args:
            config: Конфигурация фильтра
        """
        self.config = config or {}
        self.stats = FilterStatistics()

    @abstractmethod
    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Применить фильтр к данным.

        Args:
            data: Входные данные

        Returns:
            Отфильтрованные данные
        """
        pass

    @abstractmethod
    def get_filter_mask(self, data: pd.DataFrame) -> pd.Series:
        """
        Получить маску фильтрации.

        Args:
            data: Входные данные

        Returns:
            Boolean Series, где True означает "оставить строку"
        """
        pass

    def log_statistics(self) -> None:
        """Логировать статистику фильтрации."""
        logger.info(
            f"{self.__class__.__name__}: filtered {self.stats.rows_filtered} rows "
            f"({self.stats.filter_percentage:.2f}%) from {self.stats.rows_before} total"
        )

        if self.stats.reasons:
            logger.info(f"  Reasons: {self.stats.reasons}")

    def get_statistics(self) -> dict[str, Any]:
        """
        Получить статистику в виде словаря.

        Returns:
            Словарь со статистикой
        """
        return {
            "filter_name": self.__class__.__name__,
            **self.stats.to_dict(),
        }
