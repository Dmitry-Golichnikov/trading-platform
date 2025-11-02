"""
Обработка дубликатов.
"""

import logging
from typing import Any, Literal

import pandas as pd

from src.common.exceptions import PreprocessingError

logger = logging.getLogger(__name__)


class DuplicateHandler:
    """
    Обработка дубликатов timestamp.

    Поддерживает различные стратегии разрешения конфликтов.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Инициализировать обработчик.

        Args:
            config: Конфигурация:
                - strategy: 'first' | 'last' | 'mean' | 'validate'
                - subset: колонки для проверки дубликатов (default: ['timestamp'])
        """
        self.config = config or {}
        self.strategy: Literal["first", "last", "mean", "validate"] = self.config.get(
            "strategy", "last"
        )
        self.subset: list[str] = self.config.get("subset", ["timestamp"])

    def handle(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Обработать дубликаты.

        Args:
            data: Входные данные

        Returns:
            Данные без дубликатов

        Raises:
            PreprocessingError: Если strategy='validate' и найдены дубликаты
        """
        if data.empty:
            return data

        duplicates = data.duplicated(subset=self.subset, keep=False)
        num_duplicates = duplicates.sum()

        if num_duplicates == 0:
            logger.info("No duplicates found")
            return data

        logger.info(f"Found {num_duplicates} duplicate rows")

        if self.strategy == "validate":
            raise PreprocessingError(
                f"Found {num_duplicates} duplicate rows (validation failed)"
            )
        elif self.strategy == "first":
            return self._keep_first(data)
        elif self.strategy == "last":
            return self._keep_last(data)
        elif self.strategy == "mean":
            return self._aggregate_mean(data)
        else:
            raise PreprocessingError(f"Unknown strategy: {self.strategy}")

    def _keep_first(self, data: pd.DataFrame) -> pd.DataFrame:
        """Оставить первое вхождение."""
        return data.drop_duplicates(subset=self.subset, keep="first").reset_index(
            drop=True
        )

    def _keep_last(self, data: pd.DataFrame) -> pd.DataFrame:
        """Оставить последнее вхождение."""
        return data.drop_duplicates(subset=self.subset, keep="last").reset_index(
            drop=True
        )

    def _aggregate_mean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Усреднить дубликаты."""
        # Группировать по subset и агрегировать
        agg_dict = {
            "open": "mean",
            "high": "max",
            "low": "min",
            "close": "mean",
            "volume": "sum",
        }

        # Оставить только существующие колонки
        agg_dict = {k: v for k, v in agg_dict.items() if k in data.columns}

        # Добавить остальные колонки (взять первое значение)
        for col in data.columns:
            if col not in agg_dict and col not in self.subset:
                agg_dict[col] = "first"

        result = data.groupby(self.subset, as_index=False).agg(agg_dict)

        # Сортировать если есть timestamp
        if "timestamp" in result.columns:
            result = result.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"Aggregated duplicates: {len(data)} -> {len(result)} rows")
        return result
