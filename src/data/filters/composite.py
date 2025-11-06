"""
Композиция и комбинирование фильтров.
"""

import logging
from typing import Any, Callable

import pandas as pd

from src.data.filters.base import DataFilter, FilterStatistics

logger = logging.getLogger(__name__)


class FilterPipeline:
    """
    Последовательное применение нескольких фильтров.

    Позволяет создавать цепочки фильтров и применять их к данным,
    собирая статистику на каждом этапе.
    """

    def __init__(self, filters: list[DataFilter]):
        """
        Инициализировать пайплайн.

        Args:
            filters: Список фильтров для последовательного применения
        """
        self.filters = filters
        self.statistics: list[dict[str, Any]] = []

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Применить все фильтры последовательно.

        Args:
            data: Входные данные

        Returns:
            Отфильтрованные данные
        """
        result = data.copy()
        self.statistics = []

        logger.info(f"Applying filter pipeline with {len(self.filters)} filters")
        initial_rows = len(result)

        for i, filter_obj in enumerate(self.filters, 1):
            rows_before = len(result)
            result = filter_obj.filter(result)
            rows_after = len(result)

            stats = filter_obj.get_statistics()
            self.statistics.append(stats)

            logger.info(
                f"  Step {i}/{len(self.filters)}: {filter_obj.__class__.__name__} "
                f"filtered {rows_before - rows_after} rows "
                f"({(rows_before - rows_after) / rows_before * 100:.2f}%)"
            )

        final_rows = len(result)
        total_filtered = initial_rows - final_rows
        logger.info(
            f"Pipeline complete: {total_filtered} rows filtered "
            f"({total_filtered / initial_rows * 100:.2f}% of {initial_rows})"
        )

        return result

    def get_statistics(self) -> dict[str, Any]:
        """
        Получить совокупную статистику всех фильтров.

        Returns:
            Словарь со статистикой каждого фильтра
        """
        return {
            "pipeline_steps": len(self.filters),
            "filters": self.statistics,
            "total_rows_before": (self.statistics[0]["rows_before"] if self.statistics else 0),
            "total_rows_after": (self.statistics[-1]["rows_after"] if self.statistics else 0),
        }


class ConditionalFilter:
    """
    Применение фильтра по условию.

    Фильтр применяется только к строкам, удовлетворяющим условию.
    """

    def __init__(
        self,
        condition: Callable[[pd.DataFrame], pd.Series],
        filter_obj: DataFilter,
    ):
        """
        Инициализировать условный фильтр.

        Args:
            condition: Функция, возвращающая boolean Series
            filter_obj: Фильтр для применения
        """
        self.condition = condition
        self.filter_obj = filter_obj
        self.stats = FilterStatistics()

    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Применить фильтр условно.

        Args:
            data: Входные данные

        Returns:
            Отфильтрованные данные
        """
        if data.empty:
            return data

        data_before = data.copy()

        # Применить условие
        condition_mask = self.condition(data)
        rows_matching = condition_mask.sum()

        logger.info(
            f"Conditional filter: {rows_matching} rows match condition " f"({rows_matching / len(data) * 100:.2f}%)"
        )

        if rows_matching == 0:
            logger.info("No rows match condition, skipping filter")
            return data

        # Применить фильтр только к matching rows
        matching_data = data[condition_mask].copy()
        non_matching_data = data[~condition_mask].copy()

        filtered_matching = self.filter_obj.filter(matching_data)

        # Объединить обратно
        result = pd.concat([filtered_matching, non_matching_data], ignore_index=True)

        # Сортировать если есть timestamp
        if "timestamp" in result.columns:
            result = result.sort_values("timestamp").reset_index(drop=True)

        self.stats.calculate_stats(data_before, result)
        self.stats.reasons.update(self.filter_obj.stats.reasons)

        logger.info(
            f"Conditional filter removed {self.stats.rows_filtered} rows " f"({self.stats.filter_percentage:.2f}%)"
        )

        return result

    def get_statistics(self) -> dict[str, Any]:
        """Получить статистику."""
        return {
            "filter_name": f"Conditional[{self.filter_obj.__class__.__name__}]",
            **self.stats.to_dict(),
        }


class ParallelFilterGroup:
    """
    Применение нескольких фильтров параллельно с объединением результатов.

    Поддерживает AND и OR логику объединения масок фильтров.
    """

    def __init__(self, filters: list[DataFilter], logic: str = "AND"):
        """
        Инициализировать группу фильтров.

        Args:
            filters: Список фильтров
            logic: Логика объединения - 'AND' или 'OR'
        """
        self.filters = filters
        self.logic = logic.upper()
        self.stats = FilterStatistics()

        if self.logic not in ["AND", "OR"]:
            raise ValueError(f"Logic must be 'AND' or 'OR', got {logic}")

    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Применить фильтры параллельно.

        Args:
            data: Входные данные

        Returns:
            Отфильтрованные данные
        """
        if data.empty:
            return data

        data_before = data.copy()

        # Получить маски от всех фильтров
        masks = []
        for filter_obj in self.filters:
            mask = filter_obj.get_filter_mask(data)
            masks.append(mask)
            logger.debug(f"  {filter_obj.__class__.__name__}: " f"{(~mask).sum()} rows would be filtered")

        # Объединить маски
        if self.logic == "AND":
            combined_mask = pd.Series([True] * len(data), index=data.index)
            for mask in masks:
                combined_mask = combined_mask & mask
        else:  # OR
            combined_mask = pd.Series([False] * len(data), index=data.index)
            for mask in masks:
                combined_mask = combined_mask | mask

        # Применить результирующую маску
        result = data[combined_mask].reset_index(drop=True)

        self.stats.calculate_stats(data_before, result)
        self.stats.reasons["parallel_filter_" + self.logic.lower()] = (~combined_mask).sum()

        logger.info(
            f"Parallel filter ({self.logic}): filtered {self.stats.rows_filtered} rows "
            f"({self.stats.filter_percentage:.2f}%)"
        )

        return result

    def get_statistics(self) -> dict[str, Any]:
        """Получить статистику."""
        return {
            "filter_name": f"ParallelGroup[{self.logic}]",
            "num_filters": len(self.filters),
            **self.stats.to_dict(),
        }
