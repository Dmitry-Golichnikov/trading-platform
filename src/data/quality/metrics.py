"""
Метрики качества данных.
"""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class DataQualityMetrics:
    """
    Вычисление метрик качества OHLCV данных.

    Метрики:
    - Completeness: % присутствующих данных
    - Validity: % валидных записей
    - Consistency: консистентность OHLCV relationships
    - Uniqueness: отсутствие дубликатов
    """

    def __init__(self, expected_rows: int | None = None):
        """
        Инициализировать метрики.

        Args:
            expected_rows: Ожидаемое количество строк (для completeness)
        """
        self.expected_rows = expected_rows

    def calculate_completeness(self, data: pd.DataFrame) -> float:
        """
        Вычислить полноту данных.

        Args:
            data: Входные данные

        Returns:
            Процент полноты (0-100)
        """
        if data.empty:
            return 0.0

        total_cells = data.size
        missing_cells = data.isna().sum().sum()
        present_cells = total_cells - missing_cells

        completeness = (present_cells / total_cells) * 100

        logger.debug(f"Completeness: {completeness:.2f}% " f"({present_cells}/{total_cells} cells present)")

        return float(completeness)

    def calculate_validity(self, data: pd.DataFrame) -> float:
        """
        Вычислить валидность данных.

        Args:
            data: Входные данные

        Returns:
            Процент валидных записей (0-100)
        """
        if data.empty:
            return 0.0

        total_rows = len(data)
        valid_rows = 0

        # Проверки валидности
        checks = []

        # Неотрицательные цены
        if all(col in data.columns for col in ["open", "high", "low", "close"]):
            price_positive = (data["open"] >= 0) & (data["high"] >= 0) & (data["low"] >= 0) & (data["close"] >= 0)
            checks.append(price_positive)

        # Неотрицательный volume
        if "volume" in data.columns:
            volume_positive = data["volume"] >= 0
            checks.append(volume_positive)

        # OHLC relationships
        if all(col in data.columns for col in ["open", "high", "low", "close"]):
            ohlc_valid = (data["high"] >= data[["open", "close"]].max(axis=1)) & (
                data["low"] <= data[["open", "close"]].min(axis=1)
            )
            checks.append(ohlc_valid)

        # Объединить все проверки
        if checks:
            valid_mask = checks[0]
            for check in checks[1:]:
                valid_mask = valid_mask & check
            valid_rows = valid_mask.sum()
        else:
            valid_rows = total_rows

        validity = (valid_rows / total_rows) * 100

        logger.debug(f"Validity: {validity:.2f}% ({valid_rows}/{total_rows} valid rows)")

        return validity

    def calculate_consistency(self, data: pd.DataFrame) -> float:
        """
        Вычислить консистентность OHLCV relationships.

        Args:
            data: Входные данные

        Returns:
            Процент консистентных записей (0-100)
        """
        if data.empty:
            return 0.0

        if not all(col in data.columns for col in ["open", "high", "low", "close"]):
            logger.warning("Missing OHLC columns, cannot calculate consistency")
            return 100.0

        total_rows = len(data)

        # high >= max(open, close)
        high_valid = data["high"] >= data[["open", "close"]].max(axis=1)

        # low <= min(open, close)
        low_valid = data["low"] <= data[["open", "close"]].min(axis=1)

        # high >= low
        spread_valid = data["high"] >= data["low"]

        consistent = high_valid & low_valid & spread_valid
        consistent_rows = consistent.sum()

        consistency = (consistent_rows / total_rows) * 100

        logger.debug(f"Consistency: {consistency:.2f}% " f"({consistent_rows}/{total_rows} consistent rows)")

        return consistency

    def calculate_uniqueness(self, data: pd.DataFrame) -> float:
        """
        Вычислить уникальность (отсутствие дубликатов).

        Args:
            data: Входные данные

        Returns:
            Процент уникальных записей (0-100)
        """
        if data.empty:
            return 100.0

        total_rows = len(data)

        if "timestamp" in data.columns:
            unique_rows = data["timestamp"].nunique()
        else:
            unique_rows = len(data.drop_duplicates())

        uniqueness = (unique_rows / total_rows) * 100

        logger.debug(f"Uniqueness: {uniqueness:.2f}% ({unique_rows}/{total_rows} unique)")

        return uniqueness

    def calculate_quality_score(self, data: pd.DataFrame, weights: dict[str, float] | None = None) -> float:
        """
        Вычислить агрегированный score качества.

        Args:
            data: Входные данные
            weights: Веса метрик (default: равные веса)

        Returns:
            Общий score качества (0-100)
        """
        if weights is None:
            weights = {
                "completeness": 0.3,
                "validity": 0.3,
                "consistency": 0.2,
                "uniqueness": 0.2,
            }

        completeness = self.calculate_completeness(data)
        validity = self.calculate_validity(data)
        consistency = self.calculate_consistency(data)
        uniqueness = self.calculate_uniqueness(data)

        score = (
            completeness * weights["completeness"]
            + validity * weights["validity"]
            + consistency * weights["consistency"]
            + uniqueness * weights["uniqueness"]
        )

        logger.info(f"Overall quality score: {score:.2f}/100")

        return score

    def get_all_metrics(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Получить все метрики в виде словаря.

        Args:
            data: Входные данные

        Returns:
            Словарь с метриками
        """
        return {
            "completeness": round(self.calculate_completeness(data), 2),
            "validity": round(self.calculate_validity(data), 2),
            "consistency": round(self.calculate_consistency(data), 2),
            "uniqueness": round(self.calculate_uniqueness(data), 2),
            "quality_score": round(self.calculate_quality_score(data), 2),
            "total_rows": len(data),
            "total_columns": len(data.columns),
        }
