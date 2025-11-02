"""
Автоматическое исправление ошибок в данных.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataCorrector:
    """
    Автоматическое исправление распространённых ошибок в OHLCV данных.
    """

    def __init__(self) -> None:
        """Инициализировать корректор."""
        self.corrections_log: list[dict[str, Any]] = []

    def apply_all_corrections(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Применить все исправления.

        Args:
            data: Входные данные

        Returns:
            Исправленные данные
        """
        if data.empty:
            return data

        data = data.copy()
        self.corrections_log = []

        data = self.fix_negative_values(data)
        data = self.fix_inverted_ohlc(data)
        data = self.fix_decimal_errors(data)

        if self.corrections_log:
            logger.info(f"Applied {len(self.corrections_log)} corrections")

        return data

    def fix_negative_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Заменить отрицательные значения на NaN."""
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in data.columns:
                negative_mask = data[col] < 0
                if negative_mask.any():
                    count = negative_mask.sum()
                    self.corrections_log.append(
                        {
                            "type": "negative_value",
                            "column": col,
                            "count": count,
                        }
                    )
                    data.loc[negative_mask, col] = np.nan
                    logger.warning(f"Fixed {count} negative values in {col}")

        if "volume" in data.columns:
            negative_mask = data["volume"] < 0
            if negative_mask.any():
                count = negative_mask.sum()
                self.corrections_log.append({"type": "negative_volume", "count": count})
                data.loc[negative_mask, "volume"] = 0
                logger.warning(f"Fixed {count} negative volumes")

        return data

    def fix_inverted_ohlc(self, data: pd.DataFrame) -> pd.DataFrame:
        """Исправить инвертированные OHLC (high < low)."""
        if not all(col in data.columns for col in ["high", "low"]):
            return data

        inverted_mask = data["high"] < data["low"]
        if inverted_mask.any():
            count = inverted_mask.sum()
            self.corrections_log.append({"type": "inverted_ohlc", "count": count})

            # Swap high and low
            data.loc[inverted_mask, ["high", "low"]] = data.loc[
                inverted_mask, ["low", "high"]
            ].values

            logger.warning(f"Fixed {count} inverted OHLC rows")

        return data

    def fix_decimal_errors(self, data: pd.DataFrame) -> pd.DataFrame:
        """Детектировать и исправить десятичные ошибки (10x, 100x, 1000x)."""
        price_cols = ["open", "high", "low", "close"]

        for col in price_cols:
            if col not in data.columns:
                continue

            # Вычислить rolling median для reference
            rolling_median = data[col].rolling(window=20, min_periods=1).median()

            # Детектировать outliers
            ratio = data[col] / rolling_median

            # 10x error
            mask_10x = (ratio > 8) & (ratio < 12)
            if mask_10x.any():
                count = mask_10x.sum()
                data.loc[mask_10x, col] = data.loc[mask_10x, col] / 10
                self.corrections_log.append(
                    {"type": "decimal_10x", "column": col, "count": count}
                )
                logger.warning(f"Fixed {count} 10x errors in {col}")

            # 100x error
            mask_100x = (ratio > 80) & (ratio < 120)
            if mask_100x.any():
                count = mask_100x.sum()
                data.loc[mask_100x, col] = data.loc[mask_100x, col] / 100
                self.corrections_log.append(
                    {"type": "decimal_100x", "column": col, "count": count}
                )
                logger.warning(f"Fixed {count} 100x errors in {col}")

        return data

    def get_corrections_report(self) -> dict[str, Any]:
        """
        Получить отчёт о всех исправлениях.

        Returns:
            Словарь с информацией об исправлениях
        """
        if not self.corrections_log:
            return {"total_corrections": 0, "corrections": []}

        return {
            "total_corrections": len(self.corrections_log),
            "corrections": self.corrections_log,
        }


class PriceConsistencyChecker:
    """
    Проверка консистентности цен между последовательными барами.
    """

    def __init__(self, max_change_pct: float = 20.0):
        """
        Инициализировать проверку.

        Args:
            max_change_pct: Максимальное допустимое изменение цены в %
        """
        self.max_change_pct = max_change_pct

    def check_max_change(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Проверить максимальное изменение цены между барами.

        Args:
            data: Входные данные

        Returns:
            DataFrame с информацией о подозрительных изменениях
        """
        if "close" not in data.columns or len(data) < 2:
            return pd.DataFrame()

        price_change_pct = data["close"].pct_change().abs() * 100
        suspicious = price_change_pct > self.max_change_pct

        if suspicious.any():
            suspicious_data = data[suspicious].copy()
            suspicious_data["price_change_pct"] = price_change_pct[suspicious]

            logger.warning(
                f"Found {len(suspicious_data)} bars with suspicious price changes "
                f"(>{self.max_change_pct}%)"
            )

            return suspicious_data[["timestamp", "close", "price_change_pct"]]

        return pd.DataFrame()

    def validate_consistency(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Общая проверка консистентности.

        Args:
            data: Входные данные

        Returns:
            Отчёт о консистентности
        """
        issues = []

        # Проверка OHLC relationships
        if all(col in data.columns for col in ["open", "high", "low", "close"]):
            invalid_high = data["high"] < data[["open", "close"]].max(axis=1)
            invalid_low = data["low"] > data[["open", "close"]].min(axis=1)

            if invalid_high.any():
                issues.append({"type": "invalid_high", "count": invalid_high.sum()})

            if invalid_low.any():
                issues.append({"type": "invalid_low", "count": invalid_low.sum()})

        # Проверка gaps
        suspicious_changes = self.check_max_change(data)
        if not suspicious_changes.empty:
            issues.append(
                {
                    "type": "suspicious_price_change",
                    "count": len(suspicious_changes),
                }
            )

        return {
            "is_consistent": len(issues) == 0,
            "issues": issues,
            "total_issues": len(issues),
        }
