"""
Валидация качества данных.
"""

import logging

import numpy as np
import pandas as pd

from src.data.validators.schema import ValidationResult

logger = logging.getLogger(__name__)


class QualityValidator:
    """Валидатор качества данных."""

    def detect_price_anomalies(self, data: pd.DataFrame, threshold: float = 3.0) -> ValidationResult:
        """Детекция аномалий в ценах (z-score)."""
        result = ValidationResult(is_valid=True)

        # Вычислить returns
        returns = data["close"].pct_change()

        # Z-score
        z_scores = np.abs((returns - returns.mean()) / returns.std())
        anomalies = (z_scores > threshold).sum()

        if anomalies > 0:
            result.add_warning(f"Found {anomalies} price anomalies (z-score > {threshold})")

        result.statistics["anomalies"] = int(anomalies)
        result.statistics["max_z_score"] = float(z_scores.max())

        return result

    def check_volume_sanity(self, data: pd.DataFrame) -> ValidationResult:
        """Проверка объёмов."""
        result = ValidationResult(is_valid=True)

        if "volume" not in data.columns:
            result.add_error("Missing 'volume' column")
            return result

        # Проверить на отрицательные объёмы
        negative = (data["volume"] < 0).sum()
        if negative > 0:
            result.add_error(f"Found {negative} negative volumes")

        # Проверить на нулевые объёмы
        zero = (data["volume"] == 0).sum()
        if zero > 0:
            result.add_warning(f"Found {zero} zero volumes")

        result.statistics["negative_volumes"] = int(negative)
        result.statistics["zero_volumes"] = int(zero)

        return result

    def check_spread(self, data: pd.DataFrame, max_spread_pct: float = 10.0) -> ValidationResult:
        """Проверка spread (high-low)."""
        result = ValidationResult(is_valid=True)

        spread_pct = (data["high"] - data["low"]) / data["low"] * 100
        large_spreads = (spread_pct > max_spread_pct).sum()

        if large_spreads > 0:
            result.add_warning(f"Found {large_spreads} bars with spread > {max_spread_pct}%")

        result.statistics["large_spreads"] = int(large_spreads)
        result.statistics["max_spread_pct"] = float(spread_pct.max())

        return result

    def generate_quality_report(self, data: pd.DataFrame) -> dict:
        """Генерация отчёта о качестве данных."""
        report = {
            "total_bars": len(data),
            "date_range": (data["timestamp"].min(), data["timestamp"].max()),
            "price_stats": {
                "min": float(data["close"].min()),
                "max": float(data["close"].max()),
                "mean": float(data["close"].mean()),
                "std": float(data["close"].std()),
            },
            "volume_stats": {
                "min": int(data["volume"].min()),
                "max": int(data["volume"].max()),
                "mean": float(data["volume"].mean()),
            },
        }

        # Добавить результаты проверок
        report["anomalies"] = self.detect_price_anomalies(data).statistics
        report["volume_issues"] = self.check_volume_sanity(data).statistics
        report["spread_issues"] = self.check_spread(data).statistics

        return report
