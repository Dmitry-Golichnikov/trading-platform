"""
Валидация целостности данных.
"""

import logging

import pandas as pd

from src.data.validators.schema import ValidationResult

logger = logging.getLogger(__name__)


class IntegrityValidator:
    """Валидатор целостности данных."""

    def check_duplicates(self, data: pd.DataFrame) -> ValidationResult:
        """Проверка на дубликаты timestamp."""
        result = ValidationResult(is_valid=True)

        if "timestamp" not in data.columns:
            result.add_error("Missing 'timestamp' column")
            return result

        duplicates = data["timestamp"].duplicated().sum()
        if duplicates > 0:
            result.add_error(f"Found {duplicates} duplicate timestamps")

        result.statistics["duplicates"] = int(duplicates)
        return result

    def check_monotonic_timestamps(self, data: pd.DataFrame) -> ValidationResult:
        """Проверка монотонности timestamp."""
        result = ValidationResult(is_valid=True)

        if "timestamp" not in data.columns:
            result.add_error("Missing 'timestamp' column")
            return result

        timestamps = pd.to_datetime(data["timestamp"])
        is_monotonic = timestamps.is_monotonic_increasing

        if not is_monotonic:
            result.add_error("Timestamps are not monotonic")

        result.statistics["is_monotonic"] = is_monotonic
        return result

    def check_missing_bars(
        self, data: pd.DataFrame, timeframe: str
    ) -> ValidationResult:
        """Проверка на пропуски баров."""
        result = ValidationResult(is_valid=True)

        if len(data) < 2:
            return result

        freq_map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "1h": "1h",
            "4h": "4h",
            "1d": "1D",
        }

        if timeframe not in freq_map:
            result.add_warning(f"Unknown timeframe: {timeframe}")
            return result

        timestamps = pd.to_datetime(data["timestamp"])
        freq = freq_map[timeframe]

        # Создать полный диапазон
        full_range = pd.date_range(
            start=timestamps.min(), end=timestamps.max(), freq=freq
        )

        missing_count = len(full_range) - len(timestamps)

        if missing_count > 0:
            result.add_warning(f"Found {missing_count} missing bars")

        result.statistics["expected_bars"] = len(full_range)
        result.statistics["actual_bars"] = len(timestamps)
        result.statistics["missing_bars"] = missing_count

        return result

    def check_completeness(self, data: pd.DataFrame) -> ValidationResult:
        """Проверка полноты данных."""
        result = ValidationResult(is_valid=True)

        for col in ["open", "high", "low", "close", "volume"]:
            if col not in data.columns:
                result.add_error(f"Missing column: {col}")
                continue

            null_count = data[col].isna().sum()
            if null_count > 0:
                result.add_error(f"Column '{col}' has {null_count} null values")

        return result

    def validate_all(
        self, data: pd.DataFrame, timeframe: str = "1m"
    ) -> ValidationResult:
        """Выполнить все проверки целостности."""
        result = ValidationResult(is_valid=True)

        checks = [
            self.check_duplicates(data),
            self.check_monotonic_timestamps(data),
            self.check_missing_bars(data, timeframe),
            self.check_completeness(data),
        ]

        for check in checks:
            if not check.is_valid:
                result.is_valid = False
            result.errors.extend(check.errors)
            result.warnings.extend(check.warnings)
            result.statistics.update(check.statistics)

        return result
