"""
Валидация схемы данных.
"""

import logging
from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Результат валидации.

    Attributes:
        is_valid: Прошла ли валидация успешно
        errors: Список критических ошибок
        warnings: Список предупреждений
        statistics: Дополнительная статистика
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    statistics: dict = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        """Добавить ошибку."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Добавить предупреждение."""
        self.warnings.append(message)


class SchemaValidator:
    """Валидатор схемы DataFrame."""

    REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

    def validate_columns(
        self, data: pd.DataFrame, required_columns: Sequence[str] | None = None
    ) -> ValidationResult:
        """Проверить наличие обязательных колонок."""
        result = ValidationResult(is_valid=True)

        if required_columns is None:
            required_columns = self.REQUIRED_COLUMNS

        missing = set(required_columns) - set(data.columns)
        if missing:
            result.add_error(f"Missing required columns: {missing}")

        result.statistics["total_columns"] = len(data.columns)
        result.statistics["required_columns"] = len(required_columns)

        return result

    def validate_dtypes(self, data: pd.DataFrame) -> ValidationResult:
        """Проверить типы данных колонок."""
        result = ValidationResult(is_valid=True)

        # timestamp должен быть datetime
        if "timestamp" in data.columns:
            if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
                result.add_error("'timestamp' must be datetime type")

        # OHLC должны быть numeric
        for col in ["open", "high", "low", "close"]:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    result.add_error(f"'{col}' must be numeric type")

        # volume должен быть numeric
        if "volume" in data.columns:
            if not pd.api.types.is_numeric_dtype(data["volume"]):
                result.add_error("'volume' must be numeric type")

        return result

    def validate_ohlc_relationships(self, data: pd.DataFrame) -> ValidationResult:
        """Проверить соотношения OHLC."""
        result = ValidationResult(is_valid=True)

        try:
            # high >= max(open, close)
            invalid_high = data["high"] < data[["open", "close"]].max(axis=1)
            if invalid_high.any():
                count = invalid_high.sum()
                result.add_error(f"{count} bars have high < max(open, close)")

            # low <= min(open, close)
            invalid_low = data["low"] > data[["open", "close"]].min(axis=1)
            if invalid_low.any():
                count = invalid_low.sum()
                result.add_error(f"{count} bars have low > min(open, close)")

            result.statistics["invalid_bars"] = invalid_high.sum() + invalid_low.sum()

        except Exception as e:
            result.add_error(f"Failed to validate OHLC relationships: {e}")

        return result

    def validate_timestamp_format(self, data: pd.DataFrame) -> ValidationResult:
        """Проверить формат timestamp."""
        result = ValidationResult(is_valid=True)

        if "timestamp" not in data.columns:
            result.add_error("Missing 'timestamp' column")
            return result

        timestamps = pd.to_datetime(data["timestamp"])

        # Проверить timezone awareness
        if timestamps.dt.tz is None:
            result.add_warning("Timestamps are not timezone-aware")

        # Проверить на NaT
        nat_count = timestamps.isna().sum()
        if nat_count > 0:
            result.add_error(f"{nat_count} timestamps are NaT")

        result.statistics["nat_count"] = int(nat_count)

        return result

    def validate_all(self, data: pd.DataFrame) -> ValidationResult:
        """Выполнить все проверки схемы."""
        result = ValidationResult(is_valid=True)

        checks = [
            self.validate_columns(data),
            self.validate_dtypes(data),
            self.validate_ohlc_relationships(data),
            self.validate_timestamp_format(data),
        ]

        for check in checks:
            if not check.is_valid:
                result.is_valid = False
            result.errors.extend(check.errors)
            result.warnings.extend(check.warnings)
            result.statistics.update(check.statistics)

        return result
