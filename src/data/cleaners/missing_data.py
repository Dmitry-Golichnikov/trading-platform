"""
Обработка пропусков в данных.
"""

import logging
from typing import Any, Literal

import pandas as pd

from src.common.exceptions import PreprocessingError

logger = logging.getLogger(__name__)


class MissingDataHandler:
    """
    Обработка пропущенных данных (NaN values).

    Поддерживает различные стратегии заполнения пропусков
    с учётом специфики OHLCV данных.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Инициализировать обработчик.

        Args:
            config: Конфигурация, включающая параметры:
                - method: стратегия обработки пропусков
                  ('drop' | 'forward_fill' | 'backward_fill' |
                  'interpolate' | 'mean' | 'median')
                - max_gap: максимальный размер пропуска
                  для заполнения (в барах)
                - fill_volume_with_zero: заполнять volume нулями
                  вместо forward_fill
        """
        self.config = config or {}
        self.method: Literal[
            "drop",
            "forward_fill",
            "backward_fill",
            "interpolate",
            "mean",
            "median",
        ] = self.config.get("method", "forward_fill")
        self.max_gap: int = self.config.get("max_gap", 5)
        self.fill_volume_with_zero: bool = self.config.get(
            "fill_volume_with_zero", True
        )

    def handle(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Обработать пропущенные данные.

        Args:
            data: Входные данные

        Returns:
            Данные с обработанными пропусками
        """
        if data.empty:
            return data

        data = data.copy()
        missing_before = data.isna().sum().sum()

        if missing_before == 0:
            logger.info("No missing values found")
            return data

        logger.info(f"Found {missing_before} missing values, applying {self.method}")

        if self.method == "drop":
            data = self._drop_missing(data)
        elif self.method == "forward_fill":
            data = self._forward_fill(data)
        elif self.method == "backward_fill":
            data = self._backward_fill(data)
        elif self.method == "interpolate":
            data = self._interpolate(data)
        elif self.method == "mean":
            data = self._fill_mean(data)
        elif self.method == "median":
            data = self._fill_median(data)
        else:
            raise PreprocessingError(f"Unknown missing data method: {self.method}")

        missing_after = data.isna().sum().sum()
        logger.info(f"After handling: {missing_after} missing values remain")

        return data

    def _drop_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Удалить строки с NaN."""
        return data.dropna().reset_index(drop=True)

    def _forward_fill(self, data: pd.DataFrame) -> pd.DataFrame:
        """Заполнить предыдущим значением."""
        # Для OHLCV: цены - forward fill, volume - опционально zero
        price_cols = ["open", "high", "low", "close"]

        for col in price_cols:
            if col in data.columns:
                data[col] = data[col].ffill(limit=self.max_gap)

        if "volume" in data.columns:
            if self.fill_volume_with_zero:
                data["volume"] = data["volume"].fillna(0)
            else:
                data["volume"] = data["volume"].ffill(limit=self.max_gap)

        # Остальные колонки - forward fill
        other_cols = [
            col
            for col in data.columns
            if col not in price_cols + ["volume", "timestamp", "ticker"]
        ]
        for col in other_cols:
            data[col] = data[col].ffill(limit=self.max_gap)

        return data

    def _backward_fill(self, data: pd.DataFrame) -> pd.DataFrame:
        """Заполнить следующим значением."""
        return data.bfill(limit=self.max_gap)

    def _interpolate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Линейная интерполяция."""
        numeric_cols = data.select_dtypes(include=["number"]).columns

        for col in numeric_cols:
            # Проверить размер пропусков
            mask = data[col].isna()
            gaps = self._get_gap_sizes(mask)

            # Интерполировать только малые пропуски
            for start, size in gaps:
                if size <= self.max_gap:
                    data.loc[start : start + size - 1, col] = (
                        data[col]
                        .iloc[start : start + size]
                        .interpolate(method="linear", limit_area="inside")
                    )

        return data

    def _fill_mean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Заполнить средним значением."""
        numeric_cols = data.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].mean())
        return data

    def _fill_median(self, data: pd.DataFrame) -> pd.DataFrame:
        """Заполнить медианой."""
        numeric_cols = data.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].median())
        return data

    def _get_gap_sizes(self, mask: pd.Series) -> list[tuple[int, int]]:
        """
        Получить размеры пропусков.

        Args:
            mask: Boolean series (True = пропуск)

        Returns:
            Список (начальный_индекс, размер_пропуска)
        """
        gaps = []
        in_gap = False
        gap_start = 0
        gap_size = 0

        for i, is_missing in enumerate(mask):
            if is_missing:
                if not in_gap:
                    in_gap = True
                    gap_start = i
                    gap_size = 1
                else:
                    gap_size += 1
            else:
                if in_gap:
                    gaps.append((gap_start, gap_size))
                    in_gap = False

        # Последний пропуск
        if in_gap:
            gaps.append((gap_start, gap_size))

        return gaps


class GapDetector:
    """
    Детектор пропусков в временных рядах.

    Анализирует временные пропуски (отсутствующие бары)
    и категоризирует их по размеру и причинам.
    """

    def __init__(self, expected_frequency: str = "1min"):
        """
        Инициализировать детектор.

        Args:
            expected_frequency: Ожидаемая частота данных
                (pandas frequency string)
        """
        self.expected_frequency = expected_frequency

    def detect_gaps(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Детектировать пропуски во временной серии.

        Args:
            data: Данные с колонкой timestamp

        Returns:
            DataFrame с информацией о пропусках:
            - gap_start: начало пропуска
            - gap_end: конец пропуска
            - gap_size: размер пропуска в барах
            - gap_duration: длительность пропуска
        """
        if "timestamp" not in data.columns:
            logger.warning("No timestamp column, cannot detect time gaps")
            return pd.DataFrame()

        if data.empty or len(data) < 2:
            return pd.DataFrame()

        # Преобразовать к datetime
        timestamps = pd.to_datetime(data["timestamp"])

        # Создать ожидаемый диапазон
        full_range = pd.date_range(
            start=timestamps.min(),
            end=timestamps.max(),
            freq=self.expected_frequency,
        )

        # Найти отсутствующие timestamps
        missing = full_range.difference(pd.DatetimeIndex(timestamps))

        if len(missing) == 0:
            logger.info("No gaps detected")
            return pd.DataFrame()

        # Сгруппировать последовательные пропуски
        gaps = []
        if len(missing) > 0:
            gap_start = missing[0]
            gap_size = 1
            freq_delta = pd.Timedelta(self.expected_frequency)

            for i in range(1, len(missing)):
                expected_next = gap_start + freq_delta * gap_size
                if missing[i] == expected_next:
                    gap_size += 1
                else:
                    gaps.append(
                        {
                            "gap_start": gap_start,
                            "gap_end": gap_start + freq_delta * (gap_size - 1),
                            "gap_size": gap_size,
                            "gap_duration": freq_delta * gap_size,
                        }
                    )
                    gap_start = missing[i]
                    gap_size = 1

            # Последний пропуск
            gaps.append(
                {
                    "gap_start": gap_start,
                    "gap_end": gap_start + freq_delta * (gap_size - 1),
                    "gap_size": gap_size,
                    "gap_duration": freq_delta * gap_size,
                }
            )

        gaps_df = pd.DataFrame(gaps)
        logger.info(f"Detected {len(gaps_df)} gaps, total {len(missing)} missing bars")

        return gaps_df

    def analyze_gaps(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Анализировать паттерны пропусков.

        Args:
            data: Данные с timestamp

        Returns:
            Статистика пропусков
        """
        gaps = self.detect_gaps(data)

        if gaps.empty:
            return {
                "total_gaps": 0,
                "total_missing_bars": 0,
                "max_gap_size": 0,
                "avg_gap_size": 0,
                "small_gaps": 0,
                "large_gaps": 0,
            }

        return {
            "total_gaps": len(gaps),
            "total_missing_bars": gaps["gap_size"].sum(),
            "max_gap_size": gaps["gap_size"].max(),
            "avg_gap_size": gaps["gap_size"].mean(),
            "median_gap_size": gaps["gap_size"].median(),
            "small_gaps": (gaps["gap_size"] <= 5).sum(),  # <= 5 bars
            "large_gaps": (gaps["gap_size"] > 20).sum(),  # > 20 bars
        }
