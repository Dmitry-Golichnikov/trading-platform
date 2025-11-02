"""
Фильтры статистических выбросов.
"""

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from src.common.exceptions import PreprocessingError
from src.data.filters.base import DataFilter

logger = logging.getLogger(__name__)


class StatisticalOutlierFilter(DataFilter):
    """
    Статистический фильтр выбросов.

    Поддерживаемые методы:
    - mad: Median Absolute Deviation
    - tukey: Tukey's fences (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    - grubbs: Grubbs test
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Инициализировать фильтр.

        Args:
            config: Конфигурация:
                - method: 'mad' | 'tukey' | 'grubbs'
                - threshold: порог (для mad - кратность MAD, для tukey - множитель IQR)
                - column: колонка для анализа ('close' по умолчанию)
                - action: 'remove' | 'mark'
        """
        super().__init__(config)
        self.method: Literal["mad", "tukey", "grubbs"] = self.config.get(
            "method", "mad"
        )
        self.threshold: float = self.config.get("threshold", 3.5)
        self.column: str = self.config.get("column", "close")
        self.action: Literal["remove", "mark"] = self.config.get("action", "remove")

    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """Применить фильтр выбросов."""
        if data.empty:
            return data

        if self.column not in data.columns:
            logger.warning(f"Column {self.column} not found, skipping outlier filter")
            return data

        data_before = data.copy()
        mask = self.get_filter_mask(data)

        if self.action == "remove":
            data = data[mask].reset_index(drop=True)
        elif self.action == "mark":
            data = data.copy()
            data["statistical_outlier"] = ~mask

        self.stats.calculate_stats(data_before, data)
        self.stats.reasons["statistical_outlier"] = (~mask).sum()
        self.log_statistics()

        return data

    def get_filter_mask(self, data: pd.DataFrame) -> pd.Series:
        """Получить маску выбросов."""
        if self.method == "mad":
            return self._detect_mad(data)
        elif self.method == "tukey":
            return self._detect_tukey(data)
        elif self.method == "grubbs":
            return self._detect_grubbs(data)
        else:
            raise PreprocessingError(f"Unknown outlier detection method: {self.method}")

    def _detect_mad(self, data: pd.DataFrame) -> pd.Series:
        """Детекция через Median Absolute Deviation."""
        values = data[self.column]
        median = values.median()
        mad = np.median(np.abs(values - median))

        if mad == 0:
            # Все значения одинаковые или очень близкие
            return pd.Series([True] * len(data), index=data.index)

        modified_z_score = 0.6745 * (values - median) / mad
        mask = modified_z_score.abs() <= self.threshold

        return mask  # type: ignore[no-any-return]

    def _detect_tukey(self, data: pd.DataFrame) -> pd.Series:
        """Детекция через Tukey's fences."""
        values = data[self.column]

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - self.threshold * iqr
        upper_bound = q3 + self.threshold * iqr

        mask = (values >= lower_bound) & (values <= upper_bound)
        return mask

    def _detect_grubbs(self, data: pd.DataFrame) -> pd.Series:
        """Детекция через Grubbs test."""
        values = data[self.column].to_numpy(dtype=float, copy=True)
        n = len(values)

        if n < 3:
            return pd.Series([True] * n, index=data.index)

        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))

        if std == 0:
            return pd.Series([True] * n, index=data.index)

        # G-статистика для каждого значения
        g_scores = np.abs((values - mean) / std)

        # Критическое значение Грабса
        t_dist = stats.t.ppf(1 - 0.05 / (2 * n), n - 2)
        g_critical = ((n - 1) / np.sqrt(n)) * np.sqrt(t_dist**2 / (n - 2 + t_dist**2))

        mask = pd.Series(g_scores <= g_critical * self.threshold, index=data.index)
        return mask  # type: ignore[no-any-return]


class RollingOutlierFilter(DataFilter):
    """
    Фильтр выбросов с адаптивными порогами в rolling window.

    Использует скользящее окно для адаптации к локальным условиям рынка.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Инициализировать фильтр.

        Args:
            config: Конфигурация:
                - window: размер rolling window
                - threshold: порог в кол-ве std
                - min_periods: минимальное кол-во наблюдений в окне
                - column: колонка для анализа
                - action: 'remove' | 'mark'
        """
        super().__init__(config)
        self.window: int = self.config.get("window", 100)
        self.threshold: float = self.config.get("threshold", 3.0)
        self.min_periods: int = self.config.get("min_periods", 20)
        self.column: str = self.config.get("column", "close")
        self.action: Literal["remove", "mark"] = self.config.get("action", "remove")

    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """Применить фильтр выбросов."""
        if data.empty:
            return data

        if self.column not in data.columns:
            logger.warning(f"Column {self.column} not found, skipping outlier filter")
            return data

        data_before = data.copy()
        mask = self.get_filter_mask(data)

        if self.action == "remove":
            data = data[mask].reset_index(drop=True)
        elif self.action == "mark":
            data = data.copy()
            data["rolling_outlier"] = ~mask

        self.stats.calculate_stats(data_before, data)
        self.stats.reasons["rolling_outlier"] = (~mask).sum()
        self.log_statistics()

        return data

    def get_filter_mask(self, data: pd.DataFrame) -> pd.Series:
        """Получить маску выбросов."""
        values = data[self.column]

        # Rolling mean и std
        rolling_mean = values.rolling(
            window=self.window, min_periods=self.min_periods
        ).mean()
        rolling_std = values.rolling(
            window=self.window, min_periods=self.min_periods
        ).std()

        # Избежать деления на ноль
        rolling_std = rolling_std.replace(0, np.nan)

        # Z-score относительно rolling stats
        z_score = (values - rolling_mean) / rolling_std
        mask = z_score.abs() <= self.threshold

        # Первые min_periods наблюдений всегда валидны
        mask.iloc[: self.min_periods] = True

        return mask.fillna(True)
