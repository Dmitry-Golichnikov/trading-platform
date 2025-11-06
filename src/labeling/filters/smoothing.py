"""Фильтры сглаживания меток."""

import logging
from typing import Literal

import pandas as pd
from scipy import ndimage

logger = logging.getLogger(__name__)


class SmoothingFilter:
    """
    Сглаживание меток для уменьшения шума.

    Поддерживаемые методы:
    - moving_average: скользящее среднее
    - exponential: экспоненциальное сглаживание
    - median: медианный фильтр
    """

    def __init__(
        self,
        method: Literal["moving_average", "exponential", "median"] = "median",
        window: int = 3,
        alpha: float = 0.3,
    ):
        """
        Инициализация фильтра сглаживания.

        Args:
            method: Метод сглаживания
            window: Размер окна для сглаживания
            alpha: Параметр для exponential метода
        """
        self.method = method
        self.window = window
        self.alpha = alpha

        if window < 1:
            raise ValueError("window должен быть >= 1")

        if method not in ["moving_average", "exponential", "median"]:
            raise ValueError("method должен быть 'moving_average', 'exponential' или 'median'")

    def apply(self, labels: pd.Series) -> pd.Series:
        """
        Применить сглаживание к меткам.

        Args:
            labels: Series с метками

        Returns:
            Series с сглаженными метками
        """
        if self.method == "moving_average":
            smoothed = self._moving_average(labels)
        elif self.method == "exponential":
            smoothed = self._exponential_smoothing(labels)
        else:  # median
            smoothed = self._median_filter(labels)

        # Округляем до ближайшей допустимой метки
        smoothed = self._round_to_valid_label(smoothed)

        changed = (labels != smoothed).sum()
        total = len(labels) or 1
        logger.info(
            "Smoothing filter: изменено %d меток (%.1f%%)",
            changed,
            changed / total * 100,
        )

        return smoothed

    def _moving_average(self, labels: pd.Series) -> pd.Series:
        """
        Сглаживание скользящим средним.

        Args:
            labels: Series с метками

        Returns:
            Series с сглаженными метками
        """
        # Центрированное окно для симметричности
        smoothed = labels.rolling(window=self.window, center=True, min_periods=1).mean()

        return smoothed

    def _exponential_smoothing(self, labels: pd.Series) -> pd.Series:
        """
        Экспоненциальное сглаживание.

        Args:
            labels: Series с метками

        Returns:
            Series с сглаженными метками
        """
        smoothed = labels.ewm(alpha=self.alpha, adjust=False).mean()
        return smoothed

    def _median_filter(self, labels: pd.Series) -> pd.Series:
        """
        Медианный фильтр.

        Args:
            labels: Series с метками

        Returns:
            Series с сглаженными метками
        """
        # Используем scipy median filter
        smoothed_values = ndimage.median_filter(labels.values, size=self.window, mode="nearest")

        return pd.Series(smoothed_values, index=labels.index)

    def _round_to_valid_label(self, labels: pd.Series) -> pd.Series:
        """
        Округление до допустимых значений меток.

        Args:
            labels: Series с метками

        Returns:
            Series с округлёнными метками
        """
        # Округляем до ближайшего целого
        rounded = labels.round()

        # Ограничиваем диапазон [-1, 1]
        rounded = rounded.clip(-1, 1)

        return rounded.astype(int)
