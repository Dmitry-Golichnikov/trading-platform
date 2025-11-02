"""
Фильтры аномалий для OHLCV данных.
"""

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.common.exceptions import PreprocessingError
from src.data.filters.base import DataFilter

logger = logging.getLogger(__name__)


class PriceAnomalyFilter(DataFilter):
    """
    Фильтр аномальных ценовых движений.

    Поддерживаемые методы:
    - zscore: детекция через Z-score returns
    - iqr: межквартильный размах
    - ewma: отклонение от exponential moving average
    - isolation_forest: ML-based детекция
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Инициализировать фильтр.

        Args:
            config: Конфигурация со следующими параметрами:
                - method: 'zscore' | 'iqr' | 'ewma' | 'isolation_forest'
                - threshold: порог (для zscore - кол-во std, для iqr - коэффициент)
                - window: размер окна для rolling stats
                - action: 'remove' | 'replace' | 'mark'
        """
        super().__init__(config)
        self.method: Literal["zscore", "iqr", "ewma", "isolation_forest"] = (
            self.config.get("method", "zscore")
        )
        self.threshold: float = self.config.get("threshold", 3.0)
        self.window: int = self.config.get("window", 100)
        self.action: Literal["remove", "replace", "mark"] = self.config.get(
            "action", "remove"
        )

    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """Применить фильтр аномалий."""
        if data.empty:
            return data

        data_before = data.copy()
        mask = self.get_filter_mask(data)

        if self.action == "remove":
            data = data[mask].reset_index(drop=True)
        elif self.action == "mark":
            data = data.copy()
            data["price_anomaly"] = ~mask
        elif self.action == "replace":
            data = data.copy()
            data.loc[~mask, ["open", "high", "low", "close"]] = np.nan

        self.stats.calculate_stats(data_before, data)
        self.stats.reasons["price_anomaly"] = (~mask).sum()
        self.log_statistics()

        return data

    def get_filter_mask(self, data: pd.DataFrame) -> pd.Series:
        """Получить маску аномалий."""
        if self.method == "zscore":
            return self._detect_zscore(data)
        elif self.method == "iqr":
            return self._detect_iqr(data)
        elif self.method == "ewma":
            return self._detect_ewma(data)
        elif self.method == "isolation_forest":
            return self._detect_isolation_forest(data)
        else:
            raise PreprocessingError(f"Unknown detection method: {self.method}")

    def _detect_zscore(self, data: pd.DataFrame) -> pd.Series:
        """Детекция через Z-score returns."""
        returns = data["close"].pct_change()
        rolling_mean = returns.rolling(window=self.window, min_periods=1).mean()
        rolling_std = returns.rolling(window=self.window, min_periods=1).std()

        # Избежать деления на ноль
        rolling_std = rolling_std.replace(0, np.nan)

        z_score = (returns - rolling_mean) / rolling_std
        mask = z_score.abs() <= self.threshold

        # Первые наблюдения всегда валидны (недостаточно истории)
        mask.iloc[: self.window] = True

        return mask.fillna(True)

    def _detect_iqr(self, data: pd.DataFrame) -> pd.Series:
        """Детекция через IQR."""
        returns = data["close"].pct_change()

        q1 = returns.quantile(0.25)
        q3 = returns.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - self.threshold * iqr
        upper_bound = q3 + self.threshold * iqr

        mask = (returns >= lower_bound) & (returns <= upper_bound)
        mask.iloc[0] = True  # Первая строка не имеет return

        return mask.fillna(True)

    def _detect_ewma(self, data: pd.DataFrame) -> pd.Series:
        """Детекция через EWMA."""
        prices = data["close"]
        ewma = prices.ewm(span=self.window, adjust=False).mean()
        deviation = (prices - ewma).abs() / ewma

        mask = deviation <= (self.threshold / 100)  # threshold в процентах
        return mask.fillna(True)

    def _detect_isolation_forest(self, data: pd.DataFrame) -> pd.Series:
        """Детекция через Isolation Forest."""
        # Используем returns и volume как признаки
        features = pd.DataFrame(
            {
                "returns": data["close"].pct_change(),
                "volume_change": data["volume"].pct_change(),
            }
        )

        features = features.fillna(0)

        if len(features) < 10:
            # Недостаточно данных для ML
            return pd.Series([True] * len(data), index=data.index)

        contamination = min(0.1, self.threshold / 100)
        model = IsolationForest(contamination=contamination, random_state=42)
        predictions = model.fit_predict(features)

        # 1 = normal, -1 = anomaly
        mask = pd.Series(predictions == 1, index=data.index)
        return mask  # type: ignore[no-any-return]


class VolumeAnomalyFilter(DataFilter):
    """Фильтр аномальных объёмов торгов."""

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Инициализировать фильтр.

        Args:
            config: Конфигурация:
                - min_volume: минимальный объём
                - max_volume_spike: макс. всплеск (кратность к среднему)
                - action: 'remove' | 'mark'
        """
        super().__init__(config)
        self.min_volume: int = self.config.get("min_volume", 0)
        self.max_volume_spike: float = self.config.get("max_volume_spike", 10.0)
        self.action: Literal["remove", "mark"] = self.config.get("action", "remove")

    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """Применить фильтр."""
        if data.empty:
            return data

        data_before = data.copy()
        mask = self.get_filter_mask(data)

        if self.action == "remove":
            data = data[mask].reset_index(drop=True)
        elif self.action == "mark":
            data = data.copy()
            data["volume_anomaly"] = ~mask

        self.stats.calculate_stats(data_before, data)
        self.stats.reasons["volume_anomaly"] = (~mask).sum()
        self.log_statistics()

        return data

    def get_filter_mask(self, data: pd.DataFrame) -> pd.Series:
        """Получить маску аномалий."""
        volume = data["volume"]

        # Проверка отрицательных и нулевых объёмов
        mask_positive = volume > self.min_volume

        # Проверка всплесков
        mean_volume = volume.mean()
        if mean_volume > 0:
            mask_spike = volume <= (self.max_volume_spike * mean_volume)
        else:
            mask_spike = pd.Series([True] * len(data), index=data.index)

        return mask_positive & mask_spike


class SpreadAnomalyFilter(DataFilter):
    """Фильтр баров с подозрительным spread (high-low)."""

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Инициализировать фильтр.

        Args:
            config: Конфигурация:
                - max_spread_pct: макс. spread в % от цены
                - allow_zero_spread: разрешить нулевой spread
                - action: 'remove' | 'mark'
        """
        super().__init__(config)
        self.max_spread_pct: float = self.config.get("max_spread_pct", 10.0)
        self.allow_zero_spread: bool = self.config.get("allow_zero_spread", False)
        self.action: Literal["remove", "mark"] = self.config.get("action", "remove")

    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """Применить фильтр."""
        if data.empty:
            return data

        data_before = data.copy()
        mask = self.get_filter_mask(data)

        if self.action == "remove":
            data = data[mask].reset_index(drop=True)
        elif self.action == "mark":
            data = data.copy()
            data["spread_anomaly"] = ~mask

        self.stats.calculate_stats(data_before, data)
        self.stats.reasons["spread_anomaly"] = (~mask).sum()
        self.log_statistics()

        return data

    def get_filter_mask(self, data: pd.DataFrame) -> pd.Series:
        """Получить маску аномалий."""
        spread = data["high"] - data["low"]
        spread_pct = (spread / data["close"]) * 100

        # Проверка отрицательного spread (некорректные данные)
        mask_positive = spread >= 0

        # Проверка нулевого spread
        if self.allow_zero_spread:
            mask_non_zero = pd.Series([True] * len(data), index=data.index)
        else:
            mask_non_zero = spread > 0

        # Проверка слишком большого spread
        mask_reasonable = spread_pct <= self.max_spread_pct

        return mask_positive & mask_non_zero & mask_reasonable
