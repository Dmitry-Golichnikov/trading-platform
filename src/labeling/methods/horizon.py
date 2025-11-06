"""Horizon labeling метод."""

import logging
from typing import Literal, Union

import numpy as np
import pandas as pd

from src.labeling.base import BaseLabeler

logger = logging.getLogger(__name__)


class HorizonLabeler(BaseLabeler):
    """
    Horizon labeling метод.

    Разметка основана на изменении цены на заданном горизонте.
    Поддерживает фиксированный и адаптивный горизонт.
    """

    def __init__(
        self,
        horizon: Union[int, Literal["adaptive"]] = 20,
        direction: Literal["long", "short", "long+short"] = "long+short",
        threshold_pct: float = 0.01,
        adaptive_method: Literal["atr", "volatility", "custom"] = "atr",
        adaptive_window: int = 20,
        adaptive_multiplier: float = 2.0,
        min_horizon: int = 5,
        max_horizon: int = 50,
        price_type: Literal["close", "typical"] = "close",
    ):
        """
        Инициализация Horizon labeler.

        Args:
            horizon: Горизонт в барах или 'adaptive'
            direction: Направление торговли
            threshold_pct: Порог в процентах для разметки
            adaptive_method: Метод адаптивного горизонта
            adaptive_window: Окно для адаптивного расчёта
            adaptive_multiplier: Множитель для адаптивного метода
            min_horizon: Минимальный горизонт для adaptive
            max_horizon: Максимальный горизонт для adaptive
            price_type: Тип цены для расчётов
        """
        super().__init__(
            horizon=horizon,
            direction=direction,
            threshold_pct=threshold_pct,
            adaptive_method=adaptive_method,
            adaptive_window=adaptive_window,
            adaptive_multiplier=adaptive_multiplier,
            min_horizon=min_horizon,
            max_horizon=max_horizon,
            price_type=price_type,
        )

    def _validate_params(self) -> None:
        """Валидация параметров."""
        horizon = self.params["horizon"]
        if horizon != "adaptive" and (not isinstance(horizon, int) or horizon <= 0):
            raise ValueError("horizon должен быть положительным числом или 'adaptive'")

        if self.params["direction"] not in ["long", "short", "long+short"]:
            raise ValueError("direction должен быть 'long', 'short' или 'long+short'")

        if self.params["threshold_pct"] <= 0:
            raise ValueError("threshold_pct должен быть положительным")

        if self.params["adaptive_method"] not in ["atr", "volatility", "custom"]:
            raise ValueError("adaptive_method должен быть 'atr', 'volatility' или 'custom'")

    def label(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Разметка данных horizon методом.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с добавленными колонками:
            - label: метка (-1, 0, 1)
            - horizon_used: использованный горизонт
            - future_return: фактический return
        """
        self.validate_data(data)

        result = data.copy()

        # Вычисление цены для расчётов
        if self.params["price_type"] == "close":
            price = result["close"]
        else:  # typical price
            price = (result["high"] + result["low"] + result["close"]) / 3

        # Определение горизонта
        if self.params["horizon"] == "adaptive":
            horizons = self._compute_adaptive_horizon(result)
        else:
            horizons = pd.Series(self.params["horizon"], index=result.index)

        result["horizon_used"] = horizons

        # Векторизованный расчёт future returns
        labels = np.zeros(len(result))
        future_returns = np.full(len(result), np.nan)

        for i in range(len(result)):
            horizon = int(horizons.iloc[i])
            future_idx = min(i + horizon, len(result) - 1)

            if future_idx > i:
                current_price = price.iloc[i]
                future_price = price.iloc[future_idx]
                ret = (future_price - current_price) / current_price
                future_returns[i] = ret

                # Разметка на основе direction и threshold
                labels[i] = self._classify_return(ret)
            else:
                # Недостаточно данных для горизонта
                labels[i] = 0
                future_returns[i] = 0.0

        result["label"] = labels.astype(int)
        result["future_return"] = future_returns

        logger.info(
            f"Horizon labeling завершена: "
            f"{sum(labels == 1)} long, "
            f"{sum(labels == -1)} short, "
            f"{sum(labels == 0)} neutral"
        )

        return result

    def _compute_adaptive_horizon(self, data: pd.DataFrame) -> pd.Series:
        """
        Вычисление адаптивного горизонта.

        Args:
            data: DataFrame с данными

        Returns:
            Series с горизонтами для каждого бара
        """
        method = self.params["adaptive_method"]
        window = self.params["adaptive_window"]
        multiplier = self.params["adaptive_multiplier"]
        min_h = self.params["min_horizon"]
        max_h = self.params["max_horizon"]

        if method == "atr":
            # Используем ATR для определения горизонта
            tr = pd.DataFrame(
                {
                    "hl": data["high"] - data["low"],
                    "hc": abs(data["high"] - data["close"].shift(1)),
                    "lc": abs(data["low"] - data["close"].shift(1)),
                }
            ).max(axis=1)

            metric = tr.rolling(window=window, min_periods=1).mean()
            avg_price = data["close"].rolling(window=window, min_periods=1).mean()
            volatility_pct = metric / avg_price

        elif method == "volatility":
            # Используем стандартное отклонение returns
            returns = data["close"].pct_change()
            metric = returns.rolling(window=window, min_periods=1).std()
            volatility_pct = metric.copy()

        else:  # custom - можно расширить
            return pd.Series((min_h + max_h) // 2, index=data.index)

        volatility_pct = volatility_pct.replace([np.inf, -np.inf], np.nan)
        volatility_pct = volatility_pct.bfill().ffill()

        if volatility_pct.isna().all():
            return pd.Series((min_h + max_h) // 2, index=data.index)

        filled = volatility_pct.fillna(volatility_pct.median())

        if filled.nunique(dropna=True) <= 1:
            return pd.Series((min_h + max_h) // 2, index=data.index)

        ranks = filled.rank(pct=True, method="average")
        if multiplier and multiplier > 0:
            ranks = np.clip(np.power(ranks, 1.0 / multiplier), 0, 1)

        horizons = min_h + (max_h - min_h) * (1 - ranks)

        return horizons.clip(min_h, max_h).round().astype(int)

    def _classify_return(self, ret: float) -> int:
        """
        Классификация return в метку.

        Args:
            ret: Return значение

        Returns:
            Метка: -1 (short), 0 (neutral), 1 (long)
        """
        direction = self.params["direction"]
        threshold = self.params["threshold_pct"]

        if direction == "long+short":
            if ret > threshold:
                return 1
            elif ret < -threshold:
                return -1
            else:
                return 0
        elif direction == "long":
            if ret > threshold:
                return 1
            else:
                return 0
        else:  # short
            if ret < -threshold:
                return 1
            else:
                return 0
