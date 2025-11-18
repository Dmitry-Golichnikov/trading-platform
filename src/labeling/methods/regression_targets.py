"""Генерация regression таргетов."""

import logging
from typing import Literal

import numpy as np
import pandas as pd

from src.labeling.base import BaseLabeler

logger = logging.getLogger(__name__)


class RegressionTargetsLabeler(BaseLabeler):
    """
    Генерация regression таргетов для моделей регрессии.

    Поддерживаемые таргеты:
    - Future returns на заданном горизонте
    - Max Favorable Excursion (MFE)
    - Max Adverse Excursion (MAE)
    - Sharpe ratio на скользящем окне
    """

    def __init__(
        self,
        target_type: Literal["future_return", "mfe", "mae", "sharpe"] = "future_return",
        horizon: int = 20,
        price_type: Literal["close", "typical"] = "close",
        normalize: bool = False,
        sharpe_window: int = 20,
    ):
        """
        Инициализация Regression Targets labeler.

        Args:
            target_type: Тип таргета
            horizon: Горизонт для расчёта
            price_type: Тип цены
            normalize: Нормализовать таргеты
            sharpe_window: Окно для Sharpe ratio
        """
        super().__init__(
            target_type=target_type,
            horizon=horizon,
            price_type=price_type,
            normalize=normalize,
            sharpe_window=sharpe_window,
        )

    def _validate_params(self) -> None:
        """Валидация параметров."""
        if self.params["target_type"] not in ["future_return", "mfe", "mae", "sharpe"]:
            raise ValueError("target_type должен быть 'future_return', 'mfe', 'mae' или 'sharpe'")

        if self.params["horizon"] <= 0:
            raise ValueError("horizon должен быть положительным")

    def label(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Генерация regression таргетов.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с добавленной колонкой target
        """
        self.validate_data(data)

        result = data.copy()

        # Вычисление цены для расчётов
        if self.params["price_type"] == "close":
            price = result["close"]
        else:
            price = (result["high"] + result["low"] + result["close"]) / 3

        # Генерация таргета в зависимости от типа
        target_type = self.params["target_type"]

        if target_type == "future_return":
            targets = self._compute_future_return(price)
        elif target_type == "mfe":
            targets = self._compute_mfe(result)
        elif target_type == "mae":
            targets = self._compute_mae(result)
        else:  # sharpe
            targets = self._compute_sharpe(price)

        # Нормализация если требуется
        if self.params["normalize"]:
            targets = self._normalize_targets(targets)

        result["target"] = targets

        logger.info(
            f"Regression targets ({target_type}) сгенерированы: " f"mean={targets.mean():.4f}, std={targets.std():.4f}"
        )

        return result

    def _compute_future_return(self, price: pd.Series) -> pd.Series:
        """
        Вычисление future returns на заданном горизонте.

        Args:
            price: Series цен

        Returns:
            Series с future returns
        """
        horizon = self.params["horizon"]
        future_price = price.shift(-horizon)
        returns = (future_price - price) / price

        # Последние horizon баров не имеют будущих данных
        returns.iloc[-horizon:] = np.nan

        return returns

    def _compute_mfe(self, data: pd.DataFrame) -> pd.Series:
        """
        Вычисление Max Favorable Excursion (MFE).

        Максимальный благоприятный ход цены на горизонте.

        Args:
            data: DataFrame с данными

        Returns:
            Series с MFE значениями
        """
        horizon = self.params["horizon"]
        mfe = np.full(len(data), np.nan)

        for i in range(len(data) - horizon):
            entry_price = data["close"].iloc[i]
            future_highs = data["high"].iloc[i + 1 : i + horizon + 1]

            if len(future_highs) > 0:
                max_high = future_highs.max()
                mfe[i] = (max_high - entry_price) / entry_price

        return pd.Series(mfe, index=data.index)

    def _compute_mae(self, data: pd.DataFrame) -> pd.Series:
        """
        Вычисление Max Adverse Excursion (MAE).

        Максимальный неблагоприятный ход цены на горизонте.

        Args:
            data: DataFrame с данными

        Returns:
            Series с MAE значениями
        """
        horizon = self.params["horizon"]
        mae = np.full(len(data), np.nan)

        for i in range(len(data) - horizon):
            entry_price = data["close"].iloc[i]
            future_lows = data["low"].iloc[i + 1 : i + horizon + 1]

            if len(future_lows) > 0:
                min_low = future_lows.min()
                mae[i] = (entry_price - min_low) / entry_price

        return pd.Series(mae, index=data.index)

    def _compute_sharpe(self, price: pd.Series) -> pd.Series:
        """
        Вычисление Sharpe ratio на скользящем окне.

        Args:
            price: Series цен

        Returns:
            Series с Sharpe ratio значениями
        """
        window = self.params["sharpe_window"]

        # Вычисление returns
        returns = price.pct_change()

        # Rolling Sharpe ratio
        mean_return = returns.rolling(window=window).mean()
        std_return = returns.rolling(window=window).std()

        # Избегаем деления на ноль
        sharpe = np.where(std_return > 1e-8, mean_return / std_return, 0)

        return pd.Series(sharpe, index=price.index)

    def _normalize_targets(self, targets: pd.Series) -> pd.Series:
        """
        Нормализация таргетов (z-score).

        Args:
            targets: Series с таргетами

        Returns:
            Series с нормализованными таргетами
        """
        mean = targets.mean()
        std = targets.std()

        if std > 1e-8:
            normalized = (targets - mean) / std
        else:
            normalized = targets - mean

        return normalized
