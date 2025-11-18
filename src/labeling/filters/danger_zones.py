"""Фильтр опасных зон."""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DangerZonesFilter:
    """
    Фильтр для маркировки/удаления меток в опасных зонах.

    Опасные зоны:
    - Высокая волатильность
    - Низкая ликвидность
    - Экстремальные события
    - Gap'ы в данных
    """

    def __init__(
        self,
        high_volatility_threshold: float = 3.0,
        low_liquidity_threshold: Optional[float] = None,
        gap_threshold_pct: float = 0.05,
        volatility_window: int = 20,
        neutral_label: int = 0,
        mark_only: bool = False,
    ):
        """
        Инициализация danger zones filter.

        Args:
            high_volatility_threshold: Порог высокой волатильности
                (в стандартных отклонениях)
            low_liquidity_threshold: Порог низкой ликвидности (минимальный объём)
            gap_threshold_pct: Порог для определения gap'а (%)
            volatility_window: Окно для расчёта волатильности
            neutral_label: Метка для замены в опасных зонах
            mark_only: Только маркировать зоны без замены меток
        """
        self.high_volatility_threshold = high_volatility_threshold
        self.low_liquidity_threshold = low_liquidity_threshold
        self.gap_threshold_pct = gap_threshold_pct
        self.volatility_window = volatility_window
        self.neutral_label = neutral_label
        self.mark_only = mark_only

    def apply(
        self,
        labels: pd.Series,
        data: pd.DataFrame,
    ) -> pd.Series:
        """
        Применить фильтр опасных зон.

        Args:
            labels: Series с метками
            data: DataFrame с OHLCV данными

        Returns:
            Series с отфильтрованными метками (или с маркировкой)
        """
        result = labels.copy()

        # Определяем опасные зоны
        danger_mask = self._identify_danger_zones(data)

        if self.mark_only:
            # Только добавляем информацию о опасных зонах
            total = len(danger_mask) or 1
            count = danger_mask.sum()
            logger.info(
                "Danger zones filter: найдено %d опасных точек (%.1f%%)",
                count,
                count / total * 100,
            )
        else:
            # Заменяем метки в опасных зонах
            filtered_count = danger_mask.sum()
            result[danger_mask] = self.neutral_label

            total = len(labels) or 1
            logger.info(
                "Danger zones filter: отфильтровано %d меток (%.1f%%)",
                filtered_count,
                filtered_count / total * 100,
            )

        return result

    def get_danger_zones(self, data: pd.DataFrame) -> pd.Series:
        """
        Получить маску опасных зон.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Series с булевой маской (True = опасная зона)
        """
        return self._identify_danger_zones(data)

    def _identify_danger_zones(self, data: pd.DataFrame) -> pd.Series:
        """
        Определение опасных зон.

        Args:
            data: DataFrame с данными

        Returns:
            Series с булевой маской опасных зон
        """
        danger_mask = pd.Series(False, index=data.index)

        # 1. Высокая волатильность
        high_vol_mask = self._detect_high_volatility(data)
        danger_mask |= high_vol_mask

        # 2. Низкая ликвидность (если задан порог и есть volume)
        if self.low_liquidity_threshold is not None and "volume" in data.columns:
            low_liq_mask = self._detect_low_liquidity(data)
            danger_mask |= low_liq_mask

        # 3. Gap'ы
        gap_mask = self._detect_gaps(data)
        danger_mask |= gap_mask

        return danger_mask

    def _detect_high_volatility(self, data: pd.DataFrame) -> pd.Series:
        """
        Обнаружение высокой волатильности.

        Args:
            data: DataFrame с данными

        Returns:
            Series с булевой маской
        """
        # Вычисляем returns
        returns = data["close"].pct_change()

        # Rolling volatility
        rolling_vol = returns.rolling(window=self.volatility_window).std()

        # Z-score волатильности
        vol_mean = rolling_vol.mean()
        vol_std = rolling_vol.std()

        if vol_std > 1e-8:
            vol_zscore = abs((rolling_vol - vol_mean) / vol_std)
        else:
            vol_zscore = pd.Series(0, index=data.index)

        # Высокая волатильность = z-score > threshold
        high_vol_mask = vol_zscore > self.high_volatility_threshold

        logger.debug(f"Обнаружено {high_vol_mask.sum()} точек с высокой волатильностью")

        return high_vol_mask

    def _detect_low_liquidity(self, data: pd.DataFrame) -> pd.Series:
        """
        Обнаружение низкой ликвидности.

        Args:
            data: DataFrame с данными

        Returns:
            Series с булевой маской
        """
        volume = data["volume"]

        # Низкая ликвидность = volume < threshold
        low_liq_mask = volume < self.low_liquidity_threshold

        logger.debug(f"Обнаружено {low_liq_mask.sum()} точек с низкой ликвидностью")

        return low_liq_mask

    def _detect_gaps(self, data: pd.DataFrame) -> pd.Series:
        """
        Обнаружение gap'ов в данных.

        Args:
            data: DataFrame с данными

        Returns:
            Series с булевой маской
        """
        # Gap = разница между open и предыдущим close больше порога
        prev_close = data["close"].shift(1)
        gap_pct = abs((data["open"] - prev_close) / prev_close)

        gap_mask = gap_pct > self.gap_threshold_pct

        # Первый бар не может иметь gap
        gap_mask.iloc[0] = False

        logger.debug(f"Обнаружено {gap_mask.sum()} gap'ов")

        return gap_mask
