"""Reverse Signal exit rule."""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..position import Position
from .base import BaseExitRule

logger = logging.getLogger(__name__)


class ReverseSignalExit(BaseExitRule):
    """Выход по обратному сигналу модели/индикатора."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Инициализация.

        Args:
            config: Конфигурация с параметрами:
                - signal_source: "model", "indicator", "ensemble"
                - threshold: порог вероятности или значение индикатора
                - confirmation: количество баров для подтверждения
                - opposite_position: разрешить разворот
                - cooldown: минимальное время между выходами (баров)
        """
        super().__init__(config)
        self.signal_source: str = self.config.get("signal_source", "model")
        self.threshold: float = self.config.get("threshold", 0.4)
        self.confirmation: int = self.config.get("confirmation", 1)
        self.opposite_position: bool = self.config.get("opposite_position", False)
        self.cooldown: int = self.config.get("cooldown", 0)

        # Счетчики для подтверждения и cooldown
        self._confirmation_counter: Dict[str, int] = {}
        self._last_exit_bar: Dict[str, int] = {}

    def should_exit(
        self, position: Position, bar: pd.Series, portfolio_state: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Проверить обратный сигнал.

        Args:
            position: Текущая позиция
            bar: Текущий бар
            portfolio_state: Состояние портфеля

        Returns:
            (should_exit, exit_reason)
        """
        if not self.enabled:
            return False, None

        ticker = position.ticker

        # Инициализация
        if ticker not in self._confirmation_counter:
            self._confirmation_counter[ticker] = 0
            self._last_exit_bar[ticker] = -999

        # Проверка cooldown
        current_bar = int(portfolio_state.get("current_bar_index", 0))
        last_exit_bar = self._last_exit_bar.get(ticker, -999)
        bars_since_exit = current_bar - last_exit_bar
        if bars_since_exit < self.cooldown:
            return False, None

        # Получаем сигнал
        has_reverse_signal = self._check_reverse_signal(position, bar)

        if has_reverse_signal:
            self._confirmation_counter[ticker] += 1
        else:
            self._confirmation_counter[ticker] = 0

        # Проверяем подтверждение
        if self._confirmation_counter[ticker] >= self.confirmation:
            reason = f"reverse_signal_{self.signal_source}"
            logger.debug(
                f"Reverse signal detected for {ticker}: " f"confirmed {self._confirmation_counter[ticker]} bars"
            )
            # Очищаем и обновляем состояние
            self._confirmation_counter[ticker] = 0
            self._last_exit_bar[ticker] = current_bar
            return True, reason

        return False, None

    def get_exit_price(self, position: Position, bar: pd.Series) -> Optional[float]:
        """Получить цену выхода.

        Args:
            position: Позиция
            bar: Текущий бар

        Returns:
            Цена выхода (рыночная)
        """
        return bar.get("close", bar.get("open"))

    def _check_reverse_signal(self, position: Position, bar: pd.Series) -> bool:
        """Проверить наличие обратного сигнала.

        Args:
            position: Позиция
            bar: Текущий бар

        Returns:
            True если есть обратный сигнал
        """
        if self.signal_source == "model":
            # Проверяем вероятность модели
            prob_up = bar.get("prob_up", bar.get("prediction_proba"))
            if prob_up is None:
                return False
            try:
                prob_value = float(prob_up)
            except (TypeError, ValueError):
                return False
            if np.isnan(prob_value):
                return False

            if position.is_long:
                # Long: выходим если вероятность роста < threshold
                return prob_value < self.threshold
            # Short: выходим если вероятность роста > (1 - threshold)
            return prob_value > (1 - self.threshold)

        elif self.signal_source == "indicator":
            # Проверяем индикаторные сигналы
            # Например, пересечение MA
            signal = bar.get("indicator_signal")
            if signal is not None:
                try:
                    signal_value = float(signal)
                except (TypeError, ValueError):
                    signal_value = None
                if signal_value is not None:
                    if position.is_long and signal_value < 0:
                        return True
                    if position.is_short and signal_value > 0:
                        return True

            # Или RSI
            rsi = bar.get("rsi")
            if rsi is not None:
                try:
                    rsi_value = float(rsi)
                except (TypeError, ValueError):
                    rsi_value = None
                if rsi_value is not None and not np.isnan(rsi_value):
                    if position.is_long and rsi_value > 70:
                        return True  # Перекупленность
                    if position.is_short and rsi_value < 30:
                        return True  # Перепроданность

        elif self.signal_source == "ensemble":
            # Проверяем несколько источников
            model_signal = self._check_model_signal(position, bar)
            indicator_signal = self._check_indicator_signal(position, bar)

            # Требуем подтверждение от обоих
            return model_signal and indicator_signal

        return False

    def _check_model_signal(self, position: Position, bar: pd.Series) -> bool:
        """Проверить сигнал модели."""
        prob_up = bar.get("prob_up", bar.get("prediction_proba"))
        if prob_up is None:
            return False
        try:
            prob_value = float(prob_up)
        except (TypeError, ValueError):
            return False
        if np.isnan(prob_value):
            return False
        if position.is_long:
            return prob_value < self.threshold
        return prob_value > (1 - self.threshold)

    def _check_indicator_signal(self, position: Position, bar: pd.Series) -> bool:
        """Проверить индикаторный сигнал."""
        signal = bar.get("indicator_signal")
        if signal is not None:
            try:
                signal_value = float(signal)
            except (TypeError, ValueError):
                signal_value = None
            if signal_value is not None:
                if position.is_long and signal_value < 0:
                    return True
                if position.is_short and signal_value > 0:
                    return True
        return False
