"""Take Profit exit rule."""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..position import Position
from .base import BaseExitRule

logger = logging.getLogger(__name__)


class TakeProfitExit(BaseExitRule):
    """Выход по тейк-профиту."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Инициализация.

        Args:
            config: Конфигурация с параметрами:
                - type: "percent", "atr", "volatility_band", "model_based"
                - value: целевой процент или множитель ATR
                - atr_period: период ATR (для type="atr")
        """
        super().__init__(config)
        self.tp_type = self.config.get("type", "percent")
        self.value = self.config.get("value", 2.0)
        self.atr_period = self.config.get("atr_period", 14)

    def should_exit(
        self, position: Position, bar: pd.Series, portfolio_state: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Проверить достижение тейк-профита.

        Args:
            position: Текущая позиция
            bar: Текущий бар
            portfolio_state: Состояние портфеля

        Returns:
            (should_exit, exit_reason)
        """
        if not self.enabled:
            return False, None

        current_price = bar.get("close", bar.get("open"))
        if current_price is None or np.isnan(current_price):
            return False, None

        # Вычисляем уровень тейк-профита
        tp_level = self._calculate_tp_level(position, bar)
        if tp_level is None:
            return False, None

        # Проверяем достижение уровня
        if position.is_long:
            reached = current_price >= tp_level
        else:
            reached = current_price <= tp_level

        if reached:
            reason = f"take_profit_{self.tp_type}_{self.value}"
            logger.debug(
                f"Take profit reached for {position.ticker}: " f"current={current_price:.4f}, target={tp_level:.4f}"
            )
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

    def _calculate_tp_level(self, position: Position, bar: pd.Series) -> Optional[float]:
        """Рассчитать уровень тейк-профита.

        Args:
            position: Позиция
            bar: Текущий бар

        Returns:
            Уровень тейк-профита
        """
        entry_price = position.entry_price

        if self.tp_type == "percent":
            # Процентный тейк-профит
            if position.is_long:
                return entry_price * (1 + self.value / 100)
            else:
                return entry_price * (1 - self.value / 100)

        elif self.tp_type == "atr":
            # ATR-based тейк-профит
            atr_col = f"atr_{self.atr_period}"
            atr = bar.get(atr_col, bar.get("atr"))

            if atr is None or np.isnan(atr):
                logger.warning(f"ATR not available for {position.ticker}")
                return None

            if position.is_long:
                return entry_price + self.value * atr
            else:
                return entry_price - self.value * atr

        elif self.tp_type == "volatility_band":
            # На основе волатильностных полос
            bb_upper = bar.get("bb_upper")
            bb_lower = bar.get("bb_lower")

            if position.is_long and bb_upper is not None:
                return bb_upper
            elif position.is_short and bb_lower is not None:
                return bb_lower

        elif self.tp_type == "model_based":
            # На основе предсказаний модели
            # Можно использовать predicted_return или target_price
            target_price = bar.get("target_price")
            if target_price is not None and not np.isnan(target_price):
                return target_price

        return None
