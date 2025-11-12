"""Stop Loss exit rule."""

import logging
from typing import Any, Dict, Optional, cast

import numpy as np
import pandas as pd

from ..position import Position
from .base import BaseExitRule

logger = logging.getLogger(__name__)


class StopLossExit(BaseExitRule):
    """Выход по стоп-лоссу."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Инициализация.

        Args:
            config: Конфигурация с параметрами:
                - type: "percent", "atr", "support_level", "model_based"
                - value: процент или множитель ATR
                - atr_period: период ATR (для type="atr")
                - hard_stop: обязательное исполнение (True) или soft (False)
        """
        super().__init__(config)
        self.sl_type = self.config.get("type", "percent")
        self.value = self.config.get("value", 1.0)
        self.atr_period = self.config.get("atr_period", 14)
        self.hard_stop = self.config.get("hard_stop", True)

    def should_exit(
        self, position: Position, bar: pd.Series, portfolio_state: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Проверить достижение стоп-лосса.

        Args:
            position: Текущая позиция
            bar: Текущий бар
            portfolio_state: Состояние портфеля

        Returns:
            (should_exit, exit_reason)
        """
        if not self.enabled:
            return False, None

        # Для hard stop проверяем low/high бара, для soft - close
        if self.hard_stop:
            # Hard stop: проверяем был ли достигнут уровень внутри бара
            low = bar.get("low")
            high = bar.get("high")
            if low is None or high is None:
                return False, None
            low_value = float(low)
            high_value = float(high)
        else:
            # Soft stop: проверяем только цену закрытия
            current_price = bar.get("close")
            if current_price is None or np.isnan(current_price):
                return False, None
            current_price = float(current_price)

        # Вычисляем уровень стоп-лосса
        sl_level = self._calculate_sl_level(position, bar)
        if sl_level is None:
            return False, None

        # Проверяем достижение уровня
        reached = False
        if self.hard_stop:
            if position.is_long:
                # Long: стоп сработает если low <= sl_level
                reached = low_value <= sl_level
            else:
                # Short: стоп сработает если high >= sl_level
                reached = high_value >= sl_level
        else:
            current_price = cast(float, current_price)
            if position.is_long:
                reached = current_price <= sl_level
            else:
                reached = current_price >= sl_level

        if reached:
            reason = f"stop_loss_{self.sl_type}_{self.value}"
            logger.debug(
                f"Stop loss reached for {position.ticker}: " f"level={sl_level:.4f}, side={position.side.value}"
            )
            return True, reason

        return False, None

    def get_exit_price(self, position: Position, bar: pd.Series) -> Optional[float]:
        """Получить цену выхода.

        Args:
            position: Позиция
            bar: Текущий бар

        Returns:
            Цена выхода
        """
        if self.hard_stop:
            # Hard stop: выходим по уровню стоп-лосса
            sl_level = self._calculate_sl_level(position, bar)
            return sl_level
        else:
            # Soft stop: выходим по рыночной цене
            return bar.get("close", bar.get("open"))

    def _calculate_sl_level(self, position: Position, bar: pd.Series) -> Optional[float]:
        """Рассчитать уровень стоп-лосса.

        Args:
            position: Позиция
            bar: Текущий бар

        Returns:
            Уровень стоп-лосса
        """
        entry_price = position.entry_price

        if self.sl_type == "percent":
            # Процентный стоп-лосс
            if position.is_long:
                return entry_price * (1 - self.value / 100)
            else:
                return entry_price * (1 + self.value / 100)

        elif self.sl_type == "atr":
            # ATR-based стоп-лосс
            atr_col = f"atr_{self.atr_period}"
            atr = bar.get(atr_col, bar.get("atr"))

            if atr is None or np.isnan(atr):
                logger.warning(f"ATR not available for {position.ticker}")
                return None

            if position.is_long:
                return entry_price - self.value * atr
            else:
                return entry_price + self.value * atr

        elif self.sl_type == "support_level":
            # На основе уровней поддержки/сопротивления
            if position.is_long:
                support = bar.get("support_level")
                if support is not None and not np.isnan(support):
                    return support
            else:
                resistance = bar.get("resistance_level")
                if resistance is not None and not np.isnan(resistance):
                    return resistance

        elif self.sl_type == "model_based":
            # На основе предсказаний модели
            model_sl = bar.get("model_stop_loss")
            if model_sl is not None and not np.isnan(model_sl):
                return model_sl

        return None
