"""Trailing Stop exit rule."""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..position import Position
from .base import BaseExitRule

logger = logging.getLogger(__name__)


class TrailingStopExit(BaseExitRule):
    """Выход по трейлинг-стопу."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Инициализация.

        Args:
            config: Конфигурация с параметрами:
                - type: "percent", "atr", "step", "indicator"
                - distance: процент/множитель/шаг
                - atr_period: период ATR (для type="atr")
                - step_trigger: расстояние между ступенями
                - update_frequency: "every_bar", "every_n_bars"
        """
        super().__init__(config)
        self.trail_type = self.config.get("type", "percent")
        self.distance = self.config.get("distance", 1.0)
        self.atr_period = self.config.get("atr_period", 14)
        self.step_trigger = self.config.get("step_trigger", 0.5)
        self.update_frequency = self.config.get("update_frequency", "every_bar")

        # Внутреннее состояние для отслеживания лучшей цены
        self._best_price: Dict[str, float] = {}
        self._trailing_level: Dict[str, Optional[float]] = {}
        self._bars_counter: Dict[str, int] = {}

    def should_exit(
        self, position: Position, bar: pd.Series, portfolio_state: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Проверить достижение трейлинг-стопа.

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
        current_price = bar.get("close", bar.get("open"))
        if current_price is None or np.isnan(current_price):
            return False, None

        # Инициализация для новой позиции
        if ticker not in self._best_price:
            self._best_price[ticker] = position.entry_price
            self._trailing_level[ticker] = None
            self._bars_counter[ticker] = 0

        # Обновляем лучшую цену
        if position.is_long:
            if current_price > self._best_price[ticker]:
                self._best_price[ticker] = current_price
        else:
            if current_price < self._best_price[ticker]:
                self._best_price[ticker] = current_price

        # Обновляем трейлинг уровень
        self.update(position, bar, portfolio_state)

        # Проверяем достижение трейлинг-стопа
        trailing_level = self._trailing_level.get(ticker)
        if trailing_level is None:
            return False, None

        reached = False
        if position.is_long:
            # Long: выходим если цена упала ниже трейлинга
            reached = current_price <= trailing_level
        else:
            # Short: выходим если цена поднялась выше трейлинга
            reached = current_price >= trailing_level

        if reached:
            reason = f"trailing_stop_{self.trail_type}_{self.distance}"
            logger.debug(
                f"Trailing stop reached for {ticker}: "
                f"current={current_price:.4f}, trailing={trailing_level:.4f}, "
                f"best={self._best_price[ticker]:.4f}"
            )
            # Очищаем состояние
            self._cleanup_position(ticker)
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

    def update(self, position: Position, bar: pd.Series, portfolio_state: Dict[str, Any]) -> None:
        """Обновить уровень трейлинг-стопа.

        Args:
            position: Текущая позиция
            bar: Текущий бар
            portfolio_state: Состояние портфеля
        """
        ticker = position.ticker
        if ticker not in self._best_price:
            return

        # Проверяем частоту обновления
        self._bars_counter[ticker] = self._bars_counter.get(ticker, 0) + 1
        if self.update_frequency == "every_n_bars":
            n = self.config.get("n_bars", 5)
            if self._bars_counter[ticker] % n != 0:
                return

        # Вычисляем новый уровень трейлинга
        best_price = self._best_price[ticker]
        trailing_level = self._calculate_trailing_level(position, best_price, bar)
        if trailing_level is None:
            return

        # Обновляем уровень (только в сторону прибыли!)
        current_level = self._trailing_level.get(ticker)
        if current_level is None:
            self._trailing_level[ticker] = trailing_level
        else:
            if position.is_long:
                # Long: трейлинг может только расти
                self._trailing_level[ticker] = max(current_level, trailing_level)
            else:
                # Short: трейлинг может только падать
                self._trailing_level[ticker] = min(current_level, trailing_level)

    def _calculate_trailing_level(self, position: Position, best_price: float, bar: pd.Series) -> Optional[float]:
        """Рассчитать уровень трейлинг-стопа.

        Args:
            position: Позиция
            best_price: Лучшая достигнутая цена
            bar: Текущий бар

        Returns:
            Уровень трейлинг-стопа
        """
        if self.trail_type == "percent":
            # Процентный трейлинг
            if position.is_long:
                return best_price * (1 - self.distance / 100)
            else:
                return best_price * (1 + self.distance / 100)

        elif self.trail_type == "atr":
            # ATR-based трейлинг
            atr_col = f"atr_{self.atr_period}"
            atr = bar.get(atr_col, bar.get("atr"))

            if atr is None or np.isnan(atr):
                logger.warning(f"ATR not available for {position.ticker}")
                return None

            if position.is_long:
                return best_price - self.distance * atr
            else:
                return best_price + self.distance * atr

        elif self.trail_type == "step":
            # Ступенчатый трейлинг
            price_gain = abs(best_price - position.entry_price)
            steps = int(price_gain / self.step_trigger)

            if steps > 0:
                step_size = self.distance
                if position.is_long:
                    return position.entry_price + steps * step_size
                else:
                    return position.entry_price - steps * step_size

            # Если еще не прошли первую ступень, используем начальный стоп
            return None

        elif self.trail_type == "indicator":
            # На основе индикатора (например, Parabolic SAR)
            sar = bar.get("sar")
            if sar is not None and not np.isnan(sar):
                return sar

        return None

    def _cleanup_position(self, ticker: str) -> None:
        """Очистить внутреннее состояние для позиции.

        Args:
            ticker: Тикер
        """
        if ticker in self._best_price:
            del self._best_price[ticker]
        if ticker in self._trailing_level:
            del self._trailing_level[ticker]
        if ticker in self._bars_counter:
            del self._bars_counter[ticker]
