"""Scale Out exit rule."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..position import Position
from .base import BaseExitRule

logger = logging.getLogger(__name__)


class ScaleOutExit(BaseExitRule):
    """Частичное закрытие позиции (scale-out)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Инициализация.

        Args:
            config: Конфигурация с параметрами:
                - levels: список уровней и долей
                  [{target: 1.0, qty: 0.5}, {target: 2.0, qty: 0.5}]
                - mode: "percent", "atr", "time", "signal"
                - atr_period: период ATR (для mode="atr")
        """
        super().__init__(config)
        self.levels = self.config.get(
            "levels",
            [{"target": 1.0, "qty": 0.5}, {"target": 2.0, "qty": 0.5}],
        )
        self.mode = self.config.get("mode", "percent")
        self.atr_period = self.config.get("atr_period", 14)

        # Отслеживание выполненных уровней для каждой позиции
        self._executed_levels: Dict[str, List[int]] = {}

    def should_exit(
        self, position: Position, bar: pd.Series, portfolio_state: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Проверить достижение уровня частичного закрытия.

        Args:
            position: Текущая позиция
            bar: Текущий бар
            portfolio_state: Состояние портфеля

        Returns:
            (should_exit, exit_reason) - для scale-out всегда возвращает partial info
        """
        if not self.enabled:
            return False, None

        ticker = position.ticker
        current_price = bar.get("close", bar.get("open"))

        if current_price is None or np.isnan(current_price):
            return False, None

        # Инициализация для новой позиции
        if ticker not in self._executed_levels:
            self._executed_levels[ticker] = []

        # Проверяем каждый уровень
        for i, level in enumerate(self.levels):
            # Пропускаем уже исполненные уровни
            if i in self._executed_levels[ticker]:
                continue

            # Вычисляем целевой уровень
            target_price = self._calculate_target_price(position, level["target"], bar)
            if target_price is None:
                continue

            # Проверяем достижение уровня
            reached = False
            if position.is_long:
                reached = current_price >= target_price
            else:
                reached = current_price <= target_price

            if reached:
                # Сохраняем информацию о частичном закрытии
                qty_to_close = level["qty"]
                reason = f"scale_out_level_{i}_{self.mode}_{level['target']}"

                logger.debug(
                    f"Scale-out level {i} reached for {ticker}: "
                    f"current={current_price:.4f}, target={target_price:.4f}, "
                    f"qty={qty_to_close:.2%}"
                )

                # Отмечаем уровень как исполненный
                self._executed_levels[ticker].append(i)

                # Сохраняем информацию о частичном закрытии в metadata позиции
                if "scale_out_executions" not in position.metadata:
                    position.metadata["scale_out_executions"] = []

                position.metadata["scale_out_executions"].append(
                    {
                        "level": i,
                        "target": level["target"],
                        "qty": qty_to_close,
                        "price": current_price,
                        "bar": bar.get("timestamp", bar.get("datetime")),
                    }
                )

                # Для scale-out возвращаем специальную метку
                # Обработчик должен уменьшить размер позиции, а не закрыть полностью
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

    def get_exit_quantity(self, position: Position, exit_reason: str) -> Optional[float]:
        """Получить количество для закрытия.

        Args:
            position: Позиция
            exit_reason: Причина выхода

        Returns:
            Количество для закрытия
        """
        # Извлекаем номер уровня из exit_reason
        # Формат: "scale_out_level_0_percent_1.0"
        try:
            parts = exit_reason.split("_")
            level_index = int(parts[3])
            level = self.levels[level_index]
            qty_fraction = level["qty"]

            # Возвращаем часть позиции для закрытия
            return position.size * qty_fraction
        except (IndexError, ValueError, KeyError):
            logger.error(f"Cannot parse scale-out level from {exit_reason}")
            return None

    def _calculate_target_price(self, position: Position, target_value: float, bar: pd.Series) -> Optional[float]:
        """Рассчитать целевую цену для уровня.

        Args:
            position: Позиция
            target_value: Целевое значение (процент, ATR множитель, и т.д.)
            bar: Текущий бар

        Returns:
            Целевая цена
        """
        entry_price = position.entry_price

        if self.mode == "percent":
            # Процентные уровни
            if position.is_long:
                return entry_price * (1 + target_value / 100)
            else:
                return entry_price * (1 - target_value / 100)

        elif self.mode == "atr":
            # ATR-based уровни
            atr_col = f"atr_{self.atr_period}"
            atr = bar.get(atr_col, bar.get("atr"))

            if atr is None or np.isnan(atr):
                return None

            if position.is_long:
                return entry_price + target_value * atr
            else:
                return entry_price - target_value * atr

        elif self.mode == "time":
            # Временные уровни - не используют цену, но можем вернуть текущую
            # Логика должна быть в should_exit
            return bar.get("close", bar.get("open"))

        elif self.mode == "signal":
            # На основе сигналов - используем динамические таргеты
            target_price = bar.get(f"target_price_{int(target_value)}")
            if target_price is not None and not np.isnan(target_price):
                return target_price

        return None

    def reset_position(self, ticker: str) -> None:
        """Сбросить состояние для позиции (при полном закрытии).

        Args:
            ticker: Тикер
        """
        if ticker in self._executed_levels:
            del self._executed_levels[ticker]
