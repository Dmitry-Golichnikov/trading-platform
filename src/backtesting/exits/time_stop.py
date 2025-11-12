"""Time Stop exit rule."""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from ..position import Position
from .base import BaseExitRule

logger = logging.getLogger(__name__)


class TimeStopExit(BaseExitRule):
    """Выход по времени."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Инициализация.

        Args:
            config: Конфигурация с параметрами:
                - holding_period: максимальное число баров или временной интервал
                - mode: "bars" или "time" (часы)
                - conditional: отключение если profit > порога
                - profit_threshold: порог прибыли для отключения
        """
        super().__init__(config)
        self.holding_period = self.config.get("holding_period", 24)
        self.mode = self.config.get("mode", "bars")
        self.conditional = self.config.get("conditional", False)
        self.profit_threshold = self.config.get("profit_threshold", 0.5)

        # Счетчик баров для каждой позиции
        self._bars_held: Dict[str, int] = {}

    def should_exit(
        self, position: Position, bar: pd.Series, portfolio_state: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Проверить достижение временного лимита.

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
        current_time = bar.get("timestamp", bar.get("datetime"))
        current_price = bar.get("close", bar.get("open"))

        # Инициализация счетчика для новой позиции
        if ticker not in self._bars_held:
            self._bars_held[ticker] = 0

        # Увеличиваем счетчик баров
        self._bars_held[ticker] += 1

        # Проверяем условие выхода
        should_exit = False

        if self.mode == "bars":
            # Режим по количеству баров
            if self._bars_held[ticker] >= self.holding_period:
                should_exit = True

        elif self.mode == "time":
            # Режим по времени (часы)
            if current_time is not None and position.entry_time is not None:
                current_dt = self._ensure_datetime(current_time)
                entry_dt = self._ensure_datetime(position.entry_time)
                time_held = (current_dt - entry_dt).total_seconds() / 3600
                if time_held >= self.holding_period:
                    should_exit = True

        # Проверяем условие: не выходить если прибыль выше порога
        if should_exit and self.conditional and current_price is not None:
            profit_pct = position.unrealized_pnl_percent(float(current_price))
            if profit_pct > self.profit_threshold:
                logger.debug(
                    f"Time stop disabled for {ticker}: " f"profit {profit_pct:.2f}% > {self.profit_threshold}%"
                )
                should_exit = False

        if should_exit:
            reason = f"time_stop_{self.mode}_{self.holding_period}"
            logger.debug(f"Time stop reached for {ticker}: " f"held {self._bars_held[ticker]} bars")
            # Очищаем состояние
            del self._bars_held[ticker]
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
        """Обновить состояние правила.

        Args:
            position: Текущая позиция
            bar: Текущий бар
            portfolio_state: Состояние портфеля
        """
        # Для time stop не требуется дополнительных обновлений
        pass

    @staticmethod
    def _ensure_datetime(value: Any) -> datetime:
        """Преобразовать значение к datetime."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        if isinstance(value, str):
            return pd.to_datetime(value).to_pydatetime()
        raise TypeError(f"Unsupported datetime value: {value!r}")
