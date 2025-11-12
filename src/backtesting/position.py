"""Управление позициями."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class PositionSide(Enum):
    """Направление позиции."""

    LONG = "long"
    SHORT = "short"


class PositionStatus(Enum):
    """Статус позиции."""

    OPEN = "open"
    CLOSED = "closed"


@dataclass
class Position:
    """Позиция."""

    ticker: str
    side: PositionSide
    entry_price: float
    size: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    time_stop: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    commission_entry: float = 0.0
    commission_exit: float = 0.0
    slippage_entry: float = 0.0
    slippage_exit: float = 0.0
    status: PositionStatus = PositionStatus.OPEN
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        """Позиция открыта."""
        return self.status == PositionStatus.OPEN

    @property
    def is_long(self) -> bool:
        """Лонг позиция."""
        return self.side == PositionSide.LONG

    @property
    def is_short(self) -> bool:
        """Шорт позиция."""
        return self.side == PositionSide.SHORT

    @property
    def entry_value(self) -> float:
        """Стоимость входа."""
        return self.entry_price * self.size

    def current_value(self, current_price: float) -> float:
        """Текущая стоимость позиции."""
        return current_price * self.size

    def unrealized_pnl(self, current_price: float) -> float:
        """Нереализованная прибыль/убыток."""
        if self.is_long:
            pnl = (current_price - self.entry_price) * self.size
        else:
            pnl = (self.entry_price - current_price) * self.size
        return pnl - self.commission_entry

    def unrealized_pnl_percent(self, current_price: float) -> float:
        """Нереализованная прибыль/убыток в процентах."""
        pnl = self.unrealized_pnl(current_price)
        return (pnl / self.entry_value) * 100

    @property
    def realized_pnl(self) -> float:
        """Реализованная прибыль/убыток."""
        if not self.exit_price or self.is_open:
            return 0.0

        if self.is_long:
            pnl = (self.exit_price - self.entry_price) * self.size
        else:
            pnl = (self.entry_price - self.exit_price) * self.size

        return pnl - self.commission_entry - self.commission_exit

    @property
    def realized_pnl_percent(self) -> float:
        """Реализованная прибыль/убыток в процентах."""
        if not self.exit_price or self.is_open:
            return 0.0
        return (self.realized_pnl / self.entry_value) * 100

    @property
    def holding_period(self) -> Optional[float]:
        """Длительность удержания позиции в часах."""
        if not self.exit_time:
            return None
        delta = self.exit_time - self.entry_time
        return delta.total_seconds() / 3600

    def close(
        self,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
        commission: float = 0.0,
        slippage: float = 0.0,
    ) -> None:
        """Закрыть позицию."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = exit_reason
        self.commission_exit = commission
        self.slippage_exit = slippage
        self.status = PositionStatus.CLOSED

    def update_stop_loss(self, new_stop: float) -> None:
        """Обновить стоп-лосс."""
        self.stop_loss = new_stop

    def update_take_profit(self, new_tp: float) -> None:
        """Обновить тейк-профит."""
        self.take_profit = new_tp

    def update_trailing_stop(self, new_trailing: float) -> None:
        """Обновить трейлинг-стоп."""
        self.trailing_stop = new_trailing

    def check_stop_loss(self, current_price: float) -> bool:
        """Проверить достижение стоп-лосса."""
        if self.stop_loss is None:
            return False

        if self.is_long:
            return current_price <= self.stop_loss
        else:
            return current_price >= self.stop_loss

    def check_take_profit(self, current_price: float) -> bool:
        """Проверить достижение тейк-профита."""
        if self.take_profit is None:
            return False

        if self.is_long:
            return current_price >= self.take_profit
        else:
            return current_price <= self.take_profit

    def check_trailing_stop(self, current_price: float) -> bool:
        """Проверить достижение трейлинг-стопа."""
        if self.trailing_stop is None:
            return False

        if self.is_long:
            return current_price <= self.trailing_stop
        else:
            return current_price >= self.trailing_stop

    def check_time_stop(self, current_time: datetime) -> bool:
        """Проверить достижение временного стопа."""
        if not self.time_stop:
            return False
        return current_time >= self.time_stop

    def to_dict(self) -> dict:
        """Преобразовать в словарь."""
        return {
            "ticker": self.ticker,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "size": self.size,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "trailing_stop": self.trailing_stop,
            "time_stop": self.time_stop.isoformat() if self.time_stop else None,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_reason": self.exit_reason,
            "commission_entry": self.commission_entry,
            "commission_exit": self.commission_exit,
            "slippage_entry": self.slippage_entry,
            "slippage_exit": self.slippage_exit,
            "status": self.status.value,
            "realized_pnl": self.realized_pnl,
            "realized_pnl_percent": self.realized_pnl_percent,
            "holding_period": self.holding_period,
            "metadata": self.metadata,
        }
