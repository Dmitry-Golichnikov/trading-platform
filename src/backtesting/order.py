"""Ордера и исполнение."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class OrderType(Enum):
    """Тип ордера."""

    MARKET = "market"
    LIMIT = "limit"


class OrderSide(Enum):
    """Направление ордера."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Статус ордера."""

    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Ордер."""

    ticker: str
    side: OrderSide
    order_type: OrderType
    size: float
    created_at: datetime
    price: Optional[float] = None  # Для limit orders
    status: OrderStatus = OrderStatus.PENDING
    metadata: dict = field(default_factory=dict)

    def is_buy(self) -> bool:
        """Это ордер на покупку."""
        return self.side == OrderSide.BUY

    def is_sell(self) -> bool:
        """Это ордер на продажу."""
        return self.side == OrderSide.SELL

    def is_market(self) -> bool:
        """Это рыночный ордер."""
        return self.order_type == OrderType.MARKET

    def is_limit(self) -> bool:
        """Это лимитный ордер."""
        return self.order_type == OrderType.LIMIT

    def to_dict(self) -> dict:
        """Преобразовать в словарь."""
        return {
            "ticker": self.ticker,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "size": self.size,
            "price": self.price,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "status": self.status.value,
            "metadata": self.metadata,
        }


@dataclass
class Execution:
    """Исполнение ордера."""

    order: Order
    executed_price: float
    executed_size: float
    executed_at: datetime
    commission: float = 0.0
    slippage: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def executed_value(self) -> float:
        """Исполненная стоимость."""
        return self.executed_price * self.executed_size

    @property
    def total_cost(self) -> float:
        """Полная стоимость с учетом комиссий."""
        return self.executed_value + self.commission

    def to_dict(self) -> dict:
        """Преобразовать в словарь."""
        return {
            "order": self.order.to_dict(),
            "executed_price": self.executed_price,
            "executed_size": self.executed_size,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "commission": self.commission,
            "slippage": self.slippage,
            "executed_value": self.executed_value,
            "total_cost": self.total_cost,
            "metadata": self.metadata,
        }
