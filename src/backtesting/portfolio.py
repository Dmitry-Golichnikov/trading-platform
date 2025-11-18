"""Управление портфелем."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .position import Position


@dataclass
class Portfolio:
    """Портфель."""

    initial_capital: float
    cash: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    closed_positions: List[Position] = field(default_factory=list)
    equity_history: List[Tuple[datetime, float]] = field(default_factory=list)
    cash_history: List[Tuple[datetime, float]] = field(default_factory=list)

    def __post_init__(self):
        """Инициализация."""
        if self.cash == 0.0:
            self.cash = self.initial_capital

    def total_positions_value(self, current_prices: Dict[str, float]) -> float:
        """Общая стоимость открытых позиций."""
        total = 0.0
        for ticker, position in self.positions.items():
            if position.is_open and ticker in current_prices:
                total += position.size * current_prices[ticker]
        return total

    def equity(self, current_prices: Dict[str, float]) -> float:
        """Текущий капитал."""
        positions_value = sum(
            position.size * current_prices.get(ticker, position.entry_price)
            for ticker, position in self.positions.items()
            if position.is_open
        )
        return self.cash + positions_value

    @property
    def open_positions_count(self) -> int:
        """Количество открытых позиций."""
        return len([p for p in self.positions.values() if p.is_open])

    @property
    def total_trades(self) -> int:
        """Общее количество сделок."""
        return len(self.closed_positions)

    @property
    def winning_trades(self) -> int:
        """Количество прибыльных сделок."""
        return len([p for p in self.closed_positions if p.realized_pnl > 0])

    @property
    def losing_trades(self) -> int:
        """Количество убыточных сделок."""
        return len([p for p in self.closed_positions if p.realized_pnl < 0])

    @property
    def win_rate(self) -> float:
        """Процент прибыльных сделок."""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    @property
    def total_pnl(self) -> float:
        """Общая прибыль/убыток."""
        return sum(p.realized_pnl for p in self.closed_positions)

    @property
    def total_return(self) -> float:
        """Общая доходность в процентах."""
        if self.initial_capital == 0:
            return 0.0
        return (self.total_pnl / self.initial_capital) * 100

    @property
    def average_win(self) -> float:
        """Средняя прибыль в прибыльных сделках."""
        winning = [p.realized_pnl for p in self.closed_positions if p.realized_pnl > 0]
        return sum(winning) / len(winning) if winning else 0.0

    @property
    def average_loss(self) -> float:
        """Средний убыток в убыточных сделках."""
        losing = [p.realized_pnl for p in self.closed_positions if p.realized_pnl < 0]
        return sum(losing) / len(losing) if losing else 0.0

    @property
    def profit_factor(self) -> float:
        """Profit Factor = Gross Profit / Gross Loss."""
        gross_profit = sum(p.realized_pnl for p in self.closed_positions if p.realized_pnl > 0)
        gross_loss = abs(sum(p.realized_pnl for p in self.closed_positions if p.realized_pnl < 0))
        return gross_profit / gross_loss if gross_loss > 0 else 0.0

    @property
    def payoff_ratio(self) -> float:
        """Payoff Ratio = Average Win / |Average Loss|."""
        if self.average_loss == 0:
            return 0.0
        return self.average_win / abs(self.average_loss)

    @property
    def total_commission(self) -> float:
        """Общие комиссии."""
        return sum(p.commission_entry + p.commission_exit for p in self.closed_positions)

    def has_position(self, ticker: str) -> bool:
        """Есть ли открытая позиция по тикеру."""
        return ticker in self.positions and self.positions[ticker].is_open

    def get_position(self, ticker: str) -> Optional[Position]:
        """Получить позицию по тикеру."""
        return self.positions.get(ticker)

    def open_position(self, position: Position) -> None:
        """Открыть позицию."""
        # Уменьшаем cash на стоимость позиции + комиссии
        cost = position.entry_value + position.commission_entry
        if cost > self.cash:
            raise ValueError(f"Insufficient cash: {self.cash} < {cost} for {position.ticker}")

        self.cash -= cost
        self.positions[position.ticker] = position

    def close_position(
        self,
        ticker: str,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
        commission: float = 0.0,
        slippage: float = 0.0,
    ) -> Optional[Position]:
        """Закрыть позицию."""
        if not self.has_position(ticker):
            return None

        position = self.positions[ticker]
        position.close(exit_price, exit_time, exit_reason, commission, slippage)

        # Возвращаем деньги от закрытия позиции
        proceeds = exit_price * position.size - commission
        self.cash += proceeds

        # Перемещаем в закрытые позиции
        self.closed_positions.append(position)

        return position

    def update_equity_history(self, timestamp: datetime, current_prices: Dict[str, float]) -> None:
        """Обновить историю капитала."""
        equity = self.equity(current_prices)
        self.equity_history.append((timestamp, equity))
        self.cash_history.append((timestamp, self.cash))

    def get_unrealized_pnl(self, current_prices: Dict[str, float]) -> float:
        """Нереализованная прибыль/убыток."""
        total = 0.0
        for ticker, position in self.positions.items():
            if position.is_open and ticker in current_prices:
                total += position.unrealized_pnl(current_prices[ticker])
        return total

    def to_dict(self) -> dict:
        """Преобразовать в словарь."""
        return {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "open_positions": {ticker: pos.to_dict() for ticker, pos in self.positions.items() if pos.is_open},
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "total_return": self.total_return,
            "average_win": self.average_win,
            "average_loss": self.average_loss,
            "profit_factor": self.profit_factor,
            "payoff_ratio": self.payoff_ratio,
            "total_commission": self.total_commission,
        }
