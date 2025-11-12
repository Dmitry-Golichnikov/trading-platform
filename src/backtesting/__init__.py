"""Модуль бэктестинга."""

from .engine import BacktestConfig, BacktestEngine, BacktestResult
from .execution import ExecutionConfig, ExecutionModel
from .metrics import StrategyMetrics
from .order import Execution, Order, OrderSide, OrderStatus, OrderType
from .portfolio import Portfolio
from .position import Position, PositionSide, PositionStatus
from .strategy import BaseStrategy, ModelBasedStrategy, SimpleMAStrategy
from .visualizer import BacktestVisualizer

__all__ = [
    # Engine
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    # Strategy
    "BaseStrategy",
    "ModelBasedStrategy",
    "SimpleMAStrategy",
    # Position & Portfolio
    "Position",
    "PositionSide",
    "PositionStatus",
    "Portfolio",
    # Order & Execution
    "Order",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "Execution",
    "ExecutionModel",
    "ExecutionConfig",
    # Metrics & Visualization
    "StrategyMetrics",
    "BacktestVisualizer",
]
