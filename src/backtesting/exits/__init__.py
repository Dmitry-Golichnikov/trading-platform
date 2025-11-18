"""Exit rules для стратегий."""

from .base import BaseExitRule
from .reverse_signal import ReverseSignalExit
from .risk_limit import RiskLimitExit
from .scale_out import ScaleOutExit
from .stop_loss import StopLossExit
from .take_profit import TakeProfitExit
from .time_stop import TimeStopExit
from .trailing_stop import TrailingStopExit

__all__ = [
    "BaseExitRule",
    "TakeProfitExit",
    "StopLossExit",
    "TrailingStopExit",
    "TimeStopExit",
    "ReverseSignalExit",
    "ScaleOutExit",
    "RiskLimitExit",
]
