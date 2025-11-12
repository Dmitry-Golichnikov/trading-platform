"""Метрики для оценки торговых стратегий."""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from .portfolio import Portfolio
from .position import Position

logger = logging.getLogger(__name__)


class StrategyMetrics:
    """Расчет метрик для торговых стратегий."""

    def __init__(
        self,
        portfolio: Portfolio,
        equity_curve: pd.DataFrame,
        trades: List[Position],
        risk_free_rate: float = 0.0,
    ):
        """Инициализация.

        Args:
            portfolio: Портфель
            equity_curve: DataFrame с историей капитала
            trades: Список закрытых сделок
            risk_free_rate: Безрисковая ставка (годовая)
        """
        self.portfolio = portfolio
        self.equity_curve = equity_curve
        self.trades = trades
        self.risk_free_rate = risk_free_rate

        self._returns: Optional[np.ndarray] = None
        self._drawdowns: Optional[pd.Series] = None

    @property
    def returns(self) -> np.ndarray:
        """Массив returns."""
        if self._returns is None:
            if "returns" in self.equity_curve.columns:
                self._returns = np.asarray(self.equity_curve["returns"].dropna().values, dtype=float)
            else:
                returns_series = self.equity_curve["equity"].pct_change().dropna()
                self._returns = np.asarray(returns_series.values, dtype=float)
        return self._returns

    @property
    def drawdowns(self) -> pd.Series:
        """Серия просадок."""
        if self._drawdowns is None:
            equity = self.equity_curve["equity"]
            running_max = equity.expanding().max()
            self._drawdowns = (equity - running_max) / running_max
        return self._drawdowns

    def sharpe_ratio(self, periods_per_year: int = 252) -> float:
        """Sharpe Ratio.

        Args:
            periods_per_year: Количество периодов в году (252 для дневных данных)

        Returns:
            Sharpe Ratio
        """
        if len(self.returns) == 0:
            return 0.0

        excess_returns = self.returns - self.risk_free_rate / periods_per_year
        if np.std(excess_returns) == 0:
            return 0.0

        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        return sharpe * np.sqrt(periods_per_year)

    def sortino_ratio(self, periods_per_year: int = 252) -> float:
        """Sortino Ratio (использует только downside deviation).

        Args:
            periods_per_year: Количество периодов в году

        Returns:
            Sortino Ratio
        """
        if len(self.returns) == 0:
            return 0.0

        excess_returns = self.returns - self.risk_free_rate / periods_per_year
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0

        downside_std = np.std(downside_returns)
        sortino = np.mean(excess_returns) / downside_std
        return sortino * np.sqrt(periods_per_year)

    def calmar_ratio(self) -> float:
        """Calmar Ratio = Annual Return / Max Drawdown.

        Returns:
            Calmar Ratio
        """
        max_dd = self.max_drawdown()
        if max_dd == 0:
            return 0.0

        annual_return = self.annualized_return()
        return annual_return / abs(max_dd)

    def max_drawdown(self) -> float:
        """Максимальная просадка (%).

        Returns:
            Максимальная просадка
        """
        return self.drawdowns.min() * 100

    def average_drawdown(self) -> float:
        """Средняя просадка (%).

        Returns:
            Средняя просадка
        """
        negative_dd = self.drawdowns[self.drawdowns < 0]
        if len(negative_dd) == 0:
            return 0.0
        return negative_dd.mean() * 100

    def max_drawdown_duration(self) -> int:
        """Максимальная длительность просадки (в барах).

        Returns:
            Длительность в барах
        """
        equity = self.equity_curve["equity"]
        running_max = equity.expanding().max()
        underwater = equity < running_max

        if not underwater.any():
            return 0

        # Группируем последовательные просадки
        groups = (underwater != underwater.shift()).cumsum()
        durations = underwater.groupby(groups).sum()

        return int(durations.max()) if len(durations) > 0 else 0

    def annualized_return(self, periods_per_year: int = 252) -> float:
        """Годовая доходность (%).

        Args:
            periods_per_year: Количество периодов в году

        Returns:
            Годовая доходность
        """
        if len(self.equity_curve) == 0:
            return 0.0

        initial = self.equity_curve["equity"].iloc[0]
        final = self.equity_curve["equity"].iloc[-1]
        n_periods = len(self.equity_curve)

        if initial == 0 or n_periods == 0:
            return 0.0

        total_return = (final / initial) - 1
        years = n_periods / periods_per_year
        if years <= 0:
            return 0.0
        annual_return = ((1 + total_return) ** (1 / years)) - 1

        return annual_return * 100

    def annualized_volatility(self, periods_per_year: int = 252) -> float:
        """Годовая волатильность (%).

        Args:
            periods_per_year: Количество периодов в году

        Returns:
            Годовая волатильность
        """
        if len(self.returns) == 0:
            return 0.0

        return np.std(self.returns) * np.sqrt(periods_per_year) * 100

    def var(self, confidence: float = 0.95) -> float:
        """Value at Risk на заданном уровне доверия.

        Args:
            confidence: Уровень доверия (0.95 = 95%)

        Returns:
            VaR (%)
        """
        if len(self.returns) == 0:
            return 0.0

        return float(np.percentile(self.returns, (1 - confidence) * 100) * 100)

    def cvar(self, confidence: float = 0.95) -> float:
        """Conditional Value at Risk (Expected Shortfall).

        Args:
            confidence: Уровень доверия

        Returns:
            CVaR (%)
        """
        if len(self.returns) == 0:
            return 0.0

        var_threshold = np.percentile(self.returns, (1 - confidence) * 100)
        tail_returns = self.returns[self.returns <= var_threshold]

        if len(tail_returns) == 0:
            return 0.0

        return float(np.mean(tail_returns) * 100)

    def tail_ratio(self) -> float:
        """Tail Ratio = abs(95th percentile) / abs(5th percentile).

        Returns:
            Tail Ratio
        """
        if len(self.returns) == 0:
            return 0.0

        percentile_95 = float(np.percentile(self.returns, 95))
        percentile_5 = float(np.percentile(self.returns, 5))

        if percentile_5 == 0:
            return 0.0

        return abs(percentile_95) / abs(percentile_5)

    def average_trade_return(self) -> float:
        """Средняя доходность на сделку (%).

        Returns:
            Средняя доходность
        """
        if len(self.trades) == 0:
            return 0.0

        returns = [t.realized_pnl_percent for t in self.trades]
        return float(np.mean(returns))

    def average_holding_period(self) -> float:
        """Средняя длительность удержания позиции (часов).

        Returns:
            Средняя длительность в часах
        """
        if len(self.trades) == 0:
            return 0.0

        periods = [t.holding_period for t in self.trades if t.holding_period is not None]
        if len(periods) == 0:
            return 0.0

        return float(np.mean(periods))

    def turnover(self) -> float:
        """Оборачиваемость портфеля (количество полных оборотов капитала).

        Returns:
            Turnover
        """
        if len(self.trades) == 0:
            return 0.0

        total_volume = float(sum(t.entry_value for t in self.trades))
        if self.portfolio.initial_capital == 0:
            return 0.0
        return float(total_volume / self.portfolio.initial_capital)

    def slippage_adjusted_pnl(self) -> float:
        """PnL с учетом проскальзывания.

        Returns:
            Adjusted PnL
        """
        total_slippage = float(sum(t.slippage_entry + t.slippage_exit for t in self.trades))
        return float(self.portfolio.total_pnl - total_slippage)

    def kelly_criterion(self) -> float:
        """Kelly Criterion для оптимального размера позиции.

        Returns:
            Kelly % (оптимальная доля капитала)
        """
        win_rate = self.portfolio.win_rate / 100
        if win_rate == 0 or win_rate == 1:
            return 0.0

        avg_win = abs(self.portfolio.average_win)
        avg_loss = abs(self.portfolio.average_loss)

        if avg_loss == 0:
            return 0.0

        win_loss_ratio = avg_win / avg_loss
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

        return max(0.0, kelly * 100.0)  # В процентах

    def recovery_factor(self) -> float:
        """Recovery Factor = Net Profit / Max Drawdown.

        Returns:
            Recovery Factor
        """
        max_dd = abs(self.max_drawdown())
        if max_dd == 0:
            return 0.0

        return float(self.portfolio.total_pnl / max_dd)

    def ulcer_index(self) -> float:
        """Ulcer Index - мера риска просадки.

        Returns:
            Ulcer Index
        """
        if len(self.drawdowns) == 0:
            return 0.0

        squared_drawdowns = (self.drawdowns * 100) ** 2
        return float(np.sqrt(squared_drawdowns.mean()))

    def calculate_all(self) -> dict:
        """Рассчитать все метрики.

        Returns:
            Словарь со всеми метриками
        """
        return {
            # Return metrics
            "total_return": self.portfolio.total_return,
            "annualized_return": self.annualized_return(),
            "total_pnl": self.portfolio.total_pnl,
            # Risk metrics
            "sharpe_ratio": self.sharpe_ratio(),
            "sortino_ratio": self.sortino_ratio(),
            "calmar_ratio": self.calmar_ratio(),
            "annualized_volatility": self.annualized_volatility(),
            # Drawdown metrics
            "max_drawdown": self.max_drawdown(),
            "average_drawdown": self.average_drawdown(),
            "max_drawdown_duration": self.max_drawdown_duration(),
            "recovery_factor": self.recovery_factor(),
            "ulcer_index": self.ulcer_index(),
            # Risk metrics
            "var_95": self.var(0.95),
            "cvar_95": self.cvar(0.95),
            "tail_ratio": self.tail_ratio(),
            # Trade metrics
            "total_trades": self.portfolio.total_trades,
            "winning_trades": self.portfolio.winning_trades,
            "losing_trades": self.portfolio.losing_trades,
            "win_rate": self.portfolio.win_rate,
            "profit_factor": self.portfolio.profit_factor,
            "payoff_ratio": self.portfolio.payoff_ratio,
            "average_win": self.portfolio.average_win,
            "average_loss": self.portfolio.average_loss,
            "average_trade_return": self.average_trade_return(),
            "average_holding_period": self.average_holding_period(),
            # Cost metrics
            "total_commission": self.portfolio.total_commission,
            "turnover": self.turnover(),
            "slippage_adjusted_pnl": self.slippage_adjusted_pnl(),
            # Optimization metrics
            "kelly_criterion": self.kelly_criterion(),
        }
