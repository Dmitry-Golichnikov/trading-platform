"""Event-driven backtesting engine."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from .execution import ExecutionConfig, ExecutionModel
from .exits import (
    BaseExitRule,
    ReverseSignalExit,
    RiskLimitExit,
    ScaleOutExit,
    StopLossExit,
    TakeProfitExit,
    TimeStopExit,
    TrailingStopExit,
)
from .order import Order, OrderSide, OrderStatus, OrderType
from .portfolio import Portfolio
from .position import Position, PositionSide
from .strategy import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Конфигурация бэктеста."""

    initial_capital: float = 100000
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    allow_short: bool = True
    max_positions: int = 10
    # fixed_amount, fixed_units, percent_equity, risk_based
    position_sizing: str = "percent_equity"
    position_percent: float = 10.0
    risk_percent: float = 1.0

    # Exit rules
    use_stop_loss: bool = True
    stop_loss_type: str = "percent"
    stop_loss_value: float = 1.0

    use_take_profit: bool = True
    take_profit_type: str = "percent"
    take_profit_value: float = 2.0

    use_trailing_stop: bool = False
    trailing_stop_type: str = "percent"
    trailing_stop_distance: float = 1.5

    use_time_stop: bool = False
    time_stop_period: int = 24
    time_stop_mode: str = "bars"

    use_reverse_signal: bool = False
    reverse_signal_source: str = "model"

    use_scale_out: bool = False
    scale_out_levels: List[Dict] = field(default_factory=list)

    use_risk_limit: bool = False
    risk_limit_metric: str = "daily_loss"
    risk_limit_threshold: float = 5.0


@dataclass
class BacktestResult:
    """Результат бэктеста."""

    portfolio: Portfolio
    trades: List[Position]
    equity_curve: pd.DataFrame
    metrics: Dict[str, float]
    config: BacktestConfig
    metadata: Dict[str, Any] = field(default_factory=dict)


class BacktestEngine:
    """Event-driven backtesting engine."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        """Инициализация.

        Args:
            config: Конфигурация бэктеста
        """
        self.config = config or BacktestConfig()

        # Execution model
        exec_config = ExecutionConfig(
            commission_rate=self.config.commission_rate,
            slippage_rate=self.config.slippage_rate,
        )
        self.execution_model = ExecutionModel(exec_config)

        # Exit rules
        self.exit_rules: List[BaseExitRule] = []
        self._setup_exit_rules()

        # State
        self.portfolio: Optional[Portfolio] = None
        self.current_bar_index: int = 0

    def _require_portfolio(self) -> Portfolio:
        """Вернуть инициализированный портфель."""
        if self.portfolio is None:
            raise RuntimeError("Portfolio is not initialized. Call run() first.")
        return self.portfolio

    def _setup_exit_rules(self) -> None:
        """Настроить exit rules."""
        # Stop Loss (приоритет 1 - самый высокий)
        if self.config.use_stop_loss:
            self.exit_rules.append(
                StopLossExit(
                    {
                        "type": self.config.stop_loss_type,
                        "value": self.config.stop_loss_value,
                        "enabled": True,
                    }
                )
            )

        # Take Profit (приоритет 2)
        if self.config.use_take_profit:
            self.exit_rules.append(
                TakeProfitExit(
                    {
                        "type": self.config.take_profit_type,
                        "value": self.config.take_profit_value,
                        "enabled": True,
                    }
                )
            )

        # Trailing Stop (приоритет 3)
        if self.config.use_trailing_stop:
            self.exit_rules.append(
                TrailingStopExit(
                    {
                        "type": self.config.trailing_stop_type,
                        "distance": self.config.trailing_stop_distance,
                        "enabled": True,
                    }
                )
            )

        # Time Stop (приоритет 4)
        if self.config.use_time_stop:
            self.exit_rules.append(
                TimeStopExit(
                    {
                        "holding_period": self.config.time_stop_period,
                        "mode": self.config.time_stop_mode,
                        "enabled": True,
                    }
                )
            )

        # Reverse Signal (приоритет 5)
        if self.config.use_reverse_signal:
            self.exit_rules.append(
                ReverseSignalExit(
                    {
                        "signal_source": self.config.reverse_signal_source,
                        "enabled": True,
                    }
                )
            )

        # Scale Out (приоритет 6)
        if self.config.use_scale_out and self.config.scale_out_levels:
            self.exit_rules.append(
                ScaleOutExit(
                    {
                        "levels": self.config.scale_out_levels,
                        "enabled": True,
                    }
                )
            )

        # Risk Limit (приоритет 7 - проверяется всегда)
        if self.config.use_risk_limit:
            self.exit_rules.append(
                RiskLimitExit(
                    {
                        "risk_metric": self.config.risk_limit_metric,
                        "threshold": self.config.risk_limit_threshold,
                        "enabled": True,
                    }
                )
            )

    def run(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        show_progress: bool = True,
    ) -> BacktestResult:
        """Запустить бэктест.

        Args:
            strategy: Торговая стратегия
            data: Исторические данные
            show_progress: Показывать прогресс-бар

        Returns:
            Результат бэктеста
        """
        logger.info("Starting backtest with strategy: %s", strategy.name)
        logger.info(
            "Data shape: %s, period: %s to %s",
            data.shape,
            data.index[0],
            data.index[-1],
        )

        # Инициализация портфеля
        self.portfolio = Portfolio(initial_capital=self.config.initial_capital)
        self.current_bar_index = 0

        portfolio = self._require_portfolio()

        # Итерация по барам
        if show_progress:
            iterator = tqdm(
                data.iterrows(),
                total=len(data),
                desc="Backtesting",
            )
        else:
            iterator = data.iterrows()

        for idx, bar in iterator:
            self.current_bar_index += 1

            # Обновляем exit rules
            self._update_exit_rules(bar)

            # Проверяем exit условия для открытых позиций
            self._check_exits(bar)

            # Генерируем сигнал от стратегии
            signal = strategy.generate_signal(bar, portfolio)

            # Открываем новую позицию если есть сигнал
            if signal != 0:
                self._process_signal(signal, bar, strategy)

            # Обновляем историю капитала
            ticker = bar.get("ticker", "UNKNOWN")
            price = self._extract_price(bar)
            current_prices: Dict[str, float] = {}
            if price is not None:
                current_prices[ticker] = price
            timestamp = self._extract_bar_datetime(bar, idx)
            if timestamp is not None:
                portfolio.update_equity_history(timestamp, current_prices)

            # Callback для стратегии
            strategy.on_bar(bar, portfolio)

        logger.info(f"Backtest completed. Total trades: {portfolio.total_trades}")

        # Формируем результат
        return self._create_result(strategy, data)

    def _update_exit_rules(self, bar: pd.Series) -> None:
        """Обновить состояние exit rules.

        Args:
            bar: Текущий бар
        """
        portfolio = self._require_portfolio()
        portfolio_state = self._get_portfolio_state(bar)

        for ticker, position in portfolio.positions.items():
            if position.is_open:
                for exit_rule in self.exit_rules:
                    exit_rule.update(position, bar, portfolio_state)

    def _check_exits(self, bar: pd.Series) -> None:
        """Проверить условия выхода для всех открытых позиций.

        Args:
            bar: Текущий бар
        """
        portfolio = self._require_portfolio()
        portfolio_state = self._get_portfolio_state(bar)

        # Создаем копию списка позиций для итерации
        positions_to_check = list(portfolio.positions.items())

        for ticker, position in positions_to_check:
            if not position.is_open:
                continue

            # Проверяем приоритет Stop Loss > Take Profit
            # Если оба достигнуты на одном баре, закрываем по Stop Loss
            should_exit = False
            exit_reason = None
            exit_price = None
            sl_triggered = False
            tp_triggered = False

            # Проверяем каждое правило
            for exit_rule in self.exit_rules:
                rule_exit, rule_reason = exit_rule.should_exit(position, bar, portfolio_state)

                if rule_exit:
                    # Определяем тип правила
                    if isinstance(exit_rule, StopLossExit):
                        sl_triggered = True
                        should_exit = True
                        exit_reason = rule_reason
                        exit_price = exit_rule.get_exit_price(position, bar)
                        break  # Stop Loss имеет наивысший приоритет

                    elif isinstance(exit_rule, TakeProfitExit):
                        tp_triggered = True
                        if not should_exit:  # Только если еще не решили выходить
                            should_exit = True
                            exit_reason = rule_reason
                            exit_price = exit_rule.get_exit_price(position, bar)

                    else:
                        # Другие правила
                        if not should_exit:
                            should_exit = True
                            exit_reason = rule_reason
                            exit_price = exit_rule.get_exit_price(position, bar)

            # Логирование приоритета SL над TP
            if sl_triggered and tp_triggered:
                logger.debug(
                    "Both SL and TP triggered for %s, closing by SL (priority)",
                    ticker,
                )

            # Закрываем позицию
            if should_exit:
                self._close_position(ticker, bar, exit_reason, exit_price)

    def _process_signal(self, signal: int, bar: pd.Series, strategy: BaseStrategy) -> None:
        """Обработать торговый сигнал.

        Args:
            signal: Сигнал (1 = buy, -1 = sell)
            bar: Текущий бар
            strategy: Стратегия
        """
        portfolio = self._require_portfolio()
        ticker = bar.get("ticker", "UNKNOWN")
        current_price = self._extract_price(bar)
        execution_time = self._extract_bar_datetime(bar)

        if current_price is None or execution_time is None:
            return

        # Проверяем лимиты
        if portfolio.open_positions_count >= self.config.max_positions:
            logger.debug("Max positions reached (%s)", self.config.max_positions)
            return

        # Проверяем существующую позицию
        if portfolio.has_position(ticker):
            existing_position = portfolio.get_position(ticker)
            if existing_position is None:
                return
            # Если сигнал в том же направлении - игнорируем
            if (signal > 0 and existing_position.is_long) or (signal < 0 and existing_position.is_short):
                return
            # Если противоположный - закрываем текущую (опционально)
            if self.config.allow_short:
                self._close_position(ticker, bar, "reverse_signal", current_price)
            else:
                return

        # Определяем размер позиции
        size = strategy.size_position(signal, bar, portfolio)
        if size <= 0:
            return

        # Определяем направление
        side = PositionSide.LONG if signal > 0 else PositionSide.SHORT

        # Создаем и исполняем ордер
        order = Order(
            ticker=ticker,
            side=OrderSide.BUY if signal > 0 else OrderSide.SELL,
            order_type=OrderType.MARKET,
            size=size,
            created_at=execution_time,
        )

        execution = self.execution_model.execute_order(order, bar.to_dict(), execution_time)

        if execution and order.status == OrderStatus.FILLED:
            # Создаем позицию
            position = Position(
                ticker=ticker,
                side=side,
                entry_price=execution.executed_price,
                size=execution.executed_size,
                entry_time=execution.executed_at,
                commission_entry=execution.commission,
                slippage_entry=execution.slippage,
            )

            # Устанавливаем stop loss и take profit
            position.stop_loss = strategy.calculate_stop_loss(side, execution.executed_price, bar)
            position.take_profit = strategy.calculate_take_profit(side, execution.executed_price, bar)

            # Открываем позицию в портфеле
            try:
                portfolio.open_position(position)
                logger.debug(
                    "Opened %s position for %s: price=%.4f, size=%s, SL=%s, TP=%s",
                    side.value,
                    ticker,
                    execution.executed_price,
                    size,
                    position.stop_loss,
                    position.take_profit,
                )

                # Callback для стратегии
                strategy.on_trade(
                    {
                        "type": "entry",
                        "ticker": ticker,
                        "side": side.value,
                        "price": execution.executed_price,
                        "size": size,
                    }
                )

            except ValueError as error:
                logger.warning("Cannot open position: %s", error)

    def _close_position(
        self,
        ticker: str,
        bar: pd.Series,
        exit_reason: Optional[str],
        exit_price: Optional[float] = None,
    ) -> None:
        """Закрыть позицию.

        Args:
            ticker: Тикер
            bar: Текущий бар
            exit_reason: Причина закрытия
            exit_price: Цена закрытия (или None для рыночной)
        """
        portfolio = self._require_portfolio()
        if not portfolio.has_position(ticker):
            return

        position = portfolio.get_position(ticker)
        if position is None or not position.is_open:
            return

        # Определяем цену выхода
        price_value = exit_price if exit_price is not None else self._extract_price(bar)
        if price_value is None:
            return

        current_time = self._extract_bar_datetime(bar)
        if current_time is None:
            return

        reason = exit_reason or "exit"

        # Создаем ордер на закрытие
        order = Order(
            ticker=ticker,
            side=OrderSide.SELL if position.is_long else OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=position.size,
            created_at=current_time,
        )

        execution = self.execution_model.execute_order(order, bar.to_dict(), current_time)

        if execution and order.status == OrderStatus.FILLED:
            # Закрываем позицию
            portfolio.close_position(
                ticker=ticker,
                exit_price=execution.executed_price,
                exit_time=execution.executed_at,
                exit_reason=reason,
                commission=execution.commission,
                slippage=execution.slippage,
            )

            logger.debug(
                "Closed position for %s: price=%.4f, pnl=%.2f, reason=%s",
                ticker,
                execution.executed_price,
                position.realized_pnl,
                reason,
            )

    def _get_portfolio_state(self, bar: pd.Series) -> Dict[str, Any]:
        """Получить состояние портфеля для exit rules.

        Args:
            bar: Текущий бар

        Returns:
            Словарь с состоянием портфеля
        """
        portfolio = self._require_portfolio()
        price = self._extract_price(bar)
        ticker = bar.get("ticker", "UNKNOWN")
        current_prices: Dict[str, float] = {}
        if price is not None:
            current_prices[ticker] = price
        equity = portfolio.equity(current_prices)

        # Рассчитываем дневной PnL (упрощенно)
        daily_pnl = 0.0
        if portfolio.equity_history:
            last_equity = portfolio.equity_history[-1][1]
            daily_pnl = equity - last_equity

        return {
            "current_bar_index": self.current_bar_index,
            "current_equity": equity,
            "initial_capital": portfolio.initial_capital,
            "daily_pnl": daily_pnl,
            "equity_history": portfolio.equity_history,
            "returns": self._calculate_returns(),
        }

    def _calculate_returns(self) -> List[float]:
        """Рассчитать returns для портфеля.

        Returns:
            Список returns
        """
        portfolio = self._require_portfolio()
        if len(portfolio.equity_history) < 2:
            return []

        returns = []
        for i in range(1, len(portfolio.equity_history)):
            prev_equity = portfolio.equity_history[i - 1][1]
            curr_equity = portfolio.equity_history[i][1]
            if prev_equity > 0:
                ret = (curr_equity - prev_equity) / prev_equity
                returns.append(ret)

        return returns

    def _create_result(self, strategy: BaseStrategy, data: pd.DataFrame) -> BacktestResult:
        """Создать результат бэктеста.

        Args:
            strategy: Стратегия
            data: Данные

        Returns:
            Результат бэктеста
        """
        # Создаем DataFrame с equity curve
        portfolio = self._require_portfolio()
        equity_df = pd.DataFrame(portfolio.equity_history, columns=["timestamp", "equity"])
        equity_df["returns"] = equity_df["equity"].pct_change()
        equity_df["cash"] = [cash for _, cash in portfolio.cash_history]

        # Базовые метрики (детальные метрики будут в metrics.py)
        metrics = {
            "total_return": portfolio.total_return,
            "total_pnl": portfolio.total_pnl,
            "total_trades": portfolio.total_trades,
            "winning_trades": portfolio.winning_trades,
            "losing_trades": portfolio.losing_trades,
            "win_rate": portfolio.win_rate,
            "profit_factor": portfolio.profit_factor,
            "average_win": portfolio.average_win,
            "average_loss": portfolio.average_loss,
            "payoff_ratio": portfolio.payoff_ratio,
            "total_commission": portfolio.total_commission,
        }

        return BacktestResult(
            portfolio=portfolio,
            trades=portfolio.closed_positions,
            equity_curve=equity_df,
            metrics=metrics,
            config=self.config,
            metadata={
                "strategy_name": strategy.name,
                "data_shape": data.shape,
                "period": f"{data.index[0]} to {data.index[-1]}",
            },
        )

    @staticmethod
    def _extract_price(bar: pd.Series) -> Optional[float]:
        """Извлечь цену из бара."""
        for key in ("close", "open"):
            value = bar.get(key)
            if value is None:
                continue
            try:
                price = float(value)
            except (TypeError, ValueError):
                continue
            if pd.isna(price):
                continue
            return price
        return None

    @staticmethod
    def _extract_bar_datetime(bar: pd.Series, fallback: Any = None) -> Optional[datetime]:
        """Извлечь временную метку из бара."""
        for key in ("timestamp", "datetime"):
            raw_value = bar.get(key)
            dt_value = BacktestEngine._convert_to_datetime(raw_value)
            if dt_value is not None:
                return dt_value
        return BacktestEngine._convert_to_datetime(fallback)

    @staticmethod
    def _convert_to_datetime(value: Any) -> Optional[datetime]:
        """Преобразовать значение к datetime."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        if isinstance(value, str):
            try:
                return pd.to_datetime(value).to_pydatetime()
            except (ValueError, TypeError):
                return None
        return None
