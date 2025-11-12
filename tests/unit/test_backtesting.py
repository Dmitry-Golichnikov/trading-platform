"""Тесты для модуля бэктестинга."""

import numpy as np
import pandas as pd
import pytest

from src.backtesting import (
    BacktestConfig,
    BacktestEngine,
    Portfolio,
    Position,
    PositionSide,
    SimpleMAStrategy,
    StrategyMetrics,
)
from src.backtesting.exits import StopLossExit, TakeProfitExit, TrailingStopExit


@pytest.fixture
def sample_data():
    """Создать тестовые данные."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")

    # Генерируем случайное движение цены
    price = 100
    prices = []
    for _ in range(len(dates)):
        price = price * (1 + np.random.randn() * 0.01)
        prices.append(price)

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": np.random.randint(1000, 10000, len(dates)),
            "ticker": "TEST",
        }
    )

    # Добавляем индикаторы для SimpleMAStrategy
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_30"] = df["close"].rolling(30).mean()
    df["atr"] = df["close"].rolling(14).std()

    return df


def test_position_creation():
    """Тест создания позиции."""
    pos = Position(
        ticker="TEST",
        side=PositionSide.LONG,
        entry_price=100.0,
        size=10,
        entry_time=pd.Timestamp("2024-01-01"),
    )

    assert pos.is_long
    assert not pos.is_short
    assert pos.is_open
    assert pos.entry_value == 1000.0


def test_position_pnl():
    """Тест расчета PnL."""
    pos = Position(
        ticker="TEST",
        side=PositionSide.LONG,
        entry_price=100.0,
        size=10,
        entry_time=pd.Timestamp("2024-01-01"),
        commission_entry=1.0,
    )

    # Unrealized PnL
    unrealized = pos.unrealized_pnl(105.0)
    assert unrealized == 49.0  # (105 - 100) * 10 - 1

    # Close position
    pos.close(105.0, pd.Timestamp("2024-01-02"), "test", commission=1.0)

    # Realized PnL
    assert pos.realized_pnl == 48.0  # (105 - 100) * 10 - 1 - 1


def test_portfolio():
    """Тест портфеля."""
    portfolio = Portfolio(initial_capital=10000)

    assert portfolio.cash == 10000
    assert portfolio.open_positions_count == 0

    # Открываем позицию
    pos = Position(
        ticker="TEST",
        side=PositionSide.LONG,
        entry_price=100.0,
        size=10,
        entry_time=pd.Timestamp("2024-01-01"),
        commission_entry=1.0,
    )

    portfolio.open_position(pos)

    assert portfolio.cash == 8999.0  # 10000 - (100 * 10) - 1
    assert portfolio.open_positions_count == 1
    assert portfolio.has_position("TEST")


def test_stop_loss_exit():
    """Тест StopLossExit."""
    exit_rule = StopLossExit({"type": "percent", "value": 1.0})

    pos = Position(
        ticker="TEST",
        side=PositionSide.LONG,
        entry_price=100.0,
        size=10,
        entry_time=pd.Timestamp("2024-01-01"),
    )

    # Цена не достигла стоп-лосса (1% от 100 = 99)
    # Low должен быть выше стоп-лосса
    bar = pd.Series({"close": 99.5, "low": 99.5, "high": 100.5})
    should_exit, reason = exit_rule.should_exit(pos, bar, {})
    assert not should_exit

    # Цена достигла стоп-лосса (1% от 100 = 99)
    bar = pd.Series({"close": 98.5, "low": 98.0, "high": 99.5})
    should_exit, reason = exit_rule.should_exit(pos, bar, {})
    assert should_exit
    assert "stop_loss" in reason


def test_take_profit_exit():
    """Тест TakeProfitExit."""
    exit_rule = TakeProfitExit({"type": "percent", "value": 2.0})

    pos = Position(
        ticker="TEST",
        side=PositionSide.LONG,
        entry_price=100.0,
        size=10,
        entry_time=pd.Timestamp("2024-01-01"),
    )

    # Цена не достигла тейк-профита
    bar = pd.Series({"close": 101.5})
    should_exit, reason = exit_rule.should_exit(pos, bar, {})
    assert not should_exit

    # Цена достигла тейк-профита (2% от 100 = 102)
    bar = pd.Series({"close": 102.5})
    should_exit, reason = exit_rule.should_exit(pos, bar, {})
    assert should_exit
    assert "take_profit" in reason


def test_trailing_stop_exit():
    """Тест TrailingStopExit."""
    exit_rule = TrailingStopExit({"type": "percent", "distance": 1.0})

    pos = Position(
        ticker="TEST",
        side=PositionSide.LONG,
        entry_price=100.0,
        size=10,
        entry_time=pd.Timestamp("2024-01-01"),
    )

    portfolio_state = {"current_bar_index": 0}

    # Первый бар - цена растет
    bar1 = pd.Series({"close": 105.0, "ticker": "TEST"})
    should_exit, _ = exit_rule.should_exit(pos, bar1, portfolio_state)
    assert not should_exit

    # Второй бар - цена падает, но не до трейлинга
    bar2 = pd.Series({"close": 104.5, "ticker": "TEST"})
    should_exit, _ = exit_rule.should_exit(pos, bar2, portfolio_state)
    assert not should_exit

    # Третий бар - цена падает ниже трейлинга (105 * 0.99 = 103.95)
    bar3 = pd.Series({"close": 103.0, "ticker": "TEST"})
    should_exit, reason = exit_rule.should_exit(pos, bar3, portfolio_state)
    assert should_exit
    assert "trailing_stop" in reason


def test_simple_ma_strategy():
    """Тест SimpleMAStrategy."""
    strategy = SimpleMAStrategy({"fast_period": 10, "slow_period": 30})
    portfolio = Portfolio(initial_capital=10000)

    # Бычий сигнал (fast MA > slow MA)
    bar = pd.Series(
        {
            "sma_10": 105.0,
            "sma_30": 100.0,
            "close": 105.0,
            "ticker": "TEST",
        }
    )
    signal = strategy.generate_signal(bar, portfolio)
    assert signal == 1

    # Медвежий сигнал (fast MA < slow MA)
    bar = pd.Series(
        {
            "sma_10": 95.0,
            "sma_30": 100.0,
            "close": 95.0,
            "ticker": "TEST",
        }
    )
    signal = strategy.generate_signal(bar, portfolio)
    assert signal == -1


def test_backtest_engine_simple(sample_data):
    """Тест BacktestEngine с простой стратегией."""
    config = BacktestConfig(
        initial_capital=10000,
        commission_rate=0.001,
        slippage_rate=0.0005,
        use_stop_loss=True,
        stop_loss_value=2.0,
        use_take_profit=True,
        take_profit_value=3.0,
    )

    strategy = SimpleMAStrategy({"fast_period": 10, "slow_period": 30})
    engine = BacktestEngine(config)

    # Запускаем бэктест
    result = engine.run(strategy, sample_data, show_progress=False)

    # Проверяем результат
    assert result.portfolio is not None
    assert len(result.equity_curve) > 0
    assert "total_return" in result.metrics
    assert "total_trades" in result.metrics


def test_strategy_metrics(sample_data):
    """Тест StrategyMetrics."""
    # Создаем простой портфель и сделки для теста
    portfolio = Portfolio(initial_capital=10000)

    trades = [
        Position(
            ticker="TEST",
            side=PositionSide.LONG,
            entry_price=100.0,
            size=10,
            entry_time=pd.Timestamp("2024-01-01"),
            exit_price=105.0,
            exit_time=pd.Timestamp("2024-01-02"),
            exit_reason="take_profit",
        ),
        Position(
            ticker="TEST",
            side=PositionSide.LONG,
            entry_price=105.0,
            size=10,
            entry_time=pd.Timestamp("2024-01-03"),
            exit_price=103.0,
            exit_time=pd.Timestamp("2024-01-04"),
            exit_reason="stop_loss",
        ),
    ]

    # Создаем equity curve
    equity_curve = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1D"),
            "equity": [10000 + i * 100 for i in range(10)],
        }
    )
    equity_curve["returns"] = equity_curve["equity"].pct_change()

    # Вычисляем метрики
    metrics = StrategyMetrics(portfolio, equity_curve, trades)

    # Проверяем метрики
    sharpe = metrics.sharpe_ratio()
    assert isinstance(sharpe, float)

    max_dd = metrics.max_drawdown()
    assert isinstance(max_dd, float)

    all_metrics = metrics.calculate_all()
    assert "sharpe_ratio" in all_metrics
    assert "max_drawdown" in all_metrics
    assert "sortino_ratio" in all_metrics


def test_backtest_config():
    """Тест BacktestConfig."""
    config = BacktestConfig(
        initial_capital=50000,
        commission_rate=0.002,
        use_stop_loss=True,
        stop_loss_value=1.5,
    )

    assert config.initial_capital == 50000
    assert config.commission_rate == 0.002
    assert config.use_stop_loss is True
    assert config.stop_loss_value == 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
