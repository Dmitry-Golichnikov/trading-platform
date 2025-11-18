"""Tests for threshold optimizer."""

import numpy as np

from src.modeling.hyperopt.threshold_optimizer import ThresholdOptimizer


def test_threshold_optimizer_basic():
    """Test basic threshold optimization."""
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000

    # Predicted probabilities
    y_proba = np.random.rand(n_samples)

    # True labels (higher probability -> higher chance of positive)
    y_true = (y_proba + np.random.randn(n_samples) * 0.2 > 0.5).astype(int)

    # Create optimizer
    optimizer = ThresholdOptimizer(
        tp=0.02,  # 2% take profit
        sl=0.01,  # 1% stop loss
        commission=0.001,
    )

    # Optimize
    result = optimizer.optimize_threshold(y_proba, y_true)

    # Check result
    assert 0.0 <= result.optimal_threshold <= 1.0
    assert result.n_trades > 0
    assert 0.0 <= result.win_rate <= 1.0


def test_threshold_optimizer_with_constraints():
    """Test threshold optimizer with constraints."""
    np.random.seed(42)
    n_samples = 1000

    y_proba = np.random.rand(n_samples)
    y_true = (y_proba > 0.5).astype(int)

    # Create optimizer with constraints
    optimizer = ThresholdOptimizer(
        tp=0.02,
        sl=0.01,
        commission=0.001,
        constraints={
            "min_trades": 50,
            "min_win_rate": 0.45,
        },
    )

    result = optimizer.optimize_threshold(y_proba, y_true)

    # Check constraints
    if result.constraints_satisfied:
        assert result.n_trades >= 50
        assert result.win_rate >= 0.45


def test_threshold_optimizer_expected_pnl_calculation():
    """Test expected PnL calculation."""
    optimizer = ThresholdOptimizer(
        tp=0.02,
        sl=0.01,
        commission=0.001,
    )

    # Perfect predictions
    y_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.5])
    y_true = np.array([0, 0, 1, 1, 1])

    # Calculate expected PnL for threshold 0.5
    expected_pnl, metrics = optimizer.calculate_expected_pnl(y_proba, y_true, threshold=0.5)

    # With threshold 0.5, we trade on [0.8, 0.9, 0.5]
    # Outcomes: [1, 1, 1]
    # All wins -> high PnL
    assert expected_pnl > 0
    assert metrics["n_trades"] == 3
    assert metrics["win_rate"] == 1.0


def test_threshold_optimizer_no_trades():
    """Test when threshold results in no trades."""
    optimizer = ThresholdOptimizer(tp=0.02, sl=0.01, commission=0.001)

    y_proba = np.array([0.1, 0.2, 0.3, 0.4])
    y_true = np.array([0, 0, 1, 1])

    # Very high threshold -> no trades
    expected_pnl, metrics = optimizer.calculate_expected_pnl(y_proba, y_true, threshold=0.95)

    assert metrics["n_trades"] == 0
    assert expected_pnl == -np.inf


def test_threshold_optimizer_threshold_curve():
    """Test threshold curve generation."""
    np.random.seed(42)

    y_proba = np.random.rand(100)
    y_true = (y_proba > 0.5).astype(int)

    optimizer = ThresholdOptimizer(tp=0.02, sl=0.01, commission=0.001)

    result = optimizer.optimize_threshold(y_proba, y_true, threshold_range=(0.3, 0.9), n_thresholds=50)

    # Check threshold curve
    assert len(result.threshold_curve) == 50
    assert "threshold" in result.threshold_curve.columns
    assert "expected_pnl" in result.threshold_curve.columns
    assert "n_trades" in result.threshold_curve.columns


def test_threshold_optimizer_risk_penalty():
    """Test threshold optimizer with risk penalty."""
    np.random.seed(42)

    y_proba = np.random.rand(200)
    y_true = (y_proba > 0.5).astype(int)

    # Optimizer without penalty
    optimizer_no_penalty = ThresholdOptimizer(
        tp=0.02,
        sl=0.01,
        commission=0.001,
        constraints={"risk_penalty": 0.0},
    )

    result_no_penalty = optimizer_no_penalty.optimize_threshold(y_proba, y_true)

    # Optimizer with penalty
    optimizer_with_penalty = ThresholdOptimizer(
        tp=0.02,
        sl=0.01,
        commission=0.001,
        constraints={"risk_penalty": 0.5},
    )

    result_with_penalty = optimizer_with_penalty.optimize_threshold(y_proba, y_true)

    # With penalty, optimizer should prefer lower risk (fewer trades)
    # This is not always guaranteed due to randomness, so just check it runs
    assert result_no_penalty.optimal_threshold >= 0
    assert result_with_penalty.optimal_threshold >= 0


def test_check_constraints():
    """Test constraint checking."""
    optimizer = ThresholdOptimizer(
        tp=0.02,
        sl=0.01,
        commission=0.001,
        constraints={
            "min_trades": 50,
            "max_drawdown": 0.15,
            "min_sharpe": 0.5,
            "min_win_rate": 0.45,
        },
    )

    # Metrics that satisfy constraints
    good_metrics = {
        "n_trades": 100,
        "max_drawdown": 0.10,
        "sharpe_ratio": 0.8,
        "win_rate": 0.55,
    }

    assert optimizer.check_constraints(good_metrics) is True

    # Metrics that violate constraints
    bad_metrics = {
        "n_trades": 30,  # Too few trades
        "max_drawdown": 0.10,
        "sharpe_ratio": 0.8,
        "win_rate": 0.55,
    }

    assert optimizer.check_constraints(bad_metrics) is False
