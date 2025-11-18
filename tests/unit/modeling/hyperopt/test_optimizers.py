"""Tests for different optimizers."""

from src.modeling.hyperopt.grid_search import GridSearchOptimizer
from src.modeling.hyperopt.random_search import RandomSearchOptimizer


def simple_objective(params):
    """Simple quadratic objective function."""
    x = params.get("x", 0)
    return -((x - 5) ** 2) + 10  # Maximum at x=5


def test_grid_search_optimizer():
    """Test grid search optimizer."""
    search_space = {
        "x": {"type": "int", "low": 0, "high": 10, "step": 1},
    }

    optimizer = GridSearchOptimizer(
        objective_func=simple_objective,
        search_space=search_space,
        metric_name="score",
        direction="maximize",
    )

    result = optimizer.optimize()

    # Check that we tried all values
    assert result.n_trials == 11  # 0 to 10 inclusive

    # Check that best value is near x=5
    assert result.best_params["x"] in [4, 5, 6]
    assert result.best_value >= 9.0  # Should be close to maximum


def test_random_search_optimizer():
    """Test random search optimizer."""
    search_space = {
        "x": {"type": "float", "low": 0.0, "high": 10.0},
    }

    optimizer = RandomSearchOptimizer(
        objective_func=simple_objective,
        search_space=search_space,
        metric_name="score",
        direction="maximize",
        n_trials=50,
        random_state=42,
    )

    result = optimizer.optimize()

    # Check that we ran the correct number of trials
    assert result.n_trials == 50

    # Check that best value is reasonable
    assert result.best_value >= 8.0  # Should be close to maximum

    # Check that best x is near 5
    assert 3.0 <= result.best_params["x"] <= 7.0


def test_grid_search_with_multiple_params():
    """Test grid search with multiple parameters."""

    def multi_param_objective(params):
        x = params["x"]
        y = params["y"]
        return -((x - 5) ** 2) - (y - 3) ** 2 + 20

    search_space = {
        "x": {"type": "int", "low": 0, "high": 10, "step": 2},
        "y": {"type": "int", "low": 0, "high": 6, "step": 1},
    }

    optimizer = GridSearchOptimizer(
        objective_func=multi_param_objective,
        search_space=search_space,
        metric_name="score",
        direction="maximize",
    )

    result = optimizer.optimize()

    # Check combinations
    # x: 0, 2, 4, 6, 8, 10 (6 values)
    # y: 0, 1, 2, 3, 4, 5, 6 (7 values)
    assert result.n_trials == 6 * 7  # 42 combinations

    # Check best params
    assert result.best_params["x"] in [4, 6]  # Closest to 5
    assert result.best_params["y"] == 3


def test_random_search_with_categorical():
    """Test random search with categorical parameter."""

    def categorical_objective(params):
        method = params["method"]
        lr = params["lr"]

        # Different methods have different optima
        if method == "a":
            return -((lr - 0.1) ** 2) + 1.0
        elif method == "b":
            return -((lr - 0.2) ** 2) + 1.5
        else:
            return -((lr - 0.3) ** 2) + 2.0

    search_space = {
        "method": {"type": "categorical", "choices": ["a", "b", "c"]},
        "lr": {"type": "float", "low": 0.01, "high": 0.5},
    }

    optimizer = RandomSearchOptimizer(
        objective_func=categorical_objective,
        search_space=search_space,
        metric_name="score",
        direction="maximize",
        n_trials=30,
        random_state=42,
    )

    result = optimizer.optimize()

    # Best method should be 'c' (highest maximum)
    # But with random search, we might not always find it
    assert result.best_value >= 1.0


def test_optimizer_with_timeout():
    """Test optimizer with timeout."""
    import time

    def slow_objective(params):
        time.sleep(0.1)  # Slow evaluation
        return params["x"]

    search_space = {
        "x": {"type": "int", "low": 0, "high": 100},
    }

    optimizer = RandomSearchOptimizer(
        objective_func=slow_objective,
        search_space=search_space,
        metric_name="score",
        direction="maximize",
        n_trials=100,
        timeout=0.5,  # Short timeout
        random_state=42,
    )

    result = optimizer.optimize()

    # Should have stopped early due to timeout
    assert result.n_trials < 100


def test_optimizer_minimization():
    """Test optimizer with minimization direction."""

    def objective(params):
        x = params["x"]
        return (x - 5) ** 2  # Minimum at x=5

    search_space = {
        "x": {"type": "float", "low": 0.0, "high": 10.0},
    }

    optimizer = RandomSearchOptimizer(
        objective_func=objective,
        search_space=search_space,
        metric_name="loss",
        direction="minimize",
        n_trials=50,
        random_state=42,
    )

    result = optimizer.optimize()

    # Best value should be small (close to 0)
    assert result.best_value <= 1.0

    # Best x should be near 5
    assert 4.0 <= result.best_params["x"] <= 6.0


def test_optimizer_with_failed_trial():
    """Test optimizer handling failed trials."""
    call_count = [0]

    def failing_objective(params):
        call_count[0] += 1
        if call_count[0] == 3:  # Fail on third trial
            raise ValueError("Simulated failure")
        return params["x"]

    search_space = {
        "x": {"type": "int", "low": 0, "high": 10},
    }

    optimizer = RandomSearchOptimizer(
        objective_func=failing_objective,
        search_space=search_space,
        metric_name="score",
        direction="maximize",
        n_trials=5,
        random_state=42,
    )

    result = optimizer.optimize()

    # Should have completed despite failure
    assert result.n_trials == 5

    # One trial should have failed
    failed_trials = [t for t in result.all_trials if t.state == "FAILED"]
    assert len(failed_trials) == 1
