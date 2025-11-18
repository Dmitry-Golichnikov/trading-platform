"""Tests for base optimizer."""

import numpy as np

from src.modeling.hyperopt.base import BaseOptimizer, OptimizationResult, Trial


def test_trial_creation():
    """Test trial creation."""
    trial = Trial(
        trial_id=0,
        params={"learning_rate": 0.1, "max_depth": 5},
        value=0.85,
        state="COMPLETE",
    )

    assert trial.trial_id == 0
    assert trial.params["learning_rate"] == 0.1
    assert trial.value == 0.85
    assert trial.state == "COMPLETE"


def test_optimization_result_to_dataframe():
    """Test conversion of optimization result to DataFrame."""
    trials = [
        Trial(trial_id=0, params={"lr": 0.1}, value=0.8, state="COMPLETE"),
        Trial(trial_id=1, params={"lr": 0.2}, value=0.85, state="COMPLETE"),
    ]

    result = OptimizationResult(
        best_params={"lr": 0.2},
        best_value=0.85,
        best_trial=trials[1],
        all_trials=trials,
        n_trials=2,
        optimization_time=10.0,
        search_space={"lr": {"type": "float", "low": 0.01, "high": 0.5}},
        metric_name="accuracy",
        direction="maximize",
    )

    df = result.to_dataframe()

    assert len(df) == 2
    assert "trial_id" in df.columns
    assert "value" in df.columns
    assert "lr" in df.columns


def test_optimization_result_get_best_n_trials():
    """Test getting best N trials."""
    trials = [
        Trial(trial_id=0, params={"lr": 0.1}, value=0.8, state="COMPLETE"),
        Trial(trial_id=1, params={"lr": 0.2}, value=0.85, state="COMPLETE"),
        Trial(trial_id=2, params={"lr": 0.15}, value=0.82, state="COMPLETE"),
    ]

    result = OptimizationResult(
        best_params={"lr": 0.2},
        best_value=0.85,
        best_trial=trials[1],
        all_trials=trials,
        n_trials=3,
        optimization_time=10.0,
        search_space={"lr": {"type": "float", "low": 0.01, "high": 0.5}},
        metric_name="accuracy",
        direction="maximize",
    )

    top_2 = result.get_best_n_trials(2)

    assert len(top_2) == 2
    assert top_2[0].value == 0.85
    assert top_2[1].value == 0.82


def test_parse_search_space_value_int():
    """Test parsing integer parameter from search space."""

    class DummyOptimizer(BaseOptimizer):
        def optimize(self):
            pass

    optimizer = DummyOptimizer(
        objective_func=lambda x: 0,
        search_space={},
        random_state=42,
    )

    param_config = {"type": "int", "low": 10, "high": 100}

    value = optimizer._parse_search_space_value(param_config)

    assert isinstance(value, (int, np.integer))
    assert 10 <= value <= 100


def test_parse_search_space_value_float():
    """Test parsing float parameter from search space."""

    class DummyOptimizer(BaseOptimizer):
        def optimize(self):
            pass

    optimizer = DummyOptimizer(
        objective_func=lambda x: 0,
        search_space={},
        random_state=42,
    )

    param_config = {"type": "float", "low": 0.01, "high": 0.5}

    value = optimizer._parse_search_space_value(param_config)

    assert isinstance(value, float)
    assert 0.01 <= value <= 0.5


def test_parse_search_space_value_categorical():
    """Test parsing categorical parameter from search space."""

    class DummyOptimizer(BaseOptimizer):
        def optimize(self):
            pass

    optimizer = DummyOptimizer(
        objective_func=lambda x: 0,
        search_space={},
        random_state=42,
    )

    param_config = {"type": "categorical", "choices": ["a", "b", "c"]}

    value = optimizer._parse_search_space_value(param_config)

    assert value in ["a", "b", "c"]


def test_parse_search_space_value_log_scale():
    """Test parsing float parameter with log scale."""

    class DummyOptimizer(BaseOptimizer):
        def optimize(self):
            pass

    optimizer = DummyOptimizer(
        objective_func=lambda x: 0,
        search_space={},
        random_state=42,
    )

    param_config = {"type": "float", "low": 0.001, "high": 1.0, "log": True}

    values = [optimizer._parse_search_space_value(param_config) for _ in range(100)]

    # Check that values are in the correct range
    assert all(0.001 <= v <= 1.0 for v in values)


def test_optimization_result_save_load(tmp_path):
    """Test saving and loading optimization result."""
    trials = [
        Trial(trial_id=0, params={"lr": 0.1}, value=0.8, state="COMPLETE"),
        Trial(trial_id=1, params={"lr": 0.2}, value=0.85, state="COMPLETE"),
    ]

    result = OptimizationResult(
        best_params={"lr": 0.2},
        best_value=0.85,
        best_trial=trials[1],
        all_trials=trials,
        n_trials=2,
        optimization_time=10.0,
        search_space={"lr": {"type": "float", "low": 0.01, "high": 0.5}},
        metric_name="accuracy",
        direction="maximize",
    )

    # Save
    save_path = tmp_path / "result.yaml"
    result.save(save_path)

    # Check file exists
    assert save_path.exists()

    # Load and verify
    import yaml

    with open(save_path, "r") as f:
        loaded = yaml.safe_load(f)

    assert loaded["best_value"] == 0.85
    assert loaded["best_params"]["lr"] == 0.2
    assert len(loaded["trials"]) == 2
