"""Grid search optimizer."""

import time
from itertools import product
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .base import BaseOptimizer, OptimizationResult


class GridSearchOptimizer(BaseOptimizer):
    """Grid search over hyperparameter space."""

    def __init__(
        self,
        objective_func: Callable,
        search_space: Dict[str, Any],
        metric_name: str = "score",
        direction: str = "maximize",
        timeout: Optional[float] = None,
        callbacks: Optional[List[Callable]] = None,
        mlflow_tracking: bool = True,
        random_state: Optional[int] = None,
    ):
        """Initialize grid search optimizer."""
        super().__init__(
            objective_func=objective_func,
            search_space=search_space,
            metric_name=metric_name,
            direction=direction,
            n_trials=None,  # Will be computed from grid
            timeout=timeout,
            callbacks=callbacks,
            mlflow_tracking=mlflow_tracking,
            random_state=random_state,
        )

    def _generate_grid(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        param_names = list(self.search_space.keys())
        param_values = []

        for param_name, param_config in self.search_space.items():
            param_type = param_config.get("type")

            if param_type == "int":
                low = param_config["low"]
                high = param_config["high"]
                step = param_config.get("step", 1)
                values = list(range(low, high + 1, step))

            elif param_type == "float":
                low = param_config["low"]
                high = param_config["high"]
                n_points = param_config.get("n_points", 10)
                log_scale = param_config.get("log", False)

                if log_scale:
                    values = np.logspace(np.log10(low), np.log10(high), n_points).tolist()
                else:
                    values = np.linspace(low, high, n_points).tolist()

            elif param_type == "categorical":
                values = param_config["choices"]

            else:
                raise ValueError(f"Unknown parameter type: {param_type}")

            param_values.append(values)

        # Generate all combinations
        grid = []
        for combination in product(*param_values):
            params = dict(zip(param_names, combination))
            grid.append(params)

        return grid

    def optimize(self) -> OptimizationResult:
        """Run grid search optimization."""
        self._start_time = time.time()
        self.trials = []
        self.best_trial = None

        # Generate grid
        grid = self._generate_grid()
        n_combinations = len(grid)

        print(f"Grid search: {n_combinations} combinations to evaluate")

        # Evaluate all combinations
        for trial_id, params in enumerate(grid):
            # Check timeout
            if self.timeout is not None:
                elapsed = time.time() - self._start_time
                if elapsed > self.timeout:
                    print(f"Timeout reached after {trial_id} trials")
                    break

            # Evaluate trial
            trial = self._evaluate_trial(params, trial_id)
            self.trials.append(trial)

            # Update best trial
            self._update_best_trial(trial)

            # Print progress
            if (trial_id + 1) % max(1, n_combinations // 10) == 0 and self.best_trial is not None:
                print(f"Progress: {trial_id + 1}/{n_combinations} " f"(Best: {self.best_trial.value:.4f})")

        # Create result
        elapsed_time = time.time() - self._start_time

        if self.best_trial is None:
            raise ValueError("No completed trials found")

        result = OptimizationResult(
            best_params=self.best_trial.params,
            best_value=self.best_trial.value,
            best_trial=self.best_trial,
            all_trials=self.trials,
            n_trials=len(self.trials),
            optimization_time=elapsed_time,
            search_space=self.search_space,
            metric_name=self.metric_name,
            direction=self.direction,
        )

        print(f"\nGrid search completed in {elapsed_time:.2f}s")
        print(f"Best {self.metric_name}: {result.best_value:.4f}")
        print(f"Best params: {result.best_params}")

        return result
