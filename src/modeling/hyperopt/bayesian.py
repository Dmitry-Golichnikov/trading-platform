"""Bayesian optimization using scikit-optimize."""

import time
from typing import Any, Callable, Dict, List, Optional

from .base import BaseOptimizer, OptimizationResult


class BayesianOptimizer(BaseOptimizer):
    """Bayesian optimization using Gaussian Processes."""

    def __init__(
        self,
        objective_func: Callable,
        search_space: Dict[str, Any],
        metric_name: str = "score",
        direction: str = "maximize",
        n_trials: int = 100,
        n_initial_points: int = 10,
        timeout: Optional[float] = None,
        callbacks: Optional[List[Callable]] = None,
        mlflow_tracking: bool = True,
        random_state: Optional[int] = None,
        acq_func: str = "EI",  # Expected Improvement
    ):
        """
        Initialize Bayesian optimizer.

        Args:
            objective_func: Function to optimize.
            search_space: Dictionary defining search space.
            metric_name: Name of metric.
            direction: 'maximize' or 'minimize'.
            n_trials: Number of trials.
            n_initial_points: Number of random initial points.
            timeout: Timeout in seconds.
            callbacks: Callback functions.
            mlflow_tracking: Whether to log to MLflow.
            random_state: Random seed.
            acq_func: Acquisition function ('EI', 'LCB', 'PI').
        """
        super().__init__(
            objective_func=objective_func,
            search_space=search_space,
            metric_name=metric_name,
            direction=direction,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=callbacks,
            mlflow_tracking=mlflow_tracking,
            random_state=random_state,
        )

        self.n_initial_points = n_initial_points
        self.acq_func = acq_func

    def _convert_to_skopt_space(self):
        """Convert search space to scikit-optimize format."""
        try:
            from skopt.space import Categorical, Integer, Real
        except ImportError:
            raise ImportError(
                "scikit-optimize is required for Bayesian optimization. " "Install it with: pip install scikit-optimize"
            )

        dimensions = []
        param_names = []

        for param_name, param_config in self.search_space.items():
            param_names.append(param_name)
            param_type = param_config.get("type")

            if param_type == "int":
                low = param_config["low"]
                high = param_config["high"]
                dimensions.append(Integer(low, high, name=param_name))

            elif param_type == "float":
                low = param_config["low"]
                high = param_config["high"]
                log_scale = param_config.get("log", False)

                if log_scale:
                    dimensions.append(Real(low, high, prior="log-uniform", name=param_name))
                else:
                    dimensions.append(Real(low, high, name=param_name))

            elif param_type == "categorical":
                choices = param_config["choices"]
                dimensions.append(Categorical(choices, name=param_name))

            else:
                raise ValueError(f"Unknown parameter type: {param_type}")

        return dimensions, param_names

    def _params_list_to_dict(self, params_list: List[Any], param_names: List[str]) -> Dict[str, Any]:
        """Convert parameter list to dictionary."""
        return dict(zip(param_names, params_list))

    def optimize(self) -> OptimizationResult:
        """Run Bayesian optimization."""
        try:
            from skopt import gp_minimize
        except ImportError:
            raise ImportError(
                "scikit-optimize is required for Bayesian optimization. " "Install it with: pip install scikit-optimize"
            )

        self._start_time = time.time()
        self.trials = []
        self.best_trial = None

        # Convert search space
        dimensions, param_names = self._convert_to_skopt_space()

        print(f"Bayesian optimization: {self.n_trials} trials")

        # Wrapper for objective function
        def objective_wrapper(params_list):
            trial_id = len(self.trials)

            # Check timeout
            if self.timeout is not None:
                elapsed = time.time() - self._start_time
                if elapsed > self.timeout:
                    raise TimeoutError("Optimization timeout reached")

            # Convert params to dict
            params = self._params_list_to_dict(params_list, param_names)

            # Evaluate trial
            trial = self._evaluate_trial(params, trial_id)
            self.trials.append(trial)

            # Update best trial
            self._update_best_trial(trial)

            # Print progress
            if (trial_id + 1) % max(1, self.n_trials // 10) == 0:
                print(f"Progress: {trial_id + 1}/{self.n_trials} " f"(Best: {self.best_trial.value:.4f})")

            # Return value (negate if maximizing, as skopt minimizes)
            value = trial.value
            if self.direction == "maximize":
                value = -value

            return value

        try:
            # Run optimization
            result = gp_minimize(
                func=objective_wrapper,
                dimensions=dimensions,
                n_calls=self.n_trials,
                n_initial_points=self.n_initial_points,
                acq_func=self.acq_func,
                random_state=self.random_state,
                verbose=False,
            )

        except TimeoutError:
            print("Optimization timeout reached")

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

        print(f"\nBayesian optimization completed in {elapsed_time:.2f}s")
        print(f"Best {self.metric_name}: {result.best_value:.4f}")
        print(f"Best params: {result.best_params}")

        return result
