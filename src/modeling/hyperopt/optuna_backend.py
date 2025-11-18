"""Optuna-based optimizer."""

import time
from typing import Any, Callable, Dict, List, Optional

from .base import BaseOptimizer, OptimizationResult


class OptunaOptimizer(BaseOptimizer):
    """Optimizer using Optuna framework."""

    def __init__(
        self,
        objective_func: Callable,
        search_space: Dict[str, Any],
        metric_name: str = "score",
        direction: str = "maximize",
        n_trials: int = 100,
        timeout: Optional[float] = None,
        callbacks: Optional[List[Callable]] = None,
        mlflow_tracking: bool = True,
        random_state: Optional[int] = None,
        pruning_enabled: bool = True,
        sampler: str = "TPE",  # TPE, Random, CmaEs, Grid
        pruner: str = "Median",  # Median, Hyperband, Percentile
    ):
        """
        Initialize Optuna optimizer.

        Args:
            objective_func: Function to optimize.
            search_space: Dictionary defining search space.
            metric_name: Name of metric.
            direction: 'maximize' or 'minimize'.
            n_trials: Number of trials.
            timeout: Timeout in seconds.
            callbacks: Callback functions.
            mlflow_tracking: Whether to log to MLflow.
            random_state: Random seed.
            pruning_enabled: Enable pruning of unpromising trials.
            sampler: Sampler algorithm.
            pruner: Pruner algorithm.
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

        self.pruning_enabled = pruning_enabled
        self.sampler_name = sampler
        self.pruner_name = pruner

    def _create_sampler(self):
        """Create Optuna sampler."""
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required. Install it with: pip install optuna")

        if self.sampler_name == "TPE":
            return optuna.samplers.TPESampler(seed=self.random_state)
        elif self.sampler_name == "Random":
            return optuna.samplers.RandomSampler(seed=self.random_state)
        elif self.sampler_name == "CmaEs":
            return optuna.samplers.CmaEsSampler(seed=self.random_state)
        elif self.sampler_name == "Grid":
            # Grid sampler requires search_space in different format
            return optuna.samplers.GridSampler(self._convert_grid_space())
        else:
            raise ValueError(f"Unknown sampler: {self.sampler_name}")

    def _create_pruner(self):
        """Create Optuna pruner."""
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required. Install it with: pip install optuna")

        if not self.pruning_enabled:
            return optuna.pruners.NopPruner()

        if self.pruner_name == "Median":
            return optuna.pruners.MedianPruner()
        elif self.pruner_name == "Hyperband":
            return optuna.pruners.HyperbandPruner()
        elif self.pruner_name == "Percentile":
            return optuna.pruners.PercentilePruner(percentile=25.0)
        else:
            raise ValueError(f"Unknown pruner: {self.pruner_name}")

    def _convert_grid_space(self) -> Dict[str, List[Any]]:
        """Convert search space to grid format for GridSampler."""
        import numpy as np

        grid_space = {}

        for param_name, param_config in self.search_space.items():
            param_type = param_config.get("type")

            if param_type == "int":
                low = param_config["low"]
                high = param_config["high"]
                step = param_config.get("step", 1)
                grid_space[param_name] = list(range(low, high + 1, step))

            elif param_type == "float":
                low = param_config["low"]
                high = param_config["high"]
                n_points = param_config.get("n_points", 10)
                log_scale = param_config.get("log", False)

                if log_scale:
                    grid_space[param_name] = np.logspace(np.log10(low), np.log10(high), n_points).tolist()
                else:
                    grid_space[param_name] = np.linspace(low, high, n_points).tolist()

            elif param_type == "categorical":
                grid_space[param_name] = param_config["choices"]

        return grid_space

    def _suggest_params(self, trial_optuna) -> Dict[str, Any]:
        """Suggest parameters using Optuna trial."""
        params = {}

        for param_name, param_config in self.search_space.items():
            param_type = param_config.get("type")

            if param_type == "int":
                low = param_config["low"]
                high = param_config["high"]
                params[param_name] = trial_optuna.suggest_int(param_name, low, high)

            elif param_type == "float":
                low = param_config["low"]
                high = param_config["high"]
                log_scale = param_config.get("log", False)

                params[param_name] = trial_optuna.suggest_float(param_name, low, high, log=log_scale)

            elif param_type == "categorical":
                choices = param_config["choices"]
                params[param_name] = trial_optuna.suggest_categorical(param_name, choices)

            else:
                raise ValueError(f"Unknown parameter type: {param_type}")

        return params

    def optimize(self) -> OptimizationResult:
        """Run Optuna optimization."""
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required. Install it with: pip install optuna")

        self._start_time = time.time()
        self.trials = []
        self.best_trial = None

        # Create study
        study = optuna.create_study(
            direction=self.direction,
            sampler=self._create_sampler(),
            pruner=self._create_pruner(),
        )

        print(
            f"Optuna optimization: {self.n_trials} trials " f"(sampler={self.sampler_name}, pruner={self.pruner_name})"
        )

        # Objective function wrapper
        def objective(trial_optuna):
            # Suggest parameters
            params = self._suggest_params(trial_optuna)

            # Create our trial object
            trial_id = len(self.trials)
            trial = self._evaluate_trial(params, trial_id)
            self.trials.append(trial)

            # Update best trial
            self._update_best_trial(trial)

            # Print progress
            if (trial_id + 1) % max(1, self.n_trials // 10) == 0:
                print(f"Progress: {trial_id + 1}/{self.n_trials} " f"(Best: {self.best_trial.value:.4f})")

            return trial.value

        # Run optimization
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False,
        )

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

        print(f"\nOptuna optimization completed in {elapsed_time:.2f}s")
        print(f"Best {self.metric_name}: {result.best_value:.4f}")
        print(f"Best params: {result.best_params}")

        return result
