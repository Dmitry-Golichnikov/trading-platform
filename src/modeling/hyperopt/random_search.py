"""Random search optimizer."""

import time
from typing import Any, Callable, Dict, List, Optional

from .base import BaseOptimizer, OptimizationResult


class RandomSearchOptimizer(BaseOptimizer):
    """Random search over hyperparameter space."""

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
    ):
        """Initialize random search optimizer."""
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

    def _sample_params(self) -> Dict[str, Any]:
        """Sample random parameters from search space."""
        params = {}

        for param_name, param_config in self.search_space.items():
            params[param_name] = self._parse_search_space_value(param_config)

        return params

    def optimize(self) -> OptimizationResult:
        """Run random search optimization."""
        self._start_time = time.time()
        self.trials = []
        self.best_trial = None

        if self.n_trials is None:
            raise ValueError("n_trials must be specified for RandomSearchOptimizer")

        print(f"Random search: {self.n_trials} trials")

        # Run trials
        for trial_id in range(self.n_trials):
            # Check timeout
            if self.timeout is not None:
                elapsed = time.time() - self._start_time
                if elapsed > self.timeout:
                    print(f"Timeout reached after {trial_id} trials")
                    break

            # Sample parameters
            params = self._sample_params()

            # Evaluate trial
            trial = self._evaluate_trial(params, trial_id)
            self.trials.append(trial)

            # Update best trial
            self._update_best_trial(trial)

            # Print progress
            if (trial_id + 1) % max(1, self.n_trials // 10) == 0 and self.best_trial is not None:
                print(f"Progress: {trial_id + 1}/{self.n_trials} " f"(Best: {self.best_trial.value:.4f})")

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

        print(f"\nRandom search completed in {elapsed_time:.2f}s")
        print(f"Best {self.metric_name}: {result.best_value:.4f}")
        print(f"Best params: {result.best_params}")

        return result
