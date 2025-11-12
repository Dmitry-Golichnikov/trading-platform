"""Base classes for hyperparameter optimization."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml


@dataclass
class Trial:
    """Single trial in optimization."""

    trial_id: int
    params: Dict[str, Any]
    value: float
    state: str = "COMPLETE"  # COMPLETE, PRUNED, FAILED
    user_attrs: Dict[str, Any] = field(default_factory=dict)
    system_attrs: Dict[str, Any] = field(default_factory=dict)
    datetime_start: Optional[datetime] = None
    datetime_complete: Optional[datetime] = None
    duration: Optional[float] = None

    def __post_init__(self):
        """Calculate duration if not set."""
        if self.duration is None and self.datetime_start and self.datetime_complete:
            self.duration = (self.datetime_complete - self.datetime_start).total_seconds()


@dataclass
class OptimizationResult:
    """Result of optimization."""

    best_params: Dict[str, Any]
    best_value: float
    best_trial: Trial
    all_trials: List[Trial]
    n_trials: int
    optimization_time: float
    search_space: Dict[str, Any]
    metric_name: str
    direction: str  # maximize or minimize

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trials to DataFrame."""
        records = []
        for trial in self.all_trials:
            record = {
                "trial_id": trial.trial_id,
                "value": trial.value,
                "state": trial.state,
                "duration": trial.duration,
            }
            record.update(trial.params)
            records.append(record)

        return pd.DataFrame(records)

    def get_best_n_trials(self, n: int) -> List[Trial]:
        """Get top N trials."""
        completed_trials = [t for t in self.all_trials if t.state == "COMPLETE"]
        reverse = self.direction == "maximize"
        sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=reverse)
        return sorted_trials[:n]

    def save(self, path: Union[str, Path]) -> None:
        """Save optimization result to YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        result_dict = {
            "best_params": self.best_params,
            "best_value": float(self.best_value),
            "n_trials": self.n_trials,
            "optimization_time": self.optimization_time,
            "metric_name": self.metric_name,
            "direction": self.direction,
            "trials": [
                {
                    "trial_id": t.trial_id,
                    "params": t.params,
                    "value": float(t.value),
                    "state": t.state,
                    "duration": t.duration,
                }
                for t in self.all_trials
            ],
        }

        with open(path, "w") as f:
            yaml.dump(result_dict, f, default_flow_style=False)


class BaseOptimizer(ABC):
    """Base class for hyperparameter optimizers."""

    def __init__(
        self,
        objective_func: Callable,
        search_space: Dict[str, Any],
        metric_name: str = "score",
        direction: str = "maximize",
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        callbacks: Optional[List[Callable]] = None,
        mlflow_tracking: bool = True,
        random_state: Optional[int] = None,
    ):
        """
        Initialize optimizer.

        Args:
            objective_func: Function to optimize. Should accept dict of params
                           and return float value.
            search_space: Dictionary defining search space.
            metric_name: Name of the metric being optimized.
            direction: 'maximize' or 'minimize'.
            n_trials: Maximum number of trials.
            timeout: Maximum time in seconds.
            callbacks: List of callback functions.
            mlflow_tracking: Whether to log to MLflow.
            random_state: Random seed for reproducibility.
        """
        self.objective_func = objective_func
        self.search_space = search_space
        self.metric_name = metric_name
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout
        self.callbacks = callbacks or []
        self.mlflow_tracking = mlflow_tracking
        self.random_state = random_state

        self.trials: List[Trial] = []
        self.best_trial: Optional[Trial] = None
        self._start_time: Optional[float] = None

        # Set random seeds
        if random_state is not None:
            np.random.seed(random_state)

    @abstractmethod
    def optimize(self) -> OptimizationResult:
        """Run optimization."""
        pass

    def _should_prune(self, trial: Trial, intermediate_value: float) -> bool:
        """Check if trial should be pruned."""
        # Implement pruning logic in subclasses
        return False

    def _call_callbacks(self, event: str, trial: Trial) -> None:
        """Call all callbacks."""
        for callback in self.callbacks:
            callback(event, trial, self)

    def _evaluate_trial(self, params: Dict[str, Any], trial_id: int) -> Trial:
        """Evaluate a single trial."""
        trial = Trial(
            trial_id=trial_id,
            params=params,
            value=np.nan,
            state="RUNNING",
            datetime_start=datetime.now(),
        )

        self._call_callbacks("trial_start", trial)

        try:
            # Evaluate objective function
            value = self.objective_func(params)

            trial.value = value
            trial.state = "COMPLETE"
            trial.datetime_complete = datetime.now()
            if trial.datetime_start is not None:
                trial.duration = (trial.datetime_complete - trial.datetime_start).total_seconds()

            self._call_callbacks("trial_complete", trial)

            # Log to MLflow if enabled
            if self.mlflow_tracking:
                self._log_to_mlflow(trial)

        except Exception as e:
            trial.state = "FAILED"
            trial.user_attrs["error"] = str(e)
            trial.datetime_complete = datetime.now()
            self._call_callbacks("trial_failed", trial)

        return trial

    def _log_to_mlflow(self, trial: Trial) -> None:
        """Log trial to MLflow."""
        try:
            import mlflow

            with mlflow.start_run(nested=True, run_name=f"trial_{trial.trial_id}"):
                mlflow.log_params(trial.params)
                mlflow.log_metric(self.metric_name, trial.value)
                mlflow.log_metric("duration", trial.duration or 0)
                mlflow.set_tag("state", trial.state)
        except ImportError:
            pass  # MLflow not installed

    def _update_best_trial(self, trial: Trial) -> None:
        """Update best trial if current is better."""
        if trial.state != "COMPLETE":
            return

        if self.best_trial is None:
            self.best_trial = trial
            return

        is_better = (
            trial.value > self.best_trial.value if self.direction == "maximize" else trial.value < self.best_trial.value
        )

        if is_better:
            self.best_trial = trial

    def _parse_search_space_value(self, param_config: Dict[str, Any]) -> Any:
        """Parse a single parameter from search space."""
        param_type = param_config.get("type")

        if param_type == "int":
            low = param_config["low"]
            high = param_config["high"]
            return np.random.randint(low, high + 1)

        elif param_type == "float":
            low = param_config["low"]
            high = param_config["high"]
            log_scale = param_config.get("log", False)

            if log_scale:
                log_low = np.log10(low)
                log_high = np.log10(high)
                value = 10 ** np.random.uniform(log_low, log_high)
            else:
                value = np.random.uniform(low, high)

            return float(value)

        elif param_type == "categorical":
            choices = param_config["choices"]
            return np.random.choice(choices)

        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame."""
        records = []
        for trial in self.trials:
            record = {
                "trial_id": trial.trial_id,
                "value": trial.value,
                "state": trial.state,
                "duration": trial.duration,
            }
            record.update(trial.params)
            records.append(record)

        return pd.DataFrame(records)

    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters found."""
        if self.best_trial is None:
            raise ValueError("No completed trials found")
        return self.best_trial.params

    def get_best_value(self) -> float:
        """Get best value found."""
        if self.best_trial is None:
            raise ValueError("No completed trials found")
        return self.best_trial.value
