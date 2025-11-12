"""Multi-level hierarchical optimization."""

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

import yaml

from .base import BaseOptimizer, OptimizationResult


class OptimizationLevel:
    """Single level in hierarchical optimization."""

    def __init__(
        self,
        name: str,
        search_space: Dict[str, Any],
        optimizer_type: str = "bayesian",
        optimizer_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize optimization level.

        Args:
            name: Level name (e.g., "data_prep", "features", "model").
            search_space: Search space for this level.
            optimizer_type: Type of optimizer (grid, random, bayesian, optuna).
            optimizer_config: Additional optimizer configuration.
        """
        self.name = name
        self.search_space = search_space
        self.optimizer_type = optimizer_type
        self.optimizer_config = optimizer_config or {}


class MultiLevelOptimizer:
    """
    Multi-level hierarchical optimization with freezing.

    Levels:
    1. Data preparation (filters, samplers)
    2. Features (feature selection)
    3. Model architecture
    4. Model hyperparameters
    5. Training params (LR, optimizer)
    6. Strategy params (thresholds, TP/SL)
    """

    def __init__(
        self,
        objective_func: Callable,
        levels: List[OptimizationLevel],
        metric_name: str = "score",
        direction: str = "maximize",
        mlflow_tracking: bool = True,
        save_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize multi-level optimizer.

        Args:
            objective_func: Function that takes full config and returns metric.
            levels: List of optimization levels.
            metric_name: Name of metric to optimize.
            direction: 'maximize' or 'minimize'.
            mlflow_tracking: Whether to log to MLflow.
            save_dir: Directory to save intermediate results.
        """
        self.objective_func = objective_func
        self.levels = levels
        self.metric_name = metric_name
        self.direction = direction
        self.mlflow_tracking = mlflow_tracking
        self.save_dir = Path(save_dir) if save_dir else None

        self.level_results: Dict[str, OptimizationResult] = {}
        self.best_config: Dict[str, Any] = {}

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def _create_optimizer(self, level: OptimizationLevel, frozen_params: Dict[str, Any]) -> BaseOptimizer:
        """Create optimizer for a single level."""
        from .bayesian import BayesianOptimizer
        from .grid_search import GridSearchOptimizer
        from .optuna_backend import OptunaOptimizer
        from .random_search import RandomSearchOptimizer

        # Create objective function with frozen parameters
        def level_objective(params: Dict[str, Any]) -> float:
            # Combine frozen params with current level params
            full_config = {**frozen_params, **params}
            return self.objective_func(full_config)

        # Select optimizer
        optimizer_cls: Type[BaseOptimizer]
        if level.optimizer_type == "grid":
            optimizer_cls = GridSearchOptimizer
        elif level.optimizer_type == "random":
            optimizer_cls = RandomSearchOptimizer
        elif level.optimizer_type == "bayesian":
            optimizer_cls = BayesianOptimizer
        elif level.optimizer_type == "optuna":
            optimizer_cls = OptunaOptimizer
        else:
            raise ValueError(f"Unknown optimizer type: {level.optimizer_type}")

        # Create optimizer
        optimizer = optimizer_cls(
            objective_func=level_objective,
            search_space=level.search_space,
            metric_name=self.metric_name,
            direction=self.direction,
            mlflow_tracking=self.mlflow_tracking,
            **level.optimizer_config,
        )

        return optimizer

    def optimize_hierarchical(self) -> Dict[str, Any]:
        """
        Run hierarchical optimization.

        Returns:
            Best configuration across all levels.
        """
        start_time = time.time()
        frozen_params: Dict[str, Any] = {}

        print("=" * 70)
        print("MULTI-LEVEL HIERARCHICAL OPTIMIZATION")
        print("=" * 70)

        for i, level in enumerate(self.levels, 1):
            print(f"\n{'=' * 70}")
            print(f"Level {i}/{len(self.levels)}: {level.name}")
            print(f"{'=' * 70}")

            if frozen_params:
                print("Frozen parameters from previous levels:")
                for key, value in frozen_params.items():
                    print(f"  {key}: {value}")
                print()

            # Create and run optimizer
            optimizer = self._create_optimizer(level, frozen_params)
            result = optimizer.optimize()

            # Save result
            self.level_results[level.name] = result

            # Freeze best parameters from this level
            frozen_params.update(result.best_params)

            # Save intermediate result
            if self.save_dir:
                result_path = self.save_dir / f"level_{i}_{level.name}.yaml"
                result.save(result_path)
                print(f"Saved results to: {result_path}")

            print(f"\nBest params for {level.name}:")
            for key, value in result.best_params.items():
                print(f"  {key}: {value}")

        # Store final best config
        self.best_config = frozen_params

        elapsed_time = time.time() - start_time

        print(f"\n{'=' * 70}")
        print("OPTIMIZATION COMPLETED")
        print(f"{'=' * 70}")
        print(f"Total time: {elapsed_time:.2f}s")
        print(f"Best {self.metric_name}: {result.best_value:.4f}")
        print("\nFinal configuration:")
        for key, value in self.best_config.items():
            print(f"  {key}: {value}")

        # Save final config
        if self.save_dir:
            config_path = self.save_dir / "best_config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(self.best_config, f, default_flow_style=False)
            print(f"\nSaved final config to: {config_path}")

        return self.best_config

    def get_level_result(self, level_name: str) -> OptimizationResult:
        """Get optimization result for a specific level."""
        if level_name not in self.level_results:
            raise ValueError(f"No result found for level: {level_name}")
        return self.level_results[level_name]

    def get_best_config(self) -> Dict[str, Any]:
        """Get best configuration across all levels."""
        if not self.best_config:
            raise ValueError("Optimization has not been run yet")
        return self.best_config

    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path],
        objective_func: Callable,
        **kwargs,
    ) -> "MultiLevelOptimizer":
        """
        Create multi-level optimizer from YAML config.

        Config format:
        ```yaml
        metric_name: roc_auc
        direction: maximize

        levels:
          - name: data_prep
            optimizer_type: random
            optimizer_config:
              n_trials: 50
            search_space:
              filter_threshold:
                type: float
                low: 0.01
                high: 0.1

          - name: model
            optimizer_type: bayesian
            optimizer_config:
              n_trials: 100
            search_space:
              learning_rate:
                type: float
                low: 0.001
                high: 0.3
                log: true
        ```
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Parse levels
        levels = []
        for level_config in config["levels"]:
            level = OptimizationLevel(
                name=level_config["name"],
                search_space=level_config["search_space"],
                optimizer_type=level_config.get("optimizer_type", "bayesian"),
                optimizer_config=level_config.get("optimizer_config", {}),
            )
            levels.append(level)

        return cls(
            objective_func=objective_func,
            levels=levels,
            metric_name=config.get("metric_name", "score"),
            direction=config.get("direction", "maximize"),
            **kwargs,
        )
