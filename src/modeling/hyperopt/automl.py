"""AutoML pipeline for automatic model selection and tuning."""

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..base import BaseModel


class AutoMLPipeline:
    """
    Automated machine learning pipeline.

    Automatically:
    1. Feature selection
    2. Model selection
    3. Hyperparameter tuning
    4. Ensemble construction
    """

    def __init__(
        self,
        model_types: Optional[List[str]] = None,
        metric_name: str = "roc_auc",
        direction: str = "maximize",
        time_budget: int = 3600,
        n_trials_per_model: int = 50,
        top_k_models: int = 5,
        ensemble_method: str = "voting",  # voting, stacking, weighted
        feature_selection: bool = True,
        mlflow_tracking: bool = True,
        random_state: Optional[int] = None,
    ):
        """
        Initialize AutoML pipeline.

        Args:
            model_types: List of model types to try. If None, tries all available.
            metric_name: Metric to optimize.
            direction: 'maximize' or 'minimize'.
            time_budget: Total time budget in seconds.
            n_trials_per_model: Number of hyperparameter trials per model type.
            top_k_models: Number of top models to use in ensemble.
            ensemble_method: Method to combine models.
            feature_selection: Whether to perform feature selection.
            mlflow_tracking: Whether to log to MLflow.
            random_state: Random seed.
        """
        self.model_types = model_types
        self.metric_name = metric_name
        self.direction = direction
        self.time_budget = time_budget
        self.n_trials_per_model = n_trials_per_model
        self.top_k_models = top_k_models
        self.ensemble_method = ensemble_method
        self.feature_selection = feature_selection
        self.mlflow_tracking = mlflow_tracking
        self.random_state = random_state

        self.results: List[Dict[str, Any]] = []
        self.best_model: Optional[BaseModel] = None
        self.ensemble_model: Optional[Any] = None
        self.selected_features: Optional[List[str]] = None

    def _get_default_model_types(self) -> List[str]:
        """Get default model types to try."""
        return [
            "lightgbm",
            "xgboost",
            "catboost",
            "random_forest",
            "logistic_regression",
            "tabnet",
        ]

    def _get_default_search_space(self, model_type: str) -> Dict[str, Any]:
        """Get default search space for model type."""
        if model_type == "lightgbm":
            return {
                "num_leaves": {"type": "int", "low": 20, "high": 150},
                "learning_rate": {"type": "float", "low": 0.001, "high": 0.3, "log": True},
                "feature_fraction": {"type": "float", "low": 0.5, "high": 1.0},
                "bagging_fraction": {"type": "float", "low": 0.5, "high": 1.0},
                "min_child_samples": {"type": "int", "low": 5, "high": 100},
            }

        elif model_type == "xgboost":
            return {
                "max_depth": {"type": "int", "low": 3, "high": 10},
                "learning_rate": {"type": "float", "low": 0.001, "high": 0.3, "log": True},
                "subsample": {"type": "float", "low": 0.5, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
                "min_child_weight": {"type": "int", "low": 1, "high": 10},
            }

        elif model_type == "catboost":
            return {
                "depth": {"type": "int", "low": 4, "high": 10},
                "learning_rate": {"type": "float", "low": 0.001, "high": 0.3, "log": True},
                "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 10.0},
                "bagging_temperature": {"type": "float", "low": 0.0, "high": 1.0},
            }

        elif model_type == "random_forest":
            return {
                "n_estimators": {"type": "int", "low": 50, "high": 500},
                "max_depth": {"type": "int", "low": 5, "high": 30},
                "min_samples_split": {"type": "int", "low": 2, "high": 20},
                "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
            }

        elif model_type == "logistic_regression":
            return {
                "C": {"type": "float", "low": 0.001, "high": 10.0, "log": True},
                "penalty": {"type": "categorical", "choices": ["l1", "l2", "elasticnet"]},
            }

        elif model_type == "tabnet":
            return {
                "n_d": {"type": "categorical", "choices": [8, 16, 32, 64]},
                "n_a": {"type": "categorical", "choices": [8, 16, 32, 64]},
                "n_steps": {"type": "int", "low": 3, "high": 10},
                "gamma": {"type": "float", "low": 1.0, "high": 2.0},
                "learning_rate": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
            }

        else:
            return {}

    def _select_features(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
    ) -> List[str]:
        """Select important features."""
        from ..models.tree_based.lightgbm_model import LightGBMModel

        print("Performing feature selection...")

        # Train a simple LightGBM model for feature importance
        model = LightGBMModel(
            n_estimators=100,
            num_leaves=31,
            random_state=self.random_state,
        )

        # Convert to Series if needed
        y_train_series = pd.Series(y_train) if isinstance(y_train, np.ndarray) else y_train
        y_val_series = pd.Series(y_val) if isinstance(y_val, np.ndarray) else y_val
        model.fit(X_train, y_train_series, X_val=X_val, y_val=y_val_series)

        # Get feature importance
        importance = model.model.feature_importances_

        # Select top features (keep at least 50% of features)
        n_features_to_keep = max(len(X_train.columns) // 2, 1)
        top_indices = np.argsort(importance)[-n_features_to_keep:]
        selected_features = X_train.columns[top_indices].tolist()

        print(f"Selected {len(selected_features)} features out of {len(X_train.columns)}")

        return selected_features

    def _optimize_model(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        search_space: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], float]:
        """Optimize hyperparameters for a single model type."""
        from ..registry import ModelRegistry
        from .optuna_backend import OptunaOptimizer

        print(f"\nOptimizing {model_type}...")

        # Define objective function
        def objective(params: Dict[str, Any]) -> float:
            # Create model
            model_cls = ModelRegistry.get_model(model_type)
            model = model_cls(**params, random_state=self.random_state)

            # Convert to Series if needed
            y_train_series = pd.Series(y_train) if isinstance(y_train, np.ndarray) else y_train
            y_val_series = pd.Series(y_val) if isinstance(y_val, np.ndarray) else y_val

            # Train
            model.fit(X_train, y_train_series, X_val=X_val, y_val=y_val_series)

            # Evaluate
            if hasattr(model, "predict_proba"):
                y_pred = model.predict_proba(X_val)[:, 1]
            else:
                y_pred = model.predict(X_val)

            # Calculate metric
            from sklearn.metrics import accuracy_score, roc_auc_score

            if self.metric_name == "roc_auc":
                score = roc_auc_score(y_val, y_pred)
            elif self.metric_name == "accuracy":
                score = accuracy_score(y_val, (y_pred > 0.5).astype(int))
            else:
                raise ValueError(f"Unsupported metric: {self.metric_name}")

            return score

        # Run optimization
        optimizer = OptunaOptimizer(
            objective_func=objective,
            search_space=search_space,
            metric_name=self.metric_name,
            direction=self.direction,
            n_trials=self.n_trials_per_model,
            mlflow_tracking=self.mlflow_tracking,
            random_state=self.random_state,
            pruning_enabled=True,
        )

        result = optimizer.optimize()

        return result.best_params, result.best_value

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: np.ndarray,
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: np.ndarray,
    ) -> "AutoMLPipeline":
        """
        Fit AutoML pipeline.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.

        Returns:
            Self.
        """
        start_time = time.time()

        # Convert to DataFrame if needed
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        if not isinstance(X_val, pd.DataFrame):
            X_val = pd.DataFrame(X_val)

        # Type narrowing: after conversion, X_train and X_val are DataFrames
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_val, pd.DataFrame)

        print("=" * 70)
        print("AutoML Pipeline Started")
        print("=" * 70)

        # Feature selection
        if self.feature_selection:
            # Convert y to numpy array for _select_features
            y_train_arr = y_train.values if isinstance(y_train, pd.Series) else y_train
            y_val_arr = y_val.values if isinstance(y_val, pd.Series) else y_val
            self.selected_features = self._select_features(X_train, y_train_arr, X_val, y_val_arr)
            X_train = X_train[self.selected_features]
            X_val = X_val[self.selected_features]

        # Get model types to try
        model_types = self.model_types or self._get_default_model_types()

        # Optimize each model type
        for model_type in model_types:
            # Check time budget
            elapsed = time.time() - start_time
            if elapsed > self.time_budget:
                print(f"\nTime budget exceeded ({elapsed:.0f}s > {self.time_budget}s)")
                break

            # Get search space
            search_space = self._get_default_search_space(model_type)

            if not search_space:
                print(f"No search space defined for {model_type}, skipping")
                continue

            # Optimize
            try:
                # Convert y to numpy array for _optimize_model
                y_train_arr = y_train.values if isinstance(y_train, pd.Series) else y_train
                y_val_arr = y_val.values if isinstance(y_val, pd.Series) else y_val
                # X_train and X_val are guaranteed to be DataFrame here
                best_params, best_score = self._optimize_model(
                    model_type, X_train, y_train_arr, X_val, y_val_arr, search_space
                )

                self.results.append(
                    {
                        "model_type": model_type,
                        "params": best_params,
                        "score": best_score,
                    }
                )

                print(f"Best {self.metric_name} for {model_type}: {best_score:.4f}")

            except Exception as e:
                print(f"Error optimizing {model_type}: {e}")
                continue

        # Sort results
        reverse = self.direction == "maximize"
        self.results.sort(key=lambda x: x["score"], reverse=reverse)

        # Train best model
        if self.results:
            best_result = self.results[0]
            print(f"\n{'=' * 70}")
            print(f"Best model: {best_result['model_type']}")
            print(f"Best {self.metric_name}: {best_result['score']:.4f}")
            print(f"{'=' * 70}")

            from ..registry import ModelRegistry

            model_cls = ModelRegistry.get_model(best_result["model_type"])
            self.best_model = model_cls(**best_result["params"], random_state=self.random_state)
            # Convert to Series if needed
            y_train_series = pd.Series(y_train) if isinstance(y_train, np.ndarray) else y_train
            y_val_series = pd.Series(y_val) if isinstance(y_val, np.ndarray) else y_val
            # X_train and X_val are guaranteed to be DataFrame here
            self.best_model.fit(X_train, y_train_series, X_val=X_val, y_val=y_val_series)

        elapsed_time = time.time() - start_time
        print(f"\nAutoML completed in {elapsed_time:.2f}s")

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions."""
        if self.best_model is None:
            raise ValueError("Model not fitted yet")

        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Apply feature selection
        if self.selected_features:
            X = X[self.selected_features]

        return self.best_model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict probabilities."""
        if self.best_model is None:
            raise ValueError("Model not fitted yet")

        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Apply feature selection
        if self.selected_features:
            X = X[self.selected_features]

        if hasattr(self.best_model, "predict_proba"):
            return self.best_model.predict_proba(X)
        else:
            return self.best_model.predict(X)

    def get_leaderboard(self) -> pd.DataFrame:
        """Get leaderboard of all models tried."""
        if not self.results:
            raise ValueError("No results available")

        df = pd.DataFrame(self.results)
        return df.sort_values("score", ascending=(self.direction == "minimize"))
