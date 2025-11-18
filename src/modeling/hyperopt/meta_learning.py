"""Meta-learning for warm-start hyperparameter optimization."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
from sklearn.preprocessing import StandardScaler


class MetaLearning:
    """
    Meta-learning layer for warm-start hyperparameter optimization.

    Stores history of experiments and suggests good starting points
    based on dataset meta-features.
    """

    def __init__(self, history_db_path: Union[str, Path]):
        """
        Initialize meta-learning.

        Args:
            history_db_path: Path to JSON file storing experiment history.
        """
        self.history_db_path = Path(history_db_path)
        self.history: List[Dict[str, Any]] = []
        self.scaler = StandardScaler()

        # Load history if exists
        if self.history_db_path.exists():
            self._load_history()

    def _load_history(self):
        """Load history from disk."""
        with open(self.history_db_path, "r") as f:
            self.history = json.load(f)

        print(f"Loaded {len(self.history)} experiments from history")

    def _save_history(self):
        """Save history to disk."""
        self.history_db_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.history_db_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def _compute_dataset_meta_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute meta-features for a dataset.

        Meta-features include:
        - n_samples, n_features
        - Class balance
        - Feature statistics (mean, std, skewness, etc.)
        - Correlation structure
        """
        n_samples, n_features = X.shape

        # Basic stats
        meta_features = {
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "samples_per_feature": float(n_samples / n_features),
        }

        # Target statistics
        if len(np.unique(y)) <= 10:  # Classification
            unique, counts = np.unique(y, return_counts=True)
            meta_features["n_classes"] = len(unique)
            meta_features["class_balance"] = float(counts.min() / counts.max())
            meta_features["majority_class_ratio"] = float(counts.max() / len(y))

        else:  # Regression
            meta_features["target_mean"] = float(y.mean())
            meta_features["target_std"] = float(y.std())
            meta_features["target_skewness"] = float(((y - y.mean()) ** 3).mean() / (y.std() ** 3))

        # Feature statistics
        meta_features["feature_mean_avg"] = float(X.mean(axis=1).mean())
        meta_features["feature_std_avg"] = float(X.std(axis=1).mean())
        meta_features["feature_skewness_avg"] = float(
            ((X - X.mean(axis=0)) ** 3).mean(axis=0).mean() / (X.std(axis=0).mean() ** 3 + 1e-8)
        )

        # Correlation structure
        if n_features > 1:
            corr_matrix = np.corrcoef(X.T)
            # Remove diagonal
            corr_off_diag = corr_matrix[~np.eye(n_features, dtype=bool)]
            meta_features["feature_corr_mean"] = float(np.abs(corr_off_diag).mean())
            meta_features["feature_corr_max"] = float(np.abs(corr_off_diag).max())

        return meta_features

    def _compute_similarity(
        self,
        meta_features1: Dict[str, float],
        meta_features2: Dict[str, float],
    ) -> float:
        """
        Compute similarity between two datasets based on meta-features.

        Uses normalized Euclidean distance.
        """
        # Get common keys
        common_keys = set(meta_features1.keys()) & set(meta_features2.keys())

        if not common_keys:
            return 0.0

        # Extract values
        values1 = np.array([meta_features1[k] for k in common_keys])
        values2 = np.array([meta_features2[k] for k in common_keys])

        # Normalize
        values_concat = np.vstack([values1, values2])
        values_normalized = self.scaler.fit_transform(values_concat)

        # Compute distance
        distance = np.linalg.norm(values_normalized[0] - values_normalized[1])

        # Convert to similarity (0 to 1)
        similarity = 1 / (1 + distance)

        return float(similarity)

    def _find_similar_datasets(
        self,
        dataset_meta: Dict[str, float],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find most similar datasets from history."""
        similarities = []

        for experiment in self.history:
            hist_meta = experiment.get("dataset_meta_features", {})
            similarity = self._compute_similarity(dataset_meta, hist_meta)

            similarities.append(
                {
                    "experiment": experiment,
                    "similarity": float(similarity),
                }
            )

        # Sort by similarity
        # Type: similarity is always float, but mypy sees it as object
        similarities.sort(key=lambda x: cast(float, x["similarity"]), reverse=True)

        return similarities[:top_k]

    def _aggregate_configs(
        self,
        experiments: List[Dict[str, Any]],
        model_type: str,
    ) -> Dict[str, Any]:
        """
        Aggregate hyperparameters from similar experiments.

        Uses weighted average based on similarity and performance.
        """
        # Filter by model type
        filtered = [exp for exp in experiments if exp["experiment"].get("model_type") == model_type]

        if not filtered:
            return {}

        # Collect parameters and weights
        param_values: Dict[str, List[float]] = {}
        weights: List[float] = []

        for exp_data in filtered:
            exp = exp_data["experiment"]
            params = exp.get("best_params", {})

            # Weight by similarity and performance
            similarity = exp_data["similarity"]
            performance = exp.get("best_score", 0.0)
            # Higher performance gets more weight
            weight = similarity * (1 + performance)

            weights.append(weight)

            for key, value in params.items():
                if isinstance(value, (int, float)):
                    if key not in param_values:
                        param_values[key] = []
                    param_values[key].append(value)

        # Compute weighted averages
        weights_array = np.array(weights, dtype=float)
        weights_array = weights_array / weights_array.sum()  # Normalize

        aggregated_params = {}
        for key, values in param_values.items():
            values_array = np.array(values, dtype=float)
            aggregated_params[key] = float(np.average(values_array, weights=weights_array))

        return aggregated_params

    def suggest_starting_point(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        dataset_meta: Optional[Dict[str, float]] = None,
        model_type: str = "lightgbm",
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Suggest starting hyperparameters for optimization.

        Args:
            X: Training features (optional if dataset_meta provided).
            y: Training labels (optional if dataset_meta provided).
            dataset_meta: Pre-computed meta-features.
            model_type: Type of model.
            top_k: Number of similar datasets to consider.

        Returns:
            Dictionary of suggested hyperparameters.
        """
        if not self.history:
            print("No history available, returning empty suggestion")
            return {}

        # Compute meta-features if not provided
        if dataset_meta is None:
            if X is None or y is None:
                raise ValueError("Either dataset_meta or X and y must be provided")
            dataset_meta = self._compute_dataset_meta_features(X, y)

        # Find similar datasets
        similar = self._find_similar_datasets(dataset_meta, top_k)

        if not similar:
            print("No similar datasets found")
            return {}

        # Aggregate configurations
        suggested_params = self._aggregate_configs(similar, model_type)

        if suggested_params:
            print(f"Suggested starting point based on {len(similar)} similar datasets:")
            for key, value in suggested_params.items():
                print(f"  {key}: {value}")

        return suggested_params

    def add_experiment(
        self,
        model_type: str,
        best_params: Dict[str, Any],
        best_score: float,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        dataset_meta: Optional[Dict[str, float]] = None,
        additional_info: Optional[Dict[str, Any]] = None,
    ):
        """
        Add experiment to history.

        Args:
            model_type: Type of model.
            best_params: Best hyperparameters found.
            best_score: Best score achieved.
            X: Training features (optional if dataset_meta provided).
            y: Training labels (optional if dataset_meta provided).
            dataset_meta: Pre-computed meta-features.
            additional_info: Any additional information to store.
        """
        # Compute meta-features if not provided
        if dataset_meta is None:
            if X is not None and y is not None:
                dataset_meta = self._compute_dataset_meta_features(X, y)
            else:
                dataset_meta = {}

        # Create experiment record
        experiment = {
            "model_type": model_type,
            "best_params": best_params,
            "best_score": float(best_score),
            "dataset_meta_features": dataset_meta,
        }

        if additional_info:
            experiment["additional_info"] = additional_info

        # Add to history
        self.history.append(experiment)

        # Save to disk
        self._save_history()

        print(f"Added experiment to meta-learning history (total: {len(self.history)})")

    def get_best_configs_for_model(
        self,
        model_type: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get best configurations for a specific model type."""
        # Filter by model type
        filtered = [exp for exp in self.history if exp.get("model_type") == model_type]

        # Sort by score
        filtered.sort(key=lambda x: x.get("best_score", 0), reverse=True)

        return filtered[:top_k]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the history."""
        if not self.history:
            return {"n_experiments": 0}

        # Count by model type
        model_counts: Dict[str, int] = {}
        for exp in self.history:
            model_type = exp.get("model_type", "unknown")
            model_counts[model_type] = model_counts.get(model_type, 0) + 1

        # Average scores by model type
        model_scores = {}
        for model_type in model_counts.keys():
            scores = [exp.get("best_score", 0) for exp in self.history if exp.get("model_type") == model_type]
            model_scores[model_type] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
            }

        return {
            "n_experiments": len(self.history),
            "model_counts": model_counts,
            "model_scores": model_scores,
        }

    def clear_history(self):
        """Clear all history."""
        self.history = []
        self._save_history()
        print("Cleared meta-learning history")
