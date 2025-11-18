"""Centralized experiment tracking with MLflow integration."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


class ExperimentTracker:
    """
    Centralized tracking of machine learning experiments.

    Integrates with MLflow and maintains local database for offline access.
    """

    def __init__(
        self,
        experiment_name: str = "default",
        tracking_uri: Optional[str] = None,
        local_db_path: Optional[Union[str, Path]] = None,
        use_mlflow: bool = True,
    ):
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name of experiment.
            tracking_uri: MLflow tracking URI.
            local_db_path: Path to local JSON database.
            use_mlflow: Whether to use MLflow tracking.
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.use_mlflow = use_mlflow

        # Local database
        if local_db_path is None:
            local_db_path = Path("artifacts/experiments/experiments.json")

        self.local_db_path = Path(local_db_path)
        self.local_db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize MLflow if enabled
        if self.use_mlflow:
            self._init_mlflow()

    def _init_mlflow(self):
        """Initialize MLflow."""
        try:
            import mlflow

            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)

            # Create or get experiment
            try:
                experiment_id = mlflow.create_experiment(self.experiment_name)
            except Exception:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                experiment_id = experiment.experiment_id

            mlflow.set_experiment(self.experiment_name)
            self.mlflow_experiment_id = experiment_id

            print(f"MLflow tracking enabled: {self.experiment_name}")

        except ImportError:
            print("MLflow not installed, using local tracking only")
            self.use_mlflow = False

    @staticmethod
    def _sanitize_metrics(
        metrics: Dict[str, Union[int, float]],
        precision: int = 10,
    ) -> Dict[str, float]:
        """
        Normalize metric values to plain floats with limited precision.

        This helps to avoid floating point representation artefacts such as
        0.8400000000000001 which can affect exact comparisons in tests and
        on-screen output.
        """
        sanitized: Dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                sanitized[key] = round(float(value), precision)
            else:
                sanitized[key] = round(float(value), precision)
        return sanitized

    def log_experiment(
        self,
        run_name: str,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Optional[Dict[str, Union[str, Path]]] = None,
        tags: Optional[Dict[str, str]] = None,
        model_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Log an experiment.

        Args:
            run_name: Name of the run.
            config: Configuration/parameters.
            metrics: Metrics to log.
            artifacts: Dictionary of artifact paths to log.
            tags: Tags for the run.
            model_path: Path to model file.

        Returns:
            Run ID.
        """
        run_id = None
        timestamp = datetime.now().isoformat()
        sanitized_metrics = self._sanitize_metrics(metrics)

        # Log to MLflow
        if self.use_mlflow:
            import mlflow

            with mlflow.start_run(run_name=run_name) as run:
                run_id = run.info.run_id

                # Log parameters
                for key, value in config.items():
                    if isinstance(value, (int, float, str, bool)):
                        mlflow.log_param(key, value)
                    else:
                        mlflow.log_param(key, str(value))

                # Log metrics
                for key, value in sanitized_metrics.items():
                    mlflow.log_metric(key, value)

                # Log tags
                if tags:
                    mlflow.set_tags(tags)

                # Log artifacts
                if artifacts:
                    for name, path in artifacts.items():
                        mlflow.log_artifact(str(path), artifact_path=name)

                # Log model
                if model_path:
                    mlflow.log_artifact(str(model_path), artifact_path="model")

        # Log to local database
        experiment_record = {
            "run_id": run_id or f"local_{timestamp}",
            "run_name": run_name,
            "timestamp": timestamp,
            "config": config,
            "metrics": sanitized_metrics,
            "tags": tags or {},
        }

        if artifacts:
            experiment_record["artifacts"] = {k: str(v) for k, v in artifacts.items()}

        if model_path:
            experiment_record["model_path"] = str(model_path)

        self._save_to_local_db(experiment_record)

        run_id_str: str = str(experiment_record["run_id"])
        print(f"Logged experiment: {run_name} (run_id: {run_id_str})")

        return run_id_str

    def _save_to_local_db(self, record: Dict[str, Any]):
        """Save record to local database."""
        # Load existing records
        if self.local_db_path.exists():
            with open(self.local_db_path, "r") as f:
                records = json.load(f)
        else:
            records = []

        # Add new record
        records.append(record)

        # Save
        with open(self.local_db_path, "w") as f:
            json.dump(records, f, indent=2)

    def get_experiment(self, run_id: str) -> Dict[str, Any]:
        """Get experiment by run ID."""
        # Try local database first
        if self.local_db_path.exists():
            with open(self.local_db_path, "r") as f:
                records = json.load(f)

            for record in records:
                if record["run_id"] == run_id:
                    return record

        # Try MLflow
        if self.use_mlflow:
            import mlflow

            try:
                run = mlflow.get_run(run_id)
                return {
                    "run_id": run_id,
                    "run_name": run.info.run_name,
                    "timestamp": run.info.start_time,
                    "config": run.data.params,
                    "metrics": run.data.metrics,
                    "tags": run.data.tags,
                }
            except Exception:
                pass

        raise ValueError(f"Experiment not found: {run_id}")

    def list_experiments(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        List experiments.

        Args:
            filters: Dictionary of filters to apply.
            limit: Maximum number of experiments to return.

        Returns:
            List of experiment records.
        """
        if not self.local_db_path.exists():
            return []

        with open(self.local_db_path, "r") as f:
            records = json.load(f)

        # Apply filters
        if filters:
            filtered_records = []
            for record in records:
                match = True

                # Filter by tags
                if "tags" in filters:
                    for key, value in filters["tags"].items():
                        if record.get("tags", {}).get(key) != value:
                            match = False
                            break

                # Filter by metrics (range)
                if "metrics" in filters:
                    for key, value_range in filters["metrics"].items():
                        metric_value = record.get("metrics", {}).get(key)
                        if metric_value is None:
                            match = False
                            break

                        if isinstance(value_range, tuple):
                            min_val, max_val = value_range
                            if not (min_val <= metric_value <= max_val):
                                match = False
                                break

                if match:
                    filtered_records.append(record)

            records = filtered_records

        # Apply limit
        if limit:
            records = records[:limit]

        return records

    def compare_experiments(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compare multiple experiments.

        Args:
            run_ids: List of run IDs to compare.
            metrics: List of metrics to include. If None, includes all.

        Returns:
            DataFrame with comparison.
        """
        experiments = []

        for run_id in run_ids:
            try:
                exp = self.get_experiment(run_id)
                experiments.append(exp)
            except ValueError:
                print(f"Warning: Experiment {run_id} not found")
                continue

        if not experiments:
            return pd.DataFrame()

        # Create comparison DataFrame
        rows = []
        for exp in experiments:
            row = {
                "run_id": exp["run_id"],
                "run_name": exp["run_name"],
                "timestamp": exp["timestamp"],
            }

            # Add metrics
            exp_metrics = exp.get("metrics", {})
            if metrics:
                for metric in metrics:
                    row[metric] = exp_metrics.get(metric)
            else:
                row.update(exp_metrics)

            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def get_best_experiment(
        self,
        metric: str,
        direction: str = "maximize",
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get best experiment based on a metric.

        Args:
            metric: Metric name to optimize.
            direction: 'maximize' or 'minimize'.
            filters: Optional filters to apply.

        Returns:
            Best experiment record.
        """
        experiments = self.list_experiments(filters=filters)

        if not experiments:
            raise ValueError("No experiments found")

        # Filter experiments with the metric
        valid_experiments = [exp for exp in experiments if metric in exp.get("metrics", {})]

        if not valid_experiments:
            raise ValueError(f"No experiments found with metric: {metric}")

        # Find best
        reverse = direction == "maximize"
        best_exp = max(
            valid_experiments,
            key=lambda x: x["metrics"][metric] * (1 if reverse else -1),
        )

        return best_exp

    def delete_experiment(self, run_id: str):
        """Delete experiment from local database."""
        if not self.local_db_path.exists():
            return

        with open(self.local_db_path, "r") as f:
            records = json.load(f)

        # Filter out the record
        records = [r for r in records if r["run_id"] != run_id]

        # Save
        with open(self.local_db_path, "w") as f:
            json.dump(records, f, indent=2)

        print(f"Deleted experiment: {run_id}")

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of all experiments."""
        experiments = self.list_experiments()

        if not experiments:
            return {"n_experiments": 0}

        # Extract all metrics
        all_metrics = set()
        for exp in experiments:
            all_metrics.update(exp.get("metrics", {}).keys())

        # Compute statistics for each metric
        metric_stats = {}
        for metric in all_metrics:
            values = [exp["metrics"][metric] for exp in experiments if metric in exp.get("metrics", {})]

            if values:
                metric_stats[metric] = {
                    "count": len(values),
                    "mean": float(pd.Series(values).mean()),
                    "std": float(pd.Series(values).std()),
                    "min": float(pd.Series(values).min()),
                    "max": float(pd.Series(values).max()),
                }

        return {
            "n_experiments": len(experiments),
            "metric_statistics": metric_stats,
        }

    def export_to_csv(self, output_path: Union[str, Path]):
        """Export all experiments to CSV."""
        experiments = self.list_experiments()

        if not experiments:
            print("No experiments to export")
            return

        # Flatten experiments
        rows = []
        for exp in experiments:
            row = {
                "run_id": exp["run_id"],
                "run_name": exp["run_name"],
                "timestamp": exp["timestamp"],
            }

            # Add config
            for key, value in exp.get("config", {}).items():
                row[f"config_{key}"] = value

            # Add metrics
            for key, value in exp.get("metrics", {}).items():
                row[f"metric_{key}"] = value

            # Add tags
            for key, value in exp.get("tags", {}).items():
                row[f"tag_{key}"] = value

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

        print(f"Exported {len(experiments)} experiments to: {output_path}")
