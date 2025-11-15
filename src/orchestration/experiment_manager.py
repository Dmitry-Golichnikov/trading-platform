"""Experiment manager for coordinating ML experiments."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yaml

from src.orchestration.mlflow_integration import MLflowManager

logger = logging.getLogger(__name__)


class ExperimentManager:
    """
    High-level manager for ML experiments.

    Coordinates:
    - Experiment creation and configuration
    - Pipeline execution
    - Result tracking
    - Model comparison
    """

    def __init__(
        self,
        artifacts_dir: Union[str, Path] = "artifacts",
        mlflow_tracking_uri: Optional[str] = None,
        default_experiment: str = "default",
    ):
        """
        Initialize experiment manager.

        Args:
            artifacts_dir: Base directory for artifacts.
            mlflow_tracking_uri: MLflow tracking URI.
            default_experiment: Default experiment name.
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.experiments_dir = self.artifacts_dir / "experiments"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        # Initialize MLflow manager
        self.mlflow = MLflowManager(
            tracking_uri=mlflow_tracking_uri,
            experiment_name=default_experiment,
        )

        # Experiment database
        self.db_path = self.experiments_dir / "experiments.json"
        self._init_database()

    def _init_database(self):
        """Initialize experiment database."""
        if not self.db_path.exists():
            self._save_database([])

    def _load_database(self) -> List[Dict[str, Any]]:
        """Load experiment database."""
        if not self.db_path.exists():
            return []

        with open(self.db_path, "r") as f:
            return json.load(f)

    def _save_database(self, experiments: List[Dict[str, Any]]):
        """Save experiment database."""
        with open(self.db_path, "w") as f:
            json.dump(experiments, f, indent=2)

    # ========== Experiment Creation ==========

    def create_experiment(
        self,
        name: str,
        config: Dict[str, Any],
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Create new experiment.

        Args:
            name: Experiment name.
            config: Experiment configuration.
            description: Optional description.
            tags: Optional tags.

        Returns:
            Experiment ID.
        """
        # Generate experiment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"{name}_{timestamp}"

        # Create experiment directory
        exp_dir = self.experiments_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config_path = exp_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Create experiment record
        experiment = {
            "id": exp_id,
            "name": name,
            "created_at": datetime.now().isoformat(),
            "status": "created",
            "config": config,
            "description": description or "",
            "tags": tags or {},
            "directory": str(exp_dir),
            "runs": [],
        }

        # Save to database
        experiments = self._load_database()
        experiments.append(experiment)
        self._save_database(experiments)

        logger.info(f"Created experiment: {exp_id}")
        return exp_id

    def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get experiment by ID.

        Args:
            experiment_id: Experiment ID.

        Returns:
            Experiment record.
        """
        experiments = self._load_database()

        for exp in experiments:
            if exp["id"] == experiment_id:
                return exp

        raise ValueError(f"Experiment not found: {experiment_id}")

    def list_experiments(
        self,
        status: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        List experiments.

        Args:
            status: Filter by status.
            tags: Filter by tags.
            limit: Maximum number of results.

        Returns:
            List of experiment records.
        """
        experiments = self._load_database()

        # Apply filters
        if status:
            experiments = [exp for exp in experiments if exp.get("status") == status]

        if tags:
            filtered = []
            for exp in experiments:
                exp_tags = exp.get("tags", {})
                if all(exp_tags.get(k) == v for k, v in tags.items()):
                    filtered.append(exp)
            experiments = filtered

        # Apply limit
        if limit:
            experiments = experiments[:limit]

        return experiments

    def update_experiment(
        self,
        experiment_id: str,
        status: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Update experiment.

        Args:
            experiment_id: Experiment ID.
            status: New status.
            tags: Tags to add/update.
        """
        experiments = self._load_database()

        for exp in experiments:
            if exp["id"] == experiment_id:
                if status:
                    exp["status"] = status

                if tags:
                    exp["tags"].update(tags)

                exp["updated_at"] = datetime.now().isoformat()

                self._save_database(experiments)
                logger.info(f"Updated experiment: {experiment_id}")
                return

        raise ValueError(f"Experiment not found: {experiment_id}")

    def delete_experiment(self, experiment_id: str):
        """
        Delete experiment.

        Args:
            experiment_id: Experiment ID.
        """
        experiments = self._load_database()
        experiments = [exp for exp in experiments if exp["id"] != experiment_id]
        self._save_database(experiments)

        logger.info(f"Deleted experiment: {experiment_id}")

    # ========== Experiment Execution ==========

    def run_experiment(
        self,
        experiment_id: str,
        pipeline: Optional[Any] = None,
        auto_register: bool = False,
    ) -> str:
        """
        Run experiment.

        Args:
            experiment_id: Experiment ID.
            pipeline: Pipeline to execute. If None, infers from config.
            auto_register: Whether to auto-register best model.

        Returns:
            Run ID.
        """
        # Get experiment
        experiment = self.get_experiment(experiment_id)

        # Update status
        self.update_experiment(experiment_id, status="running")

        try:
            # Start MLflow run
            run_id = self.mlflow.start_run(
                run_name=experiment["name"],
                tags={
                    "experiment_id": experiment_id,
                    **experiment.get("tags", {}),
                },
                description=experiment.get("description"),
            )

            # Log config
            self.mlflow.log_params(experiment["config"])

            # Execute pipeline
            if pipeline:
                results = self._execute_pipeline(pipeline, experiment["config"])

                # Log results
                if "metrics" in results:
                    self.mlflow.log_metrics(results["metrics"])

                if "artifacts" in results:
                    for artifact_path in results["artifacts"]:
                        self.mlflow.log_artifact(artifact_path)

                # Auto-register model if requested
                if auto_register and "model_path" in results:
                    model_uri = f"runs:/{run_id}/model"
                    self.mlflow.register_model(
                        model_uri=model_uri,
                        name=experiment["name"],
                        description=experiment.get("description"),
                    )

            # End run
            self.mlflow.end_run(status="FINISHED")

            # Update experiment
            self.update_experiment(experiment_id, status="completed")

            # Add run to experiment
            if run_id:
                self._add_run_to_experiment(experiment_id, run_id)

            logger.info(f"Completed experiment: {experiment_id} (run_id: {run_id})")
            return run_id or ""

        except Exception as e:
            logger.error(f"Experiment failed: {e}")

            # End run with failure
            self.mlflow.end_run(status="FAILED")

            # Update status
            self.update_experiment(experiment_id, status="failed")

            raise

    def _execute_pipeline(
        self,
        pipeline: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute pipeline.

        Args:
            pipeline: Pipeline object.
            config: Configuration.

        Returns:
            Results dictionary.
        """
        # Check if pipeline has execute method
        if hasattr(pipeline, "execute"):
            return pipeline.execute(config)

        # Check if pipeline is callable
        if callable(pipeline):
            return pipeline(config)

        raise ValueError("Pipeline must have execute() method or be callable")

    def _add_run_to_experiment(self, experiment_id: str, run_id: str):
        """Add run to experiment record."""
        experiments = self._load_database()

        for exp in experiments:
            if exp["id"] == experiment_id:
                exp["runs"].append(
                    {
                        "run_id": run_id,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                self._save_database(experiments)
                return

    # ========== Comparison and Analysis ==========

    def compare_experiments(
        self,
        experiment_ids: List[str],
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compare multiple experiments.

        Args:
            experiment_ids: List of experiment IDs.
            metrics: List of metrics to compare.

        Returns:
            Comparison DataFrame.
        """
        experiments = []

        for exp_id in experiment_ids:
            try:
                exp = self.get_experiment(exp_id)
                experiments.append(exp)
            except ValueError:
                logger.warning(f"Experiment not found: {exp_id}")
                continue

        if not experiments:
            return pd.DataFrame()

        # Build comparison table
        rows = []
        for exp in experiments:
            row = {
                "experiment_id": exp["id"],
                "name": exp["name"],
                "status": exp["status"],
                "created_at": exp["created_at"],
            }

            # Get metrics from MLflow runs
            if exp.get("runs"):
                latest_run_id = exp["runs"][-1]["run_id"]
                run = self.mlflow.get_run(latest_run_id)

                if run:
                    run_metrics = run.data.metrics

                    if metrics:
                        for metric in metrics:
                            row[metric] = run_metrics.get(metric)
                    else:
                        row.update(run_metrics)

            rows.append(row)

        return pd.DataFrame(rows)

    def get_best_experiment(
        self,
        metric: str,
        direction: str = "maximize",
        status: Optional[str] = "completed",
    ) -> Dict[str, Any]:
        """
        Get best experiment based on metric.

        Args:
            metric: Metric name.
            direction: 'maximize' or 'minimize'.
            status: Filter by status.

        Returns:
            Best experiment record.
        """
        experiments = self.list_experiments(status=status)

        if not experiments:
            raise ValueError("No experiments found")

        # Get metrics for each experiment
        exp_metrics = []
        for exp in experiments:
            if not exp.get("runs"):
                continue

            latest_run_id = exp["runs"][-1]["run_id"]
            run = self.mlflow.get_run(latest_run_id)

            if run and metric in run.data.metrics:
                exp_metrics.append(
                    {
                        "experiment": exp,
                        "metric_value": run.data.metrics[metric],
                    }
                )

        if not exp_metrics:
            raise ValueError(f"No experiments with metric: {metric}")

        # Find best
        reverse = direction == "maximize"
        best = max(exp_metrics, key=lambda x: x["metric_value"] * (1 if reverse else -1))

        return best["experiment"]

    def get_best_model(
        self,
        metric: str,
        direction: str = "maximize",
        status: Optional[str] = "completed",
    ):
        """
        Get best model based on metric.

        Args:
            metric: Metric name.
            direction: 'maximize' or 'minimize'.
            status: Filter by status.

        Returns:
            Model object.
        """
        best_exp = self.get_best_experiment(metric, direction, status)

        if not best_exp.get("runs"):
            raise ValueError("No runs found for best experiment")

        # Get latest run
        latest_run_id = best_exp["runs"][-1]["run_id"]

        # Load model from MLflow
        model_uri = f"runs:/{latest_run_id}/model"
        model = self.mlflow.load_model(model_uri)

        if model is None:
            raise ValueError("Failed to load model")

        return model

    # ========== Reports ==========

    def generate_summary_report(
        self,
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Generate summary report of all experiments.

        Args:
            output_path: Path to save report.

        Returns:
            Report content.
        """
        experiments = self.list_experiments()

        # Build report
        lines = [
            "# Experiment Summary Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nTotal Experiments: {len(experiments)}",
            "\n## Experiments\n",
        ]

        # Group by status
        status_counts: Dict[str, int] = {}
        for exp in experiments:
            status = exp.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        lines.append("### Status Summary\n")
        for status, count in status_counts.items():
            lines.append(f"- {status}: {count}")

        lines.append("\n### Experiment Details\n")

        for exp in experiments:
            lines.append(f"\n#### {exp['name']} ({exp['id']})")
            lines.append(f"- Status: {exp['status']}")
            lines.append(f"- Created: {exp['created_at']}")
            lines.append(f"- Runs: {len(exp.get('runs', []))}")

            if exp.get("tags"):
                lines.append(f"- Tags: {exp['tags']}")

        report = "\n".join(lines)

        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                f.write(report)

            logger.info(f"Saved summary report to: {output_path}")

        return report

    # ========== Utilities ==========

    def export_experiments(
        self,
        output_path: Union[str, Path],
        format: str = "csv",
    ):
        """
        Export experiments to file.

        Args:
            output_path: Output file path.
            format: Export format ('csv', 'json', 'yaml').
        """
        experiments = self.list_experiments()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "csv":
            # Flatten for CSV
            rows = []
            for exp in experiments:
                row = {
                    "id": exp["id"],
                    "name": exp["name"],
                    "status": exp["status"],
                    "created_at": exp["created_at"],
                    "n_runs": len(exp.get("runs", [])),
                }

                # Add tags
                for key, value in exp.get("tags", {}).items():
                    row[f"tag_{key}"] = value

                rows.append(row)

            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)

        elif format == "json":
            with open(output_path, "w") as f:
                json.dump(experiments, f, indent=2)

        elif format == "yaml":
            with open(output_path, "w") as f:
                yaml.dump(experiments, f, default_flow_style=False)

        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Exported {len(experiments)} experiments to: {output_path}")

    def cleanup_failed_experiments(self):
        """Remove failed experiments."""
        experiments = self._load_database()
        initial_count = len(experiments)

        experiments = [exp for exp in experiments if exp.get("status") != "failed"]

        self._save_database(experiments)

        removed_count = initial_count - len(experiments)
        logger.info(f"Removed {removed_count} failed experiments")
