"""MLflow integration wrapper."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class MLflowManager:
    """
    Comprehensive wrapper around MLflow.

    Provides unified interface for:
    - Experiment tracking
    - Model registry
    - Artifact storage
    - Run management
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
        experiment_name: str = "default",
        artifact_location: Optional[str] = None,
    ):
        """
        Initialize MLflow manager.

        Args:
            tracking_uri: MLflow tracking server URI. If None, uses local.
            registry_uri: Model registry URI. If None, uses tracking_uri.
            experiment_name: Default experiment name.
            artifact_location: Custom artifact storage location.
        """
        self.tracking_uri = tracking_uri or "file:./artifacts/mlruns"
        self.registry_uri = registry_uri or self.tracking_uri
        self.experiment_name = experiment_name
        self.artifact_location = artifact_location

        self.mlflow_available = False
        self.mlflow = None
        self.current_run = None

        self._init_mlflow()

    def _init_mlflow(self):
        """Initialize MLflow."""
        try:
            import mlflow

            self.mlflow = mlflow
            self.mlflow_available = True

            # Set tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)

            # Set registry URI if different
            if self.registry_uri != self.tracking_uri:
                mlflow.set_registry_uri(self.registry_uri)

            # Create or get experiment
            self._setup_experiment()

            logger.info(f"MLflow initialized: {self.experiment_name}")
            logger.info(f"Tracking URI: {self.tracking_uri}")

        except ImportError:
            logger.warning("MLflow not installed. Tracking disabled.")
            self.mlflow_available = False

    def _setup_experiment(self):
        """Setup MLflow experiment."""
        if not self.mlflow_available:
            return

        try:
            # Try to create experiment
            kwargs = {"name": self.experiment_name}
            if self.artifact_location:
                kwargs["artifact_location"] = self.artifact_location

            experiment_id = self.mlflow.create_experiment(**kwargs)
            logger.info(f"Created experiment: {self.experiment_name} (id={experiment_id})")

        except Exception:
            # Experiment exists
            experiment = self.mlflow.get_experiment_by_name(self.experiment_name)
            experiment_id = experiment.experiment_id
            logger.debug(f"Using existing experiment: {self.experiment_name} (id={experiment_id})")

        self.mlflow.set_experiment(self.experiment_name)
        self.experiment_id = experiment_id

    # ========== Experiment Management ==========

    def start_run(
        self,
        run_name: Optional[str] = None,
        nested: bool = False,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ) -> Optional[str]:
        """
        Start a new MLflow run.

        Args:
            run_name: Name for the run.
            nested: Whether this is a nested run.
            tags: Dictionary of tags.
            description: Run description.

        Returns:
            Run ID if successful, None otherwise.
        """
        if not self.mlflow_available or self.mlflow is None:
            return None

        try:
            self.current_run = self.mlflow.start_run(
                run_name=run_name,
                nested=nested,
            )

            run_id = self.current_run.info.run_id

            # Set tags
            if tags:
                self.mlflow.set_tags(tags)

            # Set description as tag
            if description:
                self.mlflow.set_tag("mlflow.note.content", description)

            logger.info(f"Started run: {run_name or 'unnamed'} (id={run_id})")
            return run_id

        except Exception as e:
            logger.error(f"Failed to start run: {e}")
            return None

    def end_run(self, status: str = "FINISHED"):
        """
        End current MLflow run.

        Args:
            status: Run status ('FINISHED', 'FAILED', 'KILLED').
        """
        if not self.mlflow_available or not self.current_run:
            return

        try:
            self.mlflow.end_run(status=status)
            logger.info(f"Ended run with status: {status}")
            self.current_run = None

        except Exception as e:
            logger.error(f"Failed to end run: {e}")

    # ========== Logging ==========

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to current run.

        Args:
            params: Dictionary of parameters.
        """
        if not self.mlflow_available or not self.current_run:
            return

        try:
            for key, value in params.items():
                # MLflow params must be strings
                if not isinstance(value, (int, float, str, bool)):
                    value = str(value)
                self.mlflow.log_param(key, value)

        except Exception as e:
            logger.error(f"Failed to log params: {e}")

    def log_param(self, key: str, value: Any):
        """Log single parameter."""
        self.log_params({key: value})

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        """
        Log metrics to current run.

        Args:
            metrics: Dictionary of metrics.
            step: Optional step number for time series metrics.
        """
        if not self.mlflow_available or not self.current_run:
            return

        try:
            for key, value in metrics.items():
                self.mlflow.log_metric(key, float(value), step=step)

        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log single metric."""
        self.log_metrics({key: value}, step=step)

    def log_artifact(
        self,
        local_path: Union[str, Path],
        artifact_path: Optional[str] = None,
    ):
        """
        Log artifact to current run.

        Args:
            local_path: Path to local file.
            artifact_path: Subdirectory in artifact storage.
        """
        if not self.mlflow_available or not self.current_run:
            return

        try:
            self.mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
            logger.debug(f"Logged artifact: {local_path}")

        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")

    def log_artifacts(
        self,
        local_dir: Union[str, Path],
        artifact_path: Optional[str] = None,
    ):
        """
        Log directory of artifacts to current run.

        Args:
            local_dir: Path to local directory.
            artifact_path: Subdirectory in artifact storage.
        """
        if not self.mlflow_available or not self.current_run:
            return

        try:
            self.mlflow.log_artifacts(str(local_dir), artifact_path=artifact_path)
            logger.debug(f"Logged artifacts from: {local_dir}")

        except Exception as e:
            logger.error(f"Failed to log artifacts: {e}")

    def log_figure(
        self,
        figure,
        artifact_file: str,
    ):
        """
        Log matplotlib/plotly figure.

        Args:
            figure: Matplotlib or Plotly figure.
            artifact_file: Filename for the figure.
        """
        if not self.mlflow_available or not self.current_run:
            return

        try:
            self.mlflow.log_figure(figure, artifact_file)
            logger.debug(f"Logged figure: {artifact_file}")

        except Exception as e:
            logger.error(f"Failed to log figure: {e}")

    def log_dict(
        self,
        dictionary: Dict,
        artifact_file: str,
    ):
        """
        Log dictionary as JSON artifact.

        Args:
            dictionary: Dictionary to log.
            artifact_file: Filename for the JSON file.
        """
        if not self.mlflow_available or not self.current_run:
            return

        try:
            self.mlflow.log_dict(dictionary, artifact_file)
            logger.debug(f"Logged dict: {artifact_file}")

        except Exception as e:
            logger.error(f"Failed to log dict: {e}")

    def set_tags(self, tags: Dict[str, str]):
        """
        Set tags for current run.

        Args:
            tags: Dictionary of tags.
        """
        if not self.mlflow_available or not self.current_run:
            return

        try:
            self.mlflow.set_tags(tags)

        except Exception as e:
            logger.error(f"Failed to set tags: {e}")

    def set_tag(self, key: str, value: str):
        """Set single tag."""
        self.set_tags({key: value})

    # ========== Model Logging ==========

    def log_model(
        self,
        model,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Log model to current run.

        Args:
            model: Model object (sklearn, pytorch, etc.).
            artifact_path: Path within artifact storage.
            registered_model_name: Name for model registry.
            **kwargs: Additional arguments for specific model flavor.
        """
        if not self.mlflow_available or not self.current_run:
            return

        try:
            if hasattr(model, "predict") and hasattr(model, "fit"):
                # Sklearn-like model
                self.mlflow.sklearn.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_model_name,
                    **kwargs,
                )

            elif "torch" in str(type(model).__module__):
                # PyTorch model
                self.mlflow.pytorch.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_model_name,
                    **kwargs,
                )

            else:
                # Generic Python model
                self.mlflow.pyfunc.log_model(
                    artifact_path,
                    python_model=model,
                    registered_model_name=registered_model_name,
                    **kwargs,
                )

            logger.info(f"Logged model: {artifact_path}")

        except Exception as e:
            logger.error(f"Failed to log model: {e}")

    # ========== Run Queries ==========

    def get_run(self, run_id: str) -> Optional[Any]:
        """
        Get run by ID.

        Args:
            run_id: Run ID.

        Returns:
            MLflow Run object.
        """
        if not self.mlflow_available or self.mlflow is None:
            return None

        try:
            return self.mlflow.get_run(run_id)

        except Exception as e:
            logger.error(f"Failed to get run {run_id}: {e}")
            return None

    def search_runs(
        self,
        filter_string: str = "",
        order_by: Optional[List[str]] = None,
        max_results: int = 100,
    ) -> pd.DataFrame:
        """
        Search runs in current experiment.

        Args:
            filter_string: Filter query string.
            order_by: List of columns to order by.
            max_results: Maximum number of results.

        Returns:
            DataFrame with run information.
        """
        if not self.mlflow_available or self.mlflow is None:
            return pd.DataFrame()

        try:
            runs = self.mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=filter_string,
                order_by=order_by,
                max_results=max_results,
            )
            return runs

        except Exception as e:
            logger.error(f"Failed to search runs: {e}")
            return pd.DataFrame()

    def get_best_run(
        self,
        metric: str,
        direction: str = "maximize",
        filter_string: str = "",
    ) -> Optional[Any]:
        """
        Get best run based on metric.

        Args:
            metric: Metric name.
            direction: 'maximize' or 'minimize'.
            filter_string: Optional filter.

        Returns:
            Best run.
        """
        order = "DESC" if direction == "maximize" else "ASC"
        order_by = [f"metrics.{metric} {order}"]

        runs = self.search_runs(
            filter_string=filter_string,
            order_by=order_by,
            max_results=1,
        )

        if runs.empty:
            return None

        run_id = runs.iloc[0]["run_id"]
        return self.get_run(run_id)

    # ========== Model Registry ==========

    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Register model in Model Registry.

        Args:
            model_uri: URI of the model (runs:/<run_id>/model).
            name: Registered model name.
            tags: Optional tags.
            description: Optional description.

        Returns:
            ModelVersion object.
        """
        if not self.mlflow_available or self.mlflow is None:
            return None

        try:
            # Register model
            model_version = self.mlflow.register_model(model_uri, name)

            # Add tags if provided
            if tags:
                from mlflow.tracking import MlflowClient

                client = MlflowClient()
                for key, value in tags.items():
                    client.set_model_version_tag(name, model_version.version, key, value)

            # Set description
            if description:
                from mlflow.tracking import MlflowClient

                client = MlflowClient()
                client.update_model_version(
                    name=name,
                    version=model_version.version,
                    description=description,
                )

            logger.info(f"Registered model: {name} (version {model_version.version})")
            return model_version

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None

    def load_model(self, model_uri: str) -> Optional[Any]:
        """
        Load model from MLflow.

        Args:
            model_uri: Model URI.

        Returns:
            Loaded model.
        """
        if not self.mlflow_available or self.mlflow is None:
            return None

        try:
            # Try different flavors
            try:
                return self.mlflow.sklearn.load_model(model_uri)
            except Exception:
                pass

            try:
                return self.mlflow.pytorch.load_model(model_uri)
            except Exception:
                pass

            return self.mlflow.pyfunc.load_model(model_uri)

        except Exception as e:
            logger.error(f"Failed to load model from {model_uri}: {e}")
            return None

    def transition_model_stage(
        self,
        name: str,
        version: int,
        stage: str,
        archive_existing: bool = False,
    ):
        """
        Transition model to a new stage.

        Args:
            name: Registered model name.
            version: Model version number.
            stage: Target stage ('Staging', 'Production', 'Archived').
            archive_existing: Whether to archive existing versions in target stage.
        """
        if not self.mlflow_available:
            return

        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            client.transition_model_version_stage(
                name=name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing,
            )

            logger.info(f"Transitioned {name} v{version} to {stage}")

        except Exception as e:
            logger.error(f"Failed to transition model: {e}")

    def get_latest_model_version(
        self,
        name: str,
        stage: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Get latest version of registered model.

        Args:
            name: Registered model name.
            stage: Optional stage filter ('Production', 'Staging', etc.).

        Returns:
            ModelVersion object.
        """
        if not self.mlflow_available:
            return None

        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient()

            if stage:
                versions = client.get_latest_versions(name, stages=[stage])
            else:
                versions = client.get_latest_versions(name)

            if versions:
                return versions[0]

            return None

        except Exception as e:
            logger.error(f"Failed to get latest model version: {e}")
            return None

    # ========== Experiment Management ==========

    def list_experiments(self) -> List[Any]:
        """List all experiments."""
        if not self.mlflow_available or self.mlflow is None:
            return []

        try:
            return self.mlflow.search_experiments()

        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            return []

    def delete_experiment(self, experiment_id: str):
        """Delete experiment."""
        if not self.mlflow_available or self.mlflow is None:
            return

        try:
            self.mlflow.delete_experiment(experiment_id)
            logger.info(f"Deleted experiment: {experiment_id}")

        except Exception as e:
            logger.error(f"Failed to delete experiment: {e}")

    def delete_run(self, run_id: str):
        """Delete run."""
        if not self.mlflow_available or self.mlflow is None:
            return

        try:
            self.mlflow.delete_run(run_id)
            logger.info(f"Deleted run: {run_id}")

        except Exception as e:
            logger.error(f"Failed to delete run: {e}")

    # ========== Context Manager ==========

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.current_run:
            status = "FAILED" if exc_type else "FINISHED"
            self.end_run(status=status)
