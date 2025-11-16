"""
Experiment Service

Business logic for experiment operations.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from src.interfaces.gui.backend.api.models import (
    ExperimentComparison,
    ExperimentInfo,
    ExperimentStatusResponse,
    TrainingPhase,
)
from src.orchestration.experiment_manager import ExperimentManager


class ExperimentService:
    """Service for experiment operations"""

    def __init__(self):
        self.manager = ExperimentManager()

    def list_experiments(
        self, status: Optional[str] = None, tags: Optional[Dict[str, str]] = None, limit: int = 100
    ) -> List[ExperimentInfo]:
        """List experiments"""
        experiments = self.manager.list_experiments()

        # Filter by status
        if status:
            experiments = [e for e in experiments if e.get("status") == status]

        # Filter by tags
        if tags:
            experiments = [e for e in experiments if all(e.get("tags", {}).get(k) == v for k, v in tags.items())]

        # Apply limit
        experiments = experiments[:limit]

        # Convert to response models
        result = []
        for exp in experiments:
            try:
                result.append(self._to_experiment_info(exp))
            except Exception as e:
                print(f"Error converting experiment {exp.get('id')}: {e}")
                continue

        return result

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentInfo]:
        """Get experiment by ID"""
        exp = self.manager.get_experiment(experiment_id)
        if not exp:
            return None

        return self._to_experiment_info(exp)

    def create_experiment(
        self,
        name: str,
        config: Dict[str, Any],
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> ExperimentInfo:
        """Create new experiment"""
        exp_id = self.manager.create_experiment(name=name, config=config, description=description, tags=tags or {})

        exp = self.manager.get_experiment(exp_id)
        return self._to_experiment_info(exp)

    def get_experiment_status(self, experiment_id: str) -> ExperimentStatusResponse:
        """Get experiment status"""
        exp = self.manager.get_experiment(experiment_id)
        if not exp:
            raise ValueError(f"Experiment not found: {experiment_id}")

        # Get latest status
        status = exp.get("status", "pending")
        phase = exp.get("phase", "data_loading")
        progress = exp.get("progress", 0.0)
        message = exp.get("message", "")
        metrics = exp.get("metrics", {})

        return ExperimentStatusResponse(
            id=experiment_id,
            status=status,
            phase=TrainingPhase(phase) if phase else None,
            progress=progress,
            message=message,
            metrics=metrics,
            updated_at=datetime.now(),
        )

    def cancel_experiment(self, experiment_id: str) -> bool:
        """Cancel running experiment"""
        return self.manager.cancel_experiment(experiment_id)

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete experiment"""
        return self.manager.delete_experiment(experiment_id)

    def compare_experiments(self, experiment_ids: List[str]) -> ExperimentComparison:
        """Compare multiple experiments"""
        experiments = []
        all_metrics: Dict[str, List[float]] = {}

        for exp_id in experiment_ids:
            exp = self.manager.get_experiment(exp_id)
            if exp:
                experiments.append(self._to_experiment_info(exp))

                # Collect metrics
                metrics = exp.get("metrics", {})
                for metric_name, metric_value in metrics.items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(metric_value)

        # Find best experiment (by primary metric, e.g., val_f1)
        best_exp_id = None
        best_score = -float("inf")

        for exp in experiments:
            score = exp.metrics.get("val_f1", exp.metrics.get("val_accuracy", 0.0))
            if score > best_score:
                best_score = score
                best_exp_id = exp.id

        return ExperimentComparison(experiments=experiments, metrics=all_metrics, best_experiment_id=best_exp_id)

    def _to_experiment_info(self, exp: Dict[str, Any]) -> ExperimentInfo:
        """Convert experiment dict to ExperimentInfo"""
        return ExperimentInfo(
            id=exp["id"],
            name=exp.get("name", exp["id"]),
            description=exp.get("description"),
            status=exp.get("status", "pending"),
            config=exp.get("config", {}),
            metrics=exp.get("metrics", {}),
            tags=exp.get("tags", {}),
            created_at=exp.get("created_at", datetime.now()),
            updated_at=exp.get("updated_at", datetime.now()),
            started_at=exp.get("started_at"),
            finished_at=exp.get("finished_at"),
            duration_seconds=exp.get("duration_seconds"),
            artifacts=exp.get("artifacts", []),
        )
