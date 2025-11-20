from __future__ import annotations

from dataclasses import dataclass

from .backtest_service import BacktestService
from .dataset_service import DatasetService
from .experiment_service import ExperimentService
from .feature_service import FeatureService
from .labeling_service import LabelingService
from .training_service import TrainingService


@dataclass(slots=True)
class GUIServiceBundle:
    dataset: DatasetService
    feature: FeatureService
    labeling: LabelingService
    training: TrainingService
    backtest: BacktestService
    experiment: ExperimentService


def create_default_services() -> GUIServiceBundle:
    dataset_service = DatasetService()
    feature_service = FeatureService()
    labeling_service = LabelingService()
    training_service = TrainingService()
    backtest_service = BacktestService()
    experiment_service = ExperimentService(training_service, backtest_service)

    return GUIServiceBundle(
        dataset=dataset_service,
        feature=feature_service,
        labeling=labeling_service,
        training=training_service,
        backtest=backtest_service,
        experiment=experiment_service,
    )


__all__ = [
    "GUIServiceBundle",
    "create_default_services",
    "DatasetService",
    "FeatureService",
    "LabelingService",
    "TrainingService",
    "BacktestService",
    "ExperimentService",
]
