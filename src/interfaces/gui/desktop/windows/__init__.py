"""
Окна приложения.
"""

from .backtest_window import BacktestWindow
from .dataset_window import DatasetWindow
from .experiment_window import ExperimentWindow
from .feature_window import FeatureWindow
from .labeling_window import LabelingWindow
from .main_window import MainWindow
from .training_window import TrainingWindow

__all__ = [
    "MainWindow",
    "DatasetWindow",
    "FeatureWindow",
    "LabelingWindow",
    "TrainingWindow",
    "BacktestWindow",
    "ExperimentWindow",
]
