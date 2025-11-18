"""Модуль оценки моделей."""

from .calibration import (
    BetaCalibrator,
    CalibrationMetrics,
    ModelCalibrator,
    compare_calibration_methods,
)
from .drift_detection import (
    AdversarialValidation,
    ChiSquaredTest,
    DriftDetector,
    KolmogorovSmirnovTest,
    PopulationStabilityIndex,
)
from .feature_importance import (
    FeatureImportanceAnalyzer,
    PartialDependence,
    PermutationImportance,
    SHAPImportance,
    TreeImportance,
)
from .metrics import (
    ClassificationMetrics,
    MetricsCalculator,
    RegressionMetrics,
)
from .reports import ModelEvaluationReport

__all__ = [
    # Metrics
    "ClassificationMetrics",
    "RegressionMetrics",
    "MetricsCalculator",
    # Calibration
    "ModelCalibrator",
    "CalibrationMetrics",
    "BetaCalibrator",
    "compare_calibration_methods",
    # Feature Importance
    "TreeImportance",
    "PermutationImportance",
    "SHAPImportance",
    "PartialDependence",
    "FeatureImportanceAnalyzer",
    # Drift Detection
    "PopulationStabilityIndex",
    "KolmogorovSmirnovTest",
    "ChiSquaredTest",
    "AdversarialValidation",
    "DriftDetector",
    # Reports
    "ModelEvaluationReport",
]
