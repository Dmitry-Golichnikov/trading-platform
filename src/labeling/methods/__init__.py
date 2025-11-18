"""Методы разметки таргетов."""

from src.labeling.methods.custom_rules import CustomRulesLabeler
from src.labeling.methods.horizon import HorizonLabeler
from src.labeling.methods.regression_targets import RegressionTargetsLabeler
from src.labeling.methods.triple_barrier import TripleBarrierLabeler

__all__ = [
    "HorizonLabeler",
    "TripleBarrierLabeler",
    "CustomRulesLabeler",
    "RegressionTargetsLabeler",
]
