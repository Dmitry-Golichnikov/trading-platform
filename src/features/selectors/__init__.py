"""Модуль отбора признаков."""

from src.features.selectors.drift_detector import DriftDetector
from src.features.selectors.embedded_methods import EmbeddedSelector
from src.features.selectors.filter_methods import FilterSelector
from src.features.selectors.wrapper_methods import WrapperSelector

__all__ = [
    "FilterSelector",
    "WrapperSelector",
    "EmbeddedSelector",
    "DriftDetector",
]
