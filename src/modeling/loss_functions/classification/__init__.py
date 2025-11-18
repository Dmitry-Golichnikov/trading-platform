"""Функции потерь для классификации."""

from src.modeling.loss_functions.classification.bce import (
    BCEWithLogitsLoss,
    BinaryCrossEntropyLoss,
    WeightedBCELoss,
)
from src.modeling.loss_functions.classification.focal import (
    FocalLoss,
    MultiClassFocalLoss,
)

__all__ = [
    "BinaryCrossEntropyLoss",
    "BCEWithLogitsLoss",
    "WeightedBCELoss",
    "FocalLoss",
    "MultiClassFocalLoss",
]
