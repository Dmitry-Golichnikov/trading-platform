"""Функции потерь для регрессии."""

from src.modeling.loss_functions.regression.standard import (
    HuberLoss,
    LogCoshLoss,
    MAELoss,
    MSELoss,
    QuantileLoss,
    RMSELoss,
)

__all__ = [
    "MSELoss",
    "MAELoss",
    "HuberLoss",
    "QuantileLoss",
    "RMSELoss",
    "LogCoshLoss",
]
