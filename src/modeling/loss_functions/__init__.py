"""Функции потерь для обучения моделей."""

from src.modeling.loss_functions.base import (
    BaseLoss,
    ClassificationLoss,
    RegressionLoss,
    TradingLoss,
)

# Classification
from src.modeling.loss_functions.classification.bce import (
    BCEWithLogitsLoss,
    BinaryCrossEntropyLoss,
    WeightedBCELoss,
)
from src.modeling.loss_functions.classification.focal import (
    FocalLoss,
    MultiClassFocalLoss,
)
from src.modeling.loss_functions.registry import LossRegistry, loss_registry

# Regression
from src.modeling.loss_functions.regression.standard import (
    HuberLoss,
    LogCoshLoss,
    MAELoss,
    MSELoss,
    QuantileLoss,
    RMSELoss,
)

# Trading
from src.modeling.loss_functions.trading_custom.directional import (
    AsymmetricDirectionalLoss,
    DirectionalLoss,
    SignLoss,
)
from src.modeling.loss_functions.trading_custom.profit_based import (
    ExpectedPnLLoss,
    ProfitBasedLoss,
    RiskAdjustedProfitLoss,
    SharpeRatioLoss,
)

__all__ = [
    # Base
    "BaseLoss",
    "ClassificationLoss",
    "RegressionLoss",
    "TradingLoss",
    # Registry
    "LossRegistry",
    "loss_registry",
    # Classification
    "BinaryCrossEntropyLoss",
    "BCEWithLogitsLoss",
    "WeightedBCELoss",
    "FocalLoss",
    "MultiClassFocalLoss",
    # Regression
    "MSELoss",
    "MAELoss",
    "HuberLoss",
    "QuantileLoss",
    "RMSELoss",
    "LogCoshLoss",
    # Trading
    "DirectionalLoss",
    "SignLoss",
    "AsymmetricDirectionalLoss",
    "ProfitBasedLoss",
    "SharpeRatioLoss",
    "ExpectedPnLLoss",
    "RiskAdjustedProfitLoss",
]
