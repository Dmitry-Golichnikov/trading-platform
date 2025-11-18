"""
Модуль моделирования.

Содержит базовые интерфейсы, реестр моделей, систему обучения, loss functions и утилиты.
"""

from typing import Any, Optional

from src.modeling.base import (
    BaseModel,
    ClassifierMixin,
    ModelProtocol,
    ModelType,
    RegressorMixin,
)
from src.modeling.callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    MLflowLogger,
    ModelCheckpoint,
    ProgressBar,
)
from src.modeling.registry import ModelRegistry, registry
from src.modeling.sanity_checks import ModelSanityChecker, SanityCheckResult
from src.modeling.serialization import ModelSerializer
from src.modeling.splitting import DataSplitter
from src.modeling.trainer import ModelTrainer, TrainingResult
from src.modeling.utils import (
    TimingContext,
    check_gpu_memory,
    clear_gpu_memory,
    ensure_reproducibility,
    get_available_devices,
    get_device,
    log_system_info,
    set_seed,
)

# Loss Functions (импортируем опционально, чтобы не требовать torch при простом импорте реестра)
LossRegistry: Optional[Any]
loss_registry: Optional[Any]

try:
    from src.modeling.loss_functions import LossRegistry as _LossRegistryClass
    from src.modeling.loss_functions import loss_registry as _loss_registry_instance
except Exception:
    LossRegistry = None
    loss_registry = None
else:
    LossRegistry = _LossRegistryClass
    loss_registry = _loss_registry_instance

__all__ = [
    # Base
    "BaseModel",
    "ModelProtocol",
    "ClassifierMixin",
    "RegressorMixin",
    "ModelType",
    # Registry
    "ModelRegistry",
    "registry",
    # Training
    "ModelTrainer",
    "TrainingResult",
    # Callbacks
    "Callback",
    "CallbackList",
    "EarlyStopping",
    "ModelCheckpoint",
    "MLflowLogger",
    "ProgressBar",
    # Loss Functions
    "LossRegistry",
    "loss_registry",
    # Data Management
    "DataSplitter",
    "ModelSerializer",
    # Sanity Checks
    "ModelSanityChecker",
    "SanityCheckResult",
    # Utils
    "set_seed",
    "get_device",
    "get_available_devices",
    "check_gpu_memory",
    "clear_gpu_memory",
    "ensure_reproducibility",
    "log_system_info",
    "TimingContext",
]
