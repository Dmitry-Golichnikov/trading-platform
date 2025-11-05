"""
Модуль моделирования.

Содержит базовые интерфейсы, реестр моделей, систему обучения, loss functions и утилиты.
"""

from src.modeling.base import (
    BaseModel,
    ClassifierMixin,
    ModelProtocol,
    ModelType,
    RegressorMixin,
)

# Callbacks
from src.modeling.callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    MLflowLogger,
    ModelCheckpoint,
    ProgressBar,
)

# Loss Functions
from src.modeling.loss_functions import (
    LossRegistry,
    loss_registry,
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
