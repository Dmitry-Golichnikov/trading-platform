from typing import Any, Dict

from src.modeling.callbacks.base import Callback
from src.modeling.trainer import ModelTrainer


class QtProgressCallback(Callback):
    """Callback, прокидывающий прогресс обучения в Qt-сигналы."""

    def __init__(self, signals):
        super().__init__()
        self.signals = signals

    def on_epoch_end(self, trainer: ModelTrainer, epoch: int, logs: Dict[str, Any]) -> None:
        self.signals.epoch_finished.emit(epoch, logs)

    def on_train_end(self, trainer: ModelTrainer, logs: Dict[str, Any]) -> None:
        self.signals.training_finished.emit(logs)
