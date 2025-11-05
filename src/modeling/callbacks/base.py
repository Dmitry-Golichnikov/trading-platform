"""
Базовый класс для callbacks.

Callbacks позволяют внедрять кастомную логику в процесс обучения.
"""

from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from src.modeling.trainer import ModelTrainer


class Callback(ABC):
    """
    Базовый класс для всех callbacks.

    Callbacks вызываются в различные моменты процесса обучения и позволяют
    добавлять кастомную логику (логирование, early stopping, сохранение и т.д.).
    """

    def on_train_begin(self, trainer: "ModelTrainer") -> None:
        """
        Вызывается в начале обучения.

        Args:
            trainer: Экземпляр ModelTrainer
        """
        pass

    def on_train_end(self, trainer: "ModelTrainer", logs: Dict[str, Any]) -> None:
        """
        Вызывается в конце обучения.

        Args:
            trainer: Экземпляр ModelTrainer
            logs: Финальные метрики
        """
        pass

    def on_epoch_begin(self, trainer: "ModelTrainer", epoch: int) -> None:
        """
        Вызывается в начале эпохи.

        Args:
            trainer: Экземпляр ModelTrainer
            epoch: Номер эпохи
        """
        pass

    def on_epoch_end(
        self, trainer: "ModelTrainer", epoch: int, logs: Dict[str, Any]
    ) -> None:
        """
        Вызывается в конце эпохи.

        Args:
            trainer: Экземпляр ModelTrainer
            epoch: Номер эпохи
            logs: Метрики эпохи
        """
        pass

    def on_batch_begin(self, trainer: "ModelTrainer", batch_idx: int) -> None:
        """
        Вызывается в начале batch.

        Args:
            trainer: Экземпляр ModelTrainer
            batch_idx: Индекс batch
        """
        pass

    def on_batch_end(
        self, trainer: "ModelTrainer", batch_idx: int, logs: Dict[str, Any]
    ) -> None:
        """
        Вызывается в конце batch.

        Args:
            trainer: Экземпляр ModelTrainer
            batch_idx: Индекс batch
            logs: Метрики batch
        """
        pass


class CallbackList:
    """
    Контейнер для списка callbacks.

    Управляет вызовом всех callbacks в правильном порядке.
    """

    def __init__(self, callbacks: Optional[list[Callback]] = None):
        """
        Args:
            callbacks: Список callbacks
        """
        self.callbacks = callbacks or []

    def append(self, callback: Callback) -> None:
        """Добавить callback."""
        self.callbacks.append(callback)

    def on_train_begin(self, trainer: "ModelTrainer") -> None:
        """Вызвать on_train_begin для всех callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(trainer)

    def on_train_end(self, trainer: "ModelTrainer", logs: Dict[str, Any]) -> None:
        """Вызвать on_train_end для всех callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(trainer, logs)

    def on_epoch_begin(self, trainer: "ModelTrainer", epoch: int) -> None:
        """Вызвать on_epoch_begin для всех callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer, epoch)

    def on_epoch_end(
        self, trainer: "ModelTrainer", epoch: int, logs: Dict[str, Any]
    ) -> None:
        """Вызвать on_epoch_end для всех callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, epoch, logs)

    def on_batch_begin(self, trainer: "ModelTrainer", batch_idx: int) -> None:
        """Вызвать on_batch_begin для всех callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(trainer, batch_idx)

    def on_batch_end(
        self, trainer: "ModelTrainer", batch_idx: int, logs: Dict[str, Any]
    ) -> None:
        """Вызвать on_batch_end для всех callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(trainer, batch_idx, logs)
