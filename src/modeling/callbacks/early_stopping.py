"""
Early Stopping callback.

Останавливает обучение, если метрика не улучшается
в течение определённого количества эпох.
"""

import logging
from typing import Any, Dict, Literal

from src.modeling.callbacks.base import Callback

logger = logging.getLogger(__name__)


class EarlyStopping(Callback):
    """
    Callback для ранней остановки обучения.

    Останавливает обучение, если отслеживаемая метрика не улучшается
    в течение `patience` эпох.

    Примеры:
        >>> early_stop = EarlyStopping(
        >>>     monitor='val_loss',
        >>>     patience=10,
        >>>     min_delta=0.001,
        >>>     mode='min'
        >>> )
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: Literal["min", "max"] = "min",
        restore_best_weights: bool = True,
        verbose: bool = True,
    ):
        """
        Args:
            monitor: Имя метрики для отслеживания
            patience: Количество эпох без улучшения до остановки
            min_delta: Минимальное изменение метрики для считания за улучшение
            mode: 'min' для минимизации метрики, 'max' для максимизации
            restore_best_weights: Восстановить веса лучшей модели при остановке
            verbose: Выводить сообщения в лог
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

        if mode == "min":
            self.monitor_op = lambda x, y: x < (y - min_delta)
            self.best_value = float("inf")
        else:
            self.monitor_op = lambda x, y: x > (y + min_delta)
            self.best_value = float("-inf")

    def on_train_begin(self, trainer) -> None:
        """Сброс счётчиков в начале обучения."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

        if self.mode == "min":
            self.best_value = float("inf")
        else:
            self.best_value = float("-inf")

        if self.verbose:
            logger.info(f"EarlyStopping: отслеживается {self.monitor}, " f"patience={self.patience}, mode={self.mode}")

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]) -> None:
        """Проверка метрики в конце эпохи."""
        current_value = logs.get(self.monitor)

        if current_value is None:
            logger.warning(
                f"EarlyStopping: метрика '{self.monitor}' не найдена в logs. " f"Доступные метрики: {list(logs.keys())}"
            )
            return

        # Проверяем улучшение
        if self.monitor_op(current_value, self.best_value):
            self.best_value = current_value
            self.wait = 0

            # Сохраняем веса лучшей модели
            if self.restore_best_weights:
                self.best_weights = self._get_model_weights(trainer)

            if self.verbose:
                logger.info(f"Эпоха {epoch}: {self.monitor} улучшилась до {current_value:.6f}")
        else:
            self.wait += 1

            if self.verbose:
                logger.info(f"Эпоха {epoch}: {self.monitor} не улучшилась " f"({self.wait}/{self.patience})")

            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.stop_training = True

                if self.verbose:
                    logger.info(
                        f"Эпоха {epoch}: Early stopping. " f"Лучшее значение {self.monitor}: {self.best_value:.6f}"
                    )

    def on_train_end(self, trainer, logs: Dict[str, Any]) -> None:
        """Восстановление лучших весов в конце обучения."""
        if self.stopped_epoch > 0 and self.verbose:
            logger.info(f"Обучение остановлено на эпохе {self.stopped_epoch}")

        if self.restore_best_weights and self.best_weights is not None:
            if self.verbose:
                logger.info("Восстановление весов лучшей модели")
            self._set_model_weights(trainer, self.best_weights)

    def _get_model_weights(self, trainer):
        """Получить веса модели (для восстановления)."""
        # Для PyTorch моделей
        if hasattr(trainer.model, "state_dict"):
            import copy

            return copy.deepcopy(trainer.model.state_dict())
        # Для sklearn моделей пытаемся клонировать
        elif hasattr(trainer.model, "__sklearn_clone__"):
            from sklearn.base import clone

            return clone(trainer.model)
        else:
            logger.warning("Не удалось сохранить веса модели для восстановления")
            return None

    def _set_model_weights(self, trainer, weights):
        """Установить веса модели."""
        if hasattr(trainer.model, "load_state_dict"):
            trainer.model.load_state_dict(weights)
        elif hasattr(weights, "get_params"):
            # Для sklearn моделей
            trainer.model = weights
        else:
            logger.warning("Не удалось восстановить веса модели")
