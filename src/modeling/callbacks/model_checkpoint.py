"""
Model Checkpoint callback.

Сохраняет модель во время обучения на основе метрики.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Literal

from src.modeling.callbacks.base import Callback

logger = logging.getLogger(__name__)


class ModelCheckpoint(Callback):
    """
    Callback для сохранения лучшей модели.

    Сохраняет модель когда отслеживаемая метрика улучшается.

    Примеры:
        >>> checkpoint = ModelCheckpoint(
        >>>     filepath='models/best_model.pt',
        >>>     monitor='val_loss',
        >>>     mode='min',
        >>>     save_best_only=True
        >>> )
    """

    def __init__(
        self,
        filepath: str | Path,
        monitor: str = "val_loss",
        mode: Literal["min", "max"] = "min",
        save_best_only: bool = True,
        save_freq: int = 1,
        verbose: bool = True,
    ):
        """
        Args:
            filepath: Путь для сохранения модели
            monitor: Имя метрики для отслеживания
            mode: 'min' для минимизации, 'max' для максимизации
            save_best_only: Сохранять только лучшую модель
            save_freq: Частота сохранения в эпохах (если не save_best_only)
            verbose: Выводить сообщения в лог
        """
        super().__init__()
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.verbose = verbose

        if mode == "min":
            self.monitor_op = lambda x, y: x < y
            self.best_value = float("inf")
        else:
            self.monitor_op = lambda x, y: x > y
            self.best_value = float("-inf")

        # Создаём директорию если не существует
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def on_train_begin(self, trainer) -> None:
        """Сброс лучшего значения в начале обучения."""
        if self.mode == "min":
            self.best_value = float("inf")
        else:
            self.best_value = float("-inf")

        if self.verbose:
            logger.info(
                f"ModelCheckpoint: сохранение в {self.filepath}, "
                f"отслеживается {self.monitor} (mode={self.mode})"
            )

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]) -> None:
        """Сохранение модели в конце эпохи."""
        current_value = logs.get(self.monitor)

        if current_value is None:
            if self.verbose:
                logger.warning(
                    f"ModelCheckpoint: метрика '{self.monitor}' не найдена. "
                    f"Доступные: {list(logs.keys())}"
                )
            return

        should_save = False

        if self.save_best_only:
            # Сохраняем только если улучшилось
            if self.monitor_op(current_value, self.best_value):
                self.best_value = current_value
                should_save = True

        if self.verbose:
            logger.info(
                f"Эпоха {epoch}: {self.monitor} " f"улучшилась до {current_value:.6f}"
            )
        else:
            # Сохраняем каждые save_freq эпох
            if (epoch + 1) % self.save_freq == 0:
                should_save = True

        if should_save:
            self._save_model(trainer, epoch, current_value)

    def _save_model(self, trainer, epoch: int, metric_value: float) -> None:
        """
        Сохранить модель.

        Args:
            trainer: Экземпляр ModelTrainer
            epoch: Номер эпохи
            metric_value: Значение метрики
        """
        try:
            # Формируем путь с эпохой и метрикой
            filepath = self._format_filepath(epoch, metric_value)

            # Сохраняем модель
            if hasattr(trainer.model, "save"):
                trainer.model.save(filepath)
            else:
                # Fallback для моделей без метода save
                import torch

                if hasattr(trainer.model, "state_dict"):
                    torch.save(trainer.model.state_dict(), filepath)
                else:
                    import joblib

                    joblib.dump(trainer.model, filepath)

            if self.verbose:
                logger.info(f"Модель сохранена: {filepath}")

        except Exception as e:
            logger.error(f"Ошибка при сохранении модели: {e}")

    def _format_filepath(self, epoch: int, metric_value: float) -> Path:
        """
        Форматировать путь к файлу с подстановкой эпохи и метрики.

        Args:
            epoch: Номер эпохи
            metric_value: Значение метрики

        Returns:
            Отформатированный путь
        """
        filepath_str = str(self.filepath)

        # Подстановка переменных
        filepath_str = filepath_str.replace("{epoch}", str(epoch))
        filepath_str = filepath_str.replace("{epoch:02d}", f"{epoch:02d}")

        # Форматирование метрики
        metric_str = f"{metric_value:.4f}".replace(".", "_")
        filepath_str = filepath_str.replace(f"{{{self.monitor}}}", metric_str)
        filepath_str = filepath_str.replace(f"{{{self.monitor}:.4f}}", metric_str)

        return Path(filepath_str)
