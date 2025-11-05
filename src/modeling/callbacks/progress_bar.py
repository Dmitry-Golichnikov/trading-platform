"""
Progress Bar callback.

Отображает progress bar во время обучения.
"""

import logging
from typing import Any, Dict, Optional

from src.modeling.callbacks.base import Callback

logger = logging.getLogger(__name__)


class ProgressBar(Callback):
    """
    Callback для отображения progress bar.

    Использует tqdm для отображения прогресса обучения.

    Примеры:
        >>> progress_bar = ProgressBar(
        >>>     total_epochs=100,
        >>>     show_metrics=True
        >>> )
    """

    def __init__(
        self,
        total_epochs: Optional[int] = None,
        show_metrics: bool = True,
        metrics_format: str = ".4f",
        desc: str = "Training",
        leave: bool = True,
        disable: bool = False,
    ):
        """
        Args:
            total_epochs: Общее количество эпох (если известно)
            show_metrics: Отображать метрики в progress bar
            metrics_format: Формат для отображения метрик
            desc: Описание для progress bar
            leave: Оставить progress bar после завершения
            disable: Отключить progress bar
        """
        super().__init__()
        self.total_epochs = total_epochs
        self.show_metrics = show_metrics
        self.metrics_format = metrics_format
        self.desc = desc
        self.leave = leave
        self.disable = disable

        self.pbar = None
        self._tqdm_available = self._check_tqdm()

    def _check_tqdm(self) -> bool:
        """Проверить доступность tqdm."""
        try:
            import tqdm

            _ = tqdm
            return True
        except ImportError:
            logger.warning("tqdm не установлен, progress bar отключён")
            return False

    def on_train_begin(self, trainer) -> None:
        """Создать progress bar в начале обучения."""
        if not self._tqdm_available or self.disable:
            return

        from tqdm import tqdm

        self.pbar = tqdm(
            total=self.total_epochs,
            desc=self.desc,
            leave=self.leave,
            unit="epoch",
            ncols=100,
        )

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]) -> None:
        """Обновить progress bar в конце эпохи."""
        if self.pbar is None:
            return

        # Формируем строку с метриками
        if self.show_metrics and logs:
            metrics_str = self._format_metrics(logs)
            self.pbar.set_postfix_str(metrics_str)

        # Обновляем прогресс
        self.pbar.update(1)

    def on_train_end(self, trainer, logs: Dict[str, Any]) -> None:
        """Закрыть progress bar в конце обучения."""
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None

    def _format_metrics(self, logs: Dict[str, Any]) -> str:
        """
        Форматировать метрики для отображения.

        Args:
            logs: Словарь с метриками

        Returns:
            Отформатированная строка
        """
        metrics_parts = []

        for key, value in logs.items():
            if isinstance(value, (int, float)):
                # Форматируем числовые значения
                formatted_value = f"{value:{self.metrics_format}}"
                metrics_parts.append(f"{key}={formatted_value}")

        return ", ".join(metrics_parts)


class TQDMProgressBar(ProgressBar):
    """
    Расширенная версия ProgressBar с дополнительными возможностями.

    Примеры:
        >>> progress_bar = TQDMProgressBar(
        >>>     total_epochs=100,
        >>>     display_keys=['train_loss', 'val_loss', 'val_acc']
        >>> )
    """

    def __init__(
        self,
        total_epochs: Optional[int] = None,
        display_keys: Optional[list[str]] = None,
        metrics_format: str = ".4f",
        desc: str = "Training",
        leave: bool = True,
        position: int = 0,
        colour: Optional[str] = None,
        disable: bool = False,
    ):
        """
        Args:
            total_epochs: Общее количество эпох
            display_keys: Список ключей метрик для отображения (None = все)
            metrics_format: Формат для метрик
            desc: Описание
            leave: Оставить progress bar после завершения
            position: Позиция progress bar (для вложенных bars)
            colour: Цвет progress bar
            disable: Отключить progress bar
        """
        super().__init__(
            total_epochs=total_epochs,
            show_metrics=True,
            metrics_format=metrics_format,
            desc=desc,
            leave=leave,
            disable=disable,
        )
        self.display_keys = display_keys
        self.position = position
        self.colour = colour

    def on_train_begin(self, trainer) -> None:
        """Создать progress bar с дополнительными параметрами."""
        if not self._tqdm_available or self.disable:
            return

        from tqdm import tqdm

        kwargs = {
            "total": self.total_epochs,
            "desc": self.desc,
            "leave": self.leave,
            "unit": "epoch",
            "position": self.position,
            "ncols": 120,
        }

        if self.colour:
            kwargs["colour"] = self.colour

        self.pbar = tqdm(**kwargs)

    def _format_metrics(self, logs: Dict[str, Any]) -> str:
        """
        Форматировать метрики с фильтрацией по display_keys.

        Args:
            logs: Словарь с метриками

        Returns:
            Отформатированная строка
        """
        metrics_parts = []

        # Определяем какие ключи отображать
        keys_to_display = self.display_keys if self.display_keys else logs.keys()

        for key in keys_to_display:
            if key in logs:
                value = logs[key]
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:{self.metrics_format}}"
                    metrics_parts.append(f"{key}={formatted_value}")

        return ", ".join(metrics_parts)
