"""
Базовый класс для loss functions.

Определяет единый интерфейс для всех функций потерь.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


class BaseLoss(ABC, nn.Module):
    """
    Базовый класс для всех loss functions.

    Поддерживает как PyTorch, так и NumPy/sklearn API.
    """

    def __init__(self, name: Optional[str] = None, **kwargs):
        """
        Args:
            name: Имя loss function
            **kwargs: Дополнительные параметры
        """
        super().__init__()
        self.name = name or self.__class__.__name__
        self.params = kwargs

    @abstractmethod
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Вычислить loss (PyTorch API).

        Args:
            predictions: Предсказания модели
            targets: Истинные значения

        Returns:
            Значение loss (скаляр)
        """
        pass

    def __call__(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Вызов loss function.

        Args:
            predictions: Предсказания
            targets: Истинные значения

        Returns:
            Значение loss
        """
        return self.forward(predictions, targets)

    def numpy_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Вычислить loss на NumPy массивах (для sklearn совместимости).

        Args:
            predictions: Предсказания модели
            targets: Истинные значения

        Returns:
            Значение loss
        """
        # Конвертируем в torch tensors
        pred_tensor = torch.from_numpy(predictions).float()
        target_tensor = torch.from_numpy(targets).float()

        # Вычисляем loss
        loss = self.forward(pred_tensor, target_tensor)

        # Конвертируем обратно в float
        return float(loss.item())

    def get_params(self) -> Dict[str, Any]:
        """
        Получить параметры loss function.

        Returns:
            Словарь параметров
        """
        return self.params.copy()

    def __repr__(self) -> str:
        """Строковое представление."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({params_str})" if params_str else self.name


class ClassificationLoss(BaseLoss):
    """Базовый класс для loss functions классификации."""

    pass


class RegressionLoss(BaseLoss):
    """Базовый класс для loss functions регрессии."""

    pass


class TradingLoss(BaseLoss):
    """Базовый класс для кастомных trading loss functions."""

    pass
