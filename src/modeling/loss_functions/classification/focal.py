"""
Focal Loss.

Функция потерь для борьбы с дисбалансом классов, фокусируется на сложных примерах.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from src.modeling.loss_functions.base import ClassificationLoss


class FocalLoss(ClassificationLoss):
    """
    Focal Loss для бинарной классификации.

    Формула:
        FL = -α * (1 - p)^γ * log(p)

    Где:
        p - предсказанная вероятность правильного класса
        α - балансирующий параметр для классов
        γ (gamma) - focusing parameter (обычно 2)

    Focal Loss уменьшает вес легко классифицируемых примеров и фокусируется
    на сложных примерах.

    Ссылка:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
        https://arxiv.org/abs/1708.02002

    Примеры:
        >>> loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(
        self,
        alpha: Optional[float] = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Args:
            alpha: Балансирующий параметр (обычно 0.25)
            gamma: Focusing parameter (обычно 2.0)
            reduction: Способ агрегации ('mean', 'sum', 'none')
        """
        super().__init__(name="FocalLoss", alpha=alpha, gamma=gamma, reduction=reduction)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Вычислить Focal Loss.

        Args:
            predictions: Предсказанные вероятности или логиты
            targets: Истинные метки (0 или 1)

        Returns:
            Loss value
        """
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Применяем sigmoid если входы - логиты
        if predictions.min() < 0 or predictions.max() > 1:
            predictions = torch.sigmoid(predictions)

        # Вычисляем BCE
        bce = F.binary_cross_entropy(predictions, targets, reduction="none")

        # Вычисляем p_t (вероятность правильного класса)
        p_t = predictions * targets + (1 - predictions) * (1 - targets)

        # Вычисляем α_t (обрабатывать None как 1.0)
        alpha_value = self.alpha if self.alpha is not None else 1.0
        alpha_t = alpha_value * targets + (1 - alpha_value) * (1 - targets)

        # Focal term: (1 - p_t)^γ
        focal_weight = (1 - p_t) ** self.gamma

        # Focal Loss
        focal_loss = alpha_t * focal_weight * bce

        # Агрегируем
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class MultiClassFocalLoss(ClassificationLoss):
    """
    Focal Loss для мультиклассовой классификации.

    Примеры:
        >>> loss_fn = MultiClassFocalLoss(alpha=[0.25, 0.25, 0.5], gamma=2.0)
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(
        self,
        alpha: Optional[list[float]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Args:
            alpha: Балансирующие параметры для каждого класса (или None)
            gamma: Focusing parameter
            reduction: Способ агрегации
        """
        super().__init__(name="MultiClassFocalLoss", alpha=alpha, gamma=gamma, reduction=reduction)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Вычислить Multi-class Focal Loss.

        Args:
            predictions: Предсказанные логиты, shape (batch_size, num_classes)
            targets: Истинные метки, shape (batch_size,)

        Returns:
            Loss value
        """
        # Применяем softmax для получения вероятностей
        probs = F.softmax(predictions, dim=1)

        # Вычисляем cross entropy
        ce_loss = F.cross_entropy(predictions, targets, reduction="none")

        # Получаем вероятности правильных классов
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal term
        focal_weight = (1 - p_t) ** self.gamma

        # Focal Loss
        focal_loss = focal_weight * ce_loss

        # Применяем alpha если указаны
        if self.alpha is not None:
            alpha_t = torch.tensor(self.alpha, device=predictions.device)[targets]
            focal_loss = alpha_t * focal_loss

        # Агрегируем
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
