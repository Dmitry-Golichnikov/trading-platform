"""
Binary Cross Entropy Loss.

Стандартная функция потерь для бинарной классификации.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from src.modeling.loss_functions.base import ClassificationLoss


class BinaryCrossEntropyLoss(ClassificationLoss):
    """
    Binary Cross Entropy Loss.

    Формула:
        BCE = -[y * log(p) + (1-y) * log(1-p)]

    Где:
        y - истинная метка (0 или 1)
        p - предсказанная вероятность

    Примеры:
        >>> loss_fn = BinaryCrossEntropyLoss()
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(self, reduction: str = "mean", pos_weight: Optional[float] = None):
        """
        Args:
            reduction: Способ агрегации ('mean', 'sum', 'none')
            pos_weight: Вес для положительных примеров (для дисбаланса классов)
        """
        super().__init__(
            name="BinaryCrossEntropy", reduction=reduction, pos_weight=pos_weight
        )
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Вычислить BCE loss.

        Args:
            predictions: Предсказанные вероятности,
                shape (batch_size,) или (batch_size, 1)
            targets: Истинные метки,
                shape (batch_size,) или (batch_size, 1)

        Returns:
            Loss value
        """
        # Убеждаемся что размерности правильные
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Применяем sigmoid если нужно
        if predictions.min() < 0 or predictions.max() > 1:
            predictions = torch.sigmoid(predictions)

        # Вычисляем BCE
        pos_weight_tensor = None
        if self.pos_weight is not None:
            pos_weight_tensor = torch.tensor(self.pos_weight, device=predictions.device)

        loss = F.binary_cross_entropy(
            predictions,
            targets,
            weight=None,
            reduction=self.reduction,
            pos_weight=pos_weight_tensor,
        )

        return loss


class BCEWithLogitsLoss(ClassificationLoss):
    """
    BCE with Logits Loss.

    Более численно стабильная версия BCE, которая принимает логиты вместо вероятностей.

    Примеры:
        >>> loss_fn = BCEWithLogitsLoss()
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(self, reduction: str = "mean", pos_weight: Optional[float] = None):
        """
        Args:
            reduction: Способ агрегации ('mean', 'sum', 'none')
            pos_weight: Вес для положительных примеров
        """
        super().__init__(
            name="BCEWithLogits", reduction=reduction, pos_weight=pos_weight
        )
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Вычислить BCE with logits loss.

        Args:
            predictions: Логиты (не вероятности),
                shape (batch_size,) или (batch_size, 1)
            targets: Истинные метки,
                shape (batch_size,) или (batch_size, 1)

        Returns:
            Loss value
        """
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        pos_weight_tensor = None
        if self.pos_weight is not None:
            pos_weight_tensor = torch.tensor(self.pos_weight, device=predictions.device)

        loss = F.binary_cross_entropy_with_logits(
            predictions,
            targets,
            pos_weight=pos_weight_tensor,
            reduction=self.reduction,
        )

        return loss


class WeightedBCELoss(ClassificationLoss):
    """
    Weighted Binary Cross Entropy Loss.

    BCE с весами для каждого класса (для борьбы с дисбалансом).

    Примеры:
        >>> loss_fn = WeightedBCELoss(pos_weight=2.0)
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(
        self,
        pos_weight: float = 1.0,
        neg_weight: float = 1.0,
        reduction: str = "mean",
    ):
        """
        Args:
            pos_weight: Вес для положительного класса
            neg_weight: Вес для отрицательного класса
            reduction: Способ агрегации
        """
        super().__init__(
            name="WeightedBCE",
            pos_weight=pos_weight,
            neg_weight=neg_weight,
            reduction=reduction,
        )
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Вычислить weighted BCE loss.

        Args:
            predictions: Предсказанные вероятности
            targets: Истинные метки

        Returns:
            Loss value
        """
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Применяем sigmoid если нужно
        if predictions.min() < 0 or predictions.max() > 1:
            predictions = torch.sigmoid(predictions)

        # Вычисляем BCE для каждого примера
        bce = -(
            targets * torch.log(predictions + 1e-7)
            + (1 - targets) * torch.log(1 - predictions + 1e-7)
        )

        # Применяем веса
        weights = targets * self.pos_weight + (1 - targets) * self.neg_weight
        weighted_bce = bce * weights

        # Агрегируем
        if self.reduction == "mean":
            return weighted_bce.mean()
        elif self.reduction == "sum":
            return weighted_bce.sum()
        else:
            return weighted_bce
