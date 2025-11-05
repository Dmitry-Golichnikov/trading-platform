"""
Стандартные функции потерь для регрессии.

MSE, MAE, Huber Loss, Quantile Loss.
"""

import torch
import torch.nn.functional as F

from src.modeling.loss_functions.base import RegressionLoss


class MSELoss(RegressionLoss):
    """
    Mean Squared Error Loss.

    Формула:
        MSE = mean((y_pred - y_true)^2)

    Примеры:
        >>> loss_fn = MSELoss()
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(self, reduction: str = "mean"):
        """
        Args:
            reduction: Способ агрегации ('mean', 'sum', 'none')
        """
        super().__init__(name="MSE", reduction=reduction)
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Вычислить MSE loss.

        Args:
            predictions: Предсказанные значения
            targets: Истинные значения

        Returns:
            Loss value
        """
        return F.mse_loss(predictions, targets, reduction=self.reduction)


class MAELoss(RegressionLoss):
    """
    Mean Absolute Error Loss (L1 Loss).

    Формула:
        MAE = mean(|y_pred - y_true|)

    Более робастна к выбросам чем MSE.

    Примеры:
        >>> loss_fn = MAELoss()
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(self, reduction: str = "mean"):
        """
        Args:
            reduction: Способ агрегации ('mean', 'sum', 'none')
        """
        super().__init__(name="MAE", reduction=reduction)
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Вычислить MAE loss.

        Args:
            predictions: Предсказанные значения
            targets: Истинные значения

        Returns:
            Loss value
        """
        return F.l1_loss(predictions, targets, reduction=self.reduction)


class HuberLoss(RegressionLoss):
    """
    Huber Loss.

    Комбинация MSE и MAE:
    - Квадратичная для малых ошибок (как MSE)
    - Линейная для больших ошибок (как MAE)

    Формула:
        L(δ) = 0.5 * (y_pred - y_true)^2           если |y_pred - y_true| <= δ
             = δ * (|y_pred - y_true| - 0.5 * δ)   иначе

    Примеры:
        >>> loss_fn = HuberLoss(delta=1.0)
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        """
        Args:
            delta: Порог переключения между квадратичной и линейной частями
            reduction: Способ агрегации
        """
        super().__init__(name="Huber", delta=delta, reduction=reduction)
        self.delta = delta
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Вычислить Huber loss.

        Args:
            predictions: Предсказанные значения
            targets: Истинные значения

        Returns:
            Loss value
        """
        return F.huber_loss(
            predictions, targets, delta=self.delta, reduction=self.reduction
        )


class QuantileLoss(RegressionLoss):
    """
    Quantile Loss (Pinball Loss).

    Для предсказания квантилей распределения.

    Формула:
        L(τ) = max(τ * (y_true - y_pred), (τ - 1) * (y_true - y_pred))

    Где τ (tau) - целевой квантиль (например, 0.5 для медианы).

    Примеры:
        >>> loss_fn = QuantileLoss(quantile=0.9)  # 90-й перцентиль
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(self, quantile: float = 0.5, reduction: str = "mean"):
        """
        Args:
            quantile: Целевой квантиль (от 0 до 1)
            reduction: Способ агрегации
        """
        super().__init__(name="Quantile", quantile=quantile, reduction=reduction)

        if not 0 < quantile < 1:
            raise ValueError(f"Quantile должен быть между 0 и 1, получено: {quantile}")

        self.quantile = quantile
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Вычислить Quantile loss.

        Args:
            predictions: Предсказанные значения
            targets: Истинные значения

        Returns:
            Loss value
        """
        errors = targets - predictions

        loss = torch.max(self.quantile * errors, (self.quantile - 1) * errors)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class RMSELoss(RegressionLoss):
    """
    Root Mean Squared Error Loss.

    Формула:
        RMSE = sqrt(mean((y_pred - y_true)^2))

    Примеры:
        >>> loss_fn = RMSELoss()
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-8):
        """
        Args:
            reduction: Способ агрегации
            eps: Малое число для численной стабильности
        """
        super().__init__(name="RMSE", reduction=reduction, eps=eps)
        self.reduction = reduction
        self.eps = eps

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Вычислить RMSE loss.

        Args:
            predictions: Предсказанные значения
            targets: Истинные значения

        Returns:
            Loss value
        """
        mse = F.mse_loss(predictions, targets, reduction=self.reduction)
        return torch.sqrt(mse + self.eps)


class LogCoshLoss(RegressionLoss):
    """
    Log-Cosh Loss.

    Формула:
        L = log(cosh(y_pred - y_true))

    Более гладкая чем L1, менее чувствительна к выбросам чем L2.

    Примеры:
        >>> loss_fn = LogCoshLoss()
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(self, reduction: str = "mean"):
        """
        Args:
            reduction: Способ агрегации
        """
        super().__init__(name="LogCosh", reduction=reduction)
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Вычислить Log-Cosh loss.

        Args:
            predictions: Предсказанные значения
            targets: Истинные значения

        Returns:
            Loss value
        """
        errors = predictions - targets
        loss = torch.log(torch.cosh(errors))

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
