"""
Directional Loss для трейдинга.

Функция потерь, которая фокусируется на правильности направления движения цены.
"""

import torch

from src.modeling.loss_functions.base import TradingLoss


class DirectionalLoss(TradingLoss):
    """
    Directional Loss для торговых моделей.

    Штрафует модель за неправильное предсказание направления движения цены,
    с дополнительным весом пропорциональным величине движения.

    Формула:
        L = mean(max(0, -sign(y_true) * y_pred) * |y_true|)

    Примеры:
        >>> loss_fn = DirectionalLoss()
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(self, weight_by_magnitude: bool = True, reduction: str = "mean"):
        """
        Args:
            weight_by_magnitude: Взвешивать ошибку по величине движения
            reduction: Способ агрегации
        """
        super().__init__(
            name="DirectionalLoss",
            weight_by_magnitude=weight_by_magnitude,
            reduction=reduction,
        )
        self.weight_by_magnitude = weight_by_magnitude
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Вычислить Directional Loss.

        Args:
            predictions: Предсказанные изменения цены
            targets: Истинные изменения цены

        Returns:
            Loss value
        """
        # Проверяем правильность направления
        direction_correct = torch.sign(predictions) == torch.sign(targets)

        # Ошибка для неправильных направлений
        errors = torch.where(
            direction_correct,
            torch.zeros_like(predictions),
            torch.abs(predictions - targets),
        )

        # Взвешиваем по величине движения если нужно
        if self.weight_by_magnitude:
            weights = torch.abs(targets)
            errors = errors * weights

        # Агрегируем
        if self.reduction == "mean":
            return errors.mean()
        elif self.reduction == "sum":
            return errors.sum()
        else:
            return errors


class SignLoss(TradingLoss):
    """
    Sign Loss - простая версия directional loss.

    Бинарная функция потерь: 0 если направление правильное, 1 если нет.

    Примеры:
        >>> loss_fn = SignLoss()
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(self, reduction: str = "mean"):
        """
        Args:
            reduction: Способ агрегации
        """
        super().__init__(name="SignLoss", reduction=reduction)
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Вычислить Sign Loss.

        Args:
            predictions: Предсказанные изменения
            targets: Истинные изменения

        Returns:
            Loss value
        """
        # 1 если направления не совпадают, 0 если совпадают
        incorrect_signs = (torch.sign(predictions) != torch.sign(targets)).float()

        if self.reduction == "mean":
            return incorrect_signs.mean()
        elif self.reduction == "sum":
            return incorrect_signs.sum()
        else:
            return incorrect_signs


class AsymmetricDirectionalLoss(TradingLoss):
    """
    Asymmetric Directional Loss.

    Разные штрафы за ложные покупки и ложные продажи.

    Примеры:
        >>> loss_fn = AsymmetricDirectionalLoss(
        >>>     false_long_penalty=2.0,
        >>>     false_short_penalty=1.0
        >>> )
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(
        self,
        false_long_penalty: float = 1.0,
        false_short_penalty: float = 1.0,
        reduction: str = "mean",
    ):
        """
        Args:
            false_long_penalty: Штраф за ложный сигнал покупки
            false_short_penalty: Штраф за ложный сигнал продажи
            reduction: Способ агрегации
        """
        super().__init__(
            name="AsymmetricDirectionalLoss",
            false_long_penalty=false_long_penalty,
            false_short_penalty=false_short_penalty,
            reduction=reduction,
        )
        self.false_long_penalty = false_long_penalty
        self.false_short_penalty = false_short_penalty
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Вычислить Asymmetric Directional Loss.

        Args:
            predictions: Предсказанные изменения
            targets: Истинные изменения

        Returns:
            Loss value
        """
        pred_sign = torch.sign(predictions)
        target_sign = torch.sign(targets)

        # Ошибки
        errors = torch.abs(predictions - targets)

        # Определяем тип ошибки
        false_long = (pred_sign > 0) & (target_sign <= 0)
        false_short = (pred_sign < 0) & (target_sign >= 0)

        # Применяем штрафы
        weighted_errors = torch.where(
            false_long,
            errors * self.false_long_penalty,
            torch.where(false_short, errors * self.false_short_penalty, errors),
        )

        if self.reduction == "mean":
            return weighted_errors.mean()
        elif self.reduction == "sum":
            return weighted_errors.sum()
        else:
            return weighted_errors
