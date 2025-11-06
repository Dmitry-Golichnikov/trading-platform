"""
Profit-based Loss для трейдинга.

Функция потерь, которая напрямую оптимизирует прибыль/убыток.
"""

import torch

from src.modeling.loss_functions.base import TradingLoss


class ProfitBasedLoss(TradingLoss):
    """
    Profit-based Loss.

    Оптимизирует модель на максимизацию прибыли с учётом комиссий.

    Формула:
        Profit = sign(prediction) * actual_return - |sign(prediction)| * commission
        Loss = -mean(Profit)

    Примеры:
        >>> loss_fn = ProfitBasedLoss(commission=0.001)
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(self, commission: float = 0.001, slippage: float = 0.0, reduction: str = "mean"):
        """
        Args:
            commission: Комиссия за сделку (в долях)
            slippage: Проскальзывание (в долях)
            reduction: Способ агрегации
        """
        super().__init__(
            name="ProfitBasedLoss",
            commission=commission,
            slippage=slippage,
            reduction=reduction,
        )
        self.commission = commission
        self.slippage = slippage
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Вычислить Profit-based Loss.

        Args:
            predictions: Предсказанные изменения (сигналы)
            targets: Истинные изменения цены

        Returns:
            Loss value (негативная прибыль)
        """
        # Бинарные позиции: long (1), short (-1), или hold (0)
        positions = torch.sign(predictions)

        # Прибыль от позиции
        returns = positions * targets

        # Вычитаем комиссии (платим при открытии позиции)
        has_position = (positions != 0).float()
        costs = has_position * (self.commission + self.slippage)

        # Чистая прибыль
        net_profit = returns - costs

        # Loss = негативная прибыль
        loss = -net_profit

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class SharpeRatioLoss(TradingLoss):
    """
    Sharpe Ratio Loss.

    Оптимизирует модель на максимизацию Sharpe Ratio.

    Формула:
        Sharpe = mean(returns) / (std(returns) + eps)
        Loss = -Sharpe

    Примеры:
        >>> loss_fn = SharpeRatioLoss()
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(self, eps: float = 1e-8, reduction: str = "mean"):
        """
        Args:
            eps: Малое число для численной стабильности
            reduction: Способ агрегации (обычно 'mean')
        """
        super().__init__(name="SharpeRatioLoss", eps=eps, reduction=reduction)
        self.eps = eps
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Вычислить Sharpe Ratio Loss.

        Args:
            predictions: Предсказанные изменения (сигналы)
            targets: Истинные изменения цены

        Returns:
            Loss value (негативный Sharpe Ratio)
        """
        # Позиции
        positions = torch.sign(predictions)

        # Returns от стратегии
        strategy_returns = positions * targets

        # Вычисляем Sharpe Ratio
        mean_return = strategy_returns.mean()
        std_return = strategy_returns.std()

        sharpe_ratio = mean_return / (std_return + self.eps)

        # Loss = негативный Sharpe
        return -sharpe_ratio


class ExpectedPnLLoss(TradingLoss):
    """
    Expected PnL Loss.

    Оптимизирует ожидаемую прибыль с учётом вероятностей:
        E[PnL] = p_win * TP - (1 - p_win) * SL - commission

    Примеры:
        >>> loss_fn = ExpectedPnLLoss(
        >>>     take_profit=0.02,
        >>>     stop_loss=0.01,
        >>>     commission=0.001
        >>> )
        >>> # predictions - вероятности роста (0-1)
        >>> # targets - бинарные метки (0 или 1)
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(
        self,
        take_profit: float = 0.02,
        stop_loss: float = 0.01,
        commission: float = 0.001,
        reduction: str = "mean",
    ):
        """
        Args:
            take_profit: Уровень тейк-профита (в долях)
            stop_loss: Уровень стоп-лосса (в долях)
            commission: Комиссия за сделку
            reduction: Способ агрегации
        """
        super().__init__(
            name="ExpectedPnLLoss",
            take_profit=take_profit,
            stop_loss=stop_loss,
            commission=commission,
            reduction=reduction,
        )
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.commission = commission
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Вычислить Expected PnL Loss.

        Args:
            predictions: Предсказанные вероятности успеха (0-1)
            targets: Бинарные метки (0 или 1)

        Returns:
            Loss value (негативный ожидаемый PnL)
        """
        # Применяем sigmoid если predictions вне [0, 1]
        if predictions.min() < 0 or predictions.max() > 1:
            predictions = torch.sigmoid(predictions)

        # Ожидаемый PnL для каждой позиции
        expected_pnl = predictions * self.take_profit - (1 - predictions) * self.stop_loss - self.commission

        # Loss = негативный ожидаемый PnL
        loss = -expected_pnl

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class RiskAdjustedProfitLoss(TradingLoss):
    """
    Risk-Adjusted Profit Loss.

    Оптимизирует прибыль с учётом риска:
        Loss = -(Profit - risk_penalty * Volatility)

    Примеры:
        >>> loss_fn = RiskAdjustedProfitLoss(risk_penalty=0.5)
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(
        self,
        risk_penalty: float = 0.5,
        commission: float = 0.001,
        reduction: str = "mean",
    ):
        """
        Args:
            risk_penalty: Коэффициент штрафа за риск
            commission: Комиссия
            reduction: Способ агрегации
        """
        super().__init__(
            name="RiskAdjustedProfitLoss",
            risk_penalty=risk_penalty,
            commission=commission,
            reduction=reduction,
        )
        self.risk_penalty = risk_penalty
        self.commission = commission
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Вычислить Risk-Adjusted Profit Loss.

        Args:
            predictions: Предсказанные изменения
            targets: Истинные изменения

        Returns:
            Loss value
        """
        # Позиции
        positions = torch.sign(predictions)

        # Returns
        returns = positions * targets

        # Комиссии
        has_position = (positions != 0).float()
        costs = has_position * self.commission

        # Чистая прибыль
        net_returns = returns - costs

        # Волатильность (риск)
        volatility = net_returns.std()

        # Risk-adjusted profit
        risk_adjusted = net_returns.mean() - self.risk_penalty * volatility

        # Loss
        return -risk_adjusted
