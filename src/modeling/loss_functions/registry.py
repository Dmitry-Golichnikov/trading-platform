"""
Реестр Loss Functions.

Централизованный реестр всех доступных функций потерь.
"""

import logging
from typing import Any, Dict, Optional, Type

from src.modeling.loss_functions.base import BaseLoss

# Classification losses
from src.modeling.loss_functions.classification.bce import (
    BCEWithLogitsLoss,
    BinaryCrossEntropyLoss,
    WeightedBCELoss,
)
from src.modeling.loss_functions.classification.focal import (
    FocalLoss,
    MultiClassFocalLoss,
)

# Regression losses
from src.modeling.loss_functions.regression.standard import (
    HuberLoss,
    LogCoshLoss,
    MAELoss,
    MSELoss,
    QuantileLoss,
    RMSELoss,
)

# Trading custom losses
from src.modeling.loss_functions.trading_custom.directional import (
    AsymmetricDirectionalLoss,
    DirectionalLoss,
    SignLoss,
)
from src.modeling.loss_functions.trading_custom.profit_based import (
    ExpectedPnLLoss,
    ProfitBasedLoss,
    RiskAdjustedProfitLoss,
    SharpeRatioLoss,
)

logger = logging.getLogger(__name__)


class LossRegistry:
    """
    Реестр функций потерь.

    Предоставляет централизованный доступ ко всем loss functions.

    Примеры:
        >>> loss_fn = LossRegistry.get('focal', alpha=0.25, gamma=2.0)
        >>> loss_fn = LossRegistry.get('mse')
    """

    _losses: Dict[str, Type[BaseLoss]] = {
        # Classification
        "bce": BinaryCrossEntropyLoss,
        "binary_crossentropy": BinaryCrossEntropyLoss,
        "bce_logits": BCEWithLogitsLoss,
        "weighted_bce": WeightedBCELoss,
        "focal": FocalLoss,
        "focal_multiclass": MultiClassFocalLoss,
        # Regression
        "mse": MSELoss,
        "mae": MAELoss,
        "l1": MAELoss,
        "huber": HuberLoss,
        "quantile": QuantileLoss,
        "rmse": RMSELoss,
        "logcosh": LogCoshLoss,
        # Trading Custom
        "directional": DirectionalLoss,
        "sign": SignLoss,
        "asymmetric_directional": AsymmetricDirectionalLoss,
        "profit": ProfitBasedLoss,
        "sharpe": SharpeRatioLoss,
        "expected_pnl": ExpectedPnLLoss,
        "risk_adjusted_profit": RiskAdjustedProfitLoss,
    }

    _metadata: Dict[str, Dict[str, Any]] = {
        # Classification
        "bce": {
            "category": "classification",
            "description": "Binary Cross Entropy Loss",
            "supports_multiclass": False,
        },
        "bce_logits": {
            "category": "classification",
            "description": "BCE with Logits (numerically stable)",
            "supports_multiclass": False,
        },
        "weighted_bce": {
            "category": "classification",
            "description": "Weighted BCE for class imbalance",
            "supports_multiclass": False,
        },
        "focal": {
            "category": "classification",
            "description": "Focal Loss for hard examples",
            "supports_multiclass": False,
        },
        "focal_multiclass": {
            "category": "classification",
            "description": "Focal Loss for multiclass",
            "supports_multiclass": True,
        },
        # Regression
        "mse": {
            "category": "regression",
            "description": "Mean Squared Error",
            "robust_to_outliers": False,
        },
        "mae": {
            "category": "regression",
            "description": "Mean Absolute Error (L1)",
            "robust_to_outliers": True,
        },
        "huber": {
            "category": "regression",
            "description": "Huber Loss (MSE + MAE hybrid)",
            "robust_to_outliers": True,
        },
        "quantile": {
            "category": "regression",
            "description": "Quantile Loss (Pinball Loss)",
            "robust_to_outliers": True,
        },
        "rmse": {
            "category": "regression",
            "description": "Root Mean Squared Error",
            "robust_to_outliers": False,
        },
        "logcosh": {
            "category": "regression",
            "description": "Log-Cosh Loss",
            "robust_to_outliers": True,
        },
        # Trading
        "directional": {
            "category": "trading",
            "description": "Optimizes direction accuracy",
        },
        "sign": {
            "category": "trading",
            "description": "Binary sign matching",
        },
        "asymmetric_directional": {
            "category": "trading",
            "description": "Asymmetric penalties for false signals",
        },
        "profit": {
            "category": "trading",
            "description": "Direct profit optimization",
        },
        "sharpe": {
            "category": "trading",
            "description": "Sharpe Ratio maximization",
        },
        "expected_pnl": {
            "category": "trading",
            "description": "Expected PnL optimization",
        },
        "risk_adjusted_profit": {
            "category": "trading",
            "description": "Profit with risk penalty",
        },
    }

    @classmethod
    def get(cls, name: str, **kwargs) -> BaseLoss:
        """
        Получить loss function по имени.

        Args:
            name: Имя loss function
            **kwargs: Параметры для инициализации

        Returns:
            Экземпляр loss function

        Raises:
            ValueError: Если loss function не найдена

        Примеры:
            >>> loss_fn = LossRegistry.get('focal', alpha=0.25, gamma=2.0)
            >>> loss_fn = LossRegistry.get('mse')
        """
        name_lower = name.lower()

        if name_lower not in cls._losses:
            available = ", ".join(sorted(cls._losses.keys()))
            raise ValueError(
                f"Loss function '{name}' не найдена. " f"Доступные: {available}"
            )

        loss_class = cls._losses[name_lower]
        return loss_class(**kwargs)

    @classmethod
    def register(
        cls,
        name: str,
        loss_class: Type[BaseLoss],
        category: str = "custom",
        description: Optional[str] = None,
        **metadata,
    ) -> None:
        """
        Зарегистрировать кастомную loss function.

        Args:
            name: Имя для регистрации
            loss_class: Класс loss function
            category: Категория ('classification', 'regression', 'trading', 'custom')
            description: Описание
            **metadata: Дополнительные метаданные

        Примеры:
            >>> LossRegistry.register(
            >>>     'my_loss',
            >>>     MyCustomLoss,
            >>>     category='custom',
            >>>     description='My custom loss function'
            >>> )
        """
        name_lower = name.lower()

        if name_lower in cls._losses:
            logger.warning(
                f"Loss function '{name}' уже зарегистрирована, перезаписываем"
            )

        cls._losses[name_lower] = loss_class
        cls._metadata[name_lower] = {
            "category": category,
            "description": description or loss_class.__doc__,
            **metadata,
        }

        logger.info(f"Зарегистрирована loss function: {name}")

    @classmethod
    def list_losses(cls, category: Optional[str] = None) -> list[str]:
        """
        Получить список доступных loss functions.

        Args:
            category: Фильтр по категории (опционально)

        Returns:
            Список имён loss functions

        Примеры:
            >>> LossRegistry.list_losses()
            ['bce', 'focal', 'mse', ...]
            >>> LossRegistry.list_losses(category='trading')
            ['directional', 'profit', 'sharpe']
        """
        if category is None:
            return sorted(cls._losses.keys())

        category_lower = category.lower()
        return [
            name
            for name, meta in cls._metadata.items()
            if meta.get("category") == category_lower
        ]

    @classmethod
    def get_metadata(cls, name: str) -> Dict[str, Any]:
        """
        Получить метаданные loss function.

        Args:
            name: Имя loss function

        Returns:
            Словарь с метаданными

        Raises:
            ValueError: Если loss function не найдена
        """
        name_lower = name.lower()

        if name_lower not in cls._metadata:
            raise ValueError(f"Loss function '{name}' не найдена")

        return cls._metadata[name_lower].copy()

    @classmethod
    def summary(cls) -> str:
        """
        Получить сводку по всем loss functions.

        Returns:
            Строка с информацией
        """
        lines = [f"Всего loss functions: {len(cls._losses)}\n"]

        # Группируем по категориям
        categories: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for name, meta in cls._metadata.items():
            category = meta.get("category", "other")
            if category not in categories:
                categories[category] = []
            categories[category].append((name, meta))

        # Выводим по категориям
        for category in sorted(categories.keys()):
            lines.append(f"\n{category.upper()}:")
            for name, meta in sorted(categories[category]):
                desc = meta.get("description", "No description")
                lines.append(f"  - {name}: {desc}")

        return "\n".join(lines)


# Создаём глобальный экземпляр реестра
loss_registry = LossRegistry()
