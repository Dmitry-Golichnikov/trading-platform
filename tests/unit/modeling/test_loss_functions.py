"""
Тесты для Loss Functions.
"""

import numpy as np
import pytest
import torch

from src.modeling.loss_functions.classification.bce import BinaryCrossEntropyLoss
from src.modeling.loss_functions.classification.focal import FocalLoss
from src.modeling.loss_functions.registry import LossRegistry
from src.modeling.loss_functions.regression.standard import MAELoss, MSELoss
from src.modeling.loss_functions.trading_custom.directional import DirectionalLoss


class TestLossRegistry:
    """Тесты для LossRegistry."""

    def test_get_loss(self):
        """Тест получения loss function по имени."""

        # BCE
        loss = LossRegistry.get("bce")
        assert isinstance(loss, BinaryCrossEntropyLoss)

        # MSE
        loss = LossRegistry.get("mse")
        assert isinstance(loss, MSELoss)

        # Focal
        loss = LossRegistry.get("focal", alpha=0.5, gamma=2.0)
        assert isinstance(loss, FocalLoss)

    def test_get_unknown_loss(self):
        """Тест получения несуществующей loss function."""

        with pytest.raises(ValueError) as exc_info:
            LossRegistry.get("nonexistent_loss")

        assert (
            "не найдена" in str(exc_info.value)
            or "not found" in str(exc_info.value).lower()
        )

    def test_list_losses(self):
        """Тест получения списка losses."""

        all_losses = LossRegistry.list_losses()

        assert "bce" in all_losses
        assert "mse" in all_losses
        assert "focal" in all_losses
        assert "directional" in all_losses

    def test_list_losses_by_category(self):
        """Тест фильтрации по категории."""

        # Classification
        clf_losses = LossRegistry.list_losses(category="classification")
        assert "bce" in clf_losses
        assert "focal" in clf_losses
        assert "mse" not in clf_losses

        # Regression
        reg_losses = LossRegistry.list_losses(category="regression")
        assert "mse" in reg_losses
        assert "mae" in reg_losses
        assert "bce" not in reg_losses

        # Trading
        trading_losses = LossRegistry.list_losses(category="trading")
        assert "directional" in trading_losses
        assert "profit" in trading_losses

    def test_get_metadata(self):
        """Тест получения метаданных."""

        metadata = LossRegistry.get_metadata("bce")

        assert metadata["category"] == "classification"
        assert "description" in metadata

    def test_summary(self):
        """Тест генерации сводки."""

        summary = LossRegistry.summary()

        assert "CLASSIFICATION" in summary.upper()
        assert "REGRESSION" in summary.upper()
        assert "TRADING" in summary.upper()


class TestClassificationLosses:
    """Тесты для classification losses."""

    def test_bce_loss(self):
        """Тест Binary Cross Entropy Loss."""

        loss_fn = BinaryCrossEntropyLoss()

        # Простой случай
        predictions = torch.tensor([0.9, 0.1, 0.8, 0.2], dtype=torch.float32)
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)

        loss = loss_fn(predictions, targets)

        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_focal_loss(self):
        """Тест Focal Loss."""

        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

        predictions = torch.tensor([0.9, 0.1, 0.8, 0.2], dtype=torch.float32)
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)

        loss = loss_fn(predictions, targets)

        assert loss.item() > 0
        assert torch.isfinite(loss)


class TestRegressionLosses:
    """Тесты для regression losses."""

    def test_mse_loss(self):
        """Тест MSE Loss."""

        loss_fn = MSELoss()

        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        targets = torch.tensor([1.1, 2.2, 2.9, 4.1], dtype=torch.float32)

        loss = loss_fn(predictions, targets)

        # Вручную вычисляем MSE
        expected = ((predictions - targets) ** 2).mean()

        assert torch.allclose(loss, expected)

    def test_mae_loss(self):
        """Тест MAE Loss."""

        loss_fn = MAELoss()

        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        targets = torch.tensor([1.1, 2.2, 2.9, 4.1], dtype=torch.float32)

        loss = loss_fn(predictions, targets)

        # Вручную вычисляем MAE
        expected = torch.abs(predictions - targets).mean()

        assert torch.allclose(loss, expected)


class TestTradingLosses:
    """Тесты для trading losses."""

    def test_directional_loss(self):
        """Тест Directional Loss."""

        loss_fn = DirectionalLoss(weight_by_magnitude=False)

        # Правильные направления
        predictions = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
        targets = torch.tensor([0.5, -0.3, 1.5], dtype=torch.float32)

        loss = loss_fn(predictions, targets)

        # Все направления правильные, loss должен быть 0
        assert loss.item() == 0.0

        # Неправильные направления
        predictions = torch.tensor([1.0, -1.0], dtype=torch.float32)
        targets = torch.tensor([-0.5, 0.5], dtype=torch.float32)

        loss = loss_fn(predictions, targets)

        # Направления неправильные, loss > 0
        assert loss.item() > 0


class TestLossNumPyAPI:
    """Тесты для NumPy API loss functions."""

    def test_numpy_loss(self):
        """Тест вызова loss на NumPy массивах."""

        loss_fn = MSELoss()

        predictions = np.array([1.0, 2.0, 3.0, 4.0])
        targets = np.array([1.1, 2.2, 2.9, 4.1])

        loss = loss_fn.numpy_loss(predictions, targets)

        assert isinstance(loss, float)
        assert loss > 0

        # Должно совпадать с torch версией
        loss_torch = loss_fn(
            torch.tensor(predictions, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
        )

        assert np.isclose(loss, loss_torch.item())
