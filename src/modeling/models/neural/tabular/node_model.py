"""
NODE (Neural Oblivious Decision Ensembles) модель для классификации и регрессии.

Упрощённая реализация дифференцируемых обливиусных деревьев решений.

Основана на статье "Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data"
https://arxiv.org/abs/1909.06312
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.modeling.base import BaseModel, ClassifierMixin, RegressorMixin
from src.modeling.registry import ModelRegistry

logger = logging.getLogger(__name__)


class NODE(nn.Module):
    """
    NODE (Neural Oblivious Decision Ensembles) архитектура.

    Ансамбль дифференцируемых обливиусных деревьев решений.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        num_layers: int = 4,
        num_trees: int = 2048,
        depth: int = 6,
        choice_function: str = "entmax15",
        bin_function: str = "entmoid15",
    ):
        """
        Инициализация NODE.

        Args:
            n_features: Количество признаков
            n_classes: Количество классов (или 1 для регрессии)
            num_layers: Количество слоёв
            num_trees: Количество деревьев в ансамбле
            depth: Глубина деревьев
            choice_function: Функция выбора (entmax15, sparsemax)
            bin_function: Бинаризация (entmoid15, sparsemoid)
        """
        super().__init__()

        self.n_features = n_features
        self.n_classes = n_classes
        self.num_layers = num_layers
        self.num_trees = num_trees
        self.depth = depth

        # Feature selection layers
        self.feature_selectors = nn.ModuleList([nn.Linear(n_features, num_trees * depth) for _ in range(num_layers)])

        # Output layer
        num_leaves = 2**depth
        self.output_layer = nn.Linear(num_layers * num_trees * num_leaves, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [batch_size, n_features]

        Returns:
            Predictions [batch_size, n_classes]
        """
        batch_size = x.shape[0]
        layer_outputs = []

        for layer_idx in range(self.num_layers):
            # Feature selection
            feature_values = self.feature_selectors[layer_idx](x)  # [batch_size, num_trees * depth]
            feature_values = feature_values.view(batch_size, self.num_trees, self.depth)

            # Compute binary choices (простая сигмоида вместо entmoid)
            feature_values = torch.sigmoid(feature_values)

            # Compute leaf probabilities
            num_leaves = 2**self.depth
            leaf_probs = torch.ones(batch_size, self.num_trees, num_leaves, device=x.device)

            # Обход по глубине дерева
            for d in range(self.depth):
                # Вероятность идти влево/вправо на уровне d
                left_prob = feature_values[:, :, d]  # [batch_size, num_trees]
                right_prob = 1 - left_prob

                # Обновляем вероятности листьев
                half = num_leaves // (2 ** (d + 1))
                for leaf_idx in range(num_leaves):
                    # Определяем идём ли влево или вправо для этого листа
                    if (leaf_idx // half) % 2 == 0:
                        leaf_probs[:, :, leaf_idx] *= left_prob
                    else:
                        leaf_probs[:, :, leaf_idx] *= right_prob

            # Flatten leaf probabilities
            layer_output = leaf_probs.view(batch_size, -1)  # [batch_size, num_trees * num_leaves]
            layer_outputs.append(layer_output)

        # Concatenate all layer outputs
        combined = torch.cat(layer_outputs, dim=1)  # [batch_size, num_layers * num_trees * num_leaves]

        # Output layer
        output = self.output_layer(combined)  # [batch_size, n_classes]

        return output


@ModelRegistry.register(
    "node",
    description="Neural Oblivious Decision Ensembles model",
    tags=["neural", "tabular", "tree-like", "gpu-support"],
)
class NODEModel(BaseModel, ClassifierMixin, RegressorMixin):
    """
    NODE обёртка для классификации и регрессии.

    Поддерживает:
    - Бинарную и мультиклассовую классификацию
    - Регрессию
    - GPU обучение
    - Early stopping

    Args:
        task: Тип задачи ('classification' или 'regression')
        device: Устройство ('cpu' или 'cuda')
        **hyperparams: Гиперпараметры
    """

    def __init__(self, task: str = "classification", device: str = "cpu", **hyperparams):
        """Инициализация NODE модели."""
        # Не передаём device в hyperparams
        super().__init__(task=task, **hyperparams)
        self.task = task
        self.device = torch.device(device)
        self.model = None
        self._classes = np.array([])
        self._feature_names = []

        # Дефолтные параметры
        self._set_default_params()

    def _set_default_params(self) -> None:
        """Установить дефолтные параметры."""
        defaults = {
            "num_layers": 4,
            "num_trees": 2048,
            "depth": 6,
            "choice_function": "entmax15",
            "bin_function": "entmoid15",
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "batch_size": 256,
            "max_epochs": 100,
            "patience": 15,
        }

        for key, value in defaults.items():
            if key not in self.hyperparams:
                self.hyperparams[key] = value

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> "NODEModel":
        """Обучить NODE модель."""
        logger.info(f"Начало обучения NODE модели (задача: {self.task})")

        # Сохраняем имена признаков
        self._feature_names = list(X.columns)

        # Для классификации сохраняем классы
        n_classes = 1
        if self.task == "classification":
            self._classes = np.unique(y)
            n_classes = len(self._classes)
            if n_classes == 2:
                n_classes = 1  # Бинарная классификация

        # Создаём модель
        self.model = NODE(
            n_features=len(self._feature_names),
            n_classes=n_classes,
            num_layers=self.hyperparams["num_layers"],
            num_trees=self.hyperparams["num_trees"],
            depth=self.hyperparams["depth"],
            choice_function=self.hyperparams["choice_function"],
            bin_function=self.hyperparams["bin_function"],
        ).to(self.device)

        # Optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hyperparams["learning_rate"],
            weight_decay=self.hyperparams["weight_decay"],
        )

        # Loss function
        if self.task == "classification":
            if len(self._classes) == 2:
                criterion = nn.BCEWithLogitsLoss()
            else:
                criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        # Prepare data
        X_train_tensor = torch.FloatTensor(X.values).to(self.device)
        if self.task == "classification" and len(self._classes) == 2:
            y_train_tensor = torch.FloatTensor(y.values).unsqueeze(1).to(self.device)
        else:
            y_train_tensor = (
                torch.LongTensor(y.values).to(self.device)
                if self.task == "classification"
                else torch.FloatTensor(y.values).unsqueeze(1).to(self.device)
            )

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.hyperparams["batch_size"],
            shuffle=True,
        )

        # Validation data
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val.values).to(self.device)
            if self.task == "classification" and len(self._classes) == 2:
                y_val_tensor = torch.FloatTensor(y_val.values).unsqueeze(1).to(self.device)
            else:
                y_val_tensor = (
                    torch.LongTensor(y_val.values).to(self.device)
                    if self.task == "classification"
                    else torch.FloatTensor(y_val.values).unsqueeze(1).to(self.device)
                )

            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.hyperparams["batch_size"])

        # Training loop
        import time

        start_time = time.time()

        best_val_loss = float("inf")
        patience_counter = 0
        best_epoch = 0

        for epoch in range(self.hyperparams["max_epochs"]):
            # Train
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        output = self.model(X_batch)
                        loss = criterion(output, y_batch)
                        val_loss += loss.item()

                val_loss /= len(val_loader)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.hyperparams["patience"]:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        training_time = time.time() - start_time

        # Обновляем метаданные
        self.metadata.update(
            {
                "training_time": training_time,
                "n_samples_trained": len(X),
                "n_features": len(self._feature_names),
                "best_epoch": best_epoch,
                "best_val_loss": float(best_val_loss) if val_loader else None,
            }
        )

        self._is_fitted = True
        logger.info(f"Обучение завершено за {training_time:.2f} сек.")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Получить предсказания модели."""
        if not self.is_fitted:
            raise ValueError("Модель не обучена.")

        self._validate_features(X)
        self.model.eval()

        X_tensor = torch.FloatTensor(X.values).to(self.device)

        with torch.no_grad():
            output = self.model(X_tensor)

            if self.task == "classification":
                if len(self._classes) == 2:
                    predictions = (torch.sigmoid(output) > 0.5).cpu().numpy().flatten().astype(int)
                else:
                    predictions = torch.argmax(output, dim=1).cpu().numpy()

                if self._classes.size > 0:
                    predictions = self._classes[predictions]
            else:
                predictions = output.cpu().numpy().flatten()

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Получить вероятности классов."""
        if not self.is_fitted:
            raise ValueError("Модель не обучена.")

        if self.task != "classification":
            raise ValueError("predict_proba доступен только для классификации")

        self._validate_features(X)
        self.model.eval()

        X_tensor = torch.FloatTensor(X.values).to(self.device)

        with torch.no_grad():
            output = self.model(X_tensor)

            if len(self._classes) == 2:
                probas = torch.sigmoid(output).cpu().numpy().flatten()
            else:
                probas = F.softmax(output, dim=1).cpu().numpy()

        return probas

    def save(self, path: Path) -> None:
        """Сохранить модель на диск."""
        if not self.is_fitted:
            raise ValueError("Модель не обучена.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Сохраняем модель PyTorch
        model_path = path / "model.pt"
        torch.save(self.model.state_dict(), model_path)

        # Сохраняем метаданные
        metadata = {
            "task": self.task,
            "device": str(self.device),
            "hyperparams": self.hyperparams,
            "metadata": self.metadata,
            "classes": self._classes.tolist() if self._classes.size > 0 else None,
            "feature_names": self._feature_names,
            "n_features": len(self._feature_names),
        }

        metadata_path = path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Модель сохранена в {path}")

    @classmethod
    def load(cls, path: Path) -> "NODEModel":
        """Загрузить модель с диска."""
        path = Path(path)

        # Загружаем метаданные
        metadata_path = path / "metadata.json"
        with open(metadata_path, "r") as f:
            saved_data = json.load(f)

        # Создаём экземпляр модели
        task = saved_data["task"]
        device = saved_data["device"]
        hyperparams = saved_data["hyperparams"]
        model_instance = cls(task=task, device=device, **hyperparams)

        # Определяем n_classes
        classes = saved_data.get("classes")
        n_classes = 1
        if task == "classification" and classes:
            n_classes = len(classes)
            if n_classes == 2:
                n_classes = 1

        # Создаём модель
        model_instance.model = NODE(
            n_features=saved_data["n_features"],
            n_classes=n_classes,
            num_layers=hyperparams["num_layers"],
            num_trees=hyperparams["num_trees"],
            depth=hyperparams["depth"],
            choice_function=hyperparams["choice_function"],
            bin_function=hyperparams["bin_function"],
        ).to(model_instance.device)

        # Загружаем веса
        model_path = path / "model.pt"
        model_instance.model.load_state_dict(torch.load(model_path, map_location=model_instance.device))

        # Восстанавливаем метаданные
        model_instance.metadata = saved_data["metadata"]
        model_instance._classes = np.array(classes) if classes else np.array([])
        model_instance._feature_names = saved_data["feature_names"]
        model_instance._is_fitted = True

        logger.info(f"Модель загружена из {path}")

        return model_instance

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        """
        Получить важности признаков.

        Note: NODE не предоставляет прямые feature importances.
        Возвращаем None.
        """
        return None

    def _validate_features(self, X: pd.DataFrame) -> None:
        """Проверить соответствие признаков."""
        if self._feature_names is None:
            return

        missing_features = set(self._feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(f"Отсутствуют признаки: {missing_features}")

    def __repr__(self) -> str:
        """Строковое представление модели."""
        fitted_status = "fitted" if self.is_fitted else "not fitted"
        num_trees = self.hyperparams.get("num_trees", "?")
        depth = self.hyperparams.get("depth", "?")
        return (
            f"NODEModel(task={self.task}, num_trees={num_trees}, depth={depth}, device={self.device}, {fitted_status})"
        )
