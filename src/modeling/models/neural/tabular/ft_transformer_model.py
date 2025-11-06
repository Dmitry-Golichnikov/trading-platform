"""
FT-Transformer модель для классификации и регрессии.

Упрощённая реализация Feature Tokenizer + Transformer архитектуры.

Основана на статье "Revisiting Deep Learning Models for Tabular Data"
https://arxiv.org/abs/2106.11959
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


class FTTransformer(nn.Module):
    """
    FT-Transformer архитектура.

    Feature Tokenizer + Transformer encoder для табличных данных.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        d_token: int = 192,
        n_blocks: int = 3,
        attention_heads: int = 8,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
    ):
        """
        Инициализация FT-Transformer.

        Args:
            n_features: Количество признаков
            n_classes: Количество классов (или 1 для регрессии)
            d_token: Размерность token embedding
            n_blocks: Количество Transformer блоков
            attention_heads: Количество attention heads
            attention_dropout: Dropout в attention
            ffn_dropout: Dropout в FFN
            residual_dropout: Dropout в residual connections
        """
        super().__init__()

        # Feature tokenizer - преобразуем каждый признак в token
        self.feature_tokenizer = nn.Linear(1, d_token)

        # CLS token для финального предсказания
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=attention_heads,
            dim_feedforward=d_token * 4,
            dropout=ffn_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)

        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_token, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [batch_size, n_features]

        Returns:
            Predictions [batch_size, n_classes]
        """
        batch_size = x.shape[0]

        # Tokenize features: [batch_size, n_features] -> [batch_size, n_features, d_token]
        x = x.unsqueeze(-1)  # [batch_size, n_features, 1]
        tokens = self.feature_tokenizer(x)  # [batch_size, n_features, d_token]

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, d_token]
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # [batch_size, n_features+1, d_token]

        # Transformer
        encoded = self.transformer(tokens)  # [batch_size, n_features+1, d_token]

        # Take CLS token for prediction
        cls_output = encoded[:, 0, :]  # [batch_size, d_token]

        # Output head
        output = self.head(cls_output)  # [batch_size, n_classes]

        return output


@ModelRegistry.register(
    "ft_transformer",
    description="Feature Tokenizer + Transformer model for tabular data",
    tags=["neural", "tabular", "transformer", "gpu-support"],
)
class FTTransformerModel(BaseModel, ClassifierMixin, RegressorMixin):
    """
    FT-Transformer обёртка для классификации и регрессии.

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
        """Инициализация FT-Transformer модели."""
        # Не передаём device в BaseModel hyperparams
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
            "d_token": 192,
            "n_blocks": 3,
            "attention_heads": 8,
            "attention_dropout": 0.2,
            "ffn_dropout": 0.1,
            "residual_dropout": 0.0,
            "learning_rate": 1e-4,
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
    ) -> "FTTransformerModel":
        """Обучить FT-Transformer модель."""
        logger.info(f"Начало обучения FT-Transformer модели (задача: {self.task})")

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
        self.model = FTTransformer(
            n_features=len(self._feature_names),
            n_classes=n_classes,
            d_token=self.hyperparams["d_token"],
            n_blocks=self.hyperparams["n_blocks"],
            attention_heads=self.hyperparams["attention_heads"],
            attention_dropout=self.hyperparams["attention_dropout"],
            ffn_dropout=self.hyperparams["ffn_dropout"],
            residual_dropout=self.hyperparams["residual_dropout"],
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
    def load(cls, path: Path) -> "FTTransformerModel":
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
        model_instance.model = FTTransformer(
            n_features=saved_data["n_features"],
            n_classes=n_classes,
            d_token=hyperparams["d_token"],
            n_blocks=hyperparams["n_blocks"],
            attention_heads=hyperparams["attention_heads"],
            attention_dropout=hyperparams["attention_dropout"],
            ffn_dropout=hyperparams["ffn_dropout"],
            residual_dropout=hyperparams["residual_dropout"],
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

        Note: FT-Transformer не предоставляет прямые feature importances.
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
        n_blocks = self.hyperparams.get("n_blocks", "?")
        return f"FTTransformerModel(task={self.task}, n_blocks={n_blocks}, device={self.device}, {fitted_status})"
