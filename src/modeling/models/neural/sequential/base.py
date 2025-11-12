"""
Базовый класс для sequence моделей.

Предоставляет общую функциональность для всех моделей временных рядов,
включая sequence preparation, training loop, callbacks, и device management.
"""

import json
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    OneCycleLR,
    ReduceLROnPlateau,
)

from src.modeling.base import BaseModel, ClassifierMixin
from src.modeling.models.neural.sequential.datasets import create_sequence_dataloader


class BaseSequentialModel(BaseModel, ClassifierMixin, nn.Module):
    """
    Базовый класс для sequential моделей (LSTM, GRU, TCN и т.д.).

    Наследует от BaseModel (наш интерфейс) и nn.Module (PyTorch).
    Предоставляет общую функциональность:
    - Sequence preparation
    - Training loop с callbacks
    - Device management (CPU/GPU)
    - Early stopping
    - Model checkpointing
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        seq_length: int,
        output_size: int = 1,
        dropout: float = 0.2,
        task: str = "classification",
        device: str = "auto",
        # Training parameters
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        optimizer: str = "adam",
        scheduler: Optional[str] = "onecycle",
        early_stopping: int = 10,
        # Sequence parameters
        stride: int = 1,
        predict_horizon: int = 0,
        # Advanced
        mixed_precision: bool = False,
        gradient_clip: Optional[float] = 1.0,
        **kwargs,
    ):
        """
        Инициализация базовой sequential модели.

        Args:
            input_size: Размерность входных признаков
            hidden_size: Размер скрытого слоя
            num_layers: Количество слоёв RNN
            seq_length: Длина последовательности
            output_size: Размер выхода (1 для бинарной классификации/регрессии)
            dropout: Dropout rate
            task: 'classification' или 'regression'
            device: 'auto', 'cuda', 'cpu'
            epochs: Количество эпох обучения
            batch_size: Размер батча
            learning_rate: Learning rate
            optimizer: 'adam', 'adamw', 'sgd'
            scheduler: LR scheduler ('onecycle', 'cosine', 'plateau', None)
            early_stopping: Patience для early stopping
            stride: Stride для sliding window
            predict_horizon: Горизонт предсказания
            mixed_precision: Использовать ли mixed precision (AMP)
            gradient_clip: Градиентный клиппинг
        """
        # Инициализация BaseModel
        BaseModel.__init__(
            self,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            seq_length=seq_length,
            output_size=output_size,
            dropout=dropout,
            task=task,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=early_stopping,
            stride=stride,
            predict_horizon=predict_horizon,
            mixed_precision=mixed_precision,
            gradient_clip=gradient_clip,
            **kwargs,
        )

        # Инициализация nn.Module
        nn.Module.__init__(self)

        # Параметры архитектуры
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.output_size = output_size
        self.dropout_rate = dropout
        self.task = task

        # Параметры обучения
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.early_stopping_patience = early_stopping

        # Параметры последовательностей
        self.stride = stride
        self.predict_horizon = predict_horizon

        # Продвинутые параметры
        self.mixed_precision = mixed_precision
        self.gradient_clip = gradient_clip

        # Device management
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # История обучения
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

        # Оптимизатор и scheduler (инициализируются в fit)
        self.optimizer_obj: Optional[torch.optim.Optimizer] = None
        self.scheduler_obj: Optional[Any] = None

        # Scaler для mixed precision
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        if self.mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass модели.

        Args:
            x: Input tensor, shape (batch_size, seq_length, input_size)

        Returns:
            Output tensor, shape (batch_size, output_size)
        """
        pass

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Создать оптимизатор."""
        if self.optimizer_name.lower() == "adam":
            return Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name.lower() == "adamw":
            return AdamW(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name.lower() == "sgd":
            return SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Неизвестный оптимизатор: {self.optimizer_name}")

    def _create_scheduler(self, optimizer: torch.optim.Optimizer, steps_per_epoch: int) -> Optional[Any]:
        """Создать learning rate scheduler."""
        if self.scheduler_name is None:
            return None

        if self.scheduler_name.lower() == "onecycle":
            return OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                epochs=self.epochs,
                steps_per_epoch=steps_per_epoch,
            )
        elif self.scheduler_name.lower() == "cosine":
            return CosineAnnealingLR(
                optimizer,
                T_max=self.epochs,
            )
        elif self.scheduler_name.lower() == "plateau":
            return ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
            )
        else:
            raise ValueError(f"Неизвестный scheduler: {self.scheduler_name}")

    def _create_loss_function(self) -> nn.Module:
        """Создать loss function в зависимости от задачи."""
        if self.task == "classification":
            if self.output_size == 1:
                return nn.BCEWithLogitsLoss()
            else:
                return nn.CrossEntropyLoss()
        elif self.task == "regression":
            return nn.MSELoss()
        else:
            raise ValueError(f"Неизвестная задача: {self.task}")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> "BaseSequentialModel":
        """
        Обучить модель.

        Args:
            X: Признаки для обучения
            y: Таргет для обучения
            X_val: Признаки для валидации (опционально)
            y_val: Таргет для валидации (опционально)
            **kwargs: Дополнительные параметры

        Returns:
            Обученная модель (self)
        """
        start_time = time.time()

        # Сохраняем имена признаков
        self._feature_names = list(X.columns)

        # Сохраняем классы для классификации
        if self.task == "classification":
            self._classes = np.unique(y)

        # Создаём DataLoaders
        train_loader = create_sequence_dataloader(
            X=X,
            y=y,
            seq_length=self.seq_length,
            batch_size=self.batch_size,
            stride=self.stride,
            predict_horizon=self.predict_horizon,
            shuffle=True,
            num_workers=0,  # Windows compatibility
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = create_sequence_dataloader(
                X=X_val,
                y=y_val,
                seq_length=self.seq_length,
                batch_size=self.batch_size,
                stride=self.stride,
                predict_horizon=self.predict_horizon,
                shuffle=False,
                num_workers=0,
            )

        # Переносим модель на device
        self.to(self.device)

        # Создаём оптимизатор и scheduler
        self.optimizer_obj = self._create_optimizer()
        self.scheduler_obj = self._create_scheduler(
            self.optimizer_obj,
            steps_per_epoch=len(train_loader),
        )

        # Loss function
        criterion = self._create_loss_function()

        # Training loop
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            # Train
            train_loss = self._train_epoch(train_loader, criterion)
            self.history["train_loss"].append(train_loss)

            # Validate
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader, criterion)
                self.history["val_loss"].append(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self.early_stopping_patience:
                    print(f"Early stopping на эпохе {epoch + 1}")
                    break

            # Learning rate
            current_lr = self.optimizer_obj.param_groups[0]["lr"]
            self.history["learning_rate"].append(current_lr)

            # Update scheduler
            if self.scheduler_obj is not None:
                if isinstance(self.scheduler_obj, ReduceLROnPlateau):
                    if val_loader is not None:
                        self.scheduler_obj.step(val_loss)
                else:
                    self.scheduler_obj.step()

            # Logging
            if (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss:.4f}"
                if val_loader is not None:
                    msg += f", Val Loss: {val_loss:.4f}"
                msg += f", LR: {current_lr:.6f}"
                print(msg)

        # Обновляем метаданные
        self._is_fitted = True
        self.metadata["training_time"] = time.time() - start_time
        self.metadata["n_samples_trained"] = len(X)
        self.metadata["n_features"] = len(self._feature_names)
        self.metadata["epochs_trained"] = len(self.history["train_loss"])

        return self

    def _train_epoch(self, dataloader, criterion) -> float:
        """Один эпох обучения."""
        self.train()
        total_loss = 0.0

        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            if self.optimizer_obj is not None:
                self.optimizer_obj.zero_grad()

            # Forward pass с mixed precision если включено
            if self.mixed_precision and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self(batch_X)
                    loss = criterion(outputs, batch_y.squeeze())

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.gradient_clip is not None:
                    self.scaler.unscale_(self.optimizer_obj)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip)

                self.scaler.step(self.optimizer_obj)
                self.scaler.update()
            else:
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y.squeeze())

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip)

                if self.optimizer_obj is not None:
                    self.optimizer_obj.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def _validate_epoch(self, dataloader, criterion) -> float:
        """Валидация."""
        self.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self(batch_X)
                loss = criterion(outputs, batch_y.squeeze())

                total_loss += loss.item()

        return total_loss / len(dataloader)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказания модели.

        Args:
            X: Признаки для предсказания

        Returns:
            Предсказания (метки классов для классификации, значения для регрессии)
        """
        self.eval()

        # Создаём DataLoader
        # Для predict используем dummy таргеты
        dummy_y = pd.Series(np.zeros(len(X)))
        dataloader = create_sequence_dataloader(
            X=X,
            y=dummy_y,
            seq_length=self.seq_length,
            batch_size=self.batch_size,
            stride=1,  # Без пропусков для предсказаний
            predict_horizon=self.predict_horizon,
            shuffle=False,
            num_workers=0,
        )

        predictions = []

        with torch.no_grad():
            for batch_X, _ in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self(batch_X)

                if self.task == "classification":
                    if self.output_size == 1:
                        # Бинарная классификация
                        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int)
                    else:
                        # Мультиклассовая классификация
                        preds = torch.argmax(outputs, dim=1).cpu().numpy()
                else:
                    # Регрессия
                    preds = outputs.cpu().numpy()

                predictions.append(preds)

        return np.concatenate(predictions).flatten()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Вероятности классов (только для классификации).

        Args:
            X: Признаки для предсказания

        Returns:
            Вероятности классов, shape (n_samples, n_classes)
        """
        if self.task != "classification":
            raise NotImplementedError("predict_proba доступен только для классификации")

        self.eval()

        # Создаём DataLoader
        dummy_y = pd.Series(np.zeros(len(X)))
        dataloader = create_sequence_dataloader(
            X=X,
            y=dummy_y,
            seq_length=self.seq_length,
            batch_size=self.batch_size,
            stride=1,
            predict_horizon=self.predict_horizon,
            shuffle=False,
            num_workers=0,
        )

        probabilities = []

        with torch.no_grad():
            for batch_X, _ in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self(batch_X)

                if self.output_size == 1:
                    # Бинарная классификация
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    # Формат [prob_class_0, prob_class_1]
                    probs = np.column_stack([1 - probs, probs])
                else:
                    # Мультиклассовая классификация
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()

                probabilities.append(probs)

        return np.concatenate(probabilities)

    def save(self, path: Path) -> None:
        """
        Сохранить модель.

        Args:
            path: Путь для сохранения (директория)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Сохраняем веса модели
        torch.save(self.state_dict(), path / "model_weights.pt")

        # Сохраняем гиперпараметры и метаданные
        config = {
            "hyperparams": self.hyperparams,
            "metadata": self.metadata,
            "history": self.history,
            "feature_names": self._feature_names,
            "classes": self._classes.tolist() if hasattr(self, "_classes") and self._classes is not None else None,
        }

        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "BaseSequentialModel":
        """
        Загрузить модель.

        Args:
            path: Путь к сохранённой модели

        Returns:
            Загруженная модель
        """
        path = Path(path)

        # Загружаем конфигурацию
        with open(path / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        # Создаём модель с гиперпараметрами
        model = cls(**config["hyperparams"])

        # Загружаем веса
        model.load_state_dict(torch.load(path / "model_weights.pt", map_location=model.device))

        # Восстанавливаем метаданные
        model.metadata = config["metadata"]
        model.history = config["history"]
        model._feature_names = config["feature_names"]
        if config["classes"] is not None:
            model._classes = np.array(config["classes"])
        model._is_fitted = True

        model.eval()

        return model

    def get_training_history(self) -> pd.DataFrame:
        """Получить историю обучения в виде DataFrame."""
        return pd.DataFrame(self.history)
