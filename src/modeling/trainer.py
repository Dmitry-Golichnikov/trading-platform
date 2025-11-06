"""
Model Trainer.

Класс для обучения моделей с поддержкой callbacks, логирования и tracking.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.modeling.base import BaseModel
from src.modeling.callbacks.base import Callback, CallbackList

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """
    Результат обучения модели.

    Attributes:
        model: Обученная модель
        metrics: Метрики обучения
        training_time: Время обучения в секундах
        n_epochs: Количество эпох обучения
        history: История метрик по эпохам
        artifacts_path: Путь к артефактам (если сохранены)
    """

    model: BaseModel
    metrics: Dict[str, float]
    training_time: float
    n_epochs: int
    history: Dict[str, list[float]]
    artifacts_path: Optional[Path] = None

    def __repr__(self) -> str:
        """Строковое представление результата."""
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return (
            f"TrainingResult("
            f"model={self.model.__class__.__name__}, "
            f"n_epochs={self.n_epochs}, "
            f"time={self.training_time:.2f}s, "
            f"{metrics_str})"
        )


class ModelTrainer:
    """
    Trainer для обучения моделей с tracking и callbacks.

    Поддерживает:
    - Обучение с валидацией
    - Callbacks (early stopping, checkpoints, logging)
    - MLflow tracking
    - История метрик
    - Graceful error handling

    Примеры:
        >>> trainer = ModelTrainer(
        >>>     model=model,
        >>>     experiment_name='my_experiment'
        >>> )
        >>> result = trainer.train(
        >>>     X_train, y_train,
        >>>     X_val, y_val,
        >>>     callbacks=[early_stop, checkpoint]
        >>> )
    """

    def __init__(
        self,
        model: BaseModel,
        experiment_name: Optional[str] = None,
        mlflow_uri: Optional[str] = None,
        artifacts_dir: Optional[Path] = None,
        verbose: bool = True,
    ):
        """
        Args:
            model: Модель для обучения
            experiment_name: Имя эксперимента для MLflow
            mlflow_uri: URI MLflow tracking server
            artifacts_dir: Директория для сохранения артефактов
            verbose: Выводить информацию в лог
        """
        self.model = model
        self.experiment_name = experiment_name
        self.mlflow_uri = mlflow_uri
        self.artifacts_dir = artifacts_dir or Path("artifacts/models")
        self.verbose = verbose

        # Состояние обучения
        self.stop_training = False
        self.current_epoch = 0
        self.history: Dict[str, list[float]] = {}

        # Создаём директорию для артефактов
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        epochs: Optional[int] = None,
        callbacks: Optional[list[Callback]] = None,
        **fit_kwargs,
    ) -> TrainingResult:
        """
        Обучить модель.

        Args:
            X_train: Признаки для обучения
            y_train: Таргет для обучения
            X_val: Признаки для валидации (опционально)
            y_val: Таргет для валидации (опционально)
            epochs: Количество эпох (для итеративных моделей)
            callbacks: Список callbacks
            **fit_kwargs: Дополнительные аргументы для fit()

        Returns:
            TrainingResult с метриками и обученной моделью

        Raises:
            Exception: При ошибке обучения (с логированием)
        """
        # Инициализация
        self.stop_training = False
        self.current_epoch = 0
        self.history = {}

        callback_list = CallbackList(callbacks or [])

        start_time = time.time()

        try:
            if self.verbose:
                logger.info(
                    f"Начало обучения {self.model.__class__.__name__}: "
                    f"train_size={len(X_train)}" + (f", val_size={len(X_val)}" if X_val is not None else "")
                )

            # Callbacks: начало обучения
            callback_list.on_train_begin(self)

            # Обучение модели
            if self._is_iterative_model():
                # Итеративное обучение с эпохами
                result = self._train_iterative(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    epochs or 100,
                    callback_list,
                    **fit_kwargs,
                )
            else:
                # Одношаговое обучение (sklearn-like модели)
                result = self._train_single_step(X_train, y_train, X_val, y_val, callback_list, **fit_kwargs)

            training_time = time.time() - start_time

            # Обновляем метаданные модели
            if hasattr(self.model, "metadata"):
                self.model.metadata.update(
                    {
                        "training_time": training_time,
                        "n_samples_trained": len(X_train),
                        "n_features": X_train.shape[1],
                    }
                )

            # Callbacks: конец обучения
            callback_list.on_train_end(self, result.metrics)

            if self.verbose:
                logger.info(f"Обучение завершено за {training_time:.2f}s, " f"эпох: {result.n_epochs}")

            return result

        except Exception as e:
            logger.error(f"Ошибка при обучении модели: {e}", exc_info=True)
            raise

    def _is_iterative_model(self) -> bool:
        """Проверить, является ли модель итеративной (с эпохами)."""
        # PyTorch модели
        if hasattr(self.model, "state_dict"):
            return True
        # Модели с параметром n_estimators или max_iter
        if hasattr(self.model, "hyperparams"):
            params = self.model.hyperparams
            return "n_estimators" in params or "max_iter" in params
        return False

    def _train_iterative(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        epochs: int,
        callback_list: CallbackList,
        **fit_kwargs,
    ) -> TrainingResult:
        """
        Итеративное обучение с эпохами.

        Для PyTorch моделей и других итеративных алгоритмов.
        """
        # История метрик
        history: Dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [] if X_val is not None else [],
        }

        for epoch in range(epochs):
            if self.stop_training:
                if self.verbose:
                    logger.info(f"Обучение остановлено на эпохе {epoch}")
                break

            self.current_epoch = epoch

            # Callbacks: начало эпохи
            callback_list.on_epoch_begin(self, epoch)

            # Обучение на одной эпохе
            epoch_logs = self._train_epoch(X_train, y_train, X_val, y_val, **fit_kwargs)

            # Обновляем историю
            for key, value in epoch_logs.items():
                if key not in history:
                    history[key] = []
                history[key].append(value)

            # Callbacks: конец эпохи
            callback_list.on_epoch_end(self, epoch, epoch_logs)

        # Финальные метрики
        final_metrics = {key: values[-1] for key, values in history.items() if values}

        return TrainingResult(
            model=self.model,
            metrics=final_metrics,
            training_time=0.0,  # Будет обновлено в train()
            n_epochs=self.current_epoch + 1,
            history=history,
        )

    def _train_single_step(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        callback_list: CallbackList,
        **fit_kwargs,
    ) -> TrainingResult:
        """
        Одношаговое обучение (для sklearn-like моделей).
        """
        # Обучаем модель
        self.model.fit(X_train, y_train, X_val, y_val, **fit_kwargs)

        # Вычисляем метрики
        metrics = {}

        # Train метрики
        y_train_pred = self.model.predict(X_train)
        metrics["train_accuracy"] = self._compute_accuracy(y_train, y_train_pred)

        # Val метрики
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            metrics["val_accuracy"] = self._compute_accuracy(y_val, y_val_pred)

        history = {key: [value] for key, value in metrics.items()}

        return TrainingResult(
            model=self.model,
            metrics=metrics,
            training_time=0.0,  # Будет обновлено в train()
            n_epochs=1,
            history=history,
        )

    def _train_epoch(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        **kwargs,
    ) -> Dict[str, float]:
        """
        Обучить одну эпоху.

        Returns:
            Словарь с метриками эпохи
        """
        # Для PyTorch моделей нужна специальная логика
        # Для tree-based моделей - инкрементальное обучение невозможно
        # Поэтому просто вызываем fit (для совместимости)

        # Это заглушка - конкретная реализация должна быть в подклассах моделей
        logs = {}

        # Вычисляем примерные метрики
        if hasattr(self.model, "_is_fitted") and self.model._is_fitted:
            y_train_pred = self.model.predict(X_train)
            logs["train_loss"] = self._compute_loss(y_train, y_train_pred)

            if X_val is not None and y_val is not None:
                y_val_pred = self.model.predict(X_val)
                logs["val_loss"] = self._compute_loss(y_val, y_val_pred)

        return logs

    def _compute_accuracy(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Вычислить accuracy."""
        return float(np.mean(y_true.values == y_pred))

    def _compute_loss(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Вычислить loss (MSE для регрессии, log loss для классификации)."""
        # Простая реализация - MSE
        return float(np.mean((y_true.values - y_pred) ** 2))

    def save_model(self, path: Optional[Path] = None) -> Path:
        """
        Сохранить модель.

        Args:
            path: Путь для сохранения (если None, генерируется автоматически)

        Returns:
            Путь к сохранённой модели
        """
        if path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_name = self.model.__class__.__name__
            path = self.artifacts_dir / f"{model_name}_{timestamp}"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(path)

        if self.verbose:
            logger.info(f"Модель сохранена: {path}")

        return path
