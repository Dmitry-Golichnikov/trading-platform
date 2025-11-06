"""
TabNet модель для классификации и регрессии.

Обёртка над pytorch-tabnet с поддержкой:
- Attention-based architecture
- Feature selection через masks
- GPU обучение
- Early stopping
- Feature importances
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from src.modeling.base import BaseModel, ClassifierMixin, RegressorMixin
from src.modeling.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register(
    "tabnet",
    description="TabNet attention-based tabular neural network",
    tags=["neural", "tabular", "attention", "gpu-support"],
)
class TabNetModel(BaseModel, ClassifierMixin, RegressorMixin):
    """
    TabNet обёртка для классификации и регрессии.

    Использует pytorch-tabnet библиотеку.

    Поддерживает:
    - Бинарную и мультиклассовую классификацию
    - Регрессию
    - Attention-based feature selection
    - GPU обучение
    - Feature importances через attention masks

    Args:
        task: Тип задачи ('classification' или 'regression')
        device: Устройство ('cpu' или 'cuda')
        **hyperparams: Гиперпараметры TabNet
    """

    def __init__(self, task: str = "classification", device: str = "cpu", **hyperparams):
        """
        Инициализация TabNet модели.

        Args:
            task: Тип задачи ('classification' или 'regression')
            device: Устройство для обучения
            **hyperparams: Гиперпараметры для TabNet
        """
        # Не передаём device в hyperparams
        super().__init__(task=task, **hyperparams)
        self.task = task
        self.device = device
        self.model = None
        self._classes = np.array([])
        self._feature_names = []

        # Дефолтные параметры
        self._set_default_params()

    def _set_default_params(self) -> None:
        """Установить дефолтные параметры если не заданы."""
        defaults = {
            "n_d": 8,  # dimension of decision prediction layer
            "n_a": 8,  # dimension of attention embedding
            "n_steps": 3,  # number of steps in the architecture
            "gamma": 1.3,  # coefficient for feature reusage
            "n_independent": 2,  # number of independent GLU layers
            "n_shared": 2,  # number of shared GLU layers
            "lambda_sparse": 1e-3,  # sparsity regularization
            "optimizer_fn": torch.optim.Adam,
            "optimizer_params": {"lr": 2e-2},
            "scheduler_fn": None,
            "scheduler_params": {},
            "mask_type": "sparsemax",  # sparsemax or entmax
            "seed": 42,
            "verbose": 1,
        }

        # Обновляем только если параметр не был задан
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
    ) -> "TabNetModel":
        """
        Обучить TabNet модель.

        Args:
            X: Признаки для обучения
            y: Таргет для обучения
            X_val: Признаки для валидации
            y_val: Таргет для валидации
            **kwargs: Дополнительные параметры (max_epochs, batch_size и т.д.)

        Returns:
            Обученная модель (self)
        """
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
        except ImportError:
            raise ImportError("pytorch-tabnet не установлен. Установите: pip install pytorch-tabnet")

        logger.info(f"Начало обучения TabNet модели (задача: {self.task})")

        # Сохраняем имена признаков
        self._feature_names = list(X.columns)

        # Для классификации сохраняем классы
        if self.task == "classification":
            self._classes = np.unique(y)
            self.model = TabNetClassifier(**self.hyperparams, device_name=self.device)
        else:
            self.model = TabNetRegressor(**self.hyperparams, device_name=self.device)

        # Параметры обучения
        max_epochs = kwargs.get("max_epochs", 100)
        batch_size = kwargs.get("batch_size", 1024)
        virtual_batch_size = kwargs.get("virtual_batch_size", 128)
        patience = kwargs.get("patience", 15)

        # Validation set
        eval_set = None
        eval_metric = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val.values, y_val.values)]
            if self.task == "classification":
                eval_metric = ["auc"]
            else:
                eval_metric = ["rmse"]

        # Обучение
        import time

        start_time = time.time()

        self.model.fit(
            X_train=X.values,
            y_train=y.values,
            eval_set=eval_set,
            eval_metric=eval_metric,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
        )

        training_time = time.time() - start_time

        # Обновляем метаданные
        self.metadata.update(
            {
                "training_time": training_time,
                "n_samples_trained": len(X),
                "n_features": len(self._feature_names),
                "best_epoch": self.model.best_epoch,
                "best_cost": (float(self.model.best_cost) if hasattr(self.model, "best_cost") else None),
            }
        )

        self._is_fitted = True
        logger.info(f"Обучение завершено за {training_time:.2f} сек. " f"Best epoch: {self.model.best_epoch}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Получить предсказания модели.

        Args:
            X: Признаки для предсказания

        Returns:
            Предсказания
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")

        self._validate_features(X)
        predictions = self.model.predict(X.values)

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Получить вероятности классов.

        Args:
            X: Признаки для предсказания

        Returns:
            Вероятности классов
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")

        if self.task != "classification":
            raise ValueError("predict_proba доступен только для классификации")

        self._validate_features(X)
        probas = self.model.predict_proba(X.values)

        # Для бинарной классификации возвращаем только вероятность положительного класса
        if probas.shape[1] == 2:
            return probas[:, 1]

        return probas

    def save(self, path: Path) -> None:
        """
        Сохранить модель на диск.

        Args:
            path: Путь к директории для сохранения
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Нечего сохранять.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Сохраняем модель TabNet
        model_path = path / "model.zip"
        self.model.save_model(str(model_path))

        # Сохраняем метаданные
        metadata = {
            "task": self.task,
            "device": self.device,
            "hyperparams": self._serialize_hyperparams(self.hyperparams),
            "metadata": self.metadata,
            "classes": self._classes.tolist() if self._classes.size > 0 else None,
            "feature_names": self._feature_names,
        }

        metadata_path = path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Модель сохранена в {path}")

    @classmethod
    def load(cls, path: Path) -> "TabNetModel":
        """
        Загрузить модель с диска.

        Args:
            path: Путь к директории с моделью

        Returns:
            Загруженная модель
        """
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
        except ImportError:
            raise ImportError("pytorch-tabnet не установлен. Установите: pip install pytorch-tabnet")

        path = Path(path)

        # Загружаем метаданные
        metadata_path = path / "metadata.json"
        with open(metadata_path, "r") as f:
            saved_data = json.load(f)

        # Создаём экземпляр модели
        task = saved_data["task"]
        device = saved_data["device"]
        model_instance = cls(task=task, device=device)

        # Загружаем модель TabNet
        model_path = path / "model.zip"
        if task == "classification":
            model_instance.model = TabNetClassifier()
        else:
            model_instance.model = TabNetRegressor()

        model_instance.model.load_model(str(model_path))

        # Восстанавливаем метаданные
        model_instance.metadata = saved_data["metadata"]
        model_instance.hyperparams = saved_data["hyperparams"]
        model_instance._classes = np.array(saved_data["classes"]) if saved_data["classes"] else np.array([])
        model_instance._feature_names = saved_data["feature_names"]
        model_instance._is_fitted = True

        logger.info(f"Модель загружена из {path}")

        return model_instance

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        """
        Получить важности признаков.

        Returns:
            Массив важностей признаков (на основе attention masks)
        """
        if not self.is_fitted:
            return None

        return self.model.feature_importances_

    def get_feature_importance_df(self) -> pd.DataFrame:
        """
        Получить DataFrame с важностями признаков.

        Returns:
            DataFrame с колонками ['feature', 'importance']
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена.")

        importances = self.feature_importances_

        df = pd.DataFrame(
            {
                "feature": self._feature_names,
                "importance": importances,
            }
        )

        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def _validate_features(self, X: pd.DataFrame) -> None:
        """Проверить соответствие признаков."""
        if self._feature_names is None:
            return

        missing_features = set(self._feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(f"Отсутствуют признаки: {missing_features}")

    def _serialize_hyperparams(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Сериализация гиперпараметров для JSON."""
        serialized = {}
        for key, value in params.items():
            if key in ["optimizer_fn", "scheduler_fn"]:
                # Сохраняем только имена классов оптимизаторов
                serialized[key] = value.__name__ if value is not None else None
            elif key in ["optimizer_params", "scheduler_params"]:
                serialized[key] = value
            else:
                serialized[key] = value
        return serialized

    def __repr__(self) -> str:
        """Строковое представление модели."""
        fitted_status = "fitted" if self.is_fitted else "not fitted"
        n_steps = self.hyperparams.get("n_steps", "?")
        return f"TabNetModel(task={self.task}, n_steps={n_steps}, device={self.device}, {fitted_status})"
