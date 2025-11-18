"""
ElasticNet модель для регрессии.

Обёртка над sklearn ElasticNet с поддержкой:
- L1 и L2 регуляризация
- Feature importances (коэффициенты)
- Единый интерфейс с BaseModel
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.modeling.base import BaseModel, RegressorMixin
from src.modeling.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register(
    "elasticnet",
    description="ElasticNet regression model with L1 and L2 regularization",
    tags=["linear", "regression", "sklearn"],
)
class ElasticNetModel(BaseModel, RegressorMixin):
    """
    ElasticNet обёртка для регрессии.

    Поддерживает:
    - Регрессию с L1 и L2 регуляризацией
    - Feature importances (коэффициенты)
    - Cross-validation для alpha

    Args:
        **hyperparams: Гиперпараметры ElasticNet
    """

    def __init__(self, task: str = "regression", **hyperparams):
        """
        Инициализация ElasticNet модели.

        Args:
            task: Тип задачи (по умолчанию 'regression')
            **hyperparams: Гиперпараметры для ElasticNet
        """
        super().__init__(task=task, **hyperparams)
        self.task = task
        self.model = None
        self._feature_names = []

        # Дефолтные параметры
        self._set_default_params()

    def _set_default_params(self) -> None:
        """Установить дефолтные параметры если не заданы."""
        defaults = {
            "alpha": 1.0,  # сила регуляризации
            "l1_ratio": 0.5,  # соотношение L1/L2 (0=L2, 1=L1)
            "max_iter": 1000,
            "tol": 1e-4,
            "random_state": 42,
            "selection": "cyclic",  # cyclic или random
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
    ) -> "ElasticNetModel":
        """
        Обучить ElasticNet модель.

        Args:
            X: Признаки для обучения
            y: Таргет для обучения
            X_val: Не используется (для совместимости API)
            y_val: Не используется (для совместимости API)
            **kwargs: Дополнительные параметры

        Returns:
            Обученная модель (self)
        """
        from sklearn.linear_model import ElasticNet

        logger.info("Начало обучения ElasticNet модели")

        # Сохраняем имена признаков
        self._feature_names = list(X.columns)

        # Создаём модель
        self.model = ElasticNet(**self.hyperparams)

        # Обучение
        import time

        start_time = time.time()

        self.model.fit(X, y)

        training_time = time.time() - start_time

        # Обновляем метаданные
        self.metadata.update(
            {
                "training_time": training_time,
                "n_samples_trained": len(X),
                "n_features": len(self._feature_names),
                "n_iter": (self.model.n_iter_ if hasattr(self.model, "n_iter_") else None),
                "n_features_nonzero": np.count_nonzero(self.model.coef_),
            }
        )

        self._is_fitted = True
        logger.info(
            f"Обучение завершено за {training_time:.2f} сек. "
            f"Non-zero features: {self.metadata['n_features_nonzero']}/{len(self._feature_names)}"
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Получить предсказания модели.

        Args:
            X: Признаки для предсказания

        Returns:
            Предсказанные значения
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")

        self._validate_features(X)
        return self.model.predict(X)

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

        # Сохраняем модель через pickle
        model_path = path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        # Сохраняем метаданные
        metadata = {
            "task": self.task,
            "hyperparams": self.hyperparams,
            "metadata": self.metadata,
            "feature_names": self._feature_names,
        }

        metadata_path = path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Модель сохранена в {path}")

    @classmethod
    def load(cls, path: Path) -> "ElasticNetModel":
        """
        Загрузить модель с диска.

        Args:
            path: Путь к директории с моделью

        Returns:
            Загруженная модель
        """
        path = Path(path)

        # Загружаем метаданные
        metadata_path = path / "metadata.json"
        with open(metadata_path, "r") as f:
            saved_data = json.load(f)

        # Создаём экземпляр модели
        hyperparams = saved_data["hyperparams"]
        model_instance = cls(**hyperparams)

        # Загружаем модель
        model_path = path / "model.pkl"
        with open(model_path, "rb") as f:
            model_instance.model = pickle.load(f)

        # Восстанавливаем метаданные
        model_instance.metadata = saved_data["metadata"]
        model_instance.hyperparams = saved_data["hyperparams"]
        model_instance._feature_names = saved_data["feature_names"]
        model_instance._is_fitted = True

        logger.info(f"Модель загружена из {path}")

        return model_instance

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        """
        Получить важности признаков (коэффициенты).

        Returns:
            Массив коэффициентов (абсолютные значения)
        """
        if not self.is_fitted:
            return None

        return np.abs(self.model.coef_)

    def get_feature_importance_df(self) -> pd.DataFrame:
        """
        Получить DataFrame с важностями признаков.

        Returns:
            DataFrame с колонками ['feature', 'importance', 'coefficient']
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена.")

        df = pd.DataFrame(
            {
                "feature": self._feature_names,
                "coefficient": self.model.coef_,
                "importance": np.abs(self.model.coef_),
                "is_zero": self.model.coef_ == 0,
            }
        )

        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def get_nonzero_features(self) -> list[str]:
        """
        Получить список признаков с ненулевыми коэффициентами.

        Returns:
            Список имён признаков
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена.")

        nonzero_idx = np.nonzero(self.model.coef_)[0]
        return [self._feature_names[i] for i in nonzero_idx]

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
        alpha = self.hyperparams.get("alpha", "?")
        l1_ratio = self.hyperparams.get("l1_ratio", "?")
        return f"ElasticNetModel(alpha={alpha}, l1_ratio={l1_ratio}, {fitted_status})"
