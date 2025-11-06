"""
RandomForest модель для классификации и регрессии.

Обёртка над sklearn RandomForest с поддержкой:
- Feature importances
- Единый интерфейс с BaseModel
- Parallel processing
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.modeling.base import BaseModel, ClassifierMixin, RegressorMixin
from src.modeling.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register(
    "random_forest",
    description="Random Forest ensemble model",
    tags=["tree-based", "ensemble", "sklearn"],
)
class RandomForestModel(BaseModel, ClassifierMixin, RegressorMixin):
    """
    RandomForest обёртка для классификации и регрессии.

    Поддерживает:
    - Бинарную и мультиклассовую классификацию
    - Регрессию
    - Feature importances
    - Parallel processing

    Args:
        task: Тип задачи ('classification' или 'regression')
        **hyperparams: Гиперпараметры RandomForest
    """

    def __init__(self, task: str = "classification", **hyperparams):
        """
        Инициализация RandomForest модели.

        Args:
            task: Тип задачи ('classification' или 'regression')
            **hyperparams: Гиперпараметры для RandomForest
        """
        super().__init__(task=task, **hyperparams)
        self.task = task
        self.model = None
        self._classes = np.array([])
        self._feature_names = []

        # Дефолтные параметры
        self._set_default_params()

        # Sanitize hyperparams for sklearn (e.g., verbose must be >=0 or bool)
        if "verbose" in self.hyperparams and isinstance(self.hyperparams["verbose"], int):
            if self.hyperparams["verbose"] < 0:
                # sklearn interprets negative verbose as invalid — set to 0
                self.hyperparams["verbose"] = 0

    def _set_default_params(self) -> None:
        """Установить дефолтные параметры если не заданы."""
        defaults = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": 0,
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
    ) -> "RandomForestModel":
        """
        Обучить RandomForest модель.

        Args:
            X: Признаки для обучения
            y: Таргет для обучения
            X_val: Не используется (для совместимости API)
            y_val: Не используется (для совместимости API)
            **kwargs: Дополнительные параметры

        Returns:
            Обученная модель (self)
        """
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        logger.info(f"Начало обучения RandomForest модели (задача: {self.task})")

        # Сохраняем имена признаков
        self._feature_names = list(X.columns)

        # Для классификации сохраняем классы
        if self.task == "classification":
            self._classes = np.unique(y)
            self.model = RandomForestClassifier(**self.hyperparams)
        else:
            self.model = RandomForestRegressor(**self.hyperparams)

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
                "n_estimators": self.model.n_estimators,
            }
        )

        self._is_fitted = True
        logger.info(f"Обучение завершено за {training_time:.2f} сек.")

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
        return self.model.predict(X)

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
        probas = self.model.predict_proba(X)

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

        # Сохраняем модель через pickle
        model_path = path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        # Сохраняем метаданные
        metadata = {
            "task": self.task,
            "hyperparams": self.hyperparams,
            "metadata": self.metadata,
            "classes": self._classes.tolist() if self._classes.size > 0 else None,
            "feature_names": self._feature_names,
        }

        metadata_path = path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Модель сохранена в {path}")

    @classmethod
    def load(cls, path: Path) -> "RandomForestModel":
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
        task = saved_data["task"]
        model_instance = cls(task=task)

        # Загружаем модель
        model_path = path / "model.pkl"
        with open(model_path, "rb") as f:
            model_instance.model = pickle.load(f)

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
            Массив важностей признаков
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

    def __repr__(self) -> str:
        """Строковое представление модели."""
        fitted_status = "fitted" if self.is_fitted else "not fitted"
        n_estimators = self.model.n_estimators if self.is_fitted else self.hyperparams.get("n_estimators", "?")
        return f"RandomForestModel(task={self.task}, n_estimators={n_estimators}, {fitted_status})"
