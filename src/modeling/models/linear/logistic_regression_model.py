"""
LogisticRegression модель для классификации.

Обёртка над sklearn LogisticRegression с поддержкой:
- Различные penalty (l1, l2, elasticnet)
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

from src.modeling.base import BaseModel, ClassifierMixin
from src.modeling.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register(
    "logistic_regression",
    description="Logistic Regression model",
    tags=["linear", "classification", "sklearn"],
)
class LogisticRegressionModel(BaseModel, ClassifierMixin):
    """
    LogisticRegression обёртка для классификации.

    Поддерживает:
    - Бинарную и мультиклассовую классификацию
    - Различные penalty (l1, l2, elasticnet, none)
    - Feature importances (коэффициенты)
    - Solver options

    Args:
        **hyperparams: Гиперпараметры LogisticRegression
    """

    def __init__(self, task: str = "classification", **hyperparams):
        """
        Инициализация LogisticRegression модели.

        Args:
            task: Тип задачи (по умолчанию 'classification')
            **hyperparams: Гиперпараметры для LogisticRegression
        """
        super().__init__(task=task, **hyperparams)
        self.task = task
        self.model = None
        self._classes = np.array([])
        self._feature_names = []

        # Дефолтные параметры
        self._set_default_params()

    def _set_default_params(self) -> None:
        """Установить дефолтные параметры если не заданы."""
        defaults = {
            "penalty": "l2",  # l1, l2, elasticnet, none
            "C": 1.0,  # обратная величина регуляризации
            "solver": "lbfgs",  # lbfgs, liblinear, newton-cg, sag, saga
            "max_iter": 1000,
            "tol": 1e-4,
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
    ) -> "LogisticRegressionModel":
        """
        Обучить LogisticRegression модель.

        Args:
            X: Признаки для обучения
            y: Таргет для обучения
            X_val: Не используется (для совместимости API)
            y_val: Не используется (для совместимости API)
            **kwargs: Дополнительные параметры

        Returns:
            Обученная модель (self)
        """
        from sklearn.linear_model import LogisticRegression

        logger.info("Начало обучения LogisticRegression модели")

        # Сохраняем имена признаков
        self._feature_names = list(X.columns)

        # Сохраняем классы
        self._classes = np.unique(y)

        # Создаём модель
        self.model = LogisticRegression(**self.hyperparams)

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
                "n_classes": len(self._classes),
                "n_iter": (self.model.n_iter_[0] if hasattr(self.model, "n_iter_") else None),
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
            Предсказания (метки классов)
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

        self._validate_features(X)
        probas = self.model.predict_proba(X)

        # Для бинарной классификации возвращаем только вероятность положительного класса
        if probas.shape[1] == 2:
            return probas[:, 1]

        return probas

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """
        Вычислить decision function.

        Args:
            X: Признаки

        Returns:
            Decision scores
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")

        self._validate_features(X)
        return self.model.decision_function(X)

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
    def load(cls, path: Path) -> "LogisticRegressionModel":
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
        model_instance._classes = np.array(saved_data["classes"]) if saved_data["classes"] else np.array([])
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

        # Для бинарной классификации берём коэффициенты
        coef = self.model.coef_[0] if len(self._classes) == 2 else np.mean(np.abs(self.model.coef_), axis=0)

        return np.abs(coef)

    def get_feature_importance_df(self) -> pd.DataFrame:
        """
        Получить DataFrame с важностями признаков.

        Returns:
            DataFrame с колонками ['feature', 'importance', 'coefficient']
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена.")

        # Коэффициенты
        if len(self._classes) == 2:
            coef = self.model.coef_[0]
        else:
            # Для мультикласса усредняем по классам
            coef = np.mean(self.model.coef_, axis=0)

        df = pd.DataFrame(
            {
                "feature": self._feature_names,
                "coefficient": coef,
                "importance": np.abs(coef),
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
        penalty = self.hyperparams.get("penalty", "?")
        C = self.hyperparams.get("C", "?")
        return f"LogisticRegressionModel(penalty={penalty}, C={C}, {fitted_status})"
