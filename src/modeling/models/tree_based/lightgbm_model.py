"""
LightGBM модель для классификации и регрессии.

Обёртка над LightGBM с поддержкой:
- Early stopping на валидации
- GPU обучение
- Feature importances
- Единый интерфейс с BaseModel
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.modeling.base import BaseModel, ClassifierMixin, RegressorMixin
from src.modeling.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register(
    "lightgbm",
    description="LightGBM gradient boosting model",
    tags=["tree-based", "gradient-boosting", "gpu-support"],
)
class LightGBMModel(BaseModel, ClassifierMixin, RegressorMixin):
    """
    LightGBM обёртка для классификации и регрессии.

    Поддерживает:
    - Бинарную и мультиклассовую классификацию
    - Регрессию
    - Early stopping на валидационном наборе
    - GPU обучение (если доступно)
    - Feature importances (gain, split)

    Args:
        task: Тип задачи ('classification' или 'regression')
        **hyperparams: Гиперпараметры LightGBM
    """

    def __init__(self, task: str = "classification", **hyperparams):
        """
        Инициализация LightGBM модели.

        Args:
            task: Тип задачи ('classification' или 'regression')
            **hyperparams: Гиперпараметры для LightGBM
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
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "max_depth": -1,
            "min_data_in_leaf": 20,
            "n_estimators": 1000,
            "verbose": -1,
            "device": "cpu",
            "random_state": 42,
        }

        # Задача-специфичные параметры
        if self.task == "classification":
            defaults.update(
                {
                    "objective": "binary",
                    "metric": "auc",
                }
            )
        elif self.task == "regression":
            defaults.update(
                {
                    "objective": "regression",
                    "metric": "rmse",
                }
            )
        else:
            raise ValueError(f"Неизвестная задача: {self.task}. Используйте 'classification' или 'regression'")

        # Обновляем только если параметр не был задан пользователем
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
    ) -> "LightGBMModel":
        """
        Обучить LightGBM модель.

        Args:
            X: Признаки для обучения
            y: Таргет для обучения
            X_val: Признаки для валидации (для early stopping)
            y_val: Таргет для валидации (для early stopping)
            **kwargs: Дополнительные параметры (callbacks, eval_metric и т.д.)

        Returns:
            Обученная модель (self)
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM не установлен. Установите: pip install lightgbm")

        logger.info(f"Начало обучения LightGBM модели (задача: {self.task})")

        # Сохраняем имена признаков
        self._feature_names = list(X.columns)

        # Для классификации сохраняем классы
        if self.task == "classification":
            self._classes = np.unique(y)
            n_classes = len(self._classes)

            # Автоматически определяем objective
            if n_classes == 2:
                if "objective" not in self.hyperparams or self.hyperparams["objective"] == "binary":
                    self.hyperparams["objective"] = "binary"
            elif n_classes > 2:
                self.hyperparams["objective"] = "multiclass"
                self.hyperparams["num_class"] = n_classes

        # Создаём датасеты LightGBM
        train_data = lgb.Dataset(X, y, feature_name=self._feature_names)
        valid_sets = [train_data]
        valid_names = ["training"]

        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(X_val, y_val, reference=train_data, feature_name=self._feature_names)
            valid_sets.append(valid_data)
            valid_names.append("validation")

        # Параметры обучения
        train_params = self.hyperparams.copy()
        n_estimators = train_params.pop("n_estimators", 1000)

        # Callbacks
        callbacks = kwargs.get("callbacks", [])
        if not callbacks and X_val is not None:
            # Дефолтные callbacks если не заданы
            callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=10),
            ]

        # Обучение
        import time

        start_time = time.time()

        self.model = lgb.train(
            train_params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        training_time = time.time() - start_time

        # Обновляем метаданные
        self.metadata.update(
            {
                "training_time": training_time,
                "n_samples_trained": len(X),
                "n_features": len(self._feature_names),
                "best_iteration": self.model.best_iteration,
                "best_score": self.model.best_score,
            }
        )

        self._is_fitted = True
        logger.info(f"Обучение завершено за {training_time:.2f} сек. " f"Best iteration: {self.model.best_iteration}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Получить предсказания модели.

        Args:
            X: Признаки для предсказания

        Returns:
            Предсказания (метки классов для классификации, значения для регрессии)
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")

        # Проверяем признаки
        self._validate_features(X)

        if self.task == "classification":
            # Для классификации возвращаем метки классов
            proba = self.predict_proba(X)
            if proba.ndim == 1:
                # Бинарная классификация
                predictions = (proba >= 0.5).astype(int)
            else:
                # Мультиклассовая классификация
                predictions = np.argmax(proba, axis=1)

            # Маппинг обратно к оригинальным меткам классов
            if self._classes.size > 0:
                predictions = self._classes[predictions]

            return predictions
        else:
            # Для регрессии возвращаем значения
            return self.model.predict(X, num_iteration=self.model.best_iteration)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Получить вероятности классов (только для классификации).

        Args:
            X: Признаки для предсказания

        Returns:
            Вероятности классов, shape (n_samples,) для бинарной или (n_samples, n_classes) для мультикласса
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")

        if self.task != "classification":
            raise ValueError("predict_proba доступен только для классификации")

        # Проверяем признаки
        self._validate_features(X)

        probas = self.model.predict(X, num_iteration=self.model.best_iteration)

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

        # Сохраняем модель LightGBM
        model_path = path / "model.txt"
        self.model.save_model(str(model_path))

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
    def load(cls, path: Path) -> "LightGBMModel":
        """
        Загрузить модель с диска.

        Args:
            path: Путь к директории с моделью

        Returns:
            Загруженная модель
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM не установлен. Установите: pip install lightgbm")

        path = Path(path)

        # Загружаем метаданные
        metadata_path = path / "metadata.json"
        with open(metadata_path, "r") as f:
            saved_data = json.load(f)

        # Создаём экземпляр модели
        task = saved_data["task"]
        hyperparams = saved_data["hyperparams"]
        model_instance = cls(task=task, **hyperparams)

        # Загружаем модель LightGBM
        model_path = path / "model.txt"
        model_instance.model = lgb.Booster(model_file=str(model_path))

        # Восстанавливаем метаданные
        model_instance.metadata = saved_data["metadata"]
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
            Массив важностей признаков (gain-based)
        """
        if not self.is_fitted:
            return None

        return self.model.feature_importance(importance_type="gain")

    def get_feature_importance_df(self, importance_type: str = "gain") -> pd.DataFrame:
        """
        Получить DataFrame с важностями признаков.

        Args:
            importance_type: Тип важности ('gain' или 'split')

        Returns:
            DataFrame с колонками ['feature', 'importance']
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена.")

        importances = self.model.feature_importance(importance_type=importance_type)

        df = pd.DataFrame(
            {
                "feature": self._feature_names,
                "importance": importances,
            }
        )

        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def _validate_features(self, X: pd.DataFrame) -> None:
        """
        Проверить что признаки соответствуют обученной модели.

        Args:
            X: DataFrame с признаками

        Raises:
            ValueError: Если признаки не совпадают
        """
        if self._feature_names is None:
            return

        missing_features = set(self._feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(f"Отсутствуют признаки: {missing_features}. " f"Ожидаются: {self._feature_names}")

        # Переупорядочиваем колонки если нужно
        if list(X.columns) != self._feature_names:
            logger.warning("Переупорядочивание признаков для соответствия модели")

    def __repr__(self) -> str:
        """Строковое представление модели."""
        fitted_status = "fitted" if self.is_fitted else "not fitted"
        n_estimators = self.model.best_iteration if self.is_fitted else self.hyperparams.get("n_estimators", "?")
        return f"LightGBMModel(task={self.task}, n_estimators={n_estimators}, {fitted_status})"
