"""
CatBoost модель для классификации и регрессии.

Обёртка над CatBoost с поддержкой:
- Early stopping на валидации
- Нативная поддержка категориальных признаков
- GPU обучение
- Feature importances
- Единый интерфейс с BaseModel
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.modeling.base import BaseModel, ClassifierMixin, RegressorMixin
from src.modeling.registry import ModelRegistry

logger = logging.getLogger(__name__)


ITERATION_PARAM_NAMES = [
    "iterations",
    "n_estimators",
    "num_boost_round",
    "num_trees",
]


@ModelRegistry.register(
    "catboost",
    description="CatBoost gradient boosting model with categorical features support",
    tags=["tree-based", "gradient-boosting", "gpu-support", "categorical"],
)
class CatBoostModel(BaseModel, ClassifierMixin, RegressorMixin):
    """
    CatBoost обёртка для классификации и регрессии.

    Поддерживает:
    - Бинарную и мультиклассовую классификацию
    - Регрессию
    - Нативная поддержка категориальных признаков
    - Early stopping на валидационном наборе
    - GPU обучение (если доступно)
    - Feature importances

    Args:
        task: Тип задачи ('classification' или 'regression')
        cat_features: Список категориальных признаков
        **hyperparams: Гиперпараметры CatBoost
    """

    def __init__(
        self,
        task: str = "classification",
        cat_features: Optional[List[str]] = None,
        **hyperparams,
    ):
        """
        Инициализация CatBoost модели.

        Args:
            task: Тип задачи ('classification' или 'regression')
            cat_features: Список категориальных признаков
            **hyperparams: Гиперпараметры для CatBoost
        """
        # Не передаём cat_features в hyperparams (CatBoost ожидает их в Pool),
        # сохраняем отдельно и передаём только hyperparams в BaseModel
        super().__init__(task=task, **hyperparams)
        self.task = task
        self.cat_features = cat_features or []

        # Удаляем возможный ключ cat_features из hyperparams, если он попал туда
        if "cat_features" in self.hyperparams:
            self.hyperparams.pop("cat_features")
        self.model = None
        self._classes = np.array([])
        self._feature_names = []

        # Дефолтные параметры
        self._set_default_params()

    def _set_default_params(self) -> None:
        """Установить дефолтные параметры если не заданы."""
        defaults = {
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3,
            "random_seed": 42,
            "task_type": "CPU",  # или 'GPU'
            "verbose": False,
            "allow_writing_files": False,
        }

        # Если не задан ни один из параметров количества итераций — устанавливаем iterations
        if not any(k in self.hyperparams for k in ITERATION_PARAM_NAMES):
            self.hyperparams["iterations"] = 1000

        # Задача-специфичные параметры
        if self.task == "classification":
            defaults.update(
                {
                    "loss_function": "Logloss",
                    "eval_metric": "AUC",
                }
            )
        elif self.task == "regression":
            defaults.update(
                {
                    "loss_function": "RMSE",
                    "eval_metric": "RMSE",
                }
            )
        else:
            raise ValueError(f"Неизвестная задача: {self.task}")

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
    ) -> "CatBoostModel":
        """
        Обучить CatBoost модель.

        Args:
            X: Признаки для обучения
            y: Таргет для обучения
            X_val: Признаки для валидации
            y_val: Таргет для валидации
            **kwargs: Дополнительные параметры

        Returns:
            Обученная модель (self)
        """
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor, Pool
        except ImportError:
            raise ImportError("CatBoost не установлен. Установите: pip install catboost")

        logger.info(f"Начало обучения CatBoost модели (задача: {self.task})")

        # Сохраняем имена признаков
        self._feature_names = list(X.columns)

        # Определяем категориальные признаки по индексам
        cat_features_idx = []
        if self.cat_features:
            for cat_feat in self.cat_features:
                if cat_feat in self._feature_names:
                    cat_features_idx.append(self._feature_names.index(cat_feat))

        # Для классификации сохраняем классы
        if self.task == "classification":
            self._classes = np.unique(y)
            n_classes = len(self._classes)

            if n_classes > 2:
                self.hyperparams["loss_function"] = "MultiClass"

        # Убедимся, что передаём только один из параметров числа итераций (synonyms)
        iter_keys = [k for k in ITERATION_PARAM_NAMES if k in self.hyperparams]
        if len(iter_keys) > 1:
            # Отдадим приоритет 'n_estimators' если он задан, иначе оставим первый
            if "n_estimators" in self.hyperparams:
                for k in iter_keys:
                    if k != "n_estimators":
                        self.hyperparams.pop(k, None)
            else:
                # Оставляем только первый ключ, удаляя остальные
                for k in iter_keys[1:]:
                    self.hyperparams.pop(k, None)

        # Санитизация verbose: CatBoost требует non-negative или bool
        if "verbose" in self.hyperparams:
            v = self.hyperparams["verbose"]
            if isinstance(v, int) and v < 0:
                # заменяем на False
                self.hyperparams["verbose"] = False

        # Создаём модель
        if self.task == "classification":
            self.model = CatBoostClassifier(**self.hyperparams)
        else:
            self.model = CatBoostRegressor(**self.hyperparams)

        # Создаём Pool для обучения
        train_pool = Pool(X, y, cat_features=cat_features_idx)

        # Параметры обучения
        fit_params = {}

        # Validation pool
        if X_val is not None and y_val is not None:
            val_pool = Pool(X_val, y_val, cat_features=cat_features_idx)
            fit_params["eval_set"] = val_pool
            fit_params["early_stopping_rounds"] = kwargs.get("early_stopping_rounds", 50)

        # Обучение
        import time

        start_time = time.time()

        self.model.fit(train_pool, **fit_params)

        training_time = time.time() - start_time

        # Обновляем метаданные
        self.metadata.update(
            {
                "training_time": training_time,
                "n_samples_trained": len(X),
                "n_features": len(self._feature_names),
                "best_iteration": self.model.get_best_iteration(),
                "cat_features": self.cat_features,
            }
        )

        self._is_fitted = True
        logger.info(
            f"Обучение завершено за {training_time:.2f} сек. " f"Best iteration: {self.model.get_best_iteration()}"
        )

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
        predictions = self.model.predict(X)

        # Для классификации маппим обратно к оригинальным классам
        if self.task == "classification" and self._classes is not None and self._classes.size > 0:
            if len(self._classes) == 2:
                predictions = self._classes[predictions.astype(int).flatten()]

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

        # Сохраняем модель CatBoost
        model_path = path / "model.cbm"
        self.model.save_model(str(model_path))

        # Сохраняем метаданные
        metadata = {
            "task": self.task,
            "hyperparams": self.hyperparams,
            "metadata": self.metadata,
            "classes": self._classes.tolist() if self._classes is not None and self._classes.size > 0 else None,
            "feature_names": self._feature_names,
            "cat_features": self.cat_features,
        }

        metadata_path = path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Модель сохранена в {path}")

    @classmethod
    def load(cls, path: Path) -> "CatBoostModel":
        """
        Загрузить модель с диска.

        Args:
            path: Путь к директории с моделью

        Returns:
            Загруженная модель
        """
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError:
            raise ImportError("CatBoost не установлен. Установите: pip install catboost")

        path = Path(path)

        # Загружаем метаданные
        metadata_path = path / "metadata.json"
        with open(metadata_path, "r") as f:
            saved_data = json.load(f)

        # Создаём экземпляр модели
        task = saved_data["task"]
        cat_features = saved_data.get("cat_features", [])
        model_instance = cls(task=task, cat_features=cat_features)

        # Загружаем модель CatBoost
        model_path = path / "model.cbm"
        if task == "classification":
            model_instance.model = CatBoostClassifier()
        else:
            model_instance.model = CatBoostRegressor()

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
            Массив важностей признаков
        """
        if not self.is_fitted:
            return None

        return self.model.get_feature_importance()

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
        iterations = self.model.get_best_iteration() if self.is_fitted else self.hyperparams.get("iterations", "?")
        return f"CatBoostModel(task={self.task}, iterations={iterations}, {fitted_status})"
