"""
XGBoost модель для классификации и регрессии.

Обёртка над XGBoost с поддержкой:
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
    "xgboost",
    description="XGBoost gradient boosting model",
    tags=["tree-based", "gradient-boosting", "gpu-support"],
)
class XGBoostModel(BaseModel, ClassifierMixin, RegressorMixin):
    """
    XGBoost обёртка для классификации и регрессии.

    Поддерживает:
    - Бинарную и мультиклассовую классификацию
    - Регрессию
    - Early stopping на валидационном наборе
    - GPU обучение (если доступно)
    - Feature importances

    Args:
        task: Тип задачи ('classification' или 'regression')
        **hyperparams: Гиперпараметры XGBoost
    """

    def __init__(self, task: str = "classification", **hyperparams):
        """
        Инициализация XGBoost модели.

        Args:
            task: Тип задачи ('classification' или 'regression')
            **hyperparams: Гиперпараметры для XGBoost
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
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 1000,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "random_state": 42,
            "tree_method": "auto",
            "n_jobs": -1,
        }

        # Задача-специфичные параметры
        if self.task == "classification":
            defaults.update(
                {
                    "objective": "binary:logistic",
                    "eval_metric": "auc",
                }
            )
        elif self.task == "regression":
            defaults.update(
                {
                    "objective": "reg:squarederror",
                    "eval_metric": "rmse",
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
    ) -> "XGBoostModel":
        """
        Обучить XGBoost модель.

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
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost не установлен. Установите: pip install xgboost")

        logger.info(f"Начало обучения XGBoost модели (задача: {self.task})")

        # Сохраняем имена признаков
        self._feature_names = list(X.columns)

        # Для классификации сохраняем классы
        if self.task == "classification":
            self._classes = np.unique(y)
            n_classes = len(self._classes)

            if n_classes == 2:
                if "objective" not in self.hyperparams:
                    self.hyperparams["objective"] = "binary:logistic"
            elif n_classes > 2:
                self.hyperparams["objective"] = "multi:softprob"
                self.hyperparams["num_class"] = n_classes

        # Используем низкоуровневый API xgboost (Booster) — совместим с разными версиями
        params = self.hyperparams.copy()
        num_boost_round = params.pop("n_estimators", 1000)

        # DMatrix
        dtrain = xgb.DMatrix(X.values, label=y.values)
        watchlist = [(dtrain, "train")]
        dvalid = None
        if X_val is not None and y_val is not None:
            dvalid = xgb.DMatrix(X_val.values, label=y_val.values)
            watchlist.append((dvalid, "validation"))

        # Early stopping
        early_stopping_rounds = kwargs.get("early_stopping_rounds", 50) if dvalid is not None else None

        import time

        start_time = time.time()

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=watchlist,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=kwargs.get("verbose", False),
        )

        training_time = time.time() - start_time

        # Обновляем метаданные
        self.metadata.update(
            {
                "training_time": training_time,
                "n_samples_trained": len(X),
                "n_features": len(self._feature_names),
                "best_iteration": (
                    getattr(self.model, "best_iteration", None)
                    if hasattr(self.model, "best_iteration")
                    else getattr(self.model, "best_ntree_limit", None)
                ),
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
        import xgboost as xgb

        dmatrix = xgb.DMatrix(X.values)
        preds = self.model.predict(dmatrix)

        if self.task == "classification":
            # Бинарная классификация — preds содержит вероятность положительного класса
            if preds.ndim == 1:
                labels = (preds >= 0.5).astype(int)
                if self._classes is not None and self._classes.size > 0:
                    return self._classes[labels]
                return labels
            else:
                # Мультиклассовая предсказания shape (n_samples, n_classes)
                labels = np.argmax(preds, axis=1)
                if self._classes is not None and self._classes.size > 0:
                    return self._classes[labels]
                return labels
        else:
            # Регрессия
            return preds

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
        import xgboost as xgb

        dmatrix = xgb.DMatrix(X.values)
        probas = self.model.predict(dmatrix)

        # Если бинарная, preds одноизмерный — вероятность положительного класса
        if probas.ndim == 1:
            return probas

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

        # Сохраняем модель XGBoost
        model_path = path / "model.json"
        self.model.save_model(str(model_path))

        # Сохраняем метаданные
        metadata = {
            "task": self.task,
            "hyperparams": self.hyperparams,
            "metadata": self.metadata,
            "classes": self._classes.tolist() if self._classes is not None and self._classes.size > 0 else None,
            "feature_names": self._feature_names,
        }

        metadata_path = path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Модель сохранена в {path}")

    @classmethod
    def load(cls, path: Path) -> "XGBoostModel":
        """
        Загрузить модель с диска.

        Args:
            path: Путь к директории с моделью

        Returns:
            Загруженная модель
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost не установлен. Установите: pip install xgboost")

        path = Path(path)

        # Загружаем метаданные
        metadata_path = path / "metadata.json"
        with open(metadata_path, "r") as f:
            saved_data = json.load(f)

        # Создаём экземпляр модели
        task = saved_data["task"]
        hyperparams = saved_data["hyperparams"]
        model_instance = cls(task=task, **hyperparams)

        # Загружаем модель XGBoost
        model_path = path / "model.json"
        # Загружаем Booster
        model_instance.model = xgb.Booster()
        model_instance.model.load_model(str(model_path))

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
            Массив важностей признаков
        """
        if not self.is_fitted:
            return None

        # Если модель — Booster из xgboost
        try:
            # get_score возвращает dict {'f0': val, 'f1': val}
            scores = self.model.get_score(importance_type="gain")
        except Exception:
            # Иногда объект может быть sklearn wrapper — попробуем получить attribute
            if hasattr(self.model, "feature_importances_"):
                return getattr(self.model, "feature_importances_")
            return None

        # Преобразуем dict в массив по порядку feature_names
        importances = np.zeros(len(self._feature_names), dtype=float)
        for key, val in scores.items():
            # key формат 'f{i}'
            if key.startswith("f"):
                try:
                    idx = int(key[1:])
                except ValueError:
                    continue
                if 0 <= idx < len(importances):
                    importances[idx] = val

        return importances

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
        n_estimators = (
            getattr(self.model, "best_iteration", self.hyperparams.get("n_estimators", "?"))
            if self.is_fitted
            else self.hyperparams.get("n_estimators", "?")
        )
        return f"XGBoostModel(task={self.task}, n_estimators={n_estimators}, {fitted_status})"
