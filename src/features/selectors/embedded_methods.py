"""Embedded методы отбора признаков."""

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression


class EmbeddedSelector:
    """Embedded отбор признаков (tree importance, L1, SHAP)."""

    def __init__(
        self,
        method: str = "tree_importance",
        top_k: Optional[int] = None,
        **kwargs,
    ):
        """
        Инициализация selector'а.

        Args:
            method: Метод отбора (tree_importance, l1, shap)
            top_k: Количество топ признаков для отбора
            **kwargs: Дополнительные параметры
        """
        self.method = method
        self.top_k = top_k
        self.params = kwargs
        self.selected_features_: Optional[List[str]] = None
        self.feature_scores_: Optional[pd.Series] = None
        self.model_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EmbeddedSelector":
        """
        Обучить selector на данных.

        Args:
            X: DataFrame с признаками
            y: Таргет

        Returns:
            self
        """
        if self.method == "tree_importance":
            self._fit_tree_importance(X, y)
        elif self.method == "l1":
            self._fit_l1(X, y)
        elif self.method == "shap":
            self._fit_shap(X, y)
        else:
            raise ValueError(f"Неизвестный метод: {self.method}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Отобрать признаки.

        Args:
            X: DataFrame с признаками

        Returns:
            DataFrame с отобранными признаками
        """
        if self.selected_features_ is None:
            raise ValueError("Сначала вызовите fit()")

        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Обучить и отобрать признаки.

        Args:
            X: DataFrame с признаками
            y: Таргет

        Returns:
            DataFrame с отобранными признаками
        """
        self.fit(X, y)
        return self.transform(X)

    def _fit_tree_importance(self, X: pd.DataFrame, y: pd.Series):
        """Отбор по важности из деревьев (RandomForest)."""
        # Заполняем NaN
        X_filled = X.fillna(0)

        # Определяем тип задачи
        is_classification = y.nunique() < 20 or y.dtype == "object"

        # Создаём модель
        if is_classification:
            model = RandomForestClassifier(
                n_estimators=self.params.get("n_estimators", 100),
                max_depth=self.params.get("max_depth", 10),
                random_state=42,
                n_jobs=-1,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=self.params.get("n_estimators", 100),
                max_depth=self.params.get("max_depth", 10),
                random_state=42,
                n_jobs=-1,
            )

        # Обучаем
        model.fit(X_filled, y)
        self.model_ = model

        # Получаем важности
        importances = pd.Series(model.feature_importances_, index=X.columns)
        self.feature_scores_ = importances.sort_values(ascending=False)

        # Отбираем top_k или все с важностью выше порога
        threshold = self.params.get("threshold", 0.0)
        selected = self.feature_scores_[self.feature_scores_ > threshold].index.tolist()

        if self.top_k:
            selected = self.feature_scores_.head(self.top_k).index.tolist()

        self.selected_features_ = selected

    def _fit_l1(self, X: pd.DataFrame, y: pd.Series):
        """Отбор с помощью L1 регуляризации (Lasso)."""
        # Заполняем NaN
        X_filled = X.fillna(0)

        # Определяем тип задачи
        is_classification = y.nunique() < 20 or y.dtype == "object"

        # Создаём модель с L1 регуляризацией
        alpha = self.params.get("alpha", 0.01)

        if is_classification:
            model = LogisticRegression(
                penalty="l1",
                C=1 / alpha,
                solver="liblinear",
                random_state=42,
                max_iter=1000,
            )
        else:
            model = Lasso(alpha=alpha, random_state=42, max_iter=1000)

        # Обучаем
        model.fit(X_filled, y)
        self.model_ = model

        # Получаем веса (коэффициенты)
        if hasattr(model, "coef_"):
            if model.coef_.ndim > 1:
                # Для мультиклассовой классификации берём максимум по классам
                coefficients = np.abs(model.coef_).max(axis=0)
            else:
                coefficients = np.abs(model.coef_)

            self.feature_scores_ = pd.Series(coefficients, index=X.columns).sort_values(ascending=False)

            # Отбираем признаки с ненулевыми коэффициентами
            selected = self.feature_scores_[self.feature_scores_ > 0].index.tolist()

            if self.top_k:
                selected = self.feature_scores_.head(self.top_k).index.tolist()

            self.selected_features_ = selected
        else:
            raise ValueError("Модель не имеет атрибута coef_")

    def _fit_shap(self, X: pd.DataFrame, y: pd.Series):
        """Отбор с помощью SHAP values."""
        try:
            import shap
        except ImportError:
            raise ImportError("Для метода SHAP требуется библиотека shap. " "Установите её: pip install shap")

        # Заполняем NaN
        X_filled = X.fillna(0)

        # Определяем тип задачи
        is_classification = y.nunique() < 20 or y.dtype == "object"

        # Создаём модель
        if is_classification:
            model = RandomForestClassifier(
                n_estimators=self.params.get("n_estimators", 100),
                max_depth=self.params.get("max_depth", 10),
                random_state=42,
                n_jobs=-1,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=self.params.get("n_estimators", 100),
                max_depth=self.params.get("max_depth", 10),
                random_state=42,
                n_jobs=-1,
            )

        # Обучаем
        model.fit(X_filled, y)
        self.model_ = model

        # Вычисляем SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_filled)

        # Для классификации может быть массив для каждого класса
        if isinstance(shap_values, list):
            shap_values = np.abs(shap_values[0])

        # Вычисляем средние абсолютные SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        self.feature_scores_ = pd.Series(mean_abs_shap, index=X.columns).sort_values(ascending=False)

        # Отбираем top_k или все с SHAP выше порога
        threshold = self.params.get("threshold", 0.0)
        selected = self.feature_scores_[self.feature_scores_ > threshold].index.tolist()

        if self.top_k:
            selected = self.feature_scores_.head(self.top_k).index.tolist()

        self.selected_features_ = selected
