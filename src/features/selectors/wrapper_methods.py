"""Wrapper-based методы отбора признаков."""

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, Ridge


class WrapperSelector:
    """Wrapper-based отбор признаков (RFE, Forward/Backward selection)."""

    def __init__(
        self,
        method: str = "rfe",
        estimator=None,
        n_features_to_select: Optional[int] = None,
        **kwargs,
    ):
        """
        Инициализация selector'а.

        Args:
            method: Метод отбора (rfe, forward, backward)
            estimator: Модель для оценки (если None, используется LogisticRegression)
            n_features_to_select: Количество признаков для отбора
            **kwargs: Дополнительные параметры
        """
        self.method = method
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.params = kwargs
        self.selected_features_: Optional[List[str]] = None
        self.feature_scores_: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WrapperSelector":
        """
        Обучить selector на данных.

        Args:
            X: DataFrame с признаками
            y: Таргет

        Returns:
            self
        """
        # Определяем тип задачи
        is_classification = y.nunique() < 20 or y.dtype == "object"

        # Создаём дефолтный estimator если не указан
        if self.estimator is None:
            if is_classification:
                self.estimator = LogisticRegression(max_iter=1000, random_state=42)
            else:
                self.estimator = Ridge(random_state=42)

        if self.method == "rfe":
            self._fit_rfe(X, y)
        elif self.method == "forward":
            self._fit_forward(X, y)
        elif self.method == "backward":
            self._fit_backward(X, y)
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

    def _fit_rfe(self, X: pd.DataFrame, y: pd.Series):
        """Recursive Feature Elimination."""
        # Заполняем NaN
        X_filled = X.fillna(0)

        # Создаём RFE selector
        rfe = RFE(
            estimator=self.estimator,
            n_features_to_select=self.n_features_to_select,
            step=self.params.get("step", 1),
        )

        # Обучаем
        rfe.fit(X_filled, y)

        # Получаем отобранные признаки
        self.selected_features_ = X.columns[rfe.support_].tolist()

        # Сохраняем рейтинги признаков (1 = лучший)
        self.feature_scores_ = pd.Series(rfe.ranking_, index=X.columns).sort_values()

    def _fit_forward(self, X: pd.DataFrame, y: pd.Series):
        """Forward feature selection."""
        # Заполняем NaN
        X_filled = X.fillna(0)

        n_features = self.n_features_to_select or X.shape[1] // 2
        selected: List[str] = []
        remaining: List[str] = list(X.columns)

        best_score = -np.inf

        for _ in range(n_features):
            scores = []

            for feature in remaining:
                current_features = selected + [feature]
                X_subset = X_filled[current_features]

                # Обучаем модель и получаем score
                self.estimator.fit(X_subset, y)
                score = self.estimator.score(X_subset, y)
                scores.append((feature, score))

            # Выбираем лучший признак
            best_feature, score = max(scores, key=lambda x: x[1])

            if score > best_score:
                selected.append(best_feature)
                remaining.remove(best_feature)
                best_score = score
            else:
                break

        self.selected_features_ = selected
        self.feature_scores_ = pd.Series(range(len(selected), 0, -1), index=selected)

    def _fit_backward(self, X: pd.DataFrame, y: pd.Series):
        """Backward feature elimination."""
        # Заполняем NaN
        X_filled = X.fillna(0)

        n_features = self.n_features_to_select or X.shape[1] // 2
        selected: List[str] = list(X.columns)

        while len(selected) > n_features:
            scores = []

            for feature in selected:
                current_features = [f for f in selected if f != feature]
                X_subset = X_filled[current_features]

                # Обучаем модель и получаем score
                self.estimator.fit(X_subset, y)
                score = self.estimator.score(X_subset, y)
                scores.append((feature, score))

            # Удаляем признак, удаление которого меньше всего вредит
            feature_to_remove, _ = max(scores, key=lambda x: x[1])
            selected.remove(feature_to_remove)

        self.selected_features_ = selected
        self.feature_scores_ = pd.Series(range(len(selected), 0, -1), index=selected)
