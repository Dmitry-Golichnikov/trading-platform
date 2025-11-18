"""Утилиты для отбора признаков на основании важности."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

TaskType = Literal["classification", "regression"]


@dataclass
class FeatureImportanceSelector:
    """
    Отбор признаков по важности модели.

    По умолчанию использует RandomForest и возвращает top-k признаков по
    важности. Подходит как для классификации, так и для регрессии.
    """

    n_features: Optional[int] = 50
    threshold: float = 0.0
    task_type: Optional[TaskType] = None
    random_state: int = 42
    n_estimators: int = 200
    max_depth: Optional[int] = None

    def select(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> list[str]:
        """
        Отобрать признаки по важности.

        Args:
            X: Матрица признаков
            y: Таргет (необязательный для фильтрации по дисперсии)

        Returns:
            Список выбранных признаков
        """
        if X.empty:
            return []

        if y is None or y.empty:
            # Если таргета нет, то просто ограничиваемся top-k по дисперсии
            variances = X.var().sort_values(ascending=False)
            if self.n_features is not None:
                return variances.head(self.n_features).index.tolist()
            return variances[variances > 0].index.tolist()

        # Заполняем NaN
        X_filled = X.fillna(0)

        task_type = self._detect_task_type(y) if self.task_type is None else self.task_type

        if task_type == "classification":
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )

        model.fit(X_filled, y)

        importances = pd.Series(model.feature_importances_, index=X.columns)
        sorted_features = importances.sort_values(ascending=False)

        if self.threshold > 0:
            selected = sorted_features[sorted_features >= self.threshold].index.tolist()
        else:
            selected = sorted_features.index.tolist()

        if self.n_features is not None:
            selected = selected[: self.n_features]

        return selected

    def _detect_task_type(self, y: pd.Series) -> TaskType:
        """Определить тип задачи по таргету."""
        if y.dtype == "object":
            return "classification"

        unique_values = y.dropna().unique()
        if len(unique_values) <= 20:
            # Небольшое количество классов – классификация
            return "classification"
        numpy_dtype = y.to_numpy().dtype
        if np.issubdtype(numpy_dtype, np.integer):
            return "classification"

        return "regression"
