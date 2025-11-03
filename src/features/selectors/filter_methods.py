"""Filter-based методы отбора признаков."""

from typing import List, Optional

import pandas as pd
from sklearn.feature_selection import chi2, mutual_info_classif, mutual_info_regression


class FilterSelector:
    """Filter-based отбор признаков."""

    def __init__(
        self,
        method: str = "variance_threshold",
        top_k: Optional[int] = None,
        **kwargs,
    ):
        """
        Инициализация selector'а.

        Args:
            method: Метод отбора (variance_threshold, correlation, mutual_info, chi2)
            top_k: Количество топ признаков для отбора (None = все прошедшие фильтр)
            **kwargs: Дополнительные параметры метода
        """
        self.method = method
        self.top_k = top_k
        self.params = kwargs
        self.selected_features_: Optional[List[str]] = None
        self.feature_scores_: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FilterSelector":
        """
        Обучить selector на данных.

        Args:
            X: DataFrame с признаками
            y: Таргет (требуется для некоторых методов)

        Returns:
            self
        """
        if self.method == "variance_threshold":
            self._fit_variance_threshold(X)
        elif self.method == "correlation":
            self._fit_correlation(X, y)
        elif self.method == "mutual_info":
            self._fit_mutual_info(X, y)
        elif self.method == "chi2":
            self._fit_chi2(X, y)
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

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
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

    def _fit_variance_threshold(self, X: pd.DataFrame):
        """Отбор по порогу дисперсии."""
        threshold = self.params.get("threshold", 0.0)

        # Вычисляем дисперсии
        variances = X.var()

        # Фильтруем по порогу
        selected = variances[variances > threshold].index.tolist()

        # Сортируем по убыванию дисперсии
        self.feature_scores_ = variances[selected].sort_values(ascending=False)

        # Отбираем top_k если указано
        if self.top_k:
            selected = self.feature_scores_.head(self.top_k).index.tolist()

        self.selected_features_ = selected

    def _fit_correlation(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        """Отбор по корреляции с таргетом."""
        if y is None:
            raise ValueError("Для метода correlation требуется таргет y")

        # Вычисляем корреляции с таргетом
        correlations = X.corrwith(y).abs()

        # Удаляем признаки с NaN корреляцией
        correlations = correlations.dropna()

        # Сортируем по убыванию
        self.feature_scores_ = correlations.sort_values(ascending=False)

        # Отбираем top_k или все с корреляцией выше порога
        threshold = self.params.get("threshold", 0.0)
        selected = self.feature_scores_[self.feature_scores_ > threshold].index.tolist()

        if self.top_k:
            selected = self.feature_scores_.head(self.top_k).index.tolist()

        self.selected_features_ = selected

    def _fit_mutual_info(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        """Отбор по взаимной информации."""
        if y is None:
            raise ValueError("Для метода mutual_info требуется таргет y")

        # Определяем тип задачи
        is_classification = y.nunique() < 20 or y.dtype == "object"

        # Заполняем NaN
        X_filled = X.fillna(0)

        # Вычисляем mutual information
        if is_classification:
            mi_scores = mutual_info_classif(
                X_filled, y, random_state=self.params.get("random_state", 42)
            )
        else:
            mi_scores = mutual_info_regression(
                X_filled, y, random_state=self.params.get("random_state", 42)
            )

        # Создаём Series со score'ами
        mi_series = pd.Series(mi_scores, index=X.columns)

        # Сортируем по убыванию
        self.feature_scores_ = mi_series.sort_values(ascending=False)

        # Отбираем top_k или все с MI выше порога
        threshold = self.params.get("threshold", 0.0)
        selected = self.feature_scores_[self.feature_scores_ > threshold].index.tolist()

        if self.top_k:
            selected = self.feature_scores_.head(self.top_k).index.tolist()

        self.selected_features_ = selected

    def _fit_chi2(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        """Отбор по chi-squared тесту (только для неотрицательных признаков)."""
        if y is None:
            raise ValueError("Для метода chi2 требуется таргет y")

        # Заполняем NaN и делаем признаки неотрицательными
        X_filled = X.fillna(0)
        X_positive = X_filled - X_filled.min() + 1e-10

        # Вычисляем chi2 статистики
        chi2_scores, _ = chi2(X_positive, y)

        # Создаём Series со score'ами
        chi2_series = pd.Series(chi2_scores, index=X.columns)

        # Сортируем по убыванию
        self.feature_scores_ = chi2_series.sort_values(ascending=False)

        # Отбираем top_k или все с chi2 выше порога
        threshold = self.params.get("threshold", 0.0)
        selected = self.feature_scores_[self.feature_scores_ > threshold].index.tolist()

        if self.top_k:
            selected = self.feature_scores_.head(self.top_k).index.tolist()

        self.selected_features_ = selected
