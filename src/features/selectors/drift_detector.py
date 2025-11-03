"""Детектор дрифта признаков."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class DriftDetector:
    """Детектор дрифта признаков (PSI, KS test, Adversarial validation)."""

    def __init__(
        self,
        method: str = "psi",
        threshold: float = 0.2,
        **kwargs,
    ):
        """
        Инициализация detector'а.

        Args:
            method: Метод детекции (psi, ks_test, adversarial)
            threshold: Порог для определения дрифта
            **kwargs: Дополнительные параметры
        """
        self.method = method
        self.threshold = threshold
        self.params = kwargs
        self.drift_scores_: Optional[pd.Series] = None
        self.drifted_features_: Optional[List[str]] = None

    def detect(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.Series, List[str]]:
        """
        Детектировать дрифт признаков.

        Args:
            X_train: DataFrame с признаками из train
            X_test: DataFrame с признаками из test

        Returns:
            Кортеж (scores, drifted_features)
        """
        if self.method == "psi":
            scores = self._detect_psi(X_train, X_test)
        elif self.method == "ks_test":
            scores = self._detect_ks_test(X_train, X_test)
        elif self.method == "adversarial":
            scores = self._detect_adversarial(X_train, X_test)
        else:
            raise ValueError(f"Неизвестный метод: {self.method}")

        self.drift_scores_ = scores
        self.drifted_features_ = scores[scores > self.threshold].index.tolist()

        return scores, self.drifted_features_

    def _detect_psi(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.Series:
        """
        Population Stability Index (PSI).

        PSI < 0.1 - нет дрифта
        0.1 <= PSI < 0.2 - небольшой дрифт
        PSI >= 0.2 - значительный дрифт
        """
        n_bins = self.params.get("n_bins", 10)
        psi_scores = {}

        for col in X_train.columns:
            if col not in X_test.columns:
                continue

            # Пропускаем константные признаки
            if X_train[col].nunique() <= 1:
                psi_scores[col] = 0.0
                continue

            # Создаём бины на основе train
            train_vals = X_train[col].dropna()
            test_vals = X_test[col].dropna()

            if len(train_vals) == 0 or len(test_vals) == 0:
                psi_scores[col] = 0.0
                continue

            # Вычисляем квантили для биннинга
            _, bin_edges = pd.qcut(
                train_vals, q=n_bins, retbins=True, duplicates="drop"
            )
            bins_list: List[float] = bin_edges.tolist()

            # Подсчитываем частоты в каждом бине
            train_freq = pd.cut(
                train_vals, bins=bins_list, include_lowest=True
            ).value_counts(normalize=True)
            test_freq = pd.cut(
                test_vals, bins=bins_list, include_lowest=True
            ).value_counts(normalize=True)

            # Выравниваем индексы
            train_freq = train_freq.reindex(test_freq.index, fill_value=0.001)
            test_freq = test_freq.fillna(0.001)

            # Вычисляем PSI
            psi = np.sum((test_freq - train_freq) * np.log(test_freq / train_freq))
            psi_scores[col] = psi

        return pd.Series(psi_scores).sort_values(ascending=False)

    def _detect_ks_test(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.Series:
        """
        Kolmogorov-Smirnov test.

        Возвращает p-value теста (низкий p-value = есть дрифт).
        Для удобства возвращаем (1 - p_value), чтобы высокие значения = дрифт.
        """
        ks_scores = {}

        for col in X_train.columns:
            if col not in X_test.columns:
                continue

            train_vals = X_train[col].dropna()
            test_vals = X_test[col].dropna()

            if len(train_vals) == 0 or len(test_vals) == 0:
                ks_scores[col] = 0.0
                continue

            # Выполняем KS test
            statistic, p_value = stats.ks_2samp(train_vals, test_vals)

            # Возвращаем (1 - p_value) для согласованности с PSI
            ks_scores[col] = 1 - p_value

        return pd.Series(ks_scores).sort_values(ascending=False)

    def _detect_adversarial(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> pd.Series:
        """
        Adversarial validation.

        Обучаем модель различать train от test.
        Высокая важность признака = он помогает различать = есть дрифт.
        """
        from sklearn.ensemble import RandomForestClassifier

        # Создаём бинарные метки (0 = train, 1 = test)
        X_train_labeled = X_train.copy()
        X_train_labeled["is_test"] = 0

        X_test_labeled = X_test.copy()
        X_test_labeled["is_test"] = 1

        # Объединяем
        X_combined = pd.concat([X_train_labeled, X_test_labeled], axis=0)

        # Заполняем NaN
        X_combined = X_combined.fillna(0)

        # Разделяем на признаки и таргет
        y = X_combined["is_test"]
        X = X_combined.drop("is_test", axis=1)

        # Обучаем Random Forest
        model = RandomForestClassifier(
            n_estimators=self.params.get("n_estimators", 100),
            max_depth=self.params.get("max_depth", 10),
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X, y)

        # Получаем важности признаков
        importances = pd.Series(model.feature_importances_, index=X.columns)

        return importances.sort_values(ascending=False)
