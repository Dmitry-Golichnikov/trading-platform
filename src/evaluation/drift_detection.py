"""Детекция дрейфа данных и моделей."""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


class PopulationStabilityIndex:
    """Population Stability Index (PSI) для детекции дрейфа."""

    @staticmethod
    def compute(
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
        epsilon: float = 1e-10,
    ) -> float:
        """
        Вычислить PSI между reference и current распределениями.

        Args:
            reference: Референсное распределение (обычно train)
            current: Текущее распределение (обычно test/production)
            n_bins: Количество bins для дискретизации
            epsilon: Малое значение для избежания деления на ноль

        Returns:
            PSI значение (0-0.1: нет дрейфа, 0.1-0.2: небольшой дрейф, >0.2: значительный дрейф)
        """
        # Создаем bins на основе reference
        _, bin_edges = np.histogram(reference, bins=n_bins)

        # Распределяем reference и current по bins
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)

        # Преобразуем в проценты
        ref_percents = ref_counts / len(reference)
        curr_percents = curr_counts / len(current)

        # Избегаем деления на ноль
        ref_percents = np.where(ref_percents == 0, epsilon, ref_percents)
        curr_percents = np.where(curr_percents == 0, epsilon, curr_percents)

        # Вычисляем PSI
        psi = np.sum((curr_percents - ref_percents) * np.log(curr_percents / ref_percents))

        return float(psi)

    @staticmethod
    def compute_feature_wise(
        reference: pd.DataFrame,
        current: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """
        Вычислить PSI для каждого признака.

        Args:
            reference: Референсный датасет
            current: Текущий датасет
            feature_names: Список признаков для анализа (по умолчанию все)
            n_bins: Количество bins

        Returns:
            DataFrame с PSI для каждого признака
        """
        if feature_names is None:
            feature_names = reference.columns.tolist()

        results = []

        for feature in feature_names:
            if feature not in reference.columns or feature not in current.columns:
                warnings.warn(f"Feature {feature} not found in both datasets")
                continue

            try:
                ref_values = np.asarray(reference[feature].values)
                curr_values = np.asarray(current[feature].values)
                psi = PopulationStabilityIndex.compute(ref_values, curr_values, n_bins=n_bins)

                results.append(
                    {
                        "feature": feature,
                        "psi": psi,
                        "status": PopulationStabilityIndex._interpret_psi(psi),
                    }
                )
            except Exception as e:
                warnings.warn(f"Failed to compute PSI for {feature}: {e}")

        return pd.DataFrame(results).sort_values("psi", ascending=False)

    @staticmethod
    def _interpret_psi(psi: float) -> str:
        """Интерпретировать значение PSI."""
        if psi < 0.1:
            return "no_drift"
        elif psi < 0.2:
            return "minor_drift"
        else:
            return "major_drift"


class KolmogorovSmirnovTest:
    """Kolmogorov-Smirnov test для детекции дрейфа."""

    @staticmethod
    def compute(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """
        Провести KS тест.

        Args:
            reference: Референсное распределение
            current: Текущее распределение

        Returns:
            Tuple (statistic, p_value)
        """
        statistic, p_value = stats.ks_2samp(reference, current)
        return float(statistic), float(p_value)

    @staticmethod
    def compute_feature_wise(
        reference: pd.DataFrame,
        current: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Провести KS тест для каждого признака.

        Args:
            reference: Референсный датасет
            current: Текущий датасет
            feature_names: Список признаков для анализа
            alpha: Уровень значимости

        Returns:
            DataFrame с результатами теста
        """
        if feature_names is None:
            feature_names = reference.columns.tolist()

        results = []

        for feature in feature_names:
            if feature not in reference.columns or feature not in current.columns:
                warnings.warn(f"Feature {feature} not found in both datasets")
                continue

            try:
                ref_values = np.asarray(reference[feature].values)
                curr_values = np.asarray(current[feature].values)
                statistic, p_value = KolmogorovSmirnovTest.compute(ref_values, curr_values)

                results.append(
                    {
                        "feature": feature,
                        "ks_statistic": statistic,
                        "p_value": p_value,
                        "drift_detected": p_value < alpha,
                    }
                )
            except Exception as e:
                warnings.warn(f"Failed to compute KS test for {feature}: {e}")

        return pd.DataFrame(results).sort_values("ks_statistic", ascending=False)


class ChiSquaredTest:
    """Chi-squared test для категориальных переменных."""

    @staticmethod
    def compute(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> Tuple[float, float]:
        """
        Провести Chi-squared тест.

        Args:
            reference: Референсное распределение
            current: Текущее распределение
            n_bins: Количество bins для непрерывных переменных

        Returns:
            Tuple (statistic, p_value)
        """
        # Создаем bins
        _, bin_edges = np.histogram(np.concatenate([reference, current]), bins=n_bins)

        # Подсчитываем частоты
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)

        # Chi-squared test
        statistic, p_value = stats.chisquare(f_obs=curr_counts, f_exp=ref_counts + 1e-10)

        return float(statistic), float(p_value)

    @staticmethod
    def compute_feature_wise(
        reference: pd.DataFrame,
        current: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        n_bins: int = 10,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Провести Chi-squared тест для каждого признака.

        Args:
            reference: Референсный датасет
            current: Текущий датасет
            feature_names: Список признаков для анализа
            n_bins: Количество bins
            alpha: Уровень значимости

        Returns:
            DataFrame с результатами теста
        """
        if feature_names is None:
            feature_names = reference.columns.tolist()

        results = []

        for feature in feature_names:
            if feature not in reference.columns or feature not in current.columns:
                warnings.warn(f"Feature {feature} not found in both datasets")
                continue

            try:
                ref_values = np.asarray(reference[feature].values)
                curr_values = np.asarray(current[feature].values)
                statistic, p_value = ChiSquaredTest.compute(ref_values, curr_values, n_bins=n_bins)

                results.append(
                    {
                        "feature": feature,
                        "chi2_statistic": statistic,
                        "p_value": p_value,
                        "drift_detected": p_value < alpha,
                    }
                )
            except Exception as e:
                warnings.warn(f"Failed to compute Chi-squared test for {feature}: {e}")

        return pd.DataFrame(results).sort_values("chi2_statistic", ascending=False)


class AdversarialValidation:
    """Adversarial Validation для детекции дрейфа."""

    @staticmethod
    def compute(
        reference: pd.DataFrame,
        current: pd.DataFrame,
        n_estimators: int = 100,
        random_state: Optional[int] = None,
        cv: int = 5,
    ) -> Dict[str, Union[float, Any]]:
        """
        Провести adversarial validation.

        Идея: обучить модель отличать reference от current.
        Если ROC-AUC близок к 0.5, дрейфа нет. Если близок к 1.0, есть значительный дрейф.

        Args:
            reference: Референсный датасет
            current: Текущий датасет
            n_estimators: Количество деревьев в лесу
            random_state: Random seed
            cv: Количество фолдов для кросс-валидации

        Returns:
            Dict с результатами
        """
        # Создаем единый датасет с метками
        reference_labeled = reference.copy()
        reference_labeled["is_current"] = 0

        current_labeled = current.copy()
        current_labeled["is_current"] = 1

        # Объединяем
        combined = pd.concat([reference_labeled, current_labeled], ignore_index=True)

        # Перемешиваем
        combined = combined.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # Разделяем на X и y
        y = combined["is_current"]
        X = combined.drop("is_current", axis=1)

        # Обучаем классификатор
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

        # Кросс-валидация
        scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")

        # Обучаем на всех данных для feature importance
        clf.fit(X, y)

        # Feature importance
        feature_importance = pd.DataFrame({"feature": X.columns, "importance": clf.feature_importances_}).sort_values(
            "importance", ascending=False
        )

        return {
            "mean_roc_auc": float(np.mean(scores)),
            "std_roc_auc": float(np.std(scores)),
            "drift_interpretation": AdversarialValidation._interpret_auc(np.mean(scores)),
            "feature_importance": feature_importance,
            "cv_scores": scores.tolist(),
        }

    @staticmethod
    def _interpret_auc(auc: float) -> str:
        """Интерпретировать AUC значение."""
        if auc < 0.55:
            return "no_drift"
        elif auc < 0.65:
            return "minor_drift"
        elif auc < 0.75:
            return "moderate_drift"
        else:
            return "major_drift"


class DriftDetector:
    """Комплексный детектор дрейфа."""

    def __init__(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Инициализация детектора.

        Args:
            reference: Референсный датасет (обычно train)
            current: Текущий датасет (обычно test/production)
            feature_names: Список признаков для анализа
        """
        self.reference = reference
        self.current = current

        if feature_names is None:
            self.feature_names = [col for col in reference.columns if col in current.columns]
        else:
            self.feature_names = feature_names

    def detect_all(
        self,
        methods: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, Any]]]:
        """
        Провести детекцию дрейфа всеми методами.

        Args:
            methods: Список методов ('psi', 'ks', 'chi2', 'adversarial')
            **kwargs: Дополнительные параметры для методов

        Returns:
            Dict с результатами для каждого метода
        """
        if methods is None:
            methods = ["psi", "ks", "chi2", "adversarial"]

        results: Dict[str, Union[pd.DataFrame, Dict[str, Any]]] = {}

        # PSI
        if "psi" in methods:
            try:
                results["psi"] = PopulationStabilityIndex.compute_feature_wise(
                    self.reference[self.feature_names],
                    self.current[self.feature_names],
                    n_bins=kwargs.get("n_bins", 10),
                )
            except Exception as e:
                warnings.warn(f"Failed to compute PSI: {e}")

        # Kolmogorov-Smirnov test
        if "ks" in methods:
            try:
                results["ks"] = KolmogorovSmirnovTest.compute_feature_wise(
                    self.reference[self.feature_names],
                    self.current[self.feature_names],
                    alpha=kwargs.get("alpha", 0.05),
                )
            except Exception as e:
                warnings.warn(f"Failed to compute KS test: {e}")

        # Chi-squared test
        if "chi2" in methods:
            try:
                results["chi2"] = ChiSquaredTest.compute_feature_wise(
                    self.reference[self.feature_names],
                    self.current[self.feature_names],
                    n_bins=kwargs.get("n_bins", 10),
                    alpha=kwargs.get("alpha", 0.05),
                )
            except Exception as e:
                warnings.warn(f"Failed to compute Chi-squared test: {e}")

        # Adversarial validation
        if "adversarial" in methods:
            try:
                results["adversarial"] = AdversarialValidation.compute(
                    self.reference[self.feature_names],
                    self.current[self.feature_names],
                    n_estimators=kwargs.get("n_estimators", 100),
                    random_state=kwargs.get("random_state", None),
                    cv=kwargs.get("cv", 5),
                )
            except Exception as e:
                warnings.warn(f"Failed to compute adversarial validation: {e}")

        return results

    def get_drifted_features(self, method: str = "psi", threshold: Optional[float] = None) -> List[str]:
        """
        Получить список признаков с дрейфом.

        Args:
            method: Метод детекции ('psi', 'ks', 'chi2')
            threshold: Порог для определения дрейфа

        Returns:
            Список признаков с дрейфом
        """
        results = self.detect_all(methods=[method])

        if method not in results:
            return []

        result = results[method]
        if not isinstance(result, pd.DataFrame):
            return []

        df = result

        if method == "psi":
            if threshold is None:
                threshold = 0.2  # Значительный дрейф
            return df[df["psi"] > threshold]["feature"].tolist()

        elif method in ["ks", "chi2"]:
            if threshold is None:
                threshold = 0.05  # p-value
            return df[df["drift_detected"]]["feature"].tolist()

        return []

    def summary(self) -> Dict[str, Any]:
        """
        Получить сводную информацию о дрейфе.

        Returns:
            Dict с основной информацией
        """
        all_results = self.detect_all()

        summary_dict: Dict[str, Any] = {
            "n_features": len(self.feature_names),
            "n_samples_reference": len(self.reference),
            "n_samples_current": len(self.current),
        }

        # PSI summary
        if "psi" in all_results and isinstance(all_results["psi"], pd.DataFrame):
            psi_df = all_results["psi"]
            if not psi_df.empty:
                summary_dict["psi_mean"] = float(psi_df["psi"].mean())
                summary_dict["psi_max"] = float(psi_df["psi"].max())
                summary_dict["n_features_with_major_drift_psi"] = int((psi_df["status"] == "major_drift").sum())

        # KS summary
        if "ks" in all_results and isinstance(all_results["ks"], pd.DataFrame):
            ks_df = all_results["ks"]
            if not ks_df.empty:
                summary_dict["n_features_with_drift_ks"] = int(ks_df["drift_detected"].sum())

        # Adversarial validation summary
        if "adversarial" in all_results:
            adv_results = all_results["adversarial"]
            if isinstance(adv_results, dict) and "mean_roc_auc" in adv_results:
                summary_dict["adversarial_auc"] = float(adv_results["mean_roc_auc"])
                summary_dict["adversarial_interpretation"] = str(adv_results["drift_interpretation"])

        return summary_dict
