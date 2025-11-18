"""Тесты для детекции дрейфа."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.evaluation.drift_detection import (
    AdversarialValidation,
    ChiSquaredTest,
    DriftDetector,
    KolmogorovSmirnovTest,
    PopulationStabilityIndex,
)


class TestPopulationStabilityIndex:
    """Тесты для PSI."""

    @pytest.fixture
    def no_drift_data(self):
        """Данные без дрейфа."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, size=1000)
        current = np.random.normal(0, 1, size=1000)
        return reference, current

    @pytest.fixture
    def drift_data(self):
        """Данные с дрейфом."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, size=1000)
        current = np.random.normal(1, 1, size=1000)  # Сдвиг среднего
        return reference, current

    def test_no_drift(self, no_drift_data):
        """Тест PSI без дрейфа."""
        reference, current = no_drift_data
        psi = PopulationStabilityIndex.compute(reference, current)

        assert psi >= 0
        assert psi < 0.1  # Должен быть маленький

    def test_with_drift(self, drift_data):
        """Тест PSI с дрейфом."""
        reference, current = drift_data
        psi = PopulationStabilityIndex.compute(reference, current)

        assert psi >= 0
        assert psi > 0.1  # Должен быть больше

    def test_feature_wise(self):
        """Тест PSI по признакам."""
        np.random.seed(42)
        df_ref = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, size=100),
                "feature2": np.random.normal(0, 1, size=100),
            }
        )
        df_curr = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, size=100),  # Нет дрейфа
                "feature2": np.random.normal(1, 1, size=100),  # Есть дрейф
            }
        )

        results = PopulationStabilityIndex.compute_feature_wise(df_ref, df_curr)

        assert len(results) == 2
        assert "feature" in results.columns
        assert "psi" in results.columns
        assert "status" in results.columns

        # feature2 должен иметь больший PSI
        psi_f1 = results[results["feature"] == "feature1"]["psi"].values[0]
        psi_f2 = results[results["feature"] == "feature2"]["psi"].values[0]
        assert psi_f2 > psi_f1

    def test_interpret_psi(self):
        """Тест интерпретации PSI."""
        assert PopulationStabilityIndex._interpret_psi(0.05) == "no_drift"
        assert PopulationStabilityIndex._interpret_psi(0.15) == "minor_drift"
        assert PopulationStabilityIndex._interpret_psi(0.25) == "major_drift"


class TestKolmogorovSmirnovTest:
    """Тесты для KS теста."""

    def test_no_drift(self):
        """Тест KS без дрейфа."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, size=1000)
        current = np.random.normal(0, 1, size=1000)

        statistic, p_value = KolmogorovSmirnovTest.compute(reference, current)

        assert 0 <= statistic <= 1
        assert 0 <= p_value <= 1
        assert p_value > 0.05  # Не отклоняем гипотезу о равенстве

    def test_with_drift(self):
        """Тест KS с дрейфом."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, size=1000)
        current = np.random.normal(2, 1, size=1000)

        statistic, p_value = KolmogorovSmirnovTest.compute(reference, current)

        assert 0 <= statistic <= 1
        assert 0 <= p_value <= 1
        assert p_value < 0.05  # Отклоняем гипотезу о равенстве

    def test_feature_wise(self):
        """Тест KS по признакам."""
        np.random.seed(42)
        df_ref = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, size=100),
                "feature2": np.random.normal(0, 1, size=100),
            }
        )
        df_curr = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, size=100),
                "feature2": np.random.normal(2, 1, size=100),
            }
        )

        results = KolmogorovSmirnovTest.compute_feature_wise(df_ref, df_curr)

        assert len(results) == 2
        assert "ks_statistic" in results.columns
        assert "p_value" in results.columns
        assert "drift_detected" in results.columns


class TestChiSquaredTest:
    """Тесты для Chi-squared теста."""

    def test_compute(self):
        """Тест вычисления Chi-squared."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, size=1000)
        current = np.random.normal(0, 1, size=1000)

        statistic, p_value = ChiSquaredTest.compute(reference, current, n_bins=10)

        assert statistic >= 0
        assert 0 <= p_value <= 1

    def test_feature_wise(self):
        """Тест Chi-squared по признакам."""
        np.random.seed(42)
        df_ref = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, size=100),
                "feature2": np.random.normal(0, 1, size=100),
            }
        )
        df_curr = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, size=100),
                "feature2": np.random.normal(1, 1, size=100),
            }
        )

        results = ChiSquaredTest.compute_feature_wise(df_ref, df_curr)

        assert len(results) == 2
        assert "chi2_statistic" in results.columns
        assert "p_value" in results.columns
        assert "drift_detected" in results.columns


class TestAdversarialValidation:
    """Тесты для adversarial validation."""

    def test_no_drift(self):
        """Тест без дрейфа."""
        np.random.seed(42)
        X, _ = make_classification(n_samples=500, n_features=10, random_state=42)

        df_ref = pd.DataFrame(X[:250], columns=[f"f{i}" for i in range(10)])
        df_curr = pd.DataFrame(X[250:], columns=[f"f{i}" for i in range(10)])

        results = AdversarialValidation.compute(df_ref, df_curr, random_state=42, cv=3)

        assert "mean_roc_auc" in results
        assert "std_roc_auc" in results
        assert "drift_interpretation" in results
        assert "feature_importance" in results

        # AUC должен быть близок к 0.5 (нет дрейфа)
        assert 0.4 <= results["mean_roc_auc"] <= 0.6

    def test_with_drift(self):
        """Тест с дрейфом."""
        np.random.seed(42)
        X_ref, _ = make_classification(n_samples=250, n_features=10, random_state=42)
        X_curr, _ = make_classification(n_samples=250, n_features=10, random_state=100)

        df_ref = pd.DataFrame(X_ref, columns=[f"f{i}" for i in range(10)])
        df_curr = pd.DataFrame(X_curr, columns=[f"f{i}" for i in range(10)])

        results = AdversarialValidation.compute(df_ref, df_curr, random_state=42, cv=3)

        # AUC должен быть выше 0.5
        assert results["mean_roc_auc"] > 0.5

    def test_interpret_auc(self):
        """Тест интерпретации AUC."""
        assert AdversarialValidation._interpret_auc(0.52) == "no_drift"
        assert AdversarialValidation._interpret_auc(0.60) == "minor_drift"
        assert AdversarialValidation._interpret_auc(0.70) == "moderate_drift"
        assert AdversarialValidation._interpret_auc(0.80) == "major_drift"


class TestDriftDetector:
    """Тесты для DriftDetector."""

    @pytest.fixture
    def sample_data(self):
        """Создать данные для тестов."""
        np.random.seed(42)
        df_ref = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, size=200),
                "feature2": np.random.normal(0, 1, size=200),
                "feature3": np.random.normal(0, 1, size=200),
            }
        )
        df_curr = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, size=200),  # Нет дрейфа
                "feature2": np.random.normal(0.5, 1, size=200),  # Небольшой дрейф
                "feature3": np.random.normal(2, 1, size=200),  # Сильный дрейф
            }
        )
        return df_ref, df_curr

    def test_detect_all(self, sample_data):
        """Тест детекции всеми методами."""
        df_ref, df_curr = sample_data

        detector = DriftDetector(df_ref, df_curr)
        results = detector.detect_all(methods=["psi", "ks"])

        assert "psi" in results
        assert "ks" in results

        # Проверяем что результаты не пустые
        assert not results["psi"].empty
        assert not results["ks"].empty

    def test_get_drifted_features(self, sample_data):
        """Тест получения признаков с дрейфом."""
        df_ref, df_curr = sample_data

        detector = DriftDetector(df_ref, df_curr)
        drifted = detector.get_drifted_features(method="psi", threshold=0.1)

        # feature3 должен иметь дрейф
        assert "feature3" in drifted or len(drifted) > 0

    def test_summary(self, sample_data):
        """Тест сводки."""
        df_ref, df_curr = sample_data

        detector = DriftDetector(df_ref, df_curr)
        summary = detector.summary()

        assert "n_features" in summary
        assert "n_samples_reference" in summary
        assert "n_samples_current" in summary
        assert summary["n_features"] == 3
        assert summary["n_samples_reference"] == 200
        assert summary["n_samples_current"] == 200

    def test_specific_features(self):
        """Тест с конкретными признаками."""
        np.random.seed(42)
        df_ref = pd.DataFrame(
            {
                "f1": np.random.normal(0, 1, size=100),
                "f2": np.random.normal(0, 1, size=100),
                "f3": np.random.normal(0, 1, size=100),
            }
        )
        df_curr = pd.DataFrame(
            {
                "f1": np.random.normal(0, 1, size=100),
                "f2": np.random.normal(0, 1, size=100),
                "f3": np.random.normal(0, 1, size=100),
            }
        )

        detector = DriftDetector(df_ref, df_curr, feature_names=["f1", "f2"])
        results = detector.detect_all(methods=["psi"])

        # Должны быть только f1 и f2
        features = results["psi"]["feature"].tolist()
        assert "f1" in features
        assert "f2" in features
        assert len(features) == 2
