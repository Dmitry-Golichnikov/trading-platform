"""Тесты для калибровки моделей."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.evaluation.calibration import (
    BetaCalibrator,
    CalibrationMetrics,
    ModelCalibrator,
    compare_calibration_methods,
)


class TestModelCalibrator:
    """Тесты для ModelCalibrator."""

    @pytest.fixture
    def calibration_data(self):
        """Создать данные для калибровки."""
        # Генерируем некалиброванные вероятности
        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=100)
        y_proba = np.random.beta(2, 5, size=100)  # Смещенные вероятности
        return y_true, y_proba

    def test_isotonic_calibration(self, calibration_data):
        """Тест isotonic калибровки."""
        y_true, y_proba = calibration_data

        calibrator = ModelCalibrator(method="isotonic")
        calibrator.fit(y_proba, y_true)
        y_calibrated = calibrator.transform(y_proba)

        assert len(y_calibrated) == len(y_proba)
        assert np.all(y_calibrated >= 0)
        assert np.all(y_calibrated <= 1)

    def test_platt_calibration(self, calibration_data):
        """Тест Platt scaling."""
        y_true, y_proba = calibration_data

        calibrator = ModelCalibrator(method="platt")
        calibrator.fit(y_proba, y_true)
        y_calibrated = calibrator.transform(y_proba)

        assert len(y_calibrated) == len(y_proba)
        assert np.all(y_calibrated >= 0)
        assert np.all(y_calibrated <= 1)

    def test_beta_calibration(self, calibration_data):
        """Тест beta калибровки."""
        y_true, y_proba = calibration_data

        calibrator = ModelCalibrator(method="beta")
        calibrator.fit(y_proba, y_true)
        y_calibrated = calibrator.transform(y_proba)

        assert len(y_calibrated) == len(y_proba)
        assert np.all(y_calibrated >= 0)
        assert np.all(y_calibrated <= 1)

    def test_fit_transform(self, calibration_data):
        """Тест fit_transform."""
        y_true, y_proba = calibration_data

        calibrator = ModelCalibrator(method="isotonic")
        y_calibrated = calibrator.fit_transform(y_proba, y_true)

        assert len(y_calibrated) == len(y_proba)

    def test_calibration_curve_data(self, calibration_data):
        """Тест получения данных для calibration curve."""
        y_true, y_proba = calibration_data

        calibrator = ModelCalibrator(method="isotonic", n_bins=5)
        calibrator.fit(y_proba, y_true)

        prob_true, prob_pred = calibrator.calibration_curve_data(y_proba, y_true)

        assert len(prob_true) <= 5
        assert len(prob_pred) <= 5
        assert np.all(prob_true >= 0)
        assert np.all(prob_true <= 1)

    def test_invalid_method(self):
        """Тест с неверным методом."""
        with pytest.raises(ValueError):
            ModelCalibrator(method="invalid")


class TestCalibrationMetrics:
    """Тесты для метрик калибровки."""

    @pytest.fixture
    def calibration_data(self):
        """Создать данные."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=100)
        y_proba = np.random.uniform(0.1, 0.9, size=100)
        return y_true, y_proba

    def test_brier_score(self, calibration_data):
        """Тест Brier score."""
        y_true, y_proba = calibration_data

        brier = CalibrationMetrics.brier_score(y_true, y_proba)

        assert brier >= 0
        assert brier <= 1

    def test_expected_calibration_error(self, calibration_data):
        """Тест ECE."""
        y_true, y_proba = calibration_data

        ece = CalibrationMetrics.expected_calibration_error(y_true, y_proba, n_bins=10)

        assert ece >= 0
        assert ece <= 1

    def test_maximum_calibration_error(self, calibration_data):
        """Тест MCE."""
        y_true, y_proba = calibration_data

        mce = CalibrationMetrics.maximum_calibration_error(y_true, y_proba, n_bins=10)

        assert mce >= 0
        assert mce <= 1

    def test_compute_all(self, calibration_data):
        """Тест вычисления всех метрик."""
        y_true, y_proba = calibration_data

        metrics = CalibrationMetrics.compute_all(y_true, y_proba, n_bins=10)

        assert "brier_score" in metrics
        assert "ece" in metrics
        assert "mce" in metrics

        for value in metrics.values():
            assert 0 <= value <= 1


class TestBetaCalibrator:
    """Тесты для BetaCalibrator."""

    def test_fit_transform(self):
        """Тест fit и transform."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=100)
        y_proba = np.random.beta(2, 5, size=100)

        calibrator = BetaCalibrator()
        calibrator.fit(y_proba, y_true)

        assert calibrator.params is not None
        assert "a" in calibrator.params
        assert "b" in calibrator.params
        assert "c" in calibrator.params

        y_calibrated = calibrator.transform(y_proba)

        assert len(y_calibrated) == len(y_proba)
        assert np.all(y_calibrated >= 0)
        assert np.all(y_calibrated <= 1)

    def test_transform_without_fit(self):
        """Тест transform без fit."""
        calibrator = BetaCalibrator()

        with pytest.raises(ValueError):
            calibrator.transform(np.array([0.5, 0.6, 0.7]))


class TestCompareCalibrationMethods:
    """Тесты для сравнения методов калибровки."""

    def test_compare_all_methods(self):
        """Тест сравнения всех методов."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=100)
        y_proba = np.random.beta(2, 5, size=100)

        results = compare_calibration_methods(y_proba, y_true)

        assert "uncalibrated" in results
        assert "isotonic" in results
        assert "platt" in results
        assert "beta" in results

        # Проверяем что у каждого метода есть метрики
        for method, metrics in results.items():
            if method == "uncalibrated" or not isinstance(metrics, dict) or "error" not in metrics:
                assert "brier_score" in metrics
                assert "ece" in metrics

    def test_compare_specific_methods(self):
        """Тест сравнения конкретных методов."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=100)
        y_proba = np.random.beta(2, 5, size=100)

        results = compare_calibration_methods(y_proba, y_true, methods=["isotonic", "platt"])

        assert "uncalibrated" in results
        assert "isotonic" in results
        assert "platt" in results
        assert "beta" not in results


class TestIntegration:
    """Интеграционные тесты калибровки."""

    def test_full_calibration_pipeline(self):
        """Тест полного пайплайна калибровки."""
        # Генерируем данные
        X, y = make_classification(n_samples=500, n_features=20, n_informative=15, random_state=42)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Обучаем модель
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Получаем вероятности
        y_proba = model.predict_proba(X_test)[:, 1]

        # Метрики до калибровки
        metrics_before = CalibrationMetrics.compute_all(y_test, y_proba)

        # Калибруем
        calibrator = ModelCalibrator(method="isotonic")
        calibrator.fit(y_proba, y_test)
        y_proba_calibrated = calibrator.transform(y_proba)

        # Метрики после калибровки
        metrics_after = CalibrationMetrics.compute_all(y_test, y_proba_calibrated)

        # Проверяем что калибровка изменила вероятности
        assert not np.allclose(y_proba, y_proba_calibrated)

        # Проверяем что метрики посчитались
        assert all(0 <= v <= 1 for v in metrics_before.values())
        assert all(0 <= v <= 1 for v in metrics_after.values())
