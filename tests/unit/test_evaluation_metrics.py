"""Тесты для модуля метрик."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.evaluation.metrics import (
    ClassificationMetrics,
    MetricsCalculator,
    RegressionMetrics,
)


class TestClassificationMetrics:
    """Тесты для метрик классификации."""

    @pytest.fixture
    def binary_data(self):
        """Создать бинарные данные."""
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.3, 0.85, 0.6])
        return y_true, y_pred, y_proba

    def test_accuracy(self, binary_data):
        """Тест accuracy."""
        y_true, y_pred, _ = binary_data
        acc = ClassificationMetrics.accuracy(y_true, y_pred)
        assert 0 <= acc <= 1
        assert acc == 0.75  # 6 правильных из 8

    def test_precision(self, binary_data):
        """Тест precision."""
        y_true, y_pred, _ = binary_data
        prec = ClassificationMetrics.precision(y_true, y_pred)
        assert 0 <= prec <= 1

    def test_recall(self, binary_data):
        """Тест recall."""
        y_true, y_pred, _ = binary_data
        rec = ClassificationMetrics.recall(y_true, y_pred)
        assert 0 <= rec <= 1

    def test_f1(self, binary_data):
        """Тест F1-score."""
        y_true, y_pred, _ = binary_data
        f1 = ClassificationMetrics.f1(y_true, y_pred)
        assert 0 <= f1 <= 1

    def test_roc_auc(self, binary_data):
        """Тест ROC-AUC."""
        y_true, _, y_proba = binary_data
        auc = ClassificationMetrics.roc_auc(y_true, y_proba)
        assert 0 <= auc <= 1

    def test_pr_auc(self, binary_data):
        """Тест PR-AUC."""
        y_true, _, y_proba = binary_data
        pr_auc = ClassificationMetrics.pr_auc(y_true, y_proba)
        assert 0 <= pr_auc <= 1

    def test_mcc(self, binary_data):
        """Тест Matthews Correlation Coefficient."""
        y_true, y_pred, _ = binary_data
        mcc = ClassificationMetrics.mcc(y_true, y_pred)
        assert -1 <= mcc <= 1

    def test_confusion_matrix(self, binary_data):
        """Тест confusion matrix."""
        y_true, y_pred, _ = binary_data
        cm_dict = ClassificationMetrics.confusion_matrix_dict(y_true, y_pred)

        assert "matrix" in cm_dict
        assert "TP" in cm_dict
        assert "TN" in cm_dict
        assert "FP" in cm_dict
        assert "FN" in cm_dict

        # Проверяем что сумма равна количеству примеров
        total = cm_dict["TP"] + cm_dict["TN"] + cm_dict["FP"] + cm_dict["FN"]
        assert total == len(y_true)

    def test_compute_all(self, binary_data):
        """Тест вычисления всех метрик."""
        y_true, y_pred, y_proba = binary_data
        metrics = ClassificationMetrics.compute_all(y_true, y_pred, y_proba)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
        assert "mcc" in metrics


class TestRegressionMetrics:
    """Тесты для метрик регрессии."""

    @pytest.fixture
    def regression_data(self):
        """Создать регрессионные данные."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
        return y_true, y_pred

    def test_mse(self, regression_data):
        """Тест MSE."""
        y_true, y_pred = regression_data
        mse = RegressionMetrics.mse(y_true, y_pred)
        assert mse >= 0

    def test_rmse(self, regression_data):
        """Тест RMSE."""
        y_true, y_pred = regression_data
        rmse = RegressionMetrics.rmse(y_true, y_pred)
        assert rmse >= 0

    def test_mae(self, regression_data):
        """Тест MAE."""
        y_true, y_pred = regression_data
        mae = RegressionMetrics.mae(y_true, y_pred)
        assert mae >= 0

    def test_mape(self, regression_data):
        """Тест MAPE."""
        y_true, y_pred = regression_data
        mape = RegressionMetrics.mape(y_true, y_pred)
        assert mape >= 0

    def test_r2(self, regression_data):
        """Тест R²."""
        y_true, y_pred = regression_data
        r2 = RegressionMetrics.r2(y_true, y_pred)
        assert r2 <= 1

    def test_adjusted_r2(self, regression_data):
        """Тест Adjusted R²."""
        y_true, y_pred = regression_data
        adj_r2 = RegressionMetrics.adjusted_r2(y_true, y_pred, n_features=5)
        assert adj_r2 <= 1

    def test_quantile_loss(self, regression_data):
        """Тест quantile loss."""
        y_true, y_pred = regression_data
        ql = RegressionMetrics.quantile_loss(y_true, y_pred, quantile=0.5)
        assert ql >= 0

    def test_compute_all(self, regression_data):
        """Тест вычисления всех метрик."""
        y_true, y_pred = regression_data
        metrics = RegressionMetrics.compute_all(y_true, y_pred, n_features=5)

        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "mape" in metrics
        assert "r2" in metrics
        assert "adjusted_r2" in metrics


class TestMetricsCalculator:
    """Тесты для универсального калькулятора."""

    def test_compute_classification_metrics(self):
        """Тест вычисления метрик классификации."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X[:80], y[:80])

        y_pred = model.predict(X[80:])
        y_proba = model.predict_proba(X[80:])

        metrics = MetricsCalculator.compute_metrics(y[80:], y_pred, task_type="classification", y_proba=y_proba)

        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics

    def test_compute_regression_metrics(self):
        """Тест вычисления метрик регрессии."""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X[:80], y[:80])

        y_pred = model.predict(X[80:])

        metrics = MetricsCalculator.compute_metrics(y[80:], y_pred, task_type="regression", n_features=10)

        assert "mse" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics

    def test_metrics_to_dataframe(self):
        """Тест преобразования метрик в DataFrame."""
        metrics = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "matrix": np.array([[10, 2], [3, 15]]),  # Не скаляр
        }

        df = MetricsCalculator.metrics_to_dataframe(metrics)

        assert len(df) == 3  # Только скалярные метрики
        assert "accuracy" in df.index
        assert "precision" in df.index
        assert "recall" in df.index
        assert "matrix" not in df.index
