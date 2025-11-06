"""
Тесты для Sanity Checks.
"""

import numpy as np
import pandas as pd
import pytest

from src.modeling.sanity_checks import ModelSanityChecker, SanityCheckResult


class TestSanityCheckResult:
    """Тесты для SanityCheckResult."""

    def test_initialization(self):
        """Тест инициализации."""
        result = SanityCheckResult()

        assert result.passed is True
        assert len(result.warnings) == 0
        assert len(result.errors) == 0
        assert len(result.info) == 0

    def test_summary(self):
        """Тест генерации сводки."""
        result = SanityCheckResult()
        result.warnings.append("Warning 1")
        result.errors.append("Error 1")
        result.info["key"] = "value"
        result.passed = False

        summary = result.summary()

        assert "FAILED" in summary or "✗" in summary
        assert "Warning 1" in summary
        assert "Error 1" in summary


class TestModelSanityChecker:
    """Тесты для ModelSanityChecker."""

    @pytest.fixture
    def good_data(self):
        """Создаём хорошие тестовые данные."""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
            }
        )
        y = pd.Series(np.random.choice([0, 1], 100, p=[0.6, 0.4]))
        return X, y

    def test_check_target_distribution_good(self, good_data):
        """Тест проверки нормального таргета."""
        X, y = good_data

        checker = ModelSanityChecker(verbose=False)
        result = checker.check_target_distribution(y)

        assert result.passed is True
        assert "target_samples" in result.info
        assert result.info["target_samples"] == 100

    def test_check_target_distribution_constant(self):
        """Тест проверки константного таргета."""
        y = pd.Series([1, 1, 1, 1, 1])

        checker = ModelSanityChecker(verbose=False)
        result = checker.check_target_distribution(y)

        assert result.passed is False
        assert any("константный" in err.lower() or "constant" in err.lower() for err in result.errors)

    def test_check_target_distribution_with_missing(self):
        """Тест проверки таргета с пропусками."""
        y = pd.Series([1, 0, np.nan, 1, 0])

        checker = ModelSanityChecker(verbose=False)
        result = checker.check_target_distribution(y)

        assert result.passed is False
        assert any("пропуск" in err.lower() or "missing" in err.lower() for err in result.errors)

    def test_check_target_distribution_imbalanced(self):
        """Тест проверки несбалансированного таргета."""
        y = pd.Series([0] * 95 + [1] * 5)  # 95/5 дисбаланс

        checker = ModelSanityChecker(verbose=False)
        result = checker.check_target_distribution(y)

        # Должно быть предупреждение о дисбалансе
        assert len(result.warnings) > 0

    def test_check_feature_quality_good(self, good_data):
        """Тест проверки хороших признаков."""
        X, y = good_data

        checker = ModelSanityChecker(verbose=False)
        result = checker.check_feature_quality(X)

        assert result.passed is True
        assert result.info["n_features"] == 3

    def test_check_feature_quality_constant(self):
        """Тест проверки константных признаков."""
        X = pd.DataFrame({"feature1": [1, 1, 1, 1, 1], "feature2": [2, 3, 4, 5, 6]})  # Нормальный

        checker = ModelSanityChecker(verbose=False)
        result = checker.check_feature_quality(X)

        assert len(result.warnings) > 0
        assert "constant_features" in result.info

    def test_check_feature_quality_inf_values(self):
        """Тест проверки бесконечных значений."""
        X = pd.DataFrame({"feature1": [1, 2, np.inf, 4, 5], "feature2": [1, 2, 3, 4, 5]})

        checker = ModelSanityChecker(verbose=False)
        result = checker.check_feature_quality(X)

        assert result.passed is False
        assert any("бесконечн" in err.lower() or "inf" in err.lower() for err in result.errors)

    def test_check_data_leakage_duplicates(self, good_data):
        """Тест проверки дубликатов между train и val."""
        X, y = good_data

        # Создаём train и val с дубликатами
        X_train = X.iloc[:50].copy()
        y_train = y.iloc[:50].copy()

        # Val содержит дубликаты из train
        X_val = pd.concat([X.iloc[45:50], X.iloc[50:60]]).reset_index(drop=True)
        y_val = pd.concat([y.iloc[45:50], y.iloc[50:60]]).reset_index(drop=True)

        checker = ModelSanityChecker(verbose=False)
        result = checker.check_data_leakage(X_train, X_val, y_train, y_val)

        # Должно быть предупреждение о дубликатах
        assert len(result.warnings) > 0

    def test_check_all(self, good_data):
        """Тест полной проверки."""
        X, y = good_data

        X_train = X.iloc[:70]
        y_train = y.iloc[:70]
        X_val = X.iloc[70:85]
        y_val = y.iloc[70:85]

        checker = ModelSanityChecker(verbose=False)
        result = checker.check_all(X_train, y_train, X_val, y_val)

        assert result.passed is True or len(result.errors) == 0
        assert "target_samples" in result.info
        assert "n_features" in result.info
