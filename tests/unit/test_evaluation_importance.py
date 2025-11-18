"""Тесты для анализа важности признаков."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.evaluation.feature_importance import (
    FeatureImportanceAnalyzer,
    PermutationImportance,
    TreeImportance,
)


class TestTreeImportance:
    """Тесты для tree-based importance."""

    @pytest.fixture
    def trained_model(self):
        """Создать обученную модель."""
        X, y = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        feature_names = [f"feature_{i}" for i in range(10)]
        return model, feature_names

    def test_get_feature_importance(self, trained_model):
        """Тест извлечения важности."""
        model, feature_names = trained_model

        df = TreeImportance.get_feature_importance(model, feature_names)

        assert not df.empty
        assert "feature" in df.columns
        assert "importance" in df.columns
        assert "importance_normalized" in df.columns
        assert len(df) == len(feature_names)

        # Проверяем что сумма нормализованных важностей = 1
        assert np.isclose(df["importance_normalized"].sum(), 1.0)

    def test_without_feature_names(self, trained_model):
        """Тест без имён признаков."""
        model, _ = trained_model

        df = TreeImportance.get_feature_importance(model)

        assert not df.empty
        assert all(df["feature"].str.startswith("feature_"))

    def test_sorted_by_importance(self, trained_model):
        """Тест сортировки по важности."""
        model, feature_names = trained_model

        df = TreeImportance.get_feature_importance(model, feature_names)

        # Проверяем что отсортировано по убыванию
        importances = df["importance"].values
        assert all(importances[i] >= importances[i + 1] for i in range(len(importances) - 1))


class TestPermutationImportance:
    """Тесты для permutation importance."""

    @pytest.fixture
    def trained_model_and_data(self):
        """Создать обученную модель и данные."""
        X, y = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X[:150], y[:150])
        X_test = X[150:]
        y_test = y[150:]
        feature_names = [f"feature_{i}" for i in range(10)]
        return model, X_test, y_test, feature_names

    def test_compute(self, trained_model_and_data):
        """Тест вычисления permutation importance."""
        model, X_test, y_test, feature_names = trained_model_and_data

        df = PermutationImportance.compute(model, X_test, y_test, feature_names, n_repeats=5, random_state=42)

        assert not df.empty
        assert "feature" in df.columns
        assert "importance_mean" in df.columns
        assert "importance_std" in df.columns
        assert len(df) == len(feature_names)

    def test_with_dataframe(self, trained_model_and_data):
        """Тест с DataFrame."""
        model, X_test, y_test, feature_names = trained_model_and_data

        X_test_df = pd.DataFrame(X_test, columns=feature_names)

        df = PermutationImportance.compute(model, X_test_df, y_test, n_repeats=5, random_state=42)

        assert not df.empty
        assert len(df) == len(feature_names)

    def test_sorted_by_importance(self, trained_model_and_data):
        """Тест сортировки."""
        model, X_test, y_test, feature_names = trained_model_and_data

        df = PermutationImportance.compute(model, X_test, y_test, feature_names, n_repeats=5, random_state=42)

        # Проверяем сортировку
        importances = df["importance_mean"].values
        assert all(importances[i] >= importances[i + 1] for i in range(len(importances) - 1))


class TestFeatureImportanceAnalyzer:
    """Тесты для комплексного анализатора."""

    @pytest.fixture
    def analyzer(self):
        """Создать анализатор."""
        X, y = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X[:150], y[:150])

        X_test = X[150:]
        y_test = y[150:]
        feature_names = [f"feature_{i}" for i in range(10)]

        analyzer = FeatureImportanceAnalyzer(model, X_test, y_test, feature_names)
        return analyzer

    def test_compute_all_importances(self, analyzer):
        """Тест вычисления всех важностей."""
        importances = analyzer.compute_all_importances(methods=["tree", "permutation"], n_repeats=5, random_state=42)

        assert "tree" in importances
        assert "permutation" in importances

        # Проверяем что результаты не пустые
        assert not importances["tree"].empty
        assert not importances["permutation"].empty

    def test_get_top_features(self, analyzer):
        """Тест получения топ признаков."""
        top_features = analyzer.get_top_features(n_top=5, method="tree")

        assert len(top_features) <= 5
        assert all(isinstance(f, str) for f in top_features)

    def test_compare_methods(self, analyzer):
        """Тест сравнения методов."""
        comparison = analyzer.compare_methods()

        assert not comparison.empty
        assert "feature" in comparison.columns
        assert "tree_rank" in comparison.columns or "permutation_rank" in comparison.columns

    def test_with_dataframe_input(self):
        """Тест с DataFrame на входе."""
        X, y = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=42)
        feature_names = [f"feature_{i}" for i in range(10)]
        X_df = pd.DataFrame(X, columns=feature_names)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_df[:150], y[:150])

        analyzer = FeatureImportanceAnalyzer(model, X_df[150:], y[150:])  # Без явных feature_names

        # Должен автоматически взять из DataFrame
        assert analyzer.feature_names == feature_names

        importances = analyzer.compute_all_importances(methods=["tree"])
        assert not importances["tree"].empty

    def test_feature_names_from_array(self):
        """Тест автоматической генерации имён для массивов."""
        X, y = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=42)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X[:150], y[:150])

        analyzer = FeatureImportanceAnalyzer(model, X[150:], y[150:])  # Без feature_names

        # Должны быть сгенерированы автоматически
        assert len(analyzer.feature_names) == 10
        assert all(name.startswith("feature_") for name in analyzer.feature_names)

    def test_with_non_tree_model(self):
        """Тест с нетри-моделью (должен работать только permutation)."""
        X, y = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=42)
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X[:150], y[:150])

        X_test = X[150:]
        y_test = y[150:]
        feature_names = [f"feature_{i}" for i in range(10)]

        analyzer = FeatureImportanceAnalyzer(model, X_test, y_test, feature_names)

        # Tree importance не должен работать
        importances = analyzer.compute_all_importances(methods=["tree", "permutation"])

        # Permutation должен быть
        assert "permutation" in importances
        assert not importances["permutation"].empty


class TestIntegration:
    """Интеграционные тесты."""

    def test_full_pipeline(self):
        """Тест полного пайплайна анализа важности."""
        # Генерируем данные с явной важностью
        np.random.seed(42)
        n_samples = 500
        n_features = 20

        X = np.random.randn(n_samples, n_features)
        # Делаем первые 5 признаков важными
        y = X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] * 1 + X[:, 3] * 0.5 + X[:, 4] * 0.3 + np.random.randn(n_samples) * 0.1
        y = (y > y.mean()).astype(int)

        # Разделяем на train/test
        X_train, X_test = X[:400], X[400:]
        y_train, y_test = y[:400], y[400:]

        # Обучаем модель
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # Анализируем важность
        feature_names = [f"f{i}" for i in range(n_features)]
        analyzer = FeatureImportanceAnalyzer(model, X_test, y_test, feature_names)

        # Получаем топ признаки
        top_features = analyzer.get_top_features(n_top=10, method="tree")

        # Первые признаки должны быть в топе (хотя бы некоторые)
        important_features = {"f0", "f1", "f2", "f3", "f4"}
        top_set = set(top_features)
        overlap = important_features & top_set

        # Хотя бы 2 из 5 важных признаков должны быть в топ-10
        assert len(overlap) >= 2, f"Expected important features in top, got {top_features}"
