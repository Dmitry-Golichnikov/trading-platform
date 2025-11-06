"""
Unit тесты для всех моделей.

Проверяет базовый функционал каждой модели:
- Инициализация
- Обучение
- Предсказание
- Сохранение/загрузка
- Feature importances
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.modeling.registry import ModelRegistry

# Список всех моделей для тестирования
TREE_BASED_MODELS = [
    "lightgbm",
    "xgboost",
    "catboost",
    "random_forest",
    "extra_trees",
]

LINEAR_MODELS = [
    "logistic_regression",
    "elasticnet",
]

TABULAR_NN_MODELS = [
    "tabnet",
    "ft_transformer",
    "node",
]

ALL_MODELS = TREE_BASED_MODELS + LINEAR_MODELS + TABULAR_NN_MODELS


# Фикстуры для тестовых данных
@pytest.fixture
def classification_data():
    """Генерация тестовых данных для классификации."""
    np.random.seed(42)
    n_samples = 200
    n_features = 10

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.randint(0, 2, n_samples), name="target")

    # Разделение на train/val
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    return X_train, y_train, X_val, y_val


@pytest.fixture
def regression_data():
    """Генерация тестовых данных для регрессии."""
    np.random.seed(42)
    n_samples = 200
    n_features = 10

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.randn(n_samples), name="target")

    # Разделение на train/val
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    return X_train, y_train, X_val, y_val


# Тесты для tree-based моделей
@pytest.mark.parametrize("model_name", TREE_BASED_MODELS)
def test_tree_based_classification(model_name, classification_data):
    """Тест классификации для tree-based моделей."""
    X_train, y_train, X_val, y_val = classification_data

    # Создаём модель
    model = ModelRegistry.create(
        model_name,
        task="classification",
        n_estimators=10 if model_name in ["random_forest", "extra_trees"] else 50,
        verbose=-1,
    )

    # Обучение
    model.fit(X_train, y_train, X_val, y_val)

    assert model.is_fitted
    assert model.metadata["n_samples_trained"] == len(X_train)

    # Предсказание
    predictions = model.predict(X_val)
    assert len(predictions) == len(X_val)
    assert set(predictions).issubset({0, 1})

    # Вероятности
    probas = model.predict_proba(X_val)
    assert len(probas) == len(X_val)
    assert np.all((probas >= 0) & (probas <= 1))

    # Feature importances
    importances = model.feature_importances_
    assert importances is not None
    assert len(importances) == X_train.shape[1]


@pytest.mark.parametrize("model_name", TREE_BASED_MODELS)
def test_tree_based_regression(model_name, regression_data):
    """Тест регрессии для tree-based моделей."""
    X_train, y_train, X_val, y_val = regression_data

    # Создаём модель
    model = ModelRegistry.create(
        model_name,
        task="regression",
        n_estimators=10 if model_name in ["random_forest", "extra_trees"] else 50,
        verbose=-1,
    )

    # Обучение
    model.fit(X_train, y_train, X_val, y_val)

    assert model.is_fitted

    # Предсказание
    predictions = model.predict(X_val)
    assert len(predictions) == len(X_val)
    assert predictions.dtype == np.float64 or predictions.dtype == np.float32


# Тесты для linear моделей
def test_logistic_regression_classification(classification_data):
    """Тест LogisticRegression для классификации."""
    X_train, y_train, X_val, y_val = classification_data

    # Создаём модель
    model = ModelRegistry.create("logistic_regression", max_iter=100)

    # Обучение
    model.fit(X_train, y_train)

    assert model.is_fitted

    # Предсказание
    predictions = model.predict(X_val)
    assert len(predictions) == len(X_val)

    # Вероятности
    probas = model.predict_proba(X_val)
    assert len(probas) == len(X_val)

    # Feature importances (coefficients)
    importances = model.feature_importances_
    assert importances is not None
    assert len(importances) == X_train.shape[1]


def test_elasticnet_regression(regression_data):
    """Тест ElasticNet для регрессии."""
    X_train, y_train, X_val, y_val = regression_data

    # Создаём модель
    model = ModelRegistry.create("elasticnet", alpha=0.1, l1_ratio=0.5)

    # Обучение
    model.fit(X_train, y_train)

    assert model.is_fitted

    # Предсказание
    predictions = model.predict(X_val)
    assert len(predictions) == len(X_val)

    # Feature importances (coefficients)
    importances = model.feature_importances_
    assert importances is not None


# Тест сохранения/загрузки
@pytest.mark.parametrize("model_name", ["lightgbm", "random_forest", "logistic_regression"])
def test_model_save_load(model_name, classification_data):
    """Тест сохранения и загрузки модели."""
    X_train, y_train, X_val, y_val = classification_data

    # Создаём и обучаем модель
    if model_name == "logistic_regression":
        model = ModelRegistry.create(model_name)
        model.fit(X_train, y_train)
    else:
        model = ModelRegistry.create(
            model_name,
            task="classification",
            n_estimators=10 if model_name == "random_forest" else 50,
            verbose=-1,
        )
        model.fit(X_train, y_train, X_val, y_val)

    # Предсказания до сохранения
    predictions_before = model.predict(X_val)

    # Сохранение
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "model"
        model.save(save_path)

        # Загрузка
        model_class = type(model)
        loaded_model = model_class.load(save_path)

        assert loaded_model.is_fitted
        assert loaded_model._feature_names == model._feature_names

        # Предсказания после загрузки
        predictions_after = loaded_model.predict(X_val)

        # Проверяем что предсказания совпадают
        np.testing.assert_array_equal(predictions_before, predictions_after)


# Тест registry
def test_model_registry():
    """Тест реестра моделей."""
    # Проверяем что все модели зарегистрированы
    registered_models = ModelRegistry.list_models()

    for model_name in ALL_MODELS:
        assert model_name in registered_models

    # Проверяем создание модели
    for model_name in TREE_BASED_MODELS + LINEAR_MODELS:
        try:
            model = ModelRegistry.create(model_name, task="classification")
            assert model is not None
        except ImportError:
            # Пропускаем если библиотека не установлена
            pytest.skip(f"{model_name} library not installed")


# Тест метаданных
def test_model_metadata(classification_data):
    """Тест метаданных модели."""
    X_train, y_train, X_val, y_val = classification_data

    model = ModelRegistry.create("random_forest", task="classification", n_estimators=10)
    model.fit(X_train, y_train)

    metadata = model.get_metadata()

    assert "training_time" in metadata
    assert "n_samples_trained" in metadata
    assert "n_features" in metadata
    assert metadata["n_samples_trained"] == len(X_train)
    assert metadata["n_features"] == X_train.shape[1]

    # Проверяем гиперпараметры
    params = model.get_params()
    assert "n_estimators" in params
    assert params["n_estimators"] == 10


# Тест ошибок
def test_model_errors(classification_data):
    """Тест обработки ошибок."""
    X_train, y_train, X_val, y_val = classification_data

    model = ModelRegistry.create("random_forest", task="classification", n_estimators=10)

    # Предсказание до обучения должно вызвать ошибку
    with pytest.raises(ValueError, match="не обучена"):
        model.predict(X_val)

    # Сохранение до обучения должно вызвать ошибку
    with pytest.raises(ValueError, match="не обучена"):
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(Path(tmpdir) / "model")

    # Обучаем модель
    model.fit(X_train, y_train)

    # Предсказание с неправильными признаками
    X_wrong = X_val.copy()
    X_wrong = X_wrong.drop(columns=X_wrong.columns[0])

    with pytest.raises(ValueError, match="Отсутствуют признаки"):
        model.predict(X_wrong)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
