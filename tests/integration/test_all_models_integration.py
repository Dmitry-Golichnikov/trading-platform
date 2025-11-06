"""
Интеграционный тест для всех моделей.

Проверяет работу моделей в реальном сценарии:
- Загрузка данных
- Обучение всех моделей
- Сравнение производительности
- Проверка корректности работы
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

from src.modeling.registry import ModelRegistry


@pytest.fixture
def classification_dataset():
    """Генерация реалистичного датасета для классификации."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42,
    )

    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="target")

    # Разделение на train/val/test
    train_idx = int(0.7 * len(X))
    val_idx = int(0.85 * len(X))

    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


@pytest.fixture
def regression_dataset():
    """Генерация реалистичного датасета для регрессии."""
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        random_state=42,
    )

    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="target")

    # Разделение на train/val/test
    train_idx = int(0.7 * len(X))
    val_idx = int(0.85 * len(X))

    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def test_all_classification_models(classification_dataset):
    """
    Тест всех моделей на задаче классификации.

    Проверяет:
    - Все модели обучаются без ошибок
    - Все модели дают разумные предсказания
    - Метрики качества > baseline
    """
    data = classification_dataset

    models_to_test = [
        ("lightgbm", {"task": "classification", "n_estimators": 100, "verbose": -1}),
        ("xgboost", {"task": "classification", "n_estimators": 100, "verbose": 0}),
        ("catboost", {"task": "classification", "iterations": 100, "verbose": False}),
        ("random_forest", {"task": "classification", "n_estimators": 50, "verbose": 0}),
        ("extra_trees", {"task": "classification", "n_estimators": 50, "verbose": 0}),
        ("logistic_regression", {"max_iter": 1000}),
    ]

    results = {}

    for model_name, params in models_to_test:
        try:
            print(f"\nТестирование модели: {model_name}")

            # Создаём модель
            model = ModelRegistry.create(model_name, **params)

            # Обучение
            if model_name == "logistic_regression":
                model.fit(data["X_train"], data["y_train"])
            else:
                model.fit(data["X_train"], data["y_train"], data["X_val"], data["y_val"])

            # Предсказания
            y_pred = model.predict(data["X_test"])
            y_proba = model.predict_proba(data["X_test"])

            # Метрики
            accuracy = accuracy_score(data["y_test"], y_pred)
            f1 = f1_score(data["y_test"], y_pred)

            results[model_name] = {
                "accuracy": accuracy,
                "f1": f1,
                "training_time": model.metadata.get("training_time", 0),
            }

            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  Training time: {results[model_name]['training_time']:.2f}s")

            # Проверки
            assert model.is_fitted
            assert accuracy > 0.5, f"{model_name}: accuracy слишком низкая"
            assert len(y_pred) == len(data["y_test"])
            assert len(y_proba) == len(data["y_test"])

        except ImportError as e:
            pytest.skip(f"{model_name} library not installed: {e}")

    # Проверяем что хотя бы одна модель обучилась
    assert len(results) > 0, "Ни одна модель не обучилась"

    # Выводим лучшую модель
    best_model = max(results.items(), key=lambda x: x[1]["accuracy"])
    print(f"\n\nЛучшая модель: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")


def test_all_regression_models(regression_dataset):
    """
    Тест всех моделей на задаче регрессии.

    Проверяет:
    - Все модели обучаются без ошибок
    - Все модели дают разумные предсказания
    - Метрики качества > baseline
    """
    data = regression_dataset

    models_to_test = [
        ("lightgbm", {"task": "regression", "n_estimators": 100, "verbose": -1}),
        ("xgboost", {"task": "regression", "n_estimators": 100, "verbose": 0}),
        ("catboost", {"task": "regression", "iterations": 100, "verbose": False}),
        ("random_forest", {"task": "regression", "n_estimators": 50, "verbose": 0}),
        ("extra_trees", {"task": "regression", "n_estimators": 50, "verbose": 0}),
        ("elasticnet", {"alpha": 0.1, "l1_ratio": 0.5}),
    ]

    results = {}

    for model_name, params in models_to_test:
        try:
            print(f"\nТестирование модели: {model_name}")

            # Создаём модель
            model = ModelRegistry.create(model_name, **params)

            # Обучение
            if model_name == "elasticnet":
                model.fit(data["X_train"], data["y_train"])
            else:
                model.fit(data["X_train"], data["y_train"], data["X_val"], data["y_val"])

            # Предсказания
            y_pred = model.predict(data["X_test"])

            # Метрики
            rmse = np.sqrt(mean_squared_error(data["y_test"], y_pred))
            r2 = r2_score(data["y_test"], y_pred)

            results[model_name] = {
                "rmse": rmse,
                "r2": r2,
                "training_time": model.metadata.get("training_time", 0),
            }

            print(f"  RMSE: {rmse:.4f}")
            print(f"  R2 Score: {r2:.4f}")
            print(f"  Training time: {results[model_name]['training_time']:.2f}s")

            # Проверки
            assert model.is_fitted
            assert r2 > 0, f"{model_name}: R2 слишком низкий"
            assert len(y_pred) == len(data["y_test"])

        except ImportError as e:
            pytest.skip(f"{model_name} library not installed: {e}")

    # Проверяем что хотя бы одна модель обучилась
    assert len(results) > 0, "Ни одна модель не обучилась"

    # Выводим лучшую модель
    best_model = max(results.items(), key=lambda x: x[1]["r2"])
    print(f"\n\nЛучшая модель: {best_model[0]} (R2: {best_model[1]['r2']:.4f})")


def test_model_comparison_and_selection(classification_dataset):
    """
    Тест сравнения моделей и выбора лучшей.

    Симулирует реальный сценарий выбора модели для задачи.
    """
    data = classification_dataset

    # Список моделей для сравнения
    models_config = {
        "lightgbm": {"task": "classification", "n_estimators": 50, "verbose": -1},
        "random_forest": {"task": "classification", "n_estimators": 50, "verbose": 0},
        "logistic_regression": {"max_iter": 1000},
    }

    trained_models = {}
    results = {}

    print("\n=== Сравнение моделей ===\n")

    for model_name, params in models_config.items():
        try:
            # Создаём и обучаем модель
            model = ModelRegistry.create(model_name, **params)

            if model_name == "logistic_regression":
                model.fit(data["X_train"], data["y_train"])
            else:
                model.fit(data["X_train"], data["y_train"], data["X_val"], data["y_val"])

            trained_models[model_name] = model

            # Оценка на validation set
            y_pred_val = model.predict(data["X_val"])

            accuracy_val = accuracy_score(data["y_val"], y_pred_val)
            f1_val = f1_score(data["y_val"], y_pred_val)

            results[model_name] = {
                "accuracy_val": accuracy_val,
                "f1_val": f1_val,
                "training_time": model.metadata.get("training_time", 0),
            }

            print(f"{model_name}:")
            print(f"  Val Accuracy: {accuracy_val:.4f}")
            print(f"  Val F1: {f1_val:.4f}")
            print(f"  Training time: {results[model_name]['training_time']:.2f}s")
            print()

        except ImportError:
            continue

    # Выбираем лучшую модель
    if results:
        best_model_name = max(results.items(), key=lambda x: x[1]["f1_val"])[0]
        best_model = trained_models[best_model_name]

        print(f"Выбрана лучшая модель: {best_model_name}")

        # Тестируем лучшую модель на test set
        y_pred_test = best_model.predict(data["X_test"])
        accuracy_test = accuracy_score(data["y_test"], y_pred_test)
        f1_test = f1_score(data["y_test"], y_pred_test)

        print("\nРезультаты на test set:")
        print(f"  Accuracy: {accuracy_test:.4f}")
        print(f"  F1 Score: {f1_test:.4f}")

        # Проверяем что модель работает хорошо
        assert accuracy_test > 0.6, "Accuracy на test set слишком низкая"

        # Сохраняем и загружаем лучшую модель
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "best_model"
            best_model.save(save_path)

            # Загружаем
            model_class = type(best_model)
            loaded_model = model_class.load(save_path)

            # Проверяем что загруженная модель даёт те же предсказания
            y_pred_loaded = loaded_model.predict(data["X_test"])
            np.testing.assert_array_equal(y_pred_test, y_pred_loaded)

            print("\nМодель успешно сохранена и загружена!")


def test_feature_importances(classification_dataset):
    """
    Тест feature importances для всех моделей.

    Проверяет что модели корректно вычисляют важности признаков.
    """
    data = classification_dataset

    models_to_test = [
        "lightgbm",
        "random_forest",
        "logistic_regression",
    ]

    for model_name in models_to_test:
        try:
            print(f"\n{model_name} feature importances:")

            # Создаём и обучаем модель
            if model_name == "logistic_regression":
                model = ModelRegistry.create(model_name, max_iter=1000)
                model.fit(data["X_train"], data["y_train"])
            else:
                model = ModelRegistry.create(model_name, task="classification", n_estimators=50, verbose=-1)
                model.fit(data["X_train"], data["y_train"], data["X_val"], data["y_val"])

            # Получаем важности
            importances = model.feature_importances_
            assert importances is not None
            assert len(importances) == data["X_train"].shape[1]

            # Проверяем что есть вариация в важностях
            assert importances.std() > 0, "Все важности одинаковые"

            # Топ-5 признаков
            top_features_idx = np.argsort(importances)[-5:][::-1]
            print(f"  Топ-5 признаков: {[data['X_train'].columns[i] for i in top_features_idx]}")

        except ImportError:
            pytest.skip(f"{model_name} library not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
