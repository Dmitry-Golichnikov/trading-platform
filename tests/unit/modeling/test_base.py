"""
Тесты для базовых интерфейсов моделирования.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.modeling.base import BaseModel, ClassifierMixin, RegressorMixin


class DummyModel(BaseModel):
    """Тестовая модель для проверки базового интерфейса."""

    def __init__(self, param1=10, param2="test"):
        super().__init__(param1=param1, param2=param2)
        self.coef_ = None

    def fit(self, X, y, X_val=None, y_val=None, **kwargs):
        """Простое 'обучение' - сохраняем средние."""
        self.coef_ = X.mean().values
        self._is_fitted = True
        self.metadata["n_samples_trained"] = len(X)
        self.metadata["n_features"] = X.shape[1]
        return self

    def predict(self, X):
        """Простое предсказание - сумма с coef."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        return (X.values * self.coef_).sum(axis=1)

    def save(self, path):
        """Сохранить модель."""
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        """Загрузить модель."""
        import joblib

        return joblib.load(path)

    @property
    def feature_importances_(self):
        """Feature importances (просто coef)."""
        return self.coef_


class TestBaseModel:
    """Тесты для BaseModel."""

    def test_initialization(self):
        """Тест инициализации."""
        model = DummyModel(param1=20, param2="custom")

        assert model.hyperparams == {"param1": 20, "param2": "custom"}
        assert not model.is_fitted
        assert "created_at" in model.metadata

    def test_fit_predict(self):
        """Тест обучения и предсказания."""
        # Создаём тестовые данные
        X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [2, 4, 6, 8, 10]})
        y = pd.Series([1, 2, 3, 4, 5])

        model = DummyModel()

        # До обучения модель не должна предсказывать
        with pytest.raises(ValueError):
            model.predict(X)

        # Обучаем
        model.fit(X, y)

        assert model.is_fitted
        assert model.metadata["n_samples_trained"] == 5
        assert model.metadata["n_features"] == 2

        # Предсказываем
        predictions = model.predict(X)

        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)

    def test_get_set_params(self):
        """Тест get/set params."""
        model = DummyModel(param1=10)

        # Get params
        params = model.get_params()
        assert params == {"param1": 10, "param2": "test"}

        # Set params
        model.set_params(param1=20)
        assert model.hyperparams["param1"] == 20

    def test_get_metadata(self):
        """Тест получения метаданных."""
        model = DummyModel()
        metadata = model.get_metadata()

        assert "created_at" in metadata
        assert metadata["training_time"] is None

    def test_save_load(self, tmp_path):
        """Тест сохранения и загрузки."""
        # Создаём и обучаем модель
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([1, 2, 3])

        model = DummyModel(param1=42)
        model.fit(X, y)

        # Сохраняем
        model_path = tmp_path / "test_model.pkl"
        model.save(model_path)

        assert model_path.exists()

        # Загружаем
        loaded_model = DummyModel.load(model_path)

        assert loaded_model.is_fitted
        assert loaded_model.hyperparams["param1"] == 42
        assert np.allclose(loaded_model.coef_, model.coef_)

        # Проверяем предсказания
        pred_original = model.predict(X)
        pred_loaded = loaded_model.predict(X)

        assert np.allclose(pred_original, pred_loaded)

    def test_feature_importances(self):
        """Тест feature importances."""
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([1, 2, 3])

        model = DummyModel()

        # До обучения None
        assert model.feature_importances_ is None

        # После обучения
        model.fit(X, y)
        importances = model.feature_importances_

        assert importances is not None
        assert len(importances) == 2

    def test_repr(self):
        """Тест строкового представления."""
        model = DummyModel()

        assert "not fitted" in str(model)

        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([1, 2, 3])
        model.fit(X, y)

        assert "fitted" in str(model)


class TestMixins:
    """Тесты для Mixin классов."""

    def test_classifier_mixin(self):
        """Тест ClassifierMixin."""

        class DummyClassifier(BaseModel, ClassifierMixin):
            def fit(self, X, y, **kwargs):
                self._classes = y.unique()
                self._is_fitted = True
                return self

            def predict(self, X):
                return np.zeros(len(X))

            def save(self, path):
                pass

            @classmethod
            def load(cls, path):
                pass

        model = DummyClassifier()
        y = pd.Series([0, 1, 0, 1, 1])

        assert model.classes_ is None

        model.fit(pd.DataFrame({"a": [1, 2, 3, 4, 5]}), y)

        assert model.classes_ is not None
        assert len(model.classes_) == 2

    def test_regressor_mixin(self):
        """Тест RegressorMixin."""

        class DummyRegressor(BaseModel, RegressorMixin):
            def fit(self, X, y, **kwargs):
                self._is_fitted = True
                return self

            def predict(self, X):
                return np.zeros(len(X))

            def save(self, path):
                pass

            @classmethod
            def load(cls, path):
                pass

        model = DummyRegressor()

        # predict_quantiles по умолчанию не реализован
        with pytest.raises(NotImplementedError):
            model.predict_quantiles(pd.DataFrame({"a": [1, 2, 3]}), [0.25, 0.75])
