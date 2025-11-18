"""
Тесты для ModelRegistry.
"""

import numpy as np
import pytest

from src.modeling.base import BaseModel
from src.modeling.registry import ModelRegistry


class DummyModelA(BaseModel):
    """Тестовая модель A."""

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


class DummyModelB(BaseModel):
    """Тестовая модель B."""

    def fit(self, X, y, **kwargs):
        self._is_fitted = True
        return self

    def predict(self, X):
        return np.ones(len(X))

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        pass


class TestModelRegistry:
    """Тесты для ModelRegistry."""

    def setup_method(self):
        """Очистка реестра перед каждым тестом."""
        ModelRegistry.clear()

    def test_register_decorator(self):
        """Тест регистрации модели через декоратор."""

        @ModelRegistry.register("test_model_a")
        class TestModel(DummyModelA):
            pass

        assert ModelRegistry.is_registered("test_model_a")
        assert "test_model_a" in ModelRegistry.list_models()

    def test_register_with_metadata(self):
        """Тест регистрации с метаданными."""

        @ModelRegistry.register("test_model_b", description="Test Model B", tags=["test", "dummy"])
        class TestModel(DummyModelB):
            pass

        metadata = ModelRegistry.get_metadata("test_model_b")

        assert metadata["description"] == "Test Model B"
        assert "test" in metadata["tags"]
        assert "dummy" in metadata["tags"]

    def test_create_model(self):
        """Тест создания модели."""

        @ModelRegistry.register("dummy_a")
        class TestModel(DummyModelA):
            def __init__(self, param1=10):
                super().__init__(param1=param1)

        # Создаём модель
        model = ModelRegistry.create("dummy_a", param1=20)

        assert isinstance(model, TestModel)
        assert model.hyperparams["param1"] == 20

    def test_create_unknown_model(self):
        """Тест создания незарегистрированной модели."""

        with pytest.raises(ValueError) as exc_info:
            ModelRegistry.create("nonexistent_model")

        assert "Unknown model" in str(exc_info.value) or "не зарегистрирована" in str(exc_info.value)

    def test_duplicate_registration(self):
        """Тест дублирования регистрации."""

        @ModelRegistry.register("duplicate")
        class ModelA(DummyModelA):
            pass

        # Попытка зарегистрировать с тем же именем
        with pytest.raises(ValueError) as exc_info:

            @ModelRegistry.register("duplicate")
            class ModelB(DummyModelB):
                pass

        assert "уже зарегистрирована" in str(exc_info.value)

    def test_list_models(self):
        """Тест получения списка моделей."""

        @ModelRegistry.register("model1", tags=["tree"])
        class Model1(DummyModelA):
            pass

        @ModelRegistry.register("model2", tags=["linear"])
        class Model2(DummyModelB):
            pass

        @ModelRegistry.register("model3", tags=["tree", "ensemble"])
        class Model3(DummyModelA):
            pass

        # Все модели
        all_models = ModelRegistry.list_models()
        assert len(all_models) == 3
        assert "model1" in all_models
        assert "model2" in all_models
        assert "model3" in all_models

        # Фильтрация по тэгам
        tree_models = ModelRegistry.list_models(tags=["tree"])
        assert len(tree_models) == 2
        assert "model1" in tree_models
        assert "model3" in tree_models

    def test_unregister(self):
        """Тест удаления модели."""

        @ModelRegistry.register("to_remove")
        class TestModel(DummyModelA):
            pass

        assert ModelRegistry.is_registered("to_remove")

        ModelRegistry.unregister("to_remove")

        assert not ModelRegistry.is_registered("to_remove")

    def test_unregister_nonexistent(self):
        """Тест удаления несуществующей модели."""

        with pytest.raises(ValueError):
            ModelRegistry.unregister("nonexistent")

    def test_get_all(self):
        """Тест получения всех моделей."""

        @ModelRegistry.register("model_a")
        class ModelA(DummyModelA):
            pass

        @ModelRegistry.register("model_b")
        class ModelB(DummyModelB):
            pass

        all_models = ModelRegistry.get_all()

        assert len(all_models) == 2
        assert "model_a" in all_models
        assert "model_b" in all_models
        assert all_models["model_a"] == ModelA
        assert all_models["model_b"] == ModelB

    def test_summary(self):
        """Тест генерации сводки."""

        @ModelRegistry.register("test_summary")
        class TestModel(DummyModelA):
            """Test model for summary."""

            pass

        summary = ModelRegistry.summary()

        assert "test_summary" in summary
        assert "Test model for summary" in summary or "моделей" in summary

    def test_clear(self):
        """Тест очистки реестра."""

        @ModelRegistry.register("model1")
        class Model1(DummyModelA):
            pass

        @ModelRegistry.register("model2")
        class Model2(DummyModelB):
            pass

        assert len(ModelRegistry.list_models()) == 2

        ModelRegistry.clear()

        assert len(ModelRegistry.list_models()) == 0

    def test_register_non_basemodel(self):
        """Тест регистрации не-BaseModel класса."""

        with pytest.raises(TypeError):

            @ModelRegistry.register("invalid")
            class NotAModel:
                pass
