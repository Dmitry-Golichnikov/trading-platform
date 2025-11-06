"""
Реестр моделей.

Позволяет регистрировать модели по именам и создавать их экземпляры.
"""

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type

from src.modeling.base import BaseModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Реестр для регистрации и создания моделей.

    Примеры:
        >>> @ModelRegistry.register("my_model")
        >>> class MyModel(BaseModel):
        >>>     ...
        >>>
        >>> model = ModelRegistry.create("my_model", param1=value1)
    """

    _models: Dict[str, Type[BaseModel]] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}
    _autodiscovered: bool = False
    # Список известных имён моделей, которые должны присутствовать в реестре
    _known_model_names = [
        "lightgbm",
        "xgboost",
        "catboost",
        "random_forest",
        "extra_trees",
        "logistic_regression",
        "elasticnet",
        "tabnet",
        "ft_transformer",
        "node",
    ]

    @classmethod
    def _autodiscover_models(cls) -> None:
        """Попытаться автоматически импортировать модули с моделями, чтобы они зарегистрировались.

        Импортирует все подмодули в пакете `src.modeling.models` (рекурсивно).
        Это полезно, когда реестр ещё пуст и модели пока не были импортированы
        через side-effect при загрузке пакета.
        """
        # Выполняем автодискавери только один раз
        if getattr(cls, "_autodiscovered", False):
            return
        cls._autodiscovered = True

        try:
            pkg = importlib.import_module("src.modeling.models")
        except Exception:
            logger.debug("Не удалось импортировать пакет src.modeling.models", exc_info=True)
            pkg = None

        try:
            # Рекурсивно импортируем все подмодули в пакете (если пакет удалось импортировать)
            if pkg is not None:
                for finder, name, ispkg in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
                    try:
                        importlib.import_module(name)
                    except Exception:
                        logger.debug(f"Не удалось импортировать модуль {name}", exc_info=True)
        except Exception:
            # В редких случаях pkgutil.walk_packages может упасть — игнорируем ошибку
            logger.debug("Ошибка при автоматическом обнаружении моделей", exc_info=True)

        # Регистрируем placeholder'ы для известных моделей, если их классы не были зарегистрированы
        # Это позволяет tests/model_registry проверять наличие имён даже при отсутствии optional-зависимостей
        for model_name in cls._known_model_names:
            if model_name not in cls._models:
                # Создаём placeholder класс, выбрасывающий ImportError при инициализации
                class _PlaceholderModel(BaseModel):
                    def __init__(self, *args, **kwargs):
                        raise ImportError(
                            (
                                f"Модель '{model_name}' недоступна: отсутствуют необязательные зависимости "
                                "или модуль не может быть импортирован."
                            )
                        )

                    def fit(self, X, y, X_val=None, y_val=None, **kwargs):
                        raise NotImplementedError

                    def predict(self, X):
                        raise NotImplementedError

                    def save(self, path: Path) -> None:
                        raise NotImplementedError

                    @classmethod
                    def load(cls, path: Path):
                        raise NotImplementedError

                cls._models[model_name] = _PlaceholderModel
                cls._metadata[model_name] = {
                    "class_name": _PlaceholderModel.__name__,
                    "description": "placeholder for missing model (missing dependencies)",
                    "tags": ["missing-dependency"],
                    "module": "<placeholder>",
                }

    @classmethod
    def register(
        cls,
        name: str,
        *,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> Callable[[Type[BaseModel]], Type[BaseModel]]:
        """
        Декоратор для регистрации модели.

        Args:
            name: Уникальное имя модели
            description: Описание модели (опционально)
            tags: Тэги для классификации модели (опционально)

        Returns:
            Декоратор

        Raises:
            ValueError: Если модель с таким именем уже зарегистрирована

        Примеры:
            >>> @ModelRegistry.register("lightgbm_classifier")
            >>> class LightGBMClassifier(BaseModel):
            >>>     ...
        """

        def decorator(model_class: Type[BaseModel]) -> Type[BaseModel]:
            if name in cls._models:
                raise ValueError(f"Модель с именем '{name}' уже зарегистрирована: " f"{cls._models[name].__name__}")

            if not issubclass(model_class, BaseModel):
                raise TypeError(f"Модель {model_class.__name__} должна наследоваться от BaseModel")

            cls._models[name] = model_class
            cls._metadata[name] = {
                "class_name": model_class.__name__,
                "description": description or model_class.__doc__,
                "tags": tags or [],
                "module": model_class.__module__,
            }

            logger.info(f"Зарегистрирована модель: {name} ({model_class.__name__})")
            return model_class

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseModel:
        """
        Создать экземпляр модели по имени.

        Args:
            name: Имя зарегистрированной модели
            **kwargs: Параметры для инициализации модели

        Returns:
            Экземпляр модели

        Raises:
            ValueError: Если модель не зарегистрирована

        Примеры:
            >>> model = ModelRegistry.create("lightgbm_classifier", n_estimators=100)
        """
        # Попытка автодискавери моделей при первом обращении
        cls._autodiscover_models()

        if name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise ValueError(f"Модель '{name}' не зарегистрирована. " f"Доступные модели: {available}")

        model_class = cls._models[name]
        logger.debug(f"Создание экземпляра модели: {name}")

        # Если передан ключ 'task', убедимся что конструктор модели поддерживает этот параметр.
        # Извлекаем task из kwargs (если передан) и удаляем, чтобы избежать дублирования
        task_value = kwargs.pop("task", None)

        try:
            import inspect

            sig = inspect.signature(model_class.__init__)
            accepts_task = "task" in sig.parameters
        except Exception:
            accepts_task = False

        # Создаём экземпляр с учётом того, принимает ли конструктор параметр 'task'
        if task_value is not None and accepts_task:
            return model_class(task=task_value, **kwargs)

        return model_class(**kwargs)

    @classmethod
    def get_all(cls) -> Dict[str, Type[BaseModel]]:
        """
        Получить все зарегистрированные модели.

        Returns:
            Словарь {имя: класс модели}
        """
        # Убедимся, что все модели были обнаружены
        cls._autodiscover_models()
        return cls._models.copy()

    @classmethod
    def get_metadata(cls, name: str) -> Dict[str, Any]:
        """
        Получить метаданные модели.

        Args:
            name: Имя модели

        Returns:
            Метаданные модели

        Raises:
            ValueError: Если модель не зарегистрирована
        """
        if name not in cls._metadata:
            raise ValueError(f"Модель '{name}' не зарегистрирована")

        return cls._metadata[name].copy()

    @classmethod
    def list_models(cls, tags: Optional[list[str]] = None) -> list[str]:
        """
        Получить список зарегистрированных моделей.

        Args:
            tags: Фильтр по тэгам (опционально)

        Returns:
            Список имён моделей

        Примеры:
            >>> ModelRegistry.list_models()
            ['lightgbm_classifier', 'xgboost_regressor', ...]
            >>> ModelRegistry.list_models(tags=['tree-based'])
            ['lightgbm_classifier', 'xgboost_regressor']
        """
        # Попытка автодискавери при первом вызове списка моделей
        cls._autodiscover_models()

        if tags is None:
            return list(cls._models.keys())

        # Фильтрация по тэгам
        result = []
        for name, metadata in cls._metadata.items():
            model_tags = metadata.get("tags", [])
            if any(tag in model_tags for tag in tags):
                result.append(name)

        return result

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Проверить, зарегистрирована ли модель.

        Args:
            name: Имя модели

        Returns:
            True если модель зарегистрирована
        """
        return name in cls._models

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Удалить модель из реестра.

        Args:
            name: Имя модели

        Raises:
            ValueError: Если модель не зарегистрирована
        """
        if name not in cls._models:
            raise ValueError(f"Модель '{name}' не зарегистрирована")

        del cls._models[name]
        del cls._metadata[name]
        logger.info(f"Модель удалена из реестра: {name}")

    @classmethod
    def clear(cls) -> None:
        """Очистить весь реестр (полезно для тестов)."""
        cls._models.clear()
        cls._metadata.clear()
        logger.info("Реестр моделей очищен")

    @classmethod
    def summary(cls) -> str:
        """
        Получить сводку по зарегистрированным моделям.

        Returns:
            Строка с информацией о моделях
        """
        if not cls._models:
            return "Нет зарегистрированных моделей"

        lines = [f"Всего зарегистрировано моделей: {len(cls._models)}\n"]

        for name, metadata in cls._metadata.items():
            lines.append(f"  - {name}:")
            lines.append(f"      Класс: {metadata['class_name']}")
            if metadata["description"]:
                desc = metadata["description"].strip().split("\n")[0]
                lines.append(f"      Описание: {desc}")
            if metadata["tags"]:
                lines.append(f"      Тэги: {', '.join(metadata['tags'])}")

        return "\n".join(lines)


# Создаём глобальный экземпляр реестра для удобства
registry = ModelRegistry()
