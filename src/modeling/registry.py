"""
Реестр моделей.

Позволяет регистрировать модели по именам и создавать их экземпляры.
"""

import logging
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
                raise ValueError(
                    f"Модель с именем '{name}' уже зарегистрирована: "
                    f"{cls._models[name].__name__}"
                )

            if not issubclass(model_class, BaseModel):
                raise TypeError(
                    f"Модель {model_class.__name__} должна наследоваться от BaseModel"
                )

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
        if name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise ValueError(
                f"Модель '{name}' не зарегистрирована. "
                f"Доступные модели: {available}"
            )

        model_class = cls._models[name]
        logger.debug(f"Создание экземпляра модели: {name}")

        return model_class(**kwargs)

    @classmethod
    def get_all(cls) -> Dict[str, Type[BaseModel]]:
        """
        Получить все зарегистрированные модели.

        Returns:
            Словарь {имя: класс модели}
        """
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
