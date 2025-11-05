"""
Централизованный реестр всех технических индикаторов.
"""

from typing import Any, Callable, Dict, List, Type

from src.features.indicators.base import Indicator


class IndicatorRegistry:
    """Централизованный реестр всех индикаторов."""

    _indicators: Dict[str, Type[Indicator]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[Indicator]], Type[Indicator]]:
        """
        Декоратор для регистрации индикатора.

        Args:
            name: Название индикатора для регистрации

        Returns:
            Декоратор

        Example:
            >>> @IndicatorRegistry.register("sma")
            >>> class SMA(Indicator):
            >>>     ...
        """

        def decorator(indicator_class: Type[Indicator]) -> Type[Indicator]:
            cls._indicators[name.lower()] = indicator_class
            return indicator_class

        return decorator

    @classmethod
    def get(cls, name: str, **params: Any) -> Indicator:
        """
        Получить экземпляр индикатора по имени.

        Args:
            name: Название индикатора
            **params: Параметры для инициализации индикатора

        Returns:
            Экземпляр индикатора

        Raises:
            ValueError: Если индикатор не найден

        Example:
            >>> sma = IndicatorRegistry.get("sma", window=20)
        """
        name_lower = name.lower()
        if name_lower not in cls._indicators:
            raise ValueError(
                f"Неизвестный индикатор: {name}. "
                f"Доступные: {', '.join(cls.list_all())}"
            )
        return cls._indicators[name_lower](**params)

    @classmethod
    def list_all(cls) -> List[str]:
        """
        Список всех зарегистрированных индикаторов.

        Returns:
            Список названий индикаторов
        """
        return sorted(cls._indicators.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Проверить зарегистрирован ли индикатор.

        Args:
            name: Название индикатора

        Returns:
            True если зарегистрирован
        """
        return name.lower() in cls._indicators
