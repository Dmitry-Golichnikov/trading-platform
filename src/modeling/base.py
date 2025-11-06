"""
Базовый интерфейс для всех моделей.

Определяет единый API для обучения, предсказания, сохранения и загрузки моделей.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Union

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    Единый интерфейс для всех моделей (классификация и регрессия).

    Все модели в системе должны наследоваться от этого класса и реализовать
    все абстрактные методы.
    """

    def __init__(self, **hyperparams):
        """
        Инициализация модели.

        Args:
            **hyperparams: Гиперпараметры модели
        """
        # Сохраняем гиперпараметры, исключая служебные ключи (например, 'task')
        hp = hyperparams.copy()
        # Убираем ключ 'task', если он случайно оказался в hyperparams — модели сами хранят task
        if "task" in hp:
            hp.pop("task")

        self.hyperparams = hp
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "training_time": None,
            "n_samples_trained": None,
            "n_features": None,
        }
        self._is_fitted = False
        self.model: Any = None
        self._classes: np.ndarray = np.array([])
        self._feature_names: list[str] = []

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> "BaseModel":
        """
        Обучить модель.

        Args:
            X: Признаки для обучения
            y: Таргет для обучения
            X_val: Признаки для валидации (опционально)
            y_val: Таргет для валидации (опционально)
            **kwargs: Дополнительные параметры обучения

        Returns:
            Обученная модель (self)
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Получить предсказания модели.

        Для классификации: метки классов (0, 1, 2, ...)
        Для регрессии: предсказанные значения

        Args:
            X: Признаки для предсказания

        Returns:
            Предсказания
        """
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Получить вероятности классов (только для классификаторов).

        Args:
            X: Признаки для предсказания

        Returns:
            Вероятности классов, shape (n_samples, n_classes)

        Raises:
            NotImplementedError: Если модель не поддерживает вероятности
        """
        raise NotImplementedError(f"{self.__class__.__name__} не поддерживает predict_proba")

    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Сохранить модель на диск.

        Args:
            path: Путь для сохранения (директория или файл)
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        """
        Загрузить модель с диска.

        Args:
            path: Путь к сохранённой модели

        Returns:
            Загруженная модель
        """
        pass

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        """
        Feature importances (если поддерживается).

        Returns:
            Массив важностей признаков или None
        """
        return None

    @property
    def is_fitted(self) -> bool:
        """Проверка, обучена ли модель."""
        return self._is_fitted

    def get_params(self) -> Dict[str, Any]:
        """
        Получить гиперпараметры модели.

        Returns:
            Словарь гиперпараметров
        """
        return self.hyperparams.copy()

    def set_params(self, **params) -> "BaseModel":
        """
        Установить гиперпараметры модели.

        Args:
            **params: Новые значения гиперпараметров

        Returns:
            self
        """
        self.hyperparams.update(params)
        return self

    def get_metadata(self) -> Dict[str, Any]:
        """
        Получить метаданные модели.

        Returns:
            Словарь с метаданными (время обучения, кол-во примеров и т.д.)
        """
        return self.metadata.copy()

    def __repr__(self) -> str:
        """Строковое представление модели."""
        fitted_status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}({fitted_status})"


class ModelProtocol(Protocol):
    """
    Protocol для type hints - описывает минимальный интерфейс модели.

    Используется для аннотаций типов без необходимости наследования от BaseModel.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "ModelProtocol":
        """Обучить модель."""
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Предсказать значения."""
        ...

    def save(self, path: Path) -> None:
        """Сохранить модель."""
        ...

    @classmethod
    def load(cls, path: Path) -> "ModelProtocol":
        """Загрузить модель."""
        ...


class ClassifierMixin:
    """
    Mixin для классификаторов.

    Добавляет методы, специфичные для задач классификации.
    """

    @property
    def classes_(self) -> Optional[np.ndarray]:
        """Уникальные классы таргета."""
        if hasattr(self, "_classes"):
            return self._classes
        return None

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """
        Вычислить decision function (если поддерживается).

        Args:
            X: Признаки

        Returns:
            Decision scores

        Raises:
            NotImplementedError: Если модель не поддерживает decision function
        """
        raise NotImplementedError(f"{self.__class__.__name__} не поддерживает decision_function")


class RegressorMixin:
    """
    Mixin для регрессоров.

    Добавляет методы, специфичные для задач регрессии.
    """

    def predict_quantiles(self, X: pd.DataFrame, quantiles: list[float]) -> np.ndarray:
        """
        Предсказать квантили (если поддерживается).

        Args:
            X: Признаки
            quantiles: Список квантилей для предсказания

        Returns:
            Предсказанные квантили, shape (n_samples, n_quantiles)

        Raises:
            NotImplementedError: Если модель не поддерживает квантили
        """
        raise NotImplementedError(f"{self.__class__.__name__} не поддерживает predict_quantiles")


# Типы для type hints
ModelType = Union[BaseModel, ModelProtocol]
