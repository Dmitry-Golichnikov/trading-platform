"""Protocol интерфейсы для основных компонентов."""

from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DataLoader(Protocol):
    """Protocol для загрузчиков данных."""

    def load(
        self,
        source: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Загружает данные из источника.

        Args:
            source: Идентификатор источника данных
            start_date: Начальная дата (опционально)
            end_date: Конечная дата (опционально)
            **kwargs: Дополнительные параметры

        Returns:
            DataFrame с загруженными данными
        """
        ...  # pragma: no cover


@runtime_checkable
class Model(Protocol):
    """Protocol для моделей машинного обучения."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Обучает модель.

        Args:
            X: Признаки
            y: Таргет
        """
        ...  # pragma: no cover
        ...  # pragma: no cover
        ...  # pragma: no cover
        ...  # pragma: no cover
        ...  # pragma: no cover

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Делает предсказания.

        Args:
            X: Признаки

        Returns:
            Предсказания
        """
        ...  # pragma: no cover

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Возвращает вероятности классов.

        Args:
            X: Признаки

        Returns:
            DataFrame с вероятностями для каждого класса
        """
        ...  # pragma: no cover

    def save(self, path: Path) -> None:
        """
        Сохраняет модель.

        Args:
            path: Путь для сохранения
        """
        ...  # pragma: no cover

    def load(self, path: Path) -> None:
        """
        Загружает модель.

        Args:
            path: Путь к сохраненной модели
        """
        ...  # pragma: no cover


@runtime_checkable
class FeatureCalculator(Protocol):
    """Protocol для вычисления признаков."""

    def calculate(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Вычисляет признаки.

        Args:
            df: Исходные данные
            config: Конфигурация признаков

        Returns:
            DataFrame с добавленными признаками
        """
        ...  # pragma: no cover
