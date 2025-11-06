"""Методы sampling для балансировки классов."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OverSampler:
    """
    Oversampling minority класса.

    ВНИМАНИЕ: Для временных рядов простое дублирование сэмплов
    может привести к leakage. Используйте с осторожностью.
    """

    def __init__(
        self,
        target_ratio: float = 1.0,
        random_state: Optional[int] = None,
    ):
        """
        Инициализация oversampler.

        Args:
            target_ratio: Целевое соотношение minority/majority
            random_state: Random seed для воспроизводимости
        """
        self.target_ratio = target_ratio
        self.random_state = random_state

        if target_ratio <= 0:
            raise ValueError("target_ratio должен быть положительным")

    def fit_resample(
        self,
        data: pd.DataFrame,
        labels: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Применить oversampling.

        Args:
            data: DataFrame с данными
            labels: Series с метками

        Returns:
            Кортеж (resampled_data, resampled_labels)
        """
        np.random.seed(self.random_state)

        # Находим классы
        class_counts = labels.value_counts()
        majority_class = class_counts.idxmax()
        majority_count = class_counts.max()

        # Список для хранения индексов
        indices = list(data.index)

        # Для каждого minority класса
        for cls in class_counts.index:
            if cls == majority_class:
                continue

            minority_count = class_counts[cls]
            target_count = int(majority_count * self.target_ratio)

            if target_count <= minority_count:
                continue

            # Индексы minority класса
            minority_indices = labels[labels == cls].index.tolist()

            # Количество сэмплов для добавления
            n_samples_to_add = target_count - minority_count

            # Случайный выбор с возвратом
            additional_indices = np.random.choice(minority_indices, size=n_samples_to_add, replace=True)

            indices.extend(additional_indices)

            logger.info(f"Oversampling класса {cls}: " f"{minority_count} -> {target_count} сэмплов")

        # Создаём новый DataFrame
        resampled_data = data.loc[indices].reset_index(drop=True)
        resampled_labels = labels.loc[indices].reset_index(drop=True)

        return resampled_data, resampled_labels


class UnderSampler:
    """
    Undersampling majority класса.
    """

    def __init__(
        self,
        target_ratio: float = 1.0,
        random_state: Optional[int] = None,
    ):
        """
        Инициализация undersampler.

        Args:
            target_ratio: Целевое соотношение majority/minority
            random_state: Random seed для воспроизводимости
        """
        self.target_ratio = target_ratio
        self.random_state = random_state

        if target_ratio <= 0:
            raise ValueError("target_ratio должен быть положительным")

    def fit_resample(
        self,
        data: pd.DataFrame,
        labels: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Применить undersampling.

        Args:
            data: DataFrame с данными
            labels: Series с метками

        Returns:
            Кортеж (resampled_data, resampled_labels)
        """
        np.random.seed(self.random_state)

        # Находим классы
        class_counts = labels.value_counts()
        minority_class = class_counts.idxmin()
        minority_count = class_counts.min()

        # Список для хранения индексов
        indices = []

        # Для каждого класса
        for cls in class_counts.index:
            if cls == minority_class:
                # Берём все сэмплы minority класса
                cls_indices = labels[labels == cls].index.tolist()
                indices.extend(cls_indices)
            else:
                # Undersampling majority класса
                target_count = int(minority_count * self.target_ratio)
                cls_indices = labels[labels == cls].index.tolist()

                if target_count < len(cls_indices):
                    # Случайный выбор без возврата
                    selected_indices = np.random.choice(cls_indices, size=target_count, replace=False)
                    indices.extend(selected_indices)

                    logger.info(f"Undersampling класса {cls}: " f"{len(cls_indices)} -> {target_count} сэмплов")
                else:
                    indices.extend(cls_indices)

        # Создаём новый DataFrame
        resampled_data = data.loc[indices].reset_index(drop=True)
        resampled_labels = labels.loc[indices].reset_index(drop=True)

        return resampled_data, resampled_labels
