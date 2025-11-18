"""Фильтр для удаления коротких последовательностей."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class SequenceFilter:
    """
    Фильтр для удаления коротких последовательностей одного класса.

    Убирает одиночные сигналы и короткие последовательности,
    которые могут быть шумом.
    """

    def __init__(
        self,
        min_length: int = 2,
        neutral_label: int = 0,
    ):
        """
        Инициализация sequence filter.

        Args:
            min_length: Минимальная длина последовательности
            neutral_label: Метка для замены коротких последовательностей
        """
        self.min_length = min_length
        self.neutral_label = neutral_label

        if min_length < 1:
            raise ValueError("min_length должен быть >= 1")

    def apply(self, labels: pd.Series) -> pd.Series:
        """
        Применить фильтр к меткам.

        Args:
            labels: Series с метками

        Returns:
            Series с отфильтрованными метками
        """
        if self.min_length <= 1:
            return labels.copy()

        result = labels.copy()

        # Находим последовательности одинаковых меток
        sequences = self._find_sequences(labels)

        # Фильтруем короткие последовательности
        filtered_count = 0
        for start, end, label in sequences:
            length = end - start + 1
            if length < self.min_length:
                result.iloc[start : end + 1] = self.neutral_label
                filtered_count += 1

        logger.info(
            "Sequence filter: удалено %d коротких последовательностей (min_length=%d)",
            filtered_count,
            self.min_length,
        )

        return result

    def _find_sequences(self, labels: pd.Series) -> list:
        """
        Поиск последовательностей одинаковых меток.

        Args:
            labels: Series с метками

        Returns:
            Список кортежей (start_idx, end_idx, label)
        """
        if len(labels) == 0:
            return []

        sequences = []
        current_label = labels.iloc[0]
        start_idx = 0

        for i in range(1, len(labels)):
            if labels.iloc[i] != current_label:
                # Завершаем текущую последовательность
                sequences.append((start_idx, i - 1, current_label))
                # Начинаем новую
                current_label = labels.iloc[i]
                start_idx = i

        # Добавляем последнюю последовательность
        sequences.append((start_idx, len(labels) - 1, current_label))

        return sequences
