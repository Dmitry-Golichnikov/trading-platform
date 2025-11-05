"""Majority vote фильтр."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MajorityVoteFilter:
    """
    Majority vote фильтр.

    Для каждой точки берёт голос большинства среди N соседних баров.
    """

    def __init__(
        self,
        window: int = 3,
        weighted: bool = False,
        center: bool = True,
    ):
        """
        Инициализация majority vote filter.

        Args:
            window: Размер окна для голосования
            weighted: Использовать взвешенное голосование
            center: Центрировать окно относительно текущей точки
        """
        self.window = window
        self.weighted = weighted
        self.center = center

        if window < 1:
            raise ValueError("window должен быть >= 1")

        if window % 2 == 0 and center:
            logger.warning(
                f"Для центрированного окна рекомендуется нечётное окно. "
                f"Текущее: {window}"
            )

    def apply(self, labels: pd.Series) -> pd.Series:
        """
        Применить majority vote к меткам.

        Args:
            labels: Series с метками

        Returns:
            Series с отфильтрованными метками
        """
        if self.window == 1:
            return labels.copy()

        if self.weighted:
            result = self._weighted_majority_vote(labels)
        else:
            result = self._simple_majority_vote(labels)

        changed = (labels != result).sum()
        total = len(labels) or 1
        logger.info(
            "Majority vote filter: изменено %d меток (%.1f%%)",
            changed,
            changed / total * 100,
        )

        return result

    def _simple_majority_vote(self, labels: pd.Series) -> pd.Series:
        """
        Простое мажоритарное голосование.

        Args:
            labels: Series с метками

        Returns:
            Series с отфильтрованными метками
        """
        result = []

        for i in range(len(labels)):
            # Определяем границы окна
            if self.center:
                start = max(0, i - self.window // 2)
                end = min(len(labels), i + self.window // 2 + 1)
            else:
                start = max(0, i - self.window + 1)
                end = i + 1

            # Берём срез меток
            window_labels = labels.iloc[start:end]

            # Находим самую частую метку
            majority_label = window_labels.mode()

            if len(majority_label) > 0:
                result.append(majority_label.iloc[0])
            else:
                result.append(labels.iloc[i])

        return pd.Series(result, index=labels.index, dtype=int)

    def _weighted_majority_vote(self, labels: pd.Series) -> pd.Series:
        """
        Взвешенное мажоритарное голосование.

        Веса убывают от центра окна к краям (треугольное распределение).

        Args:
            labels: Series с метками

        Returns:
            Series с отфильтрованными метками
        """
        result = []

        for i in range(len(labels)):
            # Определяем границы окна
            if self.center:
                start = max(0, i - self.window // 2)
                end = min(len(labels), i + self.window // 2 + 1)
            else:
                start = max(0, i - self.window + 1)
                end = i + 1

            # Берём срез меток
            window_labels = labels.iloc[start:end].values

            # Создаём веса (треугольное распределение)
            window_size = len(window_labels)
            center_idx = (i - start) if not self.center else window_size // 2

            weights = np.array(
                [1.0 / (abs(j - center_idx) + 1) for j in range(window_size)]
            )
            weights /= weights.sum()

            # Взвешенное голосование для каждой метки
            unique_labels = pd.unique(window_labels)
            label_scores = {}

            for label in unique_labels:
                mask = window_labels == label
                label_scores[label] = weights[mask].sum()

            # Выбираем метку с максимальным весом
            majority_label = max(label_scores.items(), key=lambda x: x[1])[0]
            result.append(majority_label)

        return pd.Series(result, index=labels.index, dtype=int)
