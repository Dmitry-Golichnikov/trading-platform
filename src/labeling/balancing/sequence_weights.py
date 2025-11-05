"""Sequence-aware weighting для временных рядов."""

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SequenceWeighter:
    """
    Вычисление весов с учётом временных последовательностей.

    Для временных рядов учитывает:
    - Длину последовательности одного класса
    - Близость к текущему времени (более свежие данные важнее)
    - Значимость по PnL (если доступна)
    """

    def __init__(
        self,
        sequence_length_weight: float = 1.0,
        time_decay_weight: float = 1.0,
        pnl_weight: float = 0.0,
        time_decay_method: Literal["linear", "exponential"] = "exponential",
        time_decay_rate: float = 0.001,
    ):
        """
        Инициализация sequence weighter.

        Args:
            sequence_length_weight: Вес для длины последовательности
            time_decay_weight: Вес для временного затухания
            pnl_weight: Вес для PnL (если доступен)
            time_decay_method: Метод временного затухания
            time_decay_rate: Скорость затухания
        """
        self.sequence_length_weight = sequence_length_weight
        self.time_decay_weight = time_decay_weight
        self.pnl_weight = pnl_weight
        self.time_decay_method = time_decay_method
        self.time_decay_rate = time_decay_rate

    def compute_weights(
        self,
        labels: pd.Series,
        pnl: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Вычислить веса для каждого сэмпла.

        Args:
            labels: Series с метками
            pnl: Опциональный Series с PnL значениями

        Returns:
            Series с весами для каждого сэмпла
        """
        weights = pd.Series(1.0, index=labels.index)

        # 1. Веса на основе длины последовательности
        if self.sequence_length_weight > 0:
            seq_weights = self._compute_sequence_length_weights(labels)
            weights *= 1 + self.sequence_length_weight * (seq_weights - 1)

        # 2. Веса на основе времени (более свежие данные важнее)
        if self.time_decay_weight > 0:
            time_weights = self._compute_time_decay_weights(labels)
            weights *= 1 + self.time_decay_weight * (time_weights - 1)

        # 3. Веса на основе PnL
        if self.pnl_weight > 0 and pnl is not None:
            pnl_weights = self._compute_pnl_weights(pnl)
            weights *= 1 + self.pnl_weight * (pnl_weights - 1)

        # Нормализация весов
        weights = weights / weights.mean()

        logger.info(
            f"Sequence weights вычислены: "
            f"min={weights.min():.3f}, "
            f"max={weights.max():.3f}, "
            f"mean={weights.mean():.3f}"
        )

        return weights

    def _compute_sequence_length_weights(self, labels: pd.Series) -> pd.Series:
        """
        Вычисление весов на основе длины последовательности.

        Более короткие последовательности получают больший вес
        (они реже встречаются и потенциально более значимы).

        Args:
            labels: Series с метками

        Returns:
            Series с весами
        """
        weights = pd.Series(1.0, index=labels.index)

        # Находим последовательности
        sequences = self._find_sequences(labels)

        # Вычисляем веса для каждой последовательности
        for start, end, label in sequences:
            length = end - start + 1

            # Обратная зависимость от длины
            # Короткие последовательности получают больший вес
            seq_weight = 1.0 / np.sqrt(length)

            weights.iloc[start : end + 1] = seq_weight

        # Нормализация
        weights = weights / weights.mean()

        return weights

    def _compute_time_decay_weights(self, labels: pd.Series) -> pd.Series:
        """
        Вычисление весов с временным затуханием.

        Более свежие данные получают больший вес.

        Args:
            labels: Series с метками

        Returns:
            Series с весами
        """
        n = len(labels)
        positions = np.arange(n)

        if self.time_decay_method == "linear":
            # Линейное затухание: от 1 до (1 - rate*n)
            weights = 1.0 + self.time_decay_rate * positions
        else:  # exponential
            # Экспоненциальное затухание
            weights = np.exp(self.time_decay_rate * positions)

        # Нормализация
        weights = weights / weights.mean()

        return pd.Series(weights, index=labels.index)

    def _compute_pnl_weights(self, pnl: pd.Series) -> pd.Series:
        """
        Вычисление весов на основе PnL.

        Сделки с большим |PnL| получают больший вес.

        Args:
            pnl: Series с PnL значениями

        Returns:
            Series с весами
        """
        # Используем абсолютное значение PnL
        abs_pnl = pnl.abs()

        # Нормализация
        if abs_pnl.max() > 1e-8:
            weights = abs_pnl / abs_pnl.max()
        else:
            weights = pd.Series(1.0, index=pnl.index)

        # Минимальный вес 0.1
        weights = weights.clip(lower=0.1)

        return weights

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
                sequences.append((start_idx, i - 1, current_label))
                current_label = labels.iloc[i]
                start_idx = i

        sequences.append((start_idx, len(labels) - 1, current_label))

        return sequences
