"""Вычисление весов классов."""

import logging
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)


def compute_class_weights(
    labels: pd.Series,
    method: Literal["balanced", "effective_samples", "custom"] = "balanced",
    custom_weights: Optional[Dict[int, float]] = None,
    beta: float = 0.9999,
) -> Dict[int, float]:
    """
    Вычисление весов классов для балансировки.

    Args:
        labels: Series с метками
        method: Метод вычисления весов
            - 'balanced': sklearn balanced weights
            - 'effective_samples': Effective Number of Samples (Cui et al., 2019)
            - 'custom': Кастомные веса
        custom_weights: Кастомные веса для method='custom'
        beta: Параметр для effective_samples метода

    Returns:
        Словарь {класс: вес}

    Raises:
        ValueError: Если метод неизвестен или отсутствуют кастомные веса
    """
    if method == "balanced":
        return _compute_balanced_weights(labels)
    elif method == "effective_samples":
        return _compute_effective_samples_weights(labels, beta)
    elif method == "custom":
        if custom_weights is None:
            raise ValueError("Для method='custom' необходимо указать custom_weights")
        return custom_weights
    else:
        raise ValueError(f"Неизвестный метод: {method}")


def _compute_balanced_weights(labels: pd.Series) -> Dict[int, float]:
    """
    Вычисление balanced weights (sklearn).

    Args:
        labels: Series с метками

    Returns:
        Словарь весов классов
    """
    unique_classes = np.unique(labels)

    # Используем sklearn для вычисления
    weights = compute_class_weight(
        class_weight="balanced", classes=unique_classes, y=labels
    )

    weights_dict = dict(zip(unique_classes, weights))

    logger.info(f"Balanced weights: {weights_dict}")

    return weights_dict


def _compute_effective_samples_weights(
    labels: pd.Series, beta: float = 0.9999
) -> Dict[int, float]:
    """
    Вычисление весов на основе Effective Number of Samples.

    Основано на статье: "Class-Balanced Loss Based on Effective Number of Samples"
    (Cui et al., 2019)

    Формула: weight = (1 - beta) / (1 - beta^n)
    где n - количество примеров класса

    Args:
        labels: Series с метками
        beta: Параметр (обычно 0.9999 для больших датасетов)

    Returns:
        Словарь весов классов
    """
    class_counts = labels.value_counts().to_dict()

    weights_dict = {}
    for cls, count in class_counts.items():
        effective_num = (1.0 - beta**count) / (1.0 - beta)
        weights_dict[cls] = 1.0 / effective_num

    # Нормализуем веса
    total_weight = sum(weights_dict.values())
    weights_dict = {
        k: v / total_weight * len(weights_dict) for k, v in weights_dict.items()
    }

    logger.info(f"Effective samples weights (beta={beta}): {weights_dict}")

    return weights_dict


def apply_sample_weights(
    data: pd.DataFrame,
    labels: pd.Series,
    class_weights: Dict[int, float],
) -> pd.Series:
    """
    Применение весов классов к сэмплам.

    Args:
        data: DataFrame с данными
        labels: Series с метками
        class_weights: Словарь весов классов

    Returns:
        Series с весами для каждого сэмпла
    """
    sample_weights = labels.map(class_weights)

    logger.info(
        f"Sample weights применены: "
        f"min={sample_weights.min():.3f}, "
        f"max={sample_weights.max():.3f}, "
        f"mean={sample_weights.mean():.3f}"
    )

    return sample_weights


def get_class_distribution(labels: pd.Series) -> Dict[int, int]:
    """
    Получить распределение классов.

    Args:
        labels: Series с метками

    Returns:
        Словарь {класс: количество}
    """
    distribution = labels.value_counts().to_dict()

    logger.info(f"Class distribution: {distribution}")

    return distribution
