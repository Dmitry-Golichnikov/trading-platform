"""Кастомные правила разметки."""

import logging
from typing import Callable, Dict

import pandas as pd

from src.labeling.base import BaseLabeler

logger = logging.getLogger(__name__)


class CustomRulesLabeler(BaseLabeler):
    """
    Разметка на основе кастомных правил.

    Позволяет определить собственные правила для генерации меток,
    например, на основе индикаторов или паттернов.
    """

    def __init__(
        self,
        rules: Dict[str, Callable[[pd.DataFrame], pd.Series]],
        aggregation: str = "majority",
    ):
        """
        Инициализация Custom Rules labeler.

        Args:
            rules: Словарь {имя_правила: функция_правила}
                   Функция должна принимать DataFrame и возвращать Series с метками
            aggregation: Метод агрегации меток ('majority', 'unanimous', 'any')
        """
        super().__init__(
            rules=rules,
            aggregation=aggregation,
        )
        self.rule_functions = rules

    def _validate_params(self) -> None:
        """Валидация параметров."""
        if not self.params["rules"]:
            raise ValueError("Необходимо задать хотя бы одно правило")

        if self.params["aggregation"] not in ["majority", "unanimous", "any"]:
            raise ValueError("aggregation должен быть 'majority', 'unanimous' или 'any'")

    def label(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Разметка данных на основе кастомных правил.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с добавленными колонками:
            - label: итоговая метка
            - rule_*: метки от каждого правила
        """
        self.validate_data(data)

        result = data.copy()

        # Применение каждого правила
        rule_labels = {}
        for rule_name, rule_func in self.rule_functions.items():
            try:
                labels = rule_func(data)
                if not isinstance(labels, pd.Series):
                    labels = pd.Series(labels, index=data.index)
                rule_labels[f"rule_{rule_name}"] = labels
                result[f"rule_{rule_name}"] = labels
            except Exception as e:
                logger.error(f"Ошибка в правиле '{rule_name}': {e}")
                rule_labels[f"rule_{rule_name}"] = pd.Series(0, index=data.index)
                result[f"rule_{rule_name}"] = 0

        # Агрегация меток
        if rule_labels:
            aggregated = self._aggregate_labels(rule_labels)
            result["label"] = aggregated
        else:
            result["label"] = 0

        long_count = int((result["label"] == 1).sum())
        short_count = int((result["label"] == -1).sum())
        neutral_count = int((result["label"] == 0).sum())
        logger.info(
            "Custom Rules labeling завершена: %d long, %d short, %d neutral",
            long_count,
            short_count,
            neutral_count,
        )

        return result

    def _aggregate_labels(self, rule_labels: Dict[str, pd.Series]) -> pd.Series:
        """
        Агрегация меток от разных правил.

        Args:
            rule_labels: Словарь с метками от каждого правила

        Returns:
            Series с агрегированными метками
        """
        method = self.params["aggregation"]

        # Объединяем все метки в DataFrame
        labels_df = pd.DataFrame(rule_labels)

        if method == "majority":
            # Мажоритарное голосование
            # Для каждой строки берём самую частую метку
            result = labels_df.mode(axis=1)[0]

        elif method == "unanimous":
            # Единогласное решение
            # Метка назначается только если все правила согласны
            result = pd.Series(0, index=labels_df.index)

            # Проверяем единогласие для long (1)
            all_long = (labels_df == 1).all(axis=1)
            result[all_long] = 1

            # Проверяем единогласие для short (-1)
            all_short = (labels_df == -1).all(axis=1)
            result[all_short] = -1

        else:  # any
            # Хотя бы одно правило дало сигнал
            result = pd.Series(0, index=labels_df.index)

            # Long если хотя бы одно правило дало long
            any_long = (labels_df == 1).any(axis=1)
            result[any_long] = 1

            # Short если хотя бы одно правило дало short (и нет long)
            any_short = (labels_df == -1).any(axis=1) & ~any_long
            result[any_short] = -1

        return result.astype(int)


# Примеры готовых правил
def trend_following_rule(data: pd.DataFrame, fast_period: int = 10, slow_period: int = 30) -> pd.Series:
    """
    Правило на основе скользящих средних (trend following).

    Args:
        data: DataFrame с данными
        fast_period: Период быстрой MA
        slow_period: Период медленной MA

    Returns:
        Series с метками
    """
    fast_ma = data["close"].rolling(window=fast_period).mean()
    slow_ma = data["close"].rolling(window=slow_period).mean()

    labels = pd.Series(0, index=data.index)
    labels[fast_ma > slow_ma] = 1
    labels[fast_ma < slow_ma] = -1

    return labels


def momentum_rule(data: pd.DataFrame, period: int = 14, threshold: float = 0.02) -> pd.Series:
    """
    Правило на основе моментума.

    Args:
        data: DataFrame с данными
        period: Период для расчёта моментума
        threshold: Порог для сигнала

    Returns:
        Series с метками
    """
    momentum = data["close"].pct_change(period)

    labels = pd.Series(0, index=data.index)
    labels[momentum > threshold] = 1
    labels[momentum < -threshold] = -1

    return labels


def volatility_breakout_rule(data: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    Правило на основе прорыва волатильности (Bollinger Bands).

    Args:
        data: DataFrame с данными
        period: Период для расчёта
        num_std: Количество стандартных отклонений

    Returns:
        Series с метками
    """
    ma = data["close"].rolling(window=period).mean()
    std = data["close"].rolling(window=period).std()

    upper = ma + num_std * std
    lower = ma - num_std * std

    labels = pd.Series(0, index=data.index)
    labels[data["close"] > upper] = 1
    labels[data["close"] < lower] = -1

    return labels
