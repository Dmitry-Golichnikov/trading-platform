"""Соотношения признаков."""

from typing import List, Tuple

import numpy as np
import pandas as pd


class RatiosTransformer:
    """Вычисление соотношений между признаками."""

    def __init__(self, pairs: List[Tuple[str, str]]):
        """
        Инициализация transformer'а.

        Args:
            pairs: Список кортежей (числитель, знаменатель)
        """
        self.pairs = pairs

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Вычислить соотношения между признаками.

        Args:
            data: DataFrame с признаками

        Returns:
            DataFrame с соотношениями
        """
        features_df = pd.DataFrame(index=data.index)

        for numerator, denominator in self.pairs:
            if numerator not in data.columns:
                raise ValueError(f"Колонка {numerator} не найдена в данных")
            if denominator not in data.columns:
                raise ValueError(f"Колонка {denominator} не найдена в данных")

            feature_name = f"{numerator}_div_{denominator}"

            # Избегаем деления на ноль
            ratio = data[numerator] / data[denominator].replace(0, np.nan)
            features_df[feature_name] = ratio

        return features_df
