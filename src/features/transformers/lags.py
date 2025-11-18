"""Лаги признаков."""

from typing import List

import pandas as pd


class LagsTransformer:
    """Создание лагов для признаков."""

    def __init__(self, lags: List[int], columns: List[str]):
        """
        Инициализация transformer'а.

        Args:
            lags: Список лагов (должны быть положительными)
            columns: Колонки для применения
        """
        self.lags = lags
        self.columns = columns
        self._validate_lags()

    def _validate_lags(self):
        """Валидация лагов."""
        if any(lag <= 0 for lag in self.lags):
            raise ValueError("Все лаги должны быть положительными")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Создать лаги для признаков.

        Args:
            data: DataFrame с признаками

        Returns:
            DataFrame с лагами
        """
        features_df = pd.DataFrame(index=data.index)

        for col in self.columns:
            if col not in data.columns:
                raise ValueError(f"Колонка {col} не найдена в данных")

            for lag in self.lags:
                feature_name = f"{col}_lag_{lag}"
                features_df[feature_name] = data[col].shift(lag)

        return features_df
