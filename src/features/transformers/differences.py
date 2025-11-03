"""Разности признаков."""

from typing import List, Literal

import pandas as pd


class DifferencesTransformer:
    """Вычисление разностей признаков."""

    def __init__(
        self,
        periods: List[int],
        columns: List[str],
        method: Literal["diff", "pct_change"] = "diff",
    ):
        """
        Инициализация transformer'а.

        Args:
            periods: Периоды для вычисления разностей
            columns: Колонки для применения
            method: Метод вычисления (diff или pct_change)
        """
        self.periods = periods
        self.columns = columns
        self.method = method
        self._validate_periods()

    def _validate_periods(self):
        """Валидация периодов."""
        if any(p <= 0 for p in self.periods):
            raise ValueError("Все периоды должны быть положительными")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Вычислить разности для признаков.

        Args:
            data: DataFrame с признаками

        Returns:
            DataFrame с разностями
        """
        features_df = pd.DataFrame(index=data.index)

        for col in self.columns:
            if col not in data.columns:
                raise ValueError(f"Колонка {col} не найдена в данных")

            for period in self.periods:
                if self.method == "diff":
                    feature_name = f"{col}_diff_{period}"
                    features_df[feature_name] = data[col].diff(period)
                elif self.method == "pct_change":
                    feature_name = f"{col}_pct_change_{period}"
                    features_df[feature_name] = data[col].pct_change(period)

        return features_df
