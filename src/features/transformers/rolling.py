"""Rolling статистики для признаков."""

from typing import List, Optional

import pandas as pd


class RollingTransformer:
    """Вычисление rolling статистик для признаков."""

    def __init__(
        self,
        window: int,
        functions: List[str],
        columns: List[str],
        min_periods: Optional[int] = None,
    ):
        """
        Инициализация transformer'а.

        Args:
            window: Размер окна
            functions: Список функций для применения
            columns: Колонки для применения
            min_periods: Минимальное количество наблюдений
        """
        self.window = window
        self.functions = functions
        self.columns = columns
        self.min_periods = min_periods or 1
        self._validate_functions()

    def _validate_functions(self):
        """Валидация списка функций."""
        allowed = {
            "mean",
            "std",
            "min",
            "max",
            "median",
            "sum",
            "skew",
            "kurt",
            "quantile",
        }
        invalid = set(self.functions) - allowed
        if invalid:
            raise ValueError(f"Неизвестные функции: {invalid}")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Применить rolling статистики.

        Args:
            data: DataFrame с признаками

        Returns:
            DataFrame с новыми rolling признаками
        """
        features_df = pd.DataFrame(index=data.index)

        for col in self.columns:
            if col not in data.columns:
                raise ValueError(f"Колонка {col} не найдена в данных")

            rolling = data[col].rolling(
                window=self.window, min_periods=self.min_periods
            )

            for func in self.functions:
                feature_name = f"{col}_rolling_{func}_{self.window}"

                if func == "mean":
                    features_df[feature_name] = rolling.mean()
                elif func == "std":
                    features_df[feature_name] = rolling.std()
                elif func == "min":
                    features_df[feature_name] = rolling.min()
                elif func == "max":
                    features_df[feature_name] = rolling.max()
                elif func == "median":
                    features_df[feature_name] = rolling.median()
                elif func == "sum":
                    features_df[feature_name] = rolling.sum()
                elif func == "skew":
                    features_df[feature_name] = rolling.skew()
                elif func == "kurt":
                    features_df[feature_name] = rolling.kurt()
                elif func == "quantile":
                    # По умолчанию 0.75 квантиль
                    features_df[feature_name] = rolling.quantile(0.75)

        return features_df
