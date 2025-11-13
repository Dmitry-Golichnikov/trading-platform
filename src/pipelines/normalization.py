"""Пайплайн нормализации данных."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.pipelines.base import BasePipeline

logger = logging.getLogger(__name__)


class NormalizationPipeline(BasePipeline):
    """
    Пайплайн нормализации признаков.

    Этапы:
    1. Загрузка данных
    2. Определение числовых признаков
    3. Нормализация
    4. Сохранение результатов
    """

    @property
    def name(self) -> str:
        """Имя пайплайна."""
        return "normalization"

    def _get_steps(self) -> list[str]:
        """Получить список шагов."""
        return [
            "load_data",
            "select_numeric_features",
            "normalize",
            "save_normalized",
        ]

    def _execute_step(self, step_name: str, input_data: Any) -> Any:
        """Выполнить шаг пайплайна."""
        if step_name == "load_data":
            return self._load_data()
        elif step_name == "select_numeric_features":
            return self._select_numeric_features(input_data)
        elif step_name == "normalize":
            return self._normalize(input_data)
        elif step_name == "save_normalized":
            return self._save_normalized(input_data)
        else:
            raise ValueError(f"Unknown step: {step_name}")

    def _load_data(self) -> pd.DataFrame:
        """Загрузить данные."""
        data_path = self.config.get("data_path")
        if not data_path:
            raise ValueError("data_path not specified in config")

        data_path = Path(data_path)
        logger.info("Loading data from %s", data_path)

        if data_path.suffix == ".parquet":
            df = pd.read_parquet(data_path)
        elif data_path.suffix == ".csv":
            df = pd.read_csv(data_path, parse_dates=["timestamp"])
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

        logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
        return df

    def _select_numeric_features(self, data: pd.DataFrame) -> dict[str, Any]:
        """Определить числовые признаки для нормализации."""
        # Исключить колонки, которые не нужно нормализовать
        exclude_cols = self.config.get("exclude_columns", [])
        exclude_cols.extend(["timestamp", "ticker", "target"])

        # Выбрать числовые колонки
        numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

        logger.info("Found %d numeric features to normalize", len(numeric_cols))
        self.state["numeric_features"] = numeric_cols

        return {"data": data, "numeric_cols": numeric_cols}

    def _normalize(self, input_dict: dict[str, Any]) -> pd.DataFrame:
        """Нормализовать признаки."""
        data = input_dict["data"]
        numeric_cols = input_dict["numeric_cols"]

        method = self.config.get("method", "standard")
        logger.info("Normalizing with method: %s", method)

        # Создать копию данных
        normalized = data.copy()

        # Выбрать метод нормализации
        if method == "standard":
            scaler = StandardScaler()
        elif method == "robust":
            scaler = RobustScaler()
        elif method == "minmax":
            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Нормализовать
        if numeric_cols:
            normalized[numeric_cols] = scaler.fit_transform(data[numeric_cols])

            # Сохранить скейлер
            import pickle

            scaler_path = Path(self.config.get("scaler_path", "artifacts/models/scaler.pkl"))
            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

            self.state["scaler_path"] = str(scaler_path)
            logger.info("Saved scaler to %s", scaler_path)

        return normalized

    def _save_normalized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Сохранить нормализованные данные."""
        output_path = self.config.get("output_path")
        if not output_path:
            output_path = "artifacts/features/features_normalized.parquet"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving normalized data to %s", output_path)
        data.to_parquet(output_path)

        self.state["output_path"] = str(output_path)
        return data
