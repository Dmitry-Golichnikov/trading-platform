"""Пайплайн отбора признаков."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.selectors import FeatureImportanceSelector
from src.pipelines.base import BasePipeline

logger = logging.getLogger(__name__)


class FeatureSelectionPipeline(BasePipeline):
    """
    Пайплайн отбора признаков.

    Этапы:
    1. Загрузка данных с признаками
    2. Отбор признаков (по важности, корреляции, и т.д.)
    3. Сохранение отобранных признаков
    """

    @property
    def name(self) -> str:
        """Имя пайплайна."""
        return "feature_selection"

    def _get_steps(self) -> list[str]:
        """Получить список шагов."""
        return [
            "load_data",
            "select_features",
            "save_selected",
        ]

    def _execute_step(self, step_name: str, input_data: Any) -> Any:
        """Выполнить шаг пайплайна."""
        if step_name == "load_data":
            return self._load_data()
        elif step_name == "select_features":
            return self._select_features(input_data)
        elif step_name == "save_selected":
            return self._save_selected(input_data)
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

    def _select_features(self, data: pd.DataFrame) -> dict[str, Any]:
        """Отобрать признаки."""
        target_col = self.config.get("target_column", "target")
        if target_col not in data.columns:
            raise ValueError(f"Target column {target_col} not found in data")

        method = self.config.get("method", "importance")
        logger.info("Selecting features using method: %s", method)

        # Разделить на признаки и таргет
        exclude_cols = ["timestamp", "ticker", target_col]
        feature_cols = [col for col in data.columns if col not in exclude_cols]

        X = data[feature_cols]
        y = data[target_col]

        logger.info("Selecting from %d features", len(feature_cols))

        if method == "importance":
            # Отбор по важности признаков
            n_features = self.config.get("n_features", 50)
            selector = FeatureImportanceSelector(n_features=n_features)
            selected_features = selector.select(X, y)

        elif method == "correlation":
            # Отбор по корреляции
            threshold = self.config.get("correlation_threshold", 0.95)
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
            selected_features = [col for col in feature_cols if col not in to_drop]

        elif method == "variance":
            # Отбор по дисперсии
            from sklearn.feature_selection import VarianceThreshold

            threshold = self.config.get("variance_threshold", 0.01)
            selector = VarianceThreshold(threshold=threshold)
            selector.fit(X)
            selected_features = X.columns[selector.get_support()].tolist()

        else:
            raise ValueError(f"Unknown selection method: {method}")

        logger.info("Selected %d features", len(selected_features))
        self.state["selected_features"] = selected_features
        self.state["n_selected"] = len(selected_features)
        self.state["n_original"] = len(feature_cols)

        # Вернуть данные с выбранными признаками
        selected_cols = exclude_cols + selected_features
        selected_data = data[[col for col in selected_cols if col in data.columns]]

        return {"data": selected_data, "selected_features": selected_features}

    def _save_selected(self, input_dict: dict[str, Any]) -> pd.DataFrame:
        """Сохранить отобранные признаки."""
        data = input_dict["data"]
        selected_features = input_dict["selected_features"]

        # Сохранить данные
        output_path = self.config.get("output_path")
        if not output_path:
            output_path = "artifacts/features/features_selected.parquet"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving selected features to %s", output_path)
        data.to_parquet(output_path)

        # Сохранить список выбранных признаков
        features_list_path = output_path.parent / "selected_features.txt"
        with open(features_list_path, "w") as f:
            f.write("\n".join(selected_features))

        logger.info("Saved feature list to %s", features_list_path)

        self.state["output_path"] = str(output_path)
        self.state["features_list_path"] = str(features_list_path)

        return data
