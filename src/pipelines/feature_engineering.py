"""Пайплайн генерации признаков."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.features.generator import FeatureGenerator
from src.pipelines.base import BasePipeline

logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline(BasePipeline):
    """
    Пайплайн генерации признаков.

    Этапы:
    1. Загрузка данных
    2. Генерация признаков
    3. Сохранение результатов
    """

    @property
    def name(self) -> str:
        """Имя пайплайна."""
        return "feature_engineering"

    def _get_steps(self) -> list[str]:
        """Получить список шагов."""
        return [
            "load_data",
            "generate_features",
            "save_features",
        ]

    def _execute_step(self, step_name: str, input_data: Any) -> Any:
        """Выполнить шаг пайплайна."""
        if step_name == "load_data":
            return self._load_data()
        elif step_name == "generate_features":
            return self._generate_features(input_data)
        elif step_name == "save_features":
            return self._save_features(input_data)
        else:
            raise ValueError(f"Unknown step: {step_name}")

    def _load_data(self) -> pd.DataFrame:
        """Загрузить данные."""
        data_path = self.config.get("data_path")
        if not data_path:
            raise ValueError("data_path not specified in config")

        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        logger.info("Loading data from %s", data_path)

        if data_path.suffix == ".parquet":
            df = pd.read_parquet(data_path)
        elif data_path.suffix == ".csv":
            df = pd.read_csv(data_path, parse_dates=["timestamp"])
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

        logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
        return df

    def _generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Генерировать признаки."""
        logger.info("Generating features")

        # Получить конфигурацию признаков
        feature_config = self.config.get("feature_config")
        if not feature_config:
            raise ValueError("feature_config not specified in config")

        cache_dir_value = self.config.get("cache_dir")
        cache_dir = Path(cache_dir_value) if cache_dir_value else None
        generator = FeatureGenerator(
            config=feature_config,
            cache_enabled=self.config.get("cache", True),
            cache_dir=cache_dir,
        )

        # Генерировать признаки
        dataset_id = self.config.get("dataset_id")
        features_df = generator.generate(
            data,
            dataset_id=dataset_id,
            use_cache=self.config.get("use_cache", True),
        )

        logger.info("Generated %d features", features_df.shape[1])
        self.state["num_features"] = features_df.shape[1]
        self.state["feature_names"] = list(features_df.columns)

        return features_df

    def _save_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Сохранить признаки."""
        output_path = self.config.get("output_path")
        if not output_path:
            output_path = "artifacts/features/features.parquet"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving features to %s", output_path)
        data.to_parquet(output_path)

        self.state["output_path"] = str(output_path)
        return data
