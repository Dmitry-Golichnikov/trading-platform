"""Пайплайн разметки таргетов."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.labeling.base import BaseLabeler
from src.labeling.methods.horizon import HorizonLabeler
from src.labeling.methods.triple_barrier import TripleBarrierLabeler
from src.pipelines.base import BasePipeline

logger = logging.getLogger(__name__)


class LabelingPipeline(BasePipeline):
    """
    Пайплайн разметки таргетов.

    Этапы:
    1. Загрузка данных
    2. Генерация меток
    3. Постфильтрация
    4. Сохранение размеченных данных
    """

    @property
    def name(self) -> str:
        """Имя пайплайна."""
        return "labeling"

    def _get_steps(self) -> list[str]:
        """Получить список шагов."""
        return [
            "load_data",
            "generate_labels",
            "postprocess_labels",
            "save_labeled",
        ]

    def _execute_step(self, step_name: str, input_data: Any) -> Any:
        """Выполнить шаг пайплайна."""
        if step_name == "load_data":
            return self._load_data()
        elif step_name == "generate_labels":
            return self._generate_labels(input_data)
        elif step_name == "postprocess_labels":
            return self._postprocess_labels(input_data)
        elif step_name == "save_labeled":
            return self._save_labeled(input_data)
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

    def _generate_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Генерировать метки."""
        method = self.config.get("method", "horizon")
        logger.info("Generating labels using method: %s", method)

        if method == "horizon":
            labeler = self._create_horizon_labeler()
        elif method == "triple_barrier":
            labeler = self._create_triple_barrier_labeler()
        else:
            raise ValueError(f"Unknown labeling method: {method}")

        # Генерировать метки
        labeled_data = labeler.label(data)

        if "label" in labeled_data.columns and "target" not in labeled_data.columns:
            labeled_data = labeled_data.rename(columns={"label": "target"})

        # Статистика
        if "target" in labeled_data.columns:
            target_counts = labeled_data["target"].value_counts(dropna=False)
            logger.info("Label distribution: %s", target_counts.to_dict())
            self.state["label_distribution"] = target_counts.to_dict()

        return labeled_data

    def _create_horizon_labeler(self) -> BaseLabeler:
        """Создать horizon labeler."""
        direction = self.config.get("direction", self.config.get("mode", "long+short"))
        threshold_pct = self.config.get("threshold_pct", self.config.get("threshold", 0.02))

        params = {
            "horizon": self.config.get("horizon", 20),
            "direction": direction,
            "threshold_pct": threshold_pct,
            "adaptive_method": self.config.get("adaptive_method", "atr"),
            "adaptive_window": self.config.get("adaptive_window", 20),
            "adaptive_multiplier": self.config.get("adaptive_multiplier", 2.0),
            "min_horizon": self.config.get("min_horizon", 5),
            "max_horizon": self.config.get("max_horizon", 50),
            "price_type": self.config.get("price_type", "close"),
        }

        return HorizonLabeler(**params)

    def _create_triple_barrier_labeler(self) -> BaseLabeler:
        """Создать triple barrier labeler."""
        direction = self.config.get("direction", self.config.get("mode", "long+short"))

        return TripleBarrierLabeler(
            upper_barrier=self.config.get("upper_barrier", 0.02),
            lower_barrier=self.config.get("lower_barrier", 0.02),
            time_barrier=self.config.get("time_barrier", 20),
            direction=direction,
            min_return=self.config.get("min_return", 0.0),
            atr_window=self.config.get("atr_window", 20),
            atr_multiplier=self.config.get("atr_multiplier", 2.0),
            price_type=self.config.get("price_type", "close"),
            include_commissions=self.config.get("include_commissions", False),
            commission_pct=self.config.get("commission_pct", 0.001),
        )

    def _postprocess_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Постобработка меток."""
        if "target" not in data.columns:
            logger.warning("No target column found, skipping postprocessing")
            return data

        # Удалить строки с NaN в таргете
        before_count = len(data)
        data = data.dropna(subset=["target"])
        after_count = len(data)

        if before_count != after_count:
            logger.info("Removed %d rows with NaN targets", before_count - after_count)

        # Балансировка классов (если указано)
        balance_method = self.config.get("balance_method")
        if balance_method:
            data = self._balance_classes(data, balance_method)

        return data

    def _balance_classes(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """Балансировать классы."""
        logger.info("Balancing classes using method: %s", method)

        if method == "undersample":
            # Андерсемплинг мажоритарного класса
            from imblearn.under_sampling import RandomUnderSampler

            rus = RandomUnderSampler(random_state=42)
            X = data.drop(columns=["target"])
            y = data["target"]
            X_resampled, y_resampled = rus.fit_resample(X, y)
            data = pd.concat([X_resampled, y_resampled], axis=1)

        elif method == "oversample":
            # Оверсемплинг миноритарного класса
            from imblearn.over_sampling import RandomOverSampler

            ros = RandomOverSampler(random_state=42)
            X = data.drop(columns=["target"])
            y = data["target"]
            X_resampled, y_resampled = ros.fit_resample(X, y)
            data = pd.concat([X_resampled, y_resampled], axis=1)

        elif method == "smote":
            # SMOTE
            from imblearn.over_sampling import SMOTE

            smote = SMOTE(random_state=42)
            X = data.drop(columns=["target"])
            y = data["target"]
            X_resampled, y_resampled = smote.fit_resample(X, y)
            data = pd.concat([X_resampled, y_resampled], axis=1)

        logger.info("Balanced dataset size: %d", len(data))
        return data

    def _save_labeled(self, data: pd.DataFrame) -> pd.DataFrame:
        """Сохранить размеченные данные."""
        output_path = self.config.get("output_path")
        if not output_path:
            output_path = "artifacts/features/labeled_data.parquet"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving labeled data to %s", output_path)
        data.to_parquet(output_path)

        self.state["output_path"] = str(output_path)
        self.state["num_samples"] = len(data)

        return data
