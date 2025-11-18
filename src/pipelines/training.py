"""Пайплайн обучения моделей."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from src.modeling.registry import ModelRegistry
from src.modeling.trainer import ModelTrainer
from src.pipelines.base import BasePipeline

logger = logging.getLogger(__name__)


class TrainingPipeline(BasePipeline):
    """
    Пайплайн обучения моделей.

    Этапы:
    1. Загрузка данных
    2. Разделение на train/val/test
    3. Обучение модели
    4. Сохранение модели
    """

    @property
    def name(self) -> str:
        """Имя пайплайна."""
        return "training"

    def _get_steps(self) -> list[str]:
        """Получить список шагов."""
        return [
            "load_data",
            "split_data",
            "train_model",
            "save_model",
        ]

    def _execute_step(self, step_name: str, input_data: Any) -> Any:
        """Выполнить шаг пайплайна."""
        if step_name == "load_data":
            return self._load_data()
        elif step_name == "split_data":
            return self._split_data(input_data)
        elif step_name == "train_model":
            return self._train_model(input_data)
        elif step_name == "save_model":
            return self._save_model(input_data)
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

    def _split_data(self, data: pd.DataFrame) -> dict[str, Any]:
        """Разделить данные на train/val/test."""
        target_col = self.config.get("target_column", "target")
        if target_col not in data.columns:
            raise ValueError(f"Target column {target_col} not found in data")

        # Исключить служебные колонки
        exclude_cols = self.config.get("exclude_columns", ["timestamp", "ticker"])
        feature_cols = [col for col in data.columns if col not in exclude_cols + [target_col]]

        X = data[feature_cols]
        y = data[target_col]

        # Пропорции разделения
        test_size = self.config.get("test_size", 0.15)
        val_size = self.config.get("val_size", 0.15)
        random_state = self.config.get("random_state", 42)

        # Разделение train/(val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + val_size), random_state=random_state, stratify=y
        )

        # Разделение val/test
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio), random_state=random_state, stratify=y_temp
        )

        logger.info(
            "Split data: train=%d, val=%d, test=%d",
            len(X_train),
            len(X_val),
            len(X_test),
        )

        self.state["n_train"] = len(X_train)
        self.state["n_val"] = len(X_val)
        self.state["n_test"] = len(X_test)

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
        }

    def _train_model(self, data_dict: dict[str, Any]) -> dict[str, Any]:
        """Обучить модель."""
        model_type = self.config.get("model_type", "lightgbm")
        model_config: Dict[str, Any] = self.config.get("model_config", {})

        logger.info("Training model: %s", model_type)

        # Создать модель
        model = ModelRegistry.create(model_type, **model_config)

        # Создать тренер
        trainer_config: Dict[str, Any] = self.config.get("trainer_config", {})
        trainer_kwargs: Dict[str, Any] = {}
        allowed_trainer_keys = {"experiment_name", "mlflow_uri", "artifacts_dir", "verbose"}
        for key in allowed_trainer_keys:
            if key in trainer_config:
                trainer_kwargs[key] = trainer_config[key]

        if "artifacts_dir" in trainer_kwargs and trainer_kwargs["artifacts_dir"]:
            trainer_kwargs["artifacts_dir"] = Path(trainer_kwargs["artifacts_dir"])

        trainer = ModelTrainer(model=model, **trainer_kwargs)

        # Обучить
        X_train = data_dict["X_train"]
        y_train = data_dict["y_train"]
        X_val = data_dict.get("X_val")
        y_val = data_dict.get("y_val")

        training_result = trainer.train(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            **trainer_config.get("fit_params", {}),
        )

        logger.info("Training completed")

        self.state["train_metrics"] = training_result.metrics
        if trainer.history:
            self.state["training_history"] = trainer.history

        data_dict["model"] = training_result.model
        data_dict["trainer"] = trainer
        data_dict["training_result"] = training_result

        # Дополнительная оценка на тестовом наборе (если есть)
        X_test = data_dict.get("X_test")
        y_test = data_dict.get("y_test")
        if X_test is not None and y_test is not None:
            try:
                y_pred_test = training_result.model.predict(X_test)
                from sklearn.metrics import accuracy_score

                test_accuracy = accuracy_score(y_test, y_pred_test)
                self.state.setdefault("test_metrics", {})["accuracy"] = float(test_accuracy)
                logger.info("Test accuracy: %.4f", test_accuracy)
            except Exception as exc:  # pragma: no cover - best-effort evaluation
                logger.debug("Failed to compute test metrics: %s", exc)

        return data_dict

    def _save_model(self, data_dict: dict[str, Any]) -> dict[str, Any]:
        """Сохранить модель."""
        model = data_dict["model"]

        output_dir = self.config.get("output_dir")
        if not output_dir:
            output_dir = "artifacts/models"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Сгенерировать имя модели
        model_name = self.config.get("model_name")
        if not model_name:
            from datetime import datetime

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            model_type = self.config.get("model_type", "model")
            model_name = f"{model_type}_{timestamp}"

        model_path = output_dir / f"{model_name}.pkl"

        logger.info("Saving model to %s", model_path)
        model.save(model_path)

        self.state["model_path"] = str(model_path)
        self.state["model_name"] = model_name

        return data_dict
