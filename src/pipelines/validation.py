"""Пайплайн валидации моделей."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List

import pandas as pd

from src.modeling.base import BaseModel
from src.modeling.serialization import ModelSerializer
from src.pipelines.base import BasePipeline

logger = logging.getLogger(__name__)


class ValidationPipeline(BasePipeline):
    """
    Пайплайн валидации моделей.

    Этапы:
    1. Загрузка модели и данных
    2. Walk-forward валидация
    3. Расчет метрик
    4. Сохранение результатов
    """

    @property
    def name(self) -> str:
        """Имя пайплайна."""
        return "validation"

    def _get_steps(self) -> list[str]:
        """Получить список шагов."""
        return [
            "load_model_and_data",
            "validate",
            "calculate_metrics",
            "save_results",
        ]

    def _execute_step(self, step_name: str, input_data: Any) -> Any:
        """Выполнить шаг пайплайна."""
        if step_name == "load_model_and_data":
            return self._load_model_and_data()
        elif step_name == "validate":
            return self._validate(input_data)
        elif step_name == "calculate_metrics":
            return self._calculate_metrics(input_data)
        elif step_name == "save_results":
            return self._save_results(input_data)
        else:
            raise ValueError(f"Unknown step: {step_name}")

    def _load_model_and_data(self) -> dict[str, Any]:
        """Загрузить модель и данные."""
        # Загрузить модель
        model_path = self.config.get("model_path")
        if not model_path:
            raise ValueError("model_path not specified in config")

        model_path = Path(model_path)
        logger.info("Loading model from %s", model_path)

        model = ModelSerializer.load(model_path)

        # Загрузить данные
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

        return {"model": model, "data": df}

    def _validate(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Выполнить walk-forward валидацию."""
        model = input_dict["model"]
        data = input_dict["data"]

        validation_method = self.config.get("validation_method", "simple")

        if validation_method == "simple":
            # Простая валидация на тестовом наборе
            predictions = self._simple_validation(model, data)
        elif validation_method == "walk_forward":
            # Walk-forward валидация
            predictions = self._walk_forward_validation(model, data)
        elif validation_method == "cv":
            # Cross-validation
            predictions = self._cross_validation(model, data)
        else:
            raise ValueError(f"Unknown validation method: {validation_method}")

        input_dict["predictions"] = predictions
        return input_dict

    def _simple_validation(self, model: BaseModel, data: pd.DataFrame) -> pd.Series:
        """Простая валидация."""
        target_col = self.config.get("target_column", "target")
        exclude_cols = self.config.get("exclude_columns", ["timestamp", "ticker"])

        feature_cols = [col for col in data.columns if col not in exclude_cols + [target_col]]
        X = data[feature_cols]

        logger.info("Running simple validation on %d samples", len(X))
        predictions = model.predict(X)

        return pd.Series(predictions, index=data.index)

    def _walk_forward_validation(self, model: BaseModel, data: pd.DataFrame) -> pd.Series:
        """Walk-forward валидация."""
        logger.info("Running walk-forward validation")

        window_size = self.config.get("window_size", 1000)
        step_size = self.config.get("step_size", 100)

        target_col = self.config.get("target_column", "target")
        exclude_cols = self.config.get("exclude_columns", ["timestamp", "ticker"])
        feature_cols = [col for col in data.columns if col not in exclude_cols + [target_col]]

        all_predictions: List[Any] = []
        indices: List[Any] = []

        for i in range(window_size, len(data), step_size):
            # Обучающая выборка
            train_data = data.iloc[i - window_size : i]
            X_train = train_data[feature_cols]
            y_train = train_data[target_col]

            # Тестовая выборка (следующий батч)
            test_end = min(i + step_size, len(data))
            test_data = data.iloc[i:test_end]
            X_test = test_data[feature_cols]

            # Переобучить модель
            model.fit(X_train, y_train)

            # Предсказать
            preds = model.predict(X_test)
            all_predictions.extend(preds)
            indices.extend(test_data.index)

            logger.debug("Walk-forward step %d/%d", i, len(data))

        return pd.Series(all_predictions, index=indices)

    def _cross_validation(self, model: BaseModel, data: pd.DataFrame) -> pd.Series:
        """Cross-validation."""
        from sklearn.model_selection import cross_val_predict

        logger.info("Running cross-validation")

        target_col = self.config.get("target_column", "target")
        exclude_cols = self.config.get("exclude_columns", ["timestamp", "ticker"])
        feature_cols = [col for col in data.columns if col not in exclude_cols + [target_col]]

        X = data[feature_cols]
        y = data[target_col]

        cv_folds = self.config.get("cv_folds", 5)
        predictions = cross_val_predict(model, X, y, cv=cv_folds)

        return pd.Series(predictions, index=data.index)

    def _calculate_metrics(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Рассчитать метрики."""
        data = input_dict["data"]
        predictions = input_dict["predictions"]

        target_col = self.config.get("target_column", "target")
        y_true = data.loc[predictions.index, target_col]

        # Рассчитать метрики
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        metrics = {
            "accuracy": accuracy_score(y_true, predictions),
            "precision": precision_score(y_true, predictions, average="weighted", zero_division=0),
            "recall": recall_score(y_true, predictions, average="weighted", zero_division=0),
            "f1": f1_score(y_true, predictions, average="weighted", zero_division=0),
        }

        logger.info("Validation metrics: %s", metrics)
        self.state["metrics"] = metrics

        input_dict["metrics"] = metrics
        input_dict["y_true"] = y_true

        return input_dict

    def _save_results(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Сохранить результаты валидации."""
        output_dir = self.config.get("output_dir")
        if not output_dir:
            output_dir = "artifacts/validation"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Сохранить предсказания
        predictions_df = pd.DataFrame(
            {
                "y_true": input_dict["y_true"],
                "y_pred": input_dict["predictions"],
            }
        )
        predictions_path = output_dir / "predictions.parquet"
        predictions_df.to_parquet(predictions_path)
        logger.info("Saved predictions to %s", predictions_path)

        # Сохранить метрики
        import json

        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(input_dict["metrics"], f, indent=2)
        logger.info("Saved metrics to %s", metrics_path)

        self.state["output_dir"] = str(output_dir)
        self.state["predictions_path"] = str(predictions_path)
        self.state["metrics_path"] = str(metrics_path)

        return input_dict
