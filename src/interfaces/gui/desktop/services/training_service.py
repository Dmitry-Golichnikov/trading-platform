from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from src.pipelines.training import TrainingPipeline

from .runner import PipelineProgressUpdate, run_pipeline_with_callback


@dataclass(slots=True)
class TrainingJobResult:
    pipeline_result: Any
    metrics: dict[str, float]
    history: dict[str, list[float]]
    artifacts: dict[str, Any]


class TrainingService:
    """Сервис для запуска пайплайна обучения из GUI."""

    def __init__(self, *, configs_dir: Optional[Path] = None) -> None:
        self.configs_dir = configs_dir or Path("configs/models")

    def load_config(self, path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def run_training(
        self,
        config: dict[str, Any],
        *,
        progress_callback: Optional[Callable[[PipelineProgressUpdate], None]] = None,
        enable_checkpoints: bool = True,
    ) -> TrainingJobResult:
        pipeline = TrainingPipeline(config=config, enable_checkpoints=enable_checkpoints)
        result = run_pipeline_with_callback(pipeline, progress_callback)

        metrics = pipeline.state.get("train_metrics", {})
        history = pipeline.state.get("training_history", {})
        artifacts = pipeline.state.copy()

        return TrainingJobResult(
            pipeline_result=result,
            metrics=metrics,
            history=history,
            artifacts=artifacts,
        )
