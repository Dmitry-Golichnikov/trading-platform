from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from src.pipelines.backtest import BacktestPipeline

from .runner import PipelineProgressUpdate, run_pipeline_with_callback


@dataclass(slots=True)
class BacktestJobResult:
    pipeline_result: Any
    metrics: dict[str, Any]
    artifacts: dict[str, Any]


class BacktestService:
    """Сервис для конфигурации и запуска бэктестов из GUI."""

    def __init__(self, *, configs_dir: Optional[Path] = None) -> None:
        self.configs_dir = configs_dir or Path("configs/backtests")

    def load_config(self, path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def run_backtest(
        self,
        config: dict[str, Any],
        *,
        progress_callback: Optional[Callable[[PipelineProgressUpdate], None]] = None,
    ) -> BacktestJobResult:
        pipeline = BacktestPipeline(config=config, enable_checkpoints=False)
        result = run_pipeline_with_callback(pipeline, progress_callback)

        metrics = pipeline.state.get("metrics", {})
        artifacts = pipeline.state.copy()

        return BacktestJobResult(
            pipeline_result=result,
            metrics=metrics,
            artifacts=artifacts,
        )
