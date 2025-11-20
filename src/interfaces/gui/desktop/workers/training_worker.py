from __future__ import annotations

from typing import Any, Dict

from PySide6.QtCore import QThread, Signal

from ..services.runner import PipelineProgressUpdate
from ..services.training_service import TrainingService


class TrainingWorker(QThread):
    """Фоновый рабочий поток для обучения моделей."""

    progress = Signal(float, str)
    pipeline_update = Signal(dict)
    metrics = Signal(dict)
    history = Signal(dict)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, service: TrainingService, config: Dict[str, Any]) -> None:
        super().__init__()
        self._service = service
        self._config = config

    def run(self) -> None:  # noqa: D401 - Qt signature
        try:
            result = self._service.run_training(self._config, progress_callback=self._handle_progress)
            self.metrics.emit(result.metrics)
            self.history.emit(result.history)
            self.finished.emit(result)
        except Exception as exc:  # pragma: no cover - переключение в UI
            self.failed.emit(str(exc))

    def _handle_progress(self, update: PipelineProgressUpdate) -> None:
        self.progress.emit(update.progress, f"{update.pipeline}: {update.step} → {update.status}")
        self.pipeline_update.emit(
            {
                "pipeline": update.pipeline,
                "step": update.step,
                "status": update.status,
                "message": update.message,
                "progress": update.progress,
            }
        )
