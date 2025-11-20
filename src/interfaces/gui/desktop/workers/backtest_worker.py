from __future__ import annotations

from typing import Any, Dict

from PySide6.QtCore import QThread, Signal

from ..services.backtest_service import BacktestService
from ..services.runner import PipelineProgressUpdate


class BacktestWorker(QThread):
    """Фоновый worker для бэктестов."""

    progress = Signal(float, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, service: BacktestService, config: Dict[str, Any]) -> None:
        super().__init__()
        self._service = service
        self._config = config

    def run(self) -> None:  # noqa: D401
        try:
            result = self._service.run_backtest(self._config, progress_callback=self._handle_progress)
            self.finished.emit(result)
        except Exception as exc:  # pragma: no cover - ошибки UI
            self.failed.emit(str(exc))

    def _handle_progress(self, update: PipelineProgressUpdate) -> None:
        self.progress.emit(update.progress, f"{update.step}: {update.status}")
