from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QThread, Signal


class OptimizationWorker(QThread):
    """Worker для длительных оптимизаций (hyperopt, grid-search)."""

    progress = Signal(float, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        task: Callable[[Callable[[float, str], None]], Any],
    ) -> None:
        super().__init__()
        self._task = task

    def run(self) -> None:  # noqa: D401
        try:
            result = self._task(self._report_progress)
            self.finished.emit(result)
        except Exception as exc:  # pragma: no cover
            self.failed.emit(str(exc))

    def _report_progress(self, value: float, message: str) -> None:
        self.progress.emit(value, message)
