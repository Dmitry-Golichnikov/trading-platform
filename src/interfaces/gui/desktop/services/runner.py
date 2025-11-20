from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Literal, Optional

from src.pipelines.base import BasePipeline, PipelineResult, PipelineStep

ProgressStatus = Literal["pending", "running", "completed", "failed", "cached"]


@dataclass(slots=True)
class PipelineProgressUpdate:
    """DTO для обновлений пайплайна из фонового потока."""

    pipeline: str
    step: str
    status: ProgressStatus
    index: int
    total: int
    progress: float
    message: str = ""
    metadata: dict[str, Any] | None = None


def run_pipeline_with_callback(
    pipeline: BasePipeline,
    progress_callback: Optional[Callable[[PipelineProgressUpdate], None]] = None,
) -> PipelineResult:
    """
    Запустить пайплайн с обратными вызовами после каждого шага.

    Функция повторяет внутреннюю логику BasePipeline.run,
    добавляя события прогресса для GUI.
    """

    def emit(step: PipelineStep, status: ProgressStatus, idx: int, total: int, message: str = "") -> None:
        if not progress_callback:
            return

        progress = (
            0.0 if total == 0 else min(1.0, max(0.0, (idx + (1 if status in {"cached", "completed"} else 0)) / total))
        )
        progress_callback(
            PipelineProgressUpdate(
                pipeline=pipeline.name,
                step=step.name,
                status=status,
                index=idx,
                total=total,
                progress=progress,
                message=message,
                metadata=step.metadata,
            )
        )

    started_at = datetime.utcnow()
    pipeline._config_hash = pipeline._compute_config_hash()  # noqa: SLF001 - расширение поведения

    if not pipeline.force_rerun:
        pipeline._load_state()

    step_names = pipeline._get_steps()
    for step_name in step_names:
        if step_name not in [s.name for s in pipeline.steps]:
            pipeline.steps.append(PipelineStep(name=step_name))

    current_data: Any = None
    total_steps = len(pipeline.steps)

    try:
        for idx, step in enumerate(pipeline.steps):
            if step.status == "completed" and not pipeline.force_rerun:
                current_data = pipeline.load_checkpoint(step.name)
                emit(step, "cached", idx, total_steps, "Шаг уже выполнен, используем кэш")
                continue

            step.status = "running"
            step.started_at = datetime.utcnow()
            emit(step, "running", idx, total_steps, "Выполняем шаг")

            try:
                current_data = pipeline._execute_step(step.name, current_data)
                step.status = "completed"
                step.completed_at = datetime.utcnow()
                emit(step, "completed", idx, total_steps, "Шаг завершен успешно")

                if pipeline.enable_checkpoints:
                    pipeline.save_checkpoint(step.name, current_data)

            except Exception as exc:  # pragma: no cover - проброс в UI
                step.status = "failed"
                step.completed_at = datetime.utcnow()
                step.error = str(exc)
                emit(step, "failed", idx, total_steps, f"Ошибка: {exc}")
                raise

        completed_at = datetime.utcnow()
        result = PipelineResult(
            pipeline_name=pipeline.name,
            status="success",
            started_at=started_at,
            completed_at=completed_at,
            steps=pipeline.steps,
            artifacts=pipeline._get_artifacts(),
        )

        pipeline._save_state()
        pipeline._save_result(result)
        return result

    except Exception:
        completed_at = datetime.utcnow()
        result = PipelineResult(
            pipeline_name=pipeline.name,
            status="failed",
            started_at=started_at,
            completed_at=completed_at,
            steps=pipeline.steps,
            errors=[step.error for step in pipeline.steps if step.error],
        )
        pipeline._save_result(result)
        raise
