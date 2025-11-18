"""Базовый класс для всех пайплайнов с поддержкой чекпоинтов."""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineStep:
    """Информация о шаге пайплайна."""

    name: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed, skipped
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Результат выполнения пайплайна."""

    pipeline_name: str
    status: str  # success, failed, partial
    started_at: datetime
    completed_at: datetime
    steps: list[PipelineStep]
    artifacts: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Длительность выполнения в секундах."""
        return (self.completed_at - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Преобразовать в словарь."""
        return {
            "pipeline_name": self.pipeline_name,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": self.duration,
            "steps": [
                {
                    "name": step.name,
                    "started_at": step.started_at.isoformat() if step.started_at else None,
                    "completed_at": step.completed_at.isoformat() if step.completed_at else None,
                    "status": step.status,
                    "error": step.error,
                    "metadata": step.metadata,
                }
                for step in self.steps
            ],
            "artifacts": self.artifacts,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class BasePipeline(ABC):
    """
    Базовый класс для всех пайплайнов.

    Поддерживает:
    - Чекпоинты для сохранения промежуточных результатов
    - Идемпотентность (повторный запуск не изменяет результат)
    - Логирование всех этапов
    - Возобновление с последнего чекпоинта
    """

    def __init__(
        self,
        config: dict[str, Any],
        checkpoint_dir: Optional[Path] = None,
        enable_checkpoints: bool = True,
        force_rerun: bool = False,
    ):
        """
        Инициализировать пайплайн.

        Args:
            config: Конфигурация пайплайна
            checkpoint_dir: Директория для чекпоинтов
            enable_checkpoints: Включить сохранение чекпоинтов
            force_rerun: Игнорировать кэш и пересчитать все
        """
        self.config = config
        self.enable_checkpoints = enable_checkpoints
        self.force_rerun = force_rerun

        if checkpoint_dir is None:
            checkpoint_dir = Path("artifacts/checkpoints") / self.name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.state: dict[str, Any] = {}
        self.steps: list[PipelineStep] = []
        self._config_hash: Optional[str] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Имя пайплайна."""
        pass

    @abstractmethod
    def _get_steps(self) -> list[str]:
        """Получить список шагов пайплайна."""
        pass

    @abstractmethod
    def _execute_step(self, step_name: str, input_data: Any) -> Any:
        """
        Выполнить конкретный шаг.

        Args:
            step_name: Имя шага
            input_data: Входные данные (результат предыдущего шага)

        Returns:
            Результат выполнения шага
        """
        pass

    def run(self) -> PipelineResult:
        """
        Запустить пайплайн.

        Returns:
            Результат выполнения пайплайна
        """
        started_at = datetime.utcnow()
        logger.info("Starting pipeline: %s", self.name)

        self._config_hash = self._compute_config_hash()
        logger.debug("Config hash: %s", self._config_hash)

        try:
            # Загрузить состояние, если есть
            if not self.force_rerun:
                self._load_state()

            # Получить шаги
            step_names = self._get_steps()
            logger.info("Pipeline %s has %d steps: %s", self.name, len(step_names), step_names)

            # Инициализировать шаги
            for step_name in step_names:
                if step_name not in [s.name for s in self.steps]:
                    self.steps.append(PipelineStep(name=step_name))

            # Выполнить шаги последовательно
            current_data = None
            for step in self.steps:
                if step.status == "completed" and not self.force_rerun:
                    logger.info("Step %s already completed, skipping", step.name)
                    # Загрузить результат из чекпоинта
                    current_data = self.load_checkpoint(step.name)
                    continue

                step.status = "running"
                step.started_at = datetime.utcnow()
                logger.info("Executing step: %s", step.name)

                try:
                    current_data = self._execute_step(step.name, current_data)
                    step.status = "completed"
                    step.completed_at = datetime.utcnow()
                    logger.info("Step %s completed successfully", step.name)

                    # Сохранить чекпоинт
                    if self.enable_checkpoints:
                        self.save_checkpoint(step.name, current_data)

                except Exception as exc:
                    step.status = "failed"
                    step.completed_at = datetime.utcnow()
                    step.error = str(exc)
                    logger.error("Step %s failed: %s", step.name, exc, exc_info=True)
                    raise

            completed_at = datetime.utcnow()
            result = PipelineResult(
                pipeline_name=self.name,
                status="success",
                started_at=started_at,
                completed_at=completed_at,
                steps=self.steps,
                artifacts=self._get_artifacts(),
            )

            # Сохранить финальное состояние
            self._save_state()
            self._save_result(result)

            logger.info(
                "Pipeline %s completed successfully in %.2f seconds",
                self.name,
                result.duration,
            )
            return result

        except Exception as exc:
            completed_at = datetime.utcnow()
            result = PipelineResult(
                pipeline_name=self.name,
                status="failed",
                started_at=started_at,
                completed_at=completed_at,
                steps=self.steps,
                errors=[str(exc)],
            )
            self._save_result(result)
            logger.error("Pipeline %s failed: %s", self.name, exc)
            raise

    def save_checkpoint(self, step: str, data: Any) -> None:
        """
        Сохранить чекпоинт.

        Args:
            step: Имя шага
            data: Данные для сохранения
        """
        if not self.enable_checkpoints:
            return

        checkpoint_file = self._get_checkpoint_path(step)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Сохранить данные в формате, зависящем от типа
            if hasattr(data, "to_parquet"):  # DataFrame
                checkpoint_file = checkpoint_file.with_suffix(".parquet")
                data.to_parquet(checkpoint_file)
            elif isinstance(data, dict):
                checkpoint_file = checkpoint_file.with_suffix(".json")
                with open(checkpoint_file, "w") as f:
                    json.dump(data, f, indent=2, default=str)
            else:
                checkpoint_file = checkpoint_file.with_suffix(".pkl")
                import pickle

                with open(checkpoint_file, "wb") as f:
                    pickle.dump(data, f)

            logger.debug("Saved checkpoint for step %s to %s", step, checkpoint_file)
        except Exception as exc:
            logger.warning("Failed to save checkpoint for step %s: %s", step, exc)

    def load_checkpoint(self, step: str) -> Any:
        """
        Загрузить чекпоинт.

        Args:
            step: Имя шага

        Returns:
            Загруженные данные или None
        """
        checkpoint_path = self._get_checkpoint_path(step)

        # Попробовать разные форматы
        for suffix in [".parquet", ".json", ".pkl"]:
            checkpoint_file = checkpoint_path.with_suffix(suffix)
            if checkpoint_file.exists():
                try:
                    if suffix == ".parquet":
                        import pandas as pd

                        return pd.read_parquet(checkpoint_file)
                    elif suffix == ".json":
                        with open(checkpoint_file) as f:
                            return json.load(f)
                    else:  # .pkl
                        import pickle

                        with open(checkpoint_file, "rb") as f:
                            return pickle.load(f)
                except Exception as exc:
                    logger.warning("Failed to load checkpoint %s: %s", checkpoint_file, exc)

        return None

    def is_step_cached(self, step: str) -> bool:
        """
        Проверить наличие чекпоинта для шага.

        Args:
            step: Имя шага

        Returns:
            True если чекпоинт существует
        """
        checkpoint_path = self._get_checkpoint_path(step)
        for suffix in [".parquet", ".json", ".pkl"]:
            if checkpoint_path.with_suffix(suffix).exists():
                return True
        return False

    def clear_checkpoints(self) -> None:
        """Удалить все чекпоинты."""
        if self.checkpoint_dir.exists():
            import shutil

            shutil.rmtree(self.checkpoint_dir)
            logger.info("Cleared all checkpoints for pipeline %s", self.name)

    def _get_checkpoint_path(self, step: str) -> Path:
        """Получить путь к файлу чекпоинта."""
        return self.checkpoint_dir / f"{step}_{self._get_config_hash_prefix()}"

    def _compute_config_hash(self) -> str:
        """Вычислить хэш конфигурации для идемпотентности."""
        config_str = json.dumps(self.config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def _load_state(self) -> None:
        """Загрузить сохраненное состояние пайплайна."""
        state_file = self.checkpoint_dir / f"state_{self._get_config_hash_prefix()}.json"
        if not state_file.exists():
            return

        try:
            with open(state_file) as f:
                state_data = json.load(f)

            # Восстановить шаги
            self.steps = []
            for step_data in state_data.get("steps", []):
                step = PipelineStep(
                    name=step_data["name"],
                    started_at=(
                        datetime.fromisoformat(step_data["started_at"]) if step_data.get("started_at") else None
                    ),
                    completed_at=(
                        datetime.fromisoformat(step_data["completed_at"]) if step_data.get("completed_at") else None
                    ),
                    status=step_data["status"],
                    error=step_data.get("error"),
                    metadata=step_data.get("metadata", {}),
                )
                self.steps.append(step)

            logger.info("Loaded state for pipeline %s with %d completed steps", self.name, len(self.steps))
        except Exception as exc:
            logger.warning("Failed to load state: %s", exc)

    def _save_state(self) -> None:
        """Сохранить текущее состояние пайплайна."""
        state_file = self.checkpoint_dir / f"state_{self._get_config_hash_prefix()}.json"
        state_data = {
            "config_hash": self._config_hash,
            "steps": [
                {
                    "name": step.name,
                    "started_at": step.started_at.isoformat() if step.started_at else None,
                    "completed_at": step.completed_at.isoformat() if step.completed_at else None,
                    "status": step.status,
                    "error": step.error,
                    "metadata": step.metadata,
                }
                for step in self.steps
            ],
        }

        with open(state_file, "w") as f:
            json.dump(state_data, f, indent=2)

    def _save_result(self, result: PipelineResult) -> None:
        """Сохранить результат выполнения."""
        result_file = self.checkpoint_dir / f"result_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info("Saved pipeline result to %s", result_file)

    def _get_artifacts(self) -> dict[str, Any]:
        """Получить артефакты для финального результата."""
        return self.state.copy()

    def _get_config_hash_prefix(self) -> str:
        """Получить префикс хэша конфигурации, гарантируя его наличие."""
        if not self._config_hash:
            self._config_hash = self._compute_config_hash()
        return self._config_hash[:8]
