"""
MLflow Logger callback.

Логирует метрики, параметры и артефакты в MLflow.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.modeling.callbacks.base import Callback

logger = logging.getLogger(__name__)


class MLflowLogger(Callback):
    """
    Callback для логирования в MLflow.

    Логирует метрики обучения, гиперпараметры и артефакты модели в MLflow.

    Примеры:
        >>> mlflow_logger = MLflowLogger(
        >>>     experiment_name='my_experiment',
        >>>     run_name='run_001',
        >>>     log_model=True
        >>> )
    """

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        log_params: bool = True,
        log_metrics: bool = True,
        log_model: bool = True,
        log_artifacts: bool = True,
        artifact_location: Optional[Path] = None,
        tags: Optional[Dict[str, str]] = None,
        verbose: bool = True,
    ):
        """
        Args:
            experiment_name: Имя эксперимента в MLflow
            run_name: Имя запуска (опционально)
            tracking_uri: URI MLflow tracking server
            log_params: Логировать гиперпараметры
            log_metrics: Логировать метрики
            log_model: Логировать модель
            log_artifacts: Логировать артефакты
            artifact_location: Путь для сохранения артефактов перед логированием
            tags: Дополнительные тэги для запуска
            verbose: Выводить сообщения в лог
        """
        super().__init__()
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.log_params = log_params
        self.log_metrics = log_metrics
        self.log_model_flag = log_model
        self.log_artifacts = log_artifacts
        self.artifact_location = artifact_location
        self.tags = tags or {}
        self.verbose = verbose

        self.run: Optional[Any] = None
        self._mlflow_available = self._check_mlflow()

    def _check_mlflow(self) -> bool:
        """Проверить доступность MLflow."""
        try:
            import mlflow

            _ = mlflow  # предотвращает предупреждение линтера об неиспользуемом импорте
            return True
        except ImportError:
            logger.warning("MLflow не установлен, логирование отключено")
            return False

    def on_train_begin(self, trainer) -> None:
        """Начать MLflow run в начале обучения."""
        if not self._mlflow_available:
            return

        import mlflow

        # Устанавливаем tracking URI
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        # Устанавливаем/создаём эксперимент
        if self.experiment_name:
            try:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                if experiment is None:
                    mlflow.create_experiment(self.experiment_name)
                mlflow.set_experiment(self.experiment_name)
            except Exception as e:
                logger.warning(f"Не удалось установить эксперимент: {e}")

        # Начинаем run
        self.run = mlflow.start_run(run_name=self.run_name)

        # Логируем тэги
        if self.tags:
            mlflow.set_tags(self.tags)

        # Добавляем тэг с типом модели
        mlflow.set_tag("model_class", trainer.model.__class__.__name__)

        # Логируем гиперпараметры
        if self.log_params and hasattr(trainer.model, "get_params"):
            try:
                params = trainer.model.get_params()
                # MLflow не принимает None, заменяем на строку
                params = {k: (v if v is not None else "None") for k, v in params.items()}
                mlflow.log_params(params)
            except Exception as e:
                logger.warning(f"Не удалось залогировать параметры: {e}")

        if self.verbose:
            if self.run is not None:
                logger.info(f"MLflow run начат: {self.run.info.run_id}")

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]) -> None:
        """Логировать метрики в конце эпохи."""
        if not self._mlflow_available or self.run is None:
            return

        if not self.log_metrics:
            return

        import mlflow

        try:
            # Логируем все метрики из logs
            for metric_name, metric_value in logs.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value, step=epoch)
        except Exception as e:
            logger.warning(f"Не удалось залогировать метрики: {e}")

    def on_train_end(self, trainer, logs: Dict[str, Any]) -> None:
        """Логировать модель и завершить run в конце обучения."""
        if not self._mlflow_available or self.run is None:
            return

        import mlflow

        try:
            # Логируем финальные метрики
            if self.log_metrics:
                for metric_name, metric_value in logs.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(f"final_{metric_name}", metric_value)

            # Логируем метаданные модели
            if hasattr(trainer.model, "get_metadata"):
                metadata = trainer.model.get_metadata()
                for key, value in metadata.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"metadata_{key}", value)
                    elif value is not None:
                        mlflow.set_tag(f"metadata_{key}", str(value))

            # Сохраняем модель
            if self.log_model_flag:
                try:
                    # Создаём временную директорию для модели
                    import tempfile

                    with tempfile.TemporaryDirectory() as tmpdir:
                        model_path = Path(tmpdir) / "model"

                        # Сохраняем модель
                        if hasattr(trainer.model, "save"):
                            trainer.model.save(model_path)
                            mlflow.log_artifacts(str(model_path.parent), "model")
                        else:
                            logger.warning("Модель не поддерживает метод save()")
                except Exception as e:
                    logger.warning(f"Не удалось залогировать модель: {e}")

            # Логируем дополнительные артефакты
            if self.log_artifacts and self.artifact_location:
                if self.artifact_location.exists():
                    mlflow.log_artifacts(str(self.artifact_location))

            if self.verbose:
                if self.run is not None:
                    logger.info(f"MLflow run завершён: {self.run.info.run_id}")

        except Exception as e:
            logger.error(f"Ошибка при завершении MLflow run: {e}")

        finally:
            # Завершаем run
            mlflow.end_run()
            self.run = None
