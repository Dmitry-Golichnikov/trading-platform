"""Полный end-to-end пайплайн."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.pipelines.backtest import BacktestPipeline
from src.pipelines.base import BasePipeline
from src.pipelines.data_preparation import DataPreparationPipeline
from src.pipelines.feature_engineering import FeatureEngineeringPipeline
from src.pipelines.feature_selection import FeatureSelectionPipeline
from src.pipelines.labeling import LabelingPipeline
from src.pipelines.normalization import NormalizationPipeline
from src.pipelines.training import TrainingPipeline
from src.pipelines.validation import ValidationPipeline

logger = logging.getLogger(__name__)


class FullPipeline(BasePipeline):
    """
    Полный end-to-end пайплайн от сырых данных до бэктеста.

    Последовательность:
    1. Data Preparation - загрузка и подготовка данных
    2. Feature Engineering - генерация признаков
    3. Normalization - нормализация данных
    4. Feature Selection - отбор признаков
    5. Labeling - разметка таргетов
    6. Training - обучение модели
    7. Validation - валидация модели
    8. Backtest - бэктест стратегии
    9. Reporting - генерация отчетов
    """

    @property
    def name(self) -> str:
        """Имя пайплайна."""
        return "full_pipeline"

    def _get_steps(self) -> list[str]:
        """Получить список шагов."""
        steps = []

        if self.config.get("run_data_preparation", True):
            steps.append("data_preparation")

        if self.config.get("run_feature_engineering", True):
            steps.append("feature_engineering")

        if self.config.get("run_normalization", True):
            steps.append("normalization")

        if self.config.get("run_feature_selection", True):
            steps.append("feature_selection")

        if self.config.get("run_labeling", True):
            steps.append("labeling")

        if self.config.get("run_training", True):
            steps.append("training")

        if self.config.get("run_validation", True):
            steps.append("validation")

        if self.config.get("run_backtest", True):
            steps.append("backtest")

        steps.append("generate_report")

        return steps

    def _execute_step(self, step_name: str, input_data: Any) -> Any:
        """Выполнить шаг пайплайна."""
        if step_name == "data_preparation":
            return self._run_data_preparation()
        elif step_name == "feature_engineering":
            return self._run_feature_engineering(input_data)
        elif step_name == "normalization":
            return self._run_normalization(input_data)
        elif step_name == "feature_selection":
            return self._run_feature_selection(input_data)
        elif step_name == "labeling":
            return self._run_labeling(input_data)
        elif step_name == "training":
            return self._run_training(input_data)
        elif step_name == "validation":
            return self._run_validation(input_data)
        elif step_name == "backtest":
            return self._run_backtest(input_data)
        elif step_name == "generate_report":
            return self._generate_report(input_data)
        else:
            raise ValueError(f"Unknown step: {step_name}")

    def _run_data_preparation(self) -> dict[str, Any]:
        """Запустить подготовку данных."""
        logger.info("=" * 80)
        logger.info("Step 1/9: Data Preparation")
        logger.info("=" * 80)

        config = self.config.get("data_preparation", {})

        # Для DataPreparationPipeline используем специальную логику
        # т.к. он асинхронный
        from src.data.schemas import DatasetConfig

        dataset_config = DatasetConfig(**config)
        pipeline = DataPreparationPipeline(config=dataset_config)

        # Запустить асинхронно
        import asyncio

        results = asyncio.run(pipeline.run())

        data_path = config.get("prepared_data_path")
        if not data_path:
            downstream = self.config.get("feature_engineering", {}).get("data_path")
            if downstream:
                data_path = downstream

        if not data_path:
            logger.warning(
                "Не удалось определить путь к подготовленным данным автоматически. "
                "Убедитесь, что следующий шаг получает корректный 'data_path' через конфигурацию."
            )

        return {"data_path": data_path, "config": config, "results": results}

    def _run_feature_engineering(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Запустить генерацию признаков."""
        logger.info("=" * 80)
        logger.info("Step 2/9: Feature Engineering")
        logger.info("=" * 80)

        config = self.config.get("feature_engineering", {})
        config["data_path"] = input_data["data_path"]

        if "output_path" not in config:
            config["output_path"] = "artifacts/features/features.parquet"

        pipeline = FeatureEngineeringPipeline(
            config=config,
            checkpoint_dir=self.checkpoint_dir / "feature_engineering",
            enable_checkpoints=self.enable_checkpoints,
        )

        result = pipeline.run()
        return {"data_path": config["output_path"], "result": result}

    def _run_normalization(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Запустить нормализацию."""
        logger.info("=" * 80)
        logger.info("Step 3/9: Normalization")
        logger.info("=" * 80)

        config = self.config.get("normalization", {})
        config["data_path"] = input_data["data_path"]

        if "output_path" not in config:
            config["output_path"] = "artifacts/features/features_normalized.parquet"

        pipeline = NormalizationPipeline(
            config=config,
            checkpoint_dir=self.checkpoint_dir / "normalization",
            enable_checkpoints=self.enable_checkpoints,
        )

        result = pipeline.run()
        return {"data_path": config["output_path"], "result": result}

    def _run_feature_selection(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Запустить отбор признаков."""
        logger.info("=" * 80)
        logger.info("Step 4/9: Feature Selection")
        logger.info("=" * 80)

        config = self.config.get("feature_selection", {})
        config["data_path"] = input_data["data_path"]

        if "output_path" not in config:
            config["output_path"] = "artifacts/features/features_selected.parquet"

        pipeline = FeatureSelectionPipeline(
            config=config,
            checkpoint_dir=self.checkpoint_dir / "feature_selection",
            enable_checkpoints=self.enable_checkpoints,
        )

        result = pipeline.run()
        return {"data_path": config["output_path"], "result": result}

    def _run_labeling(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Запустить разметку."""
        logger.info("=" * 80)
        logger.info("Step 5/9: Labeling")
        logger.info("=" * 80)

        config = self.config.get("labeling", {})
        config["data_path"] = input_data["data_path"]

        if "output_path" not in config:
            config["output_path"] = "artifacts/features/labeled_data.parquet"

        pipeline = LabelingPipeline(
            config=config,
            checkpoint_dir=self.checkpoint_dir / "labeling",
            enable_checkpoints=self.enable_checkpoints,
        )

        result = pipeline.run()
        return {"data_path": config["output_path"], "result": result}

    def _run_training(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Запустить обучение."""
        logger.info("=" * 80)
        logger.info("Step 6/9: Training")
        logger.info("=" * 80)

        config = self.config.get("training", {})
        config["data_path"] = input_data["data_path"]

        if "output_dir" not in config:
            config["output_dir"] = "artifacts/models"

        pipeline = TrainingPipeline(
            config=config,
            checkpoint_dir=self.checkpoint_dir / "training",
            enable_checkpoints=self.enable_checkpoints,
        )

        result = pipeline.run()

        # Получить путь к модели из результата
        model_path = pipeline.state.get("model_path")

        return {
            "data_path": input_data["data_path"],
            "model_path": model_path,
            "result": result,
        }

    def _run_validation(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Запустить валидацию."""
        logger.info("=" * 80)
        logger.info("Step 7/9: Validation")
        logger.info("=" * 80)

        config = self.config.get("validation", {})
        config["data_path"] = input_data["data_path"]
        config["model_path"] = input_data["model_path"]

        if "output_dir" not in config:
            config["output_dir"] = "artifacts/validation"

        pipeline = ValidationPipeline(
            config=config,
            checkpoint_dir=self.checkpoint_dir / "validation",
            enable_checkpoints=self.enable_checkpoints,
        )

        result = pipeline.run()
        return {
            "data_path": input_data["data_path"],
            "model_path": input_data["model_path"],
            "result": result,
        }

    def _run_backtest(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Запустить бэктест."""
        logger.info("=" * 80)
        logger.info("Step 8/9: Backtest")
        logger.info("=" * 80)

        config = self.config.get("backtest", {})
        config["data_path"] = input_data["data_path"]
        config["model_path"] = input_data["model_path"]

        if "output_dir" not in config:
            config["output_dir"] = "artifacts/backtests"

        pipeline = BacktestPipeline(
            config=config,
            checkpoint_dir=self.checkpoint_dir / "backtest",
            enable_checkpoints=self.enable_checkpoints,
        )

        result = pipeline.run()
        return {
            "data_path": input_data["data_path"],
            "model_path": input_data["model_path"],
            "result": result,
        }

    def _generate_report(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Генерировать финальный отчет."""
        logger.info("=" * 80)
        logger.info("Step 9/9: Generate Report")
        logger.info("=" * 80)

        output_dir = Path(self.config.get("output_dir", "artifacts/reports"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Собрать информацию из всех шагов
        report_data = {
            "pipeline_name": self.name,
            "config": self.config,
            "steps": [
                {
                    "name": step.name,
                    "status": step.status,
                    "started_at": step.started_at.isoformat() if step.started_at else None,
                    "completed_at": step.completed_at.isoformat() if step.completed_at else None,
                    "metadata": step.metadata,
                }
                for step in self.steps
            ],
            "artifacts": self.state,
        }

        # Сохранить JSON отчет
        import json
        from datetime import datetime

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"full_pipeline_report_{timestamp}.json"

        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)

        logger.info("Saved full pipeline report to %s", report_path)

        # Создать HTML отчет
        html_path = output_dir / f"full_pipeline_report_{timestamp}.html"
        self._create_html_report(report_data, html_path)
        logger.info("Saved HTML report to %s", html_path)

        self.state["report_path"] = str(report_path)
        self.state["html_report_path"] = str(html_path)

        return input_data

    def _create_html_report(self, report_data: dict[str, Any], output_path: Path) -> None:
        """Создать HTML отчет."""
        from datetime import datetime

        # Собрать статистику
        total_steps = len(report_data["steps"])
        completed_steps = sum(1 for s in report_data["steps"] if s["status"] == "completed")
        failed_steps = sum(1 for s in report_data["steps"] if s["status"] == "failed")

        # Создать HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Full Pipeline Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    border-bottom: 3px solid #0066cc;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #555;
                    margin-top: 30px;
                }}
                .summary {{
                    display: flex;
                    justify-content: space-around;
                    margin: 30px 0;
                }}
                .summary-item {{
                    text-align: center;
                    padding: 20px;
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    min-width: 150px;
                }}
                .summary-value {{
                    font-size: 36px;
                    font-weight: bold;
                    color: #0066cc;
                }}
                .summary-label {{
                    color: #666;
                    margin-top: 5px;
                }}
                .step {{
                    margin: 15px 0;
                    padding: 15px;
                    background-color: #f8f9fa;
                    border-left: 4px solid #0066cc;
                    border-radius: 4px;
                }}
                .step.completed {{
                    border-left-color: #28a745;
                }}
                .step.failed {{
                    border-left-color: #dc3545;
                }}
                .step-name {{
                    font-weight: bold;
                    font-size: 18px;
                    margin-bottom: 5px;
                }}
                .step-status {{
                    display: inline-block;
                    padding: 3px 10px;
                    border-radius: 3px;
                    font-size: 12px;
                    text-transform: uppercase;
                }}
                .step-status.completed {{
                    background-color: #28a745;
                    color: white;
                }}
                .step-status.failed {{
                    background-color: #dc3545;
                    color: white;
                }}
                .config {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    font-family: monospace;
                    font-size: 14px;
                    white-space: pre-wrap;
                }}
                .timestamp {{
                    color: #999;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Full Pipeline Report</h1>
                <p class="timestamp">Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}</p>

                <div class="summary">
                    <div class="summary-item">
                        <div class="summary-value">{total_steps}</div>
                        <div class="summary-label">Total Steps</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-value">{completed_steps}</div>
                        <div class="summary-label">Completed</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-value">{failed_steps}</div>
                        <div class="summary-label">Failed</div>
                    </div>
                </div>

                <h2>Pipeline Steps</h2>
                {''.join([f'''
                <div class="step {s['status']}">
                    <div class="step-name">
                        {s['name']}
                        <span class="step-status {s['status']}">{s['status']}</span>
                    </div>
                    <div class="timestamp">
                        Started: {s.get('started_at', 'N/A')} |
                        Completed: {s.get('completed_at', 'N/A')}
                    </div>
                </div>
                ''' for s in report_data['steps']])}

                <h2>Configuration</h2>
                <div class="config">{json.dumps(report_data.get('config', {}), indent=2)}</div>

                <h2>Artifacts</h2>
                <div class="config">{json.dumps(report_data.get('artifacts', {}), indent=2)}</div>
            </div>
        </body>
        </html>
        """

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
