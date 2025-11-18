"""Пайплайн бэктестинга."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.modeling.serialization import ModelSerializer
from src.pipelines.base import BasePipeline

logger = logging.getLogger(__name__)


class BacktestPipeline(BasePipeline):
    """
    Пайплайн бэктестинга стратегий.

    Этапы:
    1. Загрузка модели и данных
    2. Создание стратегии
    3. Запуск бэктеста
    4. Расчет метрик
    5. Сохранение результатов и отчета
    """

    @property
    def name(self) -> str:
        """Имя пайплайна."""
        return "backtest"

    def _get_steps(self) -> list[str]:
        """Получить список шагов."""
        return [
            "load_model_and_data",
            "create_strategy",
            "run_backtest",
            "calculate_metrics",
            "save_results",
        ]

    def _execute_step(self, step_name: str, input_data: Any) -> Any:
        """Выполнить шаг пайплайна."""
        if step_name == "load_model_and_data":
            return self._load_model_and_data()
        elif step_name == "create_strategy":
            return self._create_strategy(input_data)
        elif step_name == "run_backtest":
            return self._run_backtest(input_data)
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
        model = None
        if model_path:
            model_path = Path(model_path)
            logger.info("Loading model from %s", model_path)
            model = ModelSerializer.load(model_path)
        else:
            logger.info("No model specified, will use indicator-based strategy")

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

        logger.info("Loaded %d rows for backtest", len(df))

        return {"model": model, "data": df}

    def _create_strategy(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Создать стратегию."""
        strategy_type = self.config.get("strategy_type", "model_based")
        strategy_config = self.config.get("strategy_config", {})

        logger.info("Creating strategy: %s", strategy_type)

        if strategy_type == "model_based":
            from src.backtesting.strategy import ModelBasedStrategy

            threshold = strategy_config.get("threshold", 0.5)
            strategy = ModelBasedStrategy(
                model=input_dict["model"],
                config=strategy_config,
                threshold=threshold,
            )
        else:
            raise ValueError(f"Unsupported strategy_type '{strategy_type}' in backtest pipeline")

        input_dict["strategy"] = strategy
        return input_dict

    def _run_backtest(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Запустить бэктест."""
        data = input_dict["data"]
        strategy = input_dict["strategy"]

        # Конфигурация бэктеста
        initial_capital = float(self.config.get("initial_capital", 100000.0))
        commission = float(self.config.get("commission", 0.001))
        slippage = float(self.config.get("slippage", 0.0005))

        logger.info("Running backtest with initial capital: %.2f", initial_capital)

        # Создать движок
        config_overrides: Dict[str, Any] = self.config.get("backtest_config", {})
        config_kwargs: Dict[str, Any] = {
            "initial_capital": initial_capital,
            "commission_rate": commission,
            "slippage_rate": slippage,
        }
        for key, value in config_overrides.items():
            if key in BacktestConfig.__dataclass_fields__:
                config_kwargs[key] = value

        engine = BacktestEngine(config=BacktestConfig(**config_kwargs))

        # Запустить
        results = engine.run(strategy=strategy, data=data, show_progress=self.config.get("show_progress", False))

        logger.info("Backtest completed: %d trades executed", len(results.trades))

        input_dict["results"] = results
        return input_dict

    def _calculate_metrics(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Рассчитать метрики."""
        results = input_dict["results"]

        # Рассчитать метрики стратегии
        metrics = results.calculate_metrics()

        logger.info("Backtest metrics:")
        logger.info("  Total return: %.2f%%", metrics.get("total_return", 0) * 100)
        logger.info("  Sharpe ratio: %.2f", metrics.get("sharpe_ratio", 0))
        logger.info("  Max drawdown: %.2f%%", metrics.get("max_drawdown", 0) * 100)
        logger.info("  Win rate: %.2f%%", metrics.get("win_rate", 0) * 100)

        self.state["metrics"] = metrics
        input_dict["metrics"] = metrics

        return input_dict

    def _save_results(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Сохранить результаты бэктеста."""
        results = input_dict["results"]
        metrics = input_dict["metrics"]

        output_dir = self.config.get("output_dir")
        if not output_dir:
            output_dir = "artifacts/backtests"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Сохранить сделки
        trades_df = pd.DataFrame([trade.__dict__ for trade in results.trades])
        if not trades_df.empty:
            trades_path = output_dir / "trades.parquet"
            trades_df.to_parquet(trades_path)
            logger.info("Saved %d trades to %s", len(trades_df), trades_path)
            self.state["trades_path"] = str(trades_path)

        # Сохранить equity curve
        equity_df = pd.DataFrame(
            {
                "timestamp": results.equity_curve.index,
                "equity": results.equity_curve.values,
            }
        )
        equity_path = output_dir / "equity_curve.parquet"
        equity_df.to_parquet(equity_path)
        logger.info("Saved equity curve to %s", equity_path)
        self.state["equity_path"] = str(equity_path)

        # Сохранить метрики
        import json

        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved metrics to %s", metrics_path)
        self.state["metrics_path"] = str(metrics_path)

        # Создать HTML отчет
        if self.config.get("generate_report", True):
            report_path = output_dir / "backtest_report.html"
            self._generate_html_report(results, metrics, report_path)
            logger.info("Generated HTML report: %s", report_path)
            self.state["report_path"] = str(report_path)

        self.state["output_dir"] = str(output_dir)

        return input_dict

    def _generate_html_report(self, results: Any, metrics: dict[str, Any], output_path: Path) -> None:
        """Генерировать HTML отчет."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Создать графики
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Equity Curve", "Drawdown"),
            vertical_spacing=0.1,
        )

        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=results.equity_curve.index,
                y=results.equity_curve.values,
                mode="lines",
                name="Equity",
            ),
            row=1,
            col=1,
        )

        # Drawdown
        if hasattr(results, "drawdown"):
            fig.add_trace(
                go.Scatter(
                    x=results.drawdown.index,
                    y=results.drawdown.values,
                    mode="lines",
                    name="Drawdown",
                    fill="tozeroy",
                ),
                row=2,
                col=1,
            )

        fig.update_layout(height=800, showlegend=True, title_text="Backtest Results")

        # Сгенерировать HTML
        metrics_html = "\n".join(
            [
                (
                    '<div class="metric">'
                    f'<span class="metric-label">{key}:</span> '
                    f'<span class="metric-value">{value:.4f}</span>'
                    "</div>"
                )
                for key, value in metrics.items()
            ]
        )

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .metrics {{ margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
                .metric-label {{ font-weight: bold; }}
                .metric-value {{ color: #0066cc; }}
            </style>
        </head>
        <body>
            <h1>Backtest Report</h1>

            <div class="metrics">
                <h2>Performance Metrics</h2>
                {metrics_html}
            </div>

            <div class="charts">
                <h2>Charts</h2>
                {fig.to_html(full_html=False)}
            </div>

            <div class="trades">
                <h2>Trade Statistics</h2>
                <p>Total trades: {len(results.trades)}</p>
            </div>
        </body>
        </html>
        """

        with open(output_path, "w") as f:
            f.write(html_content)
