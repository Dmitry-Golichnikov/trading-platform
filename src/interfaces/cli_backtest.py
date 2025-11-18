"""CLI команды для бэктестинга."""

import json
import logging
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from ..backtesting import BacktestConfig, BacktestEngine, BacktestVisualizer, SimpleMAStrategy, StrategyMetrics
from ..common.logging import setup_logging

logger = logging.getLogger(__name__)


@click.group()
def backtest_cli():
    """Команды для бэктестинга."""
    pass


@backtest_cli.command("run")
@click.option(
    "--data",
    "-d",
    required=True,
    type=click.Path(exists=True),
    help="Путь к файлу с данными (CSV/Parquet)",
)
@click.option(
    "--strategy",
    "-s",
    default="simple_ma",
    type=click.Choice(["simple_ma", "model_based"]),
    help="Тип стратегии",
)
@click.option(
    "--model",
    "-m",
    type=click.Path(exists=True),
    help="Путь к модели (для model_based стратегии)",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Путь к конфигурации бэктеста (JSON)",
)
@click.option("--initial-capital", default=100000, help="Начальный капитал")
@click.option("--commission", default=0.001, help="Комиссия (0.001 = 0.1%)")
@click.option("--slippage", default=0.0005, help="Проскальзывание")
@click.option("--stop-loss", default=1.0, help="Стоп-лосс в процентах (0 = отключить)")
@click.option("--take-profit", default=2.0, help="Тейк-профит в процентах (0 = отключить)")
@click.option("--output", "-o", default="./backtest_results", help="Директория для результатов")
@click.option("--verbose", "-v", is_flag=True, help="Подробный вывод")
def run_backtest(
    data: str,
    strategy: str,
    model: Optional[str],
    config: Optional[str],
    initial_capital: float,
    commission: float,
    slippage: float,
    stop_loss: float,
    take_profit: float,
    output: str,
    verbose: bool,
):
    """Запустить бэктест стратегии.

    Пример:
        trading-platform backtest run -d data.parquet -s simple_ma --stop-loss 1.0 --take-profit 2.0
    """
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)

    click.echo("🚀 Starting backtest...")

    # Загружаем данные
    click.echo(f"📊 Loading data from {data}...")
    data_path = Path(data)
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    elif data_path.suffix == ".csv":
        df = pd.read_csv(data_path, parse_dates=["timestamp"])
    else:
        click.echo(f"❌ Unsupported file format: {data_path.suffix}", err=True)
        return

    click.echo(f"✅ Loaded {len(df)} rows")

    # Загружаем или создаем конфигурацию
    if config:
        click.echo(f"📋 Loading config from {config}...")
        with open(config, "r") as f:
            config_dict = json.load(f)
        bt_config = BacktestConfig(**config_dict)
    else:
        bt_config = BacktestConfig(
            initial_capital=initial_capital,
            commission_rate=commission,
            slippage_rate=slippage,
            use_stop_loss=stop_loss > 0,
            stop_loss_value=stop_loss,
            use_take_profit=take_profit > 0,
            take_profit_value=take_profit,
        )

    # Создаем стратегию
    click.echo(f"🎯 Creating {strategy} strategy...")
    if strategy == "simple_ma":
        strat = SimpleMAStrategy({"fast_period": 10, "slow_period": 30})
    elif strategy == "model_based":
        if not model:
            click.echo("❌ Model path required for model_based strategy", err=True)
            return
        # TODO: Load model
        click.echo("⚠️  Model-based strategy not fully implemented yet")
        return
    else:
        click.echo(f"❌ Unknown strategy: {strategy}", err=True)
        return

    # Запускаем бэктест
    click.echo("⚙️  Running backtest...")
    engine = BacktestEngine(bt_config)
    result = engine.run(strat, df, show_progress=True)

    # Вычисляем метрики
    click.echo("📈 Calculating metrics...")
    metrics_calc = StrategyMetrics(
        portfolio=result.portfolio,
        equity_curve=result.equity_curve,
        trades=result.trades,
    )
    all_metrics = metrics_calc.calculate_all()

    # Выводим результаты
    click.echo("\n" + "=" * 60)
    click.echo("📊 BACKTEST RESULTS")
    click.echo("=" * 60)
    click.echo(f"Total Return: {all_metrics['total_return']:.2f}%")
    click.echo(f"Total PnL: {all_metrics['total_pnl']:.2f}")
    click.echo(f"Sharpe Ratio: {all_metrics['sharpe_ratio']:.3f}")
    click.echo(f"Sortino Ratio: {all_metrics['sortino_ratio']:.3f}")
    click.echo(f"Max Drawdown: {all_metrics['max_drawdown']:.2f}%")
    click.echo(f"Win Rate: {all_metrics['win_rate']:.2f}%")
    click.echo(f"Profit Factor: {all_metrics['profit_factor']:.2f}")
    click.echo(f"Total Trades: {all_metrics['total_trades']}")
    click.echo(f"Winning Trades: {all_metrics['winning_trades']}")
    click.echo(f"Losing Trades: {all_metrics['losing_trades']}")
    click.echo("=" * 60)

    # Сохраняем результаты
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    click.echo(f"\n💾 Saving results to {output_path}...")

    # Сохраняем метрики
    metrics_path = output_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    click.echo(f"  ✓ Metrics saved to {metrics_path}")

    # Сохраняем equity curve
    equity_path = output_path / "equity_curve.csv"
    result.equity_curve.to_csv(equity_path, index=False)
    click.echo(f"  ✓ Equity curve saved to {equity_path}")

    # Сохраняем сделки
    trades_path = output_path / "trades.csv"
    trades_df = pd.DataFrame([t.to_dict() for t in result.trades])
    if not trades_df.empty:
        trades_df.to_csv(trades_path, index=False)
        click.echo(f"  ✓ Trades saved to {trades_path}")

    # Создаем графики
    click.echo("\n📊 Creating visualizations...")
    visualizer = BacktestVisualizer(
        equity_curve=result.equity_curve,
        trades=result.trades,
        portfolio_metrics=all_metrics,
    )
    visualizer.create_all_plots(output_path / "charts", show=False)
    click.echo(f"  ✓ Charts saved to {output_path / 'charts'}")

    click.echo("\n✅ Backtest completed successfully!")


@backtest_cli.command("report")
@click.option(
    "--results",
    "-r",
    required=True,
    type=click.Path(exists=True),
    help="Путь к директории с результатами бэктеста",
)
@click.option("--output", "-o", help="Путь для сохранения HTML отчета")
def generate_report(results: str, output: Optional[str]):
    """Сгенерировать HTML отчет по результатам бэктеста.

    Пример:
        trading-platform backtest report -r ./backtest_results -o report.html
    """
    click.echo("📝 Generating backtest report...")

    results_path = Path(results)

    # Загружаем метрики
    metrics_path = results_path / "metrics.json"
    if not metrics_path.exists():
        click.echo(f"❌ Metrics file not found: {metrics_path}", err=True)
        return

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # Загружаем equity curve
    equity_path = results_path / "equity_curve.csv"
    if equity_path.exists():
        equity_df = pd.read_csv(equity_path)
    else:
        equity_df = None

    # Загружаем сделки
    trades_path = results_path / "trades.csv"
    if trades_path.exists():
        trades_df = pd.read_csv(trades_path)
    else:
        trades_df = None

    # Генерируем HTML отчет
    html = _generate_html_report(metrics, equity_df, trades_df, results_path)

    # Сохраняем
    if output:
        output_path = Path(output)
    else:
        output_path = results_path / "report.html"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    click.echo(f"✅ Report saved to {output_path}")


def _generate_html_report(
    metrics: dict,
    equity_df: Optional[pd.DataFrame],
    trades_df: Optional[pd.DataFrame],
    charts_dir: Path,
) -> str:
    """Сгенерировать HTML отчет."""
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }}
        .metric-label {{
            font-size: 14px;
            color: #777;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .chart {{
            margin: 20px 0;
            text-align: center;
        }}
        .chart img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Backtest Report</h1>

        <h2>Key Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value">{metrics.get('total_return', 0):.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{metrics.get('sharpe_ratio', 0):.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value">{metrics.get('max_drawdown', 0):.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{metrics.get('win_rate', 0):.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Profit Factor</div>
                <div class="metric-value">{metrics.get('profit_factor', 0):.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{metrics.get('total_trades', 0)}</div>
            </div>
        </div>

        <h2>Charts</h2>
        <div class="chart">
            <h3>Equity Curve</h3>
            <img src="charts/equity_curve.png" alt="Equity Curve">
        </div>
        <div class="chart">
            <h3>Drawdown</h3>
            <img src="charts/drawdown.png" alt="Drawdown">
        </div>
        <div class="chart">
            <h3>Trades Analysis</h3>
            <img src="charts/trades.png" alt="Trades">
        </div>

        <h2>All Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
    """

    for key, value in metrics.items():
        formatted_key = key.replace("_", " ").title()
        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)
        html += f"<tr><td>{formatted_key}</td><td>{formatted_value}</td></tr>\n"

    html += """
        </table>
    </div>
</body>
</html>
"""

    return html


if __name__ == "__main__":
    backtest_cli()
