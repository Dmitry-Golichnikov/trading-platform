"""CLI команды для бэктестинга торговых стратегий."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
import pandas as pd

from src.interfaces.cli.utils import (
    ProgressTracker,
    console,
    create_table,
    format_duration,
    format_number,
    format_percentage,
    handle_errors,
    load_config,
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
)


@click.group()
def backtest() -> None:
    """Команды для бэктестинга стратегий."""
    pass


@backtest.command("run")
@click.option(
    "--strategy",
    "-s",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Путь к конфигурации стратегии (YAML)",
)
@click.option(
    "--model-id",
    "-m",
    type=str,
    help="ID модели из MLflow",
)
@click.option(
    "--model-path",
    type=click.Path(exists=True, path_type=Path),
    help="Путь к файлу модели (альтернатива model-id)",
)
@click.option(
    "--data",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Путь к историческим данным (Parquet)",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("artifacts/backtests"),
    help="Директория для сохранения результатов",
)
@click.option(
    "--initial-capital",
    type=float,
    default=100000.0,
    help="Начальный капитал",
)
@click.option(
    "--commission",
    type=float,
    default=0.001,
    help="Комиссия (долями, например 0.001 = 0.1%)",
)
@click.option(
    "--slippage",
    type=float,
    default=0.0005,
    help="Проскальзывание (долями)",
)
@click.option(
    "--plot/--no-plot",
    default=True,
    help="Создать графики",
)
@handle_errors
def run_backtest(
    strategy: Path,
    model_id: Optional[str],
    model_path: Optional[Path],
    data: Path,
    output_dir: Path,
    initial_capital: float,
    commission: float,
    slippage: float,
    plot: bool,
) -> None:
    """
    Запустить бэктест стратегии.

    Examples:
        $ trading-cli backtest run -s configs/strategy.yaml -m abc123 -d data/test.parquet
        $ trading-cli backtest run -s configs/strategy.yaml --model-path model.pkl -d data/test.parquet
    """
    import json

    from src.backtesting.engine import BacktestConfig, BacktestEngine
    from src.orchestration.mlflow_integration import MLflowManager

    print_header("📈 Запуск бэктеста", f"Стратегия: {strategy.name}")

    # Проверка параметров модели
    if not model_id and not model_path:
        print_error("Необходимо указать либо --model-id, либо --model-path")
        raise click.Abort()

    # Загрузка стратегии
    print_info("Загрузка конфигурации стратегии...")
    strategy_config = load_config(strategy)

    # Загрузка модели
    if model_id:
        print_info(f"Загрузка модели из MLflow: {model_id}")
        mlflow_manager = MLflowManager()
        try:
            model = mlflow_manager.load_model(model_id)
        except Exception as e:
            print_error(f"Ошибка загрузки модели: {e}")
            raise click.Abort()
    else:
        print_info(f"Загрузка модели из файла: {model_path}")
        from src.modeling.base import BaseModel

        if model_path is None:
            print_error("model_path не может быть None")
            raise click.Abort()

        try:
            model = BaseModel.load(model_path)
        except Exception as e:
            print_error(f"Ошибка загрузки модели: {e}")
            raise click.Abort()

    # Загрузка данных
    print_info(f"Загрузка данных из {data}...")
    df = pd.read_parquet(data)
    print_success(f"Загружено {len(df):,} баров")

    # Создание стратегии (упрощённая версия - нужно создать BaseStrategy)
    print_info("Создание стратегии...")
    # TODO: Реализовать создание стратегии из конфига
    # strategy_obj = create_strategy_from_config(strategy_config, model, initial_capital)
    print_warning("Создание стратегии из конфига не реализовано, используем ModelBasedStrategy")
    from src.backtesting.strategy import ModelBasedStrategy

    strategy_obj = ModelBasedStrategy(
        model=model,
        config={"name": strategy_config.get("name", "CLI Strategy"), "initial_capital": initial_capital},
    )

    # Создание конфигурации бэктеста
    backtest_config = BacktestConfig(
        initial_capital=initial_capital,
        commission_rate=commission,
        slippage_rate=slippage,
    )

    # Создание движка бэктестинга
    engine = BacktestEngine(config=backtest_config)

    # Запуск бэктеста
    print_info("Выполнение бэктеста...")
    with ProgressTracker("Бэктест", total=len(df)) as progress:
        result = engine.run(strategy=strategy_obj, data=df, show_progress=False)
        progress.update(len(df))

    # Вывод результатов
    print_success("✓ Бэктест завершён")

    # Таблица с метриками
    metrics_table = create_table("Результаты бэктеста", ["Метрика", "Значение"])

    # Основные метрики из result.metrics и equity_curve
    if not result.equity_curve.empty:
        final_equity = float(result.equity_curve.iloc[-1]["equity"])
    else:
        # Fallback: используем cash + позиции
        final_equity = result.portfolio.cash
        for position in result.portfolio.positions.values():
            if position.is_open and position.exit_price:
                final_equity += position.size * position.exit_price

    total_return = (final_equity - initial_capital) / initial_capital
    total_pnl = final_equity - initial_capital

    metrics_table.add_row("Начальный капитал", f"${format_number(initial_capital, 2)}")
    metrics_table.add_row("Конечный капитал", f"${format_number(final_equity, 2)}")
    metrics_table.add_row(
        "Прибыль/Убыток",
        f"${format_number(total_pnl, 2)} ({format_percentage(total_return * 100)})",
    )

    # Метрики из result.metrics
    total_trades = len(result.trades)
    winning_trades = sum(1 for trade in result.trades if trade.realized_pnl > 0)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

    metrics_table.add_row("Всего сделок", str(total_trades))
    metrics_table.add_row("Прибыльных сделок", f"{winning_trades} ({format_percentage(win_rate * 100)})")

    # Метрики из словаря metrics
    sharpe = result.metrics.get("sharpe_ratio", 0.0)
    sortino = result.metrics.get("sortino_ratio", 0.0)
    max_dd = result.metrics.get("max_drawdown", 0.0)
    calmar = result.metrics.get("calmar_ratio", 0.0)
    profit_factor = result.metrics.get("profit_factor", 0.0)

    metrics_table.add_row("Sharpe Ratio", f"{sharpe:.3f}")
    metrics_table.add_row("Sortino Ratio", f"{sortino:.3f}")
    metrics_table.add_row("Max Drawdown", f"{format_percentage(max_dd * 100)}")
    metrics_table.add_row("Calmar Ratio", f"{calmar:.3f}")
    metrics_table.add_row("Profit Factor", f"{profit_factor:.3f}")

    console.print(metrics_table)

    # Сохранение результатов
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    backtest_id = f"backtest_{timestamp}"

    # Сохранение метрик
    results_path = output_dir / f"{backtest_id}_results.json"
    metrics_dict = {
        "final_equity": final_equity,
        "total_pnl": total_pnl,
        "total_return": total_return,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "win_rate": win_rate,
        **result.metrics,
    }
    with open(results_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print_info(f"Результаты сохранены: {results_path}")

    # Сохранение сделок
    if result.trades:
        trades_data = [
            {
                "entry_time": (
                    trade.entry_time.isoformat() if hasattr(trade.entry_time, "isoformat") else str(trade.entry_time)
                ),
                "exit_time": (
                    trade.exit_time.isoformat()
                    if trade.exit_time is not None and hasattr(trade.exit_time, "isoformat")
                    else str(trade.exit_time) if trade.exit_time is not None else ""
                ),
                "side": trade.side.value if hasattr(trade.side, "value") else str(trade.side),
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price if trade.exit_price is not None else 0.0,
                "size": trade.size,
                "realized_pnl": trade.realized_pnl,
            }
            for trade in result.trades
        ]
        trades_df = pd.DataFrame(trades_data)
        trades_path = output_dir / f"{backtest_id}_trades.parquet"
        trades_df.to_parquet(trades_path, compression="snappy")
        print_info(f"Сделки сохранены: {trades_path}")

    # Сохранение equity curve
    if not result.equity_curve.empty:
        equity_path = output_dir / f"{backtest_id}_equity.parquet"
        result.equity_curve.to_parquet(equity_path, compression="snappy")
        print_info(f"Equity curve сохранён: {equity_path}")

    # Создание графиков
    if plot:
        from src.backtesting.visualization import BacktestVisualizer

        print_info("Создание графиков...")
        visualizer = BacktestVisualizer(result)
        plot_path = output_dir / f"{backtest_id}_plots.html"
        visualizer.create_report(plot_path)
        print_success(f"Графики сохранены: {plot_path}")

    print_success(f"\n✓ Backtest ID: {backtest_id}")


@backtest.command("list")
@click.option(
    "--limit",
    "-l",
    type=int,
    default=20,
    help="Максимальное количество результатов",
)
@click.option(
    "--sort-by",
    type=str,
    default="date",
    help="Поле для сортировки (date, sharpe, return)",
)
@handle_errors
def list_backtests(limit: int, sort_by: str) -> None:
    """
    Показать список бэктестов.

    Examples:
        $ trading-cli backtest list
        $ trading-cli backtest list --limit 10 --sort-by sharpe
    """
    import json

    print_header("📋 Список бэктестов")

    backtest_dir = Path("artifacts/backtests")
    if not backtest_dir.exists():
        print_info("Директория бэктестов не найдена")
        return

    # Поиск всех результатов
    result_files = list(backtest_dir.glob("*_results.json"))

    if not result_files:
        print_info("Бэктесты не найдены")
        return

    # Загрузка данных
    backtests = []
    for result_file in result_files:
        try:
            with open(result_file, "r") as f:
                data = json.load(f)
                backtest_id = result_file.stem.replace("_results", "")
                backtests.append(
                    {
                        "id": backtest_id,
                        "date": result_file.stat().st_mtime,
                        "sharpe": data.get("sharpe_ratio", 0.0),
                        "return": data.get("total_return", 0.0),
                        "trades": data.get("total_trades", 0),
                        "win_rate": data.get("win_rate", 0.0),
                    }
                )
        except Exception as e:
            print_error(f"Ошибка загрузки {result_file.name}: {e}")

    # Сортировка
    if sort_by == "date":
        backtests.sort(key=lambda x: x["date"], reverse=True)
    elif sort_by == "sharpe":
        backtests.sort(key=lambda x: x["sharpe"], reverse=True)
    elif sort_by == "return":
        backtests.sort(key=lambda x: x["return"], reverse=True)

    # Ограничение
    backtests = backtests[:limit]

    # Вывод таблицы
    table = create_table(
        f"Бэктесты (показано: {len(backtests)})",
        ["ID", "Дата", "Return", "Sharpe", "Сделок", "Win Rate"],
    )

    for bt in backtests:
        date_str = pd.Timestamp(bt["date"], unit="s").strftime("%Y-%m-%d %H:%M")
        table.add_row(
            bt["id"][-20:],  # Показываем последние 20 символов
            date_str,
            f"{format_percentage(bt['return'] * 100)}",
            f"{bt['sharpe']:.3f}",
            str(bt["trades"]),
            f"{format_percentage(bt['win_rate'] * 100)}",
        )

    console.print(table)


@backtest.command("report")
@click.argument("backtest_id")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["html", "pdf", "json"]),
    default="html",
    help="Формат отчёта",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Путь для сохранения отчёта",
)
@handle_errors
def backtest_report(backtest_id: str, output_format: str, output: Optional[Path]) -> None:
    """
    Создать подробный отчёт по бэктесту.

    Examples:
        $ trading-cli backtest report backtest_20231215_120000
        $ trading-cli backtest report backtest_20231215_120000 --format pdf -o report.pdf
    """
    import json

    from src.backtesting.visualization import BacktestVisualizer

    print_header("📊 Генерация отчёта", f"Backtest ID: {backtest_id}")

    # Поиск результатов
    backtest_dir = Path("artifacts/backtests")
    results_file = backtest_dir / f"{backtest_id}_results.json"

    if not results_file.exists():
        print_error(f"Бэктест не найден: {backtest_id}")
        raise click.Abort()

    # Загрузка результатов
    print_info("Загрузка результатов...")
    with open(results_file, "r") as f:
        results_data = json.load(f)

    # Загрузка equity curve
    equity_file = backtest_dir / f"{backtest_id}_equity.parquet"
    equity_curve = None
    if equity_file.exists():
        equity_curve = pd.read_parquet(equity_file)

    # Загрузка сделок
    trades_file = backtest_dir / f"{backtest_id}_trades.parquet"
    trades = None
    if trades_file.exists():
        trades = pd.read_parquet(trades_file)

    # Создание визуализатора
    from src.backtesting.results import BacktestResults

    results = BacktestResults(**results_data)
    results.equity_curve = equity_curve
    results.trades = trades.to_dict("records") if trades is not None else []

    visualizer = BacktestVisualizer(results)

    # Генерация отчёта
    if output is None:
        output = backtest_dir / f"{backtest_id}_report.{output_format}"

    print_info(f"Генерация отчёта в формате {output_format}...")

    if output_format == "html":
        visualizer.create_report(output)
    elif output_format == "json":
        with open(output, "w") as f:
            json.dump(results_data, f, indent=2)
    elif output_format == "pdf":
        # Сначала создаём HTML, затем конвертируем в PDF
        html_path = output.with_suffix(".html")
        visualizer.create_report(html_path)

        try:
            import pdfkit

            pdfkit.from_file(str(html_path), str(output))
            html_path.unlink()  # Удаляем временный HTML
        except ImportError:
            print_error("Для генерации PDF требуется установить pdfkit и wkhtmltopdf")
            print_info(f"HTML отчёт сохранён: {html_path}")
            return

    print_success(f"Отчёт сохранён: {output}")


@backtest.command("compare")
@click.argument("backtest_ids", nargs=-1, required=True)
@click.option(
    "--metric",
    "-m",
    type=str,
    default="sharpe_ratio",
    help="Метрика для сравнения",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Путь для сохранения отчёта сравнения",
)
@handle_errors
def compare_backtests(backtest_ids: tuple[str, ...], metric: str, output: Optional[Path]) -> None:
    """
    Сравнить результаты нескольких бэктестов.

    Examples:
        $ trading-cli backtest compare bt1 bt2 bt3
        $ trading-cli backtest compare bt1 bt2 --metric total_return -o comparison.html
    """
    import json

    print_header("⚖️  Сравнение бэктестов", f"Метрика: {metric}")

    backtest_dir = Path("artifacts/backtests")

    # Загрузка данных всех бэктестов
    backtests_data = []
    for backtest_id in backtest_ids:
        results_file = backtest_dir / f"{backtest_id}_results.json"

        if not results_file.exists():
            print_error(f"Бэктест не найден: {backtest_id}")
            continue

        try:
            with open(results_file, "r") as f:
                data = json.load(f)
                backtests_data.append({"id": backtest_id, **data})
        except Exception as e:
            print_error(f"Ошибка загрузки {backtest_id}: {e}")

    if not backtests_data:
        print_info("Нет данных для сравнения")
        return

    # Сортировка по метрике
    backtests_data.sort(key=lambda x: x.get(metric, 0.0), reverse=True)

    # Вывод таблицы
    table = create_table(
        f"Сравнение по {metric}",
        ["#", "ID", "Return", "Sharpe", "Sortino", "Max DD", "Trades", "Win Rate"],
    )

    for i, bt in enumerate(backtests_data, 1):
        rank = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else str(i)
        table.add_row(
            rank,
            bt["id"][-20:],
            f"{format_percentage(bt.get('total_return', 0) * 100)}",
            f"{bt.get('sharpe_ratio', 0):.3f}",
            f"{bt.get('sortino_ratio', 0):.3f}",
            f"{format_percentage(bt.get('max_drawdown', 0) * 100)}",
            str(bt.get("total_trades", 0)),
            f"{format_percentage(bt.get('win_rate', 0) * 100)}",
        )

    console.print(table)

    # Вывод победителя
    best = backtests_data[0]
    print_success(f"\n🏆 Лучший бэктест: {best['id']} ({metric}={best.get(metric, 0):.4f})")

    # Создание отчёта сравнения
    if output:
        from src.backtesting.visualization import create_comparison_report

        print_info("Создание отчёта сравнения...")
        create_comparison_report(backtests_data, output)
        print_success(f"Отчёт сохранён: {output}")


@backtest.command("analyze")
@click.argument("backtest_id")
@handle_errors
def analyze_backtest(backtest_id: str) -> None:
    """
    Провести углублённый анализ бэктеста.

    Examples:
        $ trading-cli backtest analyze backtest_20231215_120000
    """
    print_header("🔍 Углублённый анализ", f"Backtest ID: {backtest_id}")

    backtest_dir = Path("artifacts/backtests")
    results_file = backtest_dir / f"{backtest_id}_results.json"
    trades_file = backtest_dir / f"{backtest_id}_trades.parquet"

    if not results_file.exists():
        print_error(f"Бэктест не найден: {backtest_id}")
        raise click.Abort()

    # Загрузка результатов (для будущего использования)
    # with open(results_file, "r") as f:
    #     results_data = json.load(f)

    # Загрузка сделок
    if not trades_file.exists():
        print_error("Файл сделок не найден")
        raise click.Abort()

    trades = pd.read_parquet(trades_file)

    # Анализ сделок
    print_info(f"\nВсего сделок: {len(trades)}")

    # Прибыльные/убыточные
    profitable = trades[trades["pnl"] > 0]
    losing = trades[trades["pnl"] < 0]

    trades_table = create_table("Анализ сделок", ["Тип", "Количество", "Средний PnL"])
    trades_table.add_row("Прибыльные", str(len(profitable)), f"${profitable['pnl'].mean():.2f}")
    trades_table.add_row("Убыточные", str(len(losing)), f"${losing['pnl'].mean():.2f}")
    trades_table.add_row("Всего", str(len(trades)), f"${trades['pnl'].mean():.2f}")

    console.print(trades_table)

    # Анализ длительности
    if "duration" in trades.columns:
        duration_table = create_table("Длительность сделок", ["Метрика", "Значение"])
        duration_table.add_row("Средняя", format_duration(trades["duration"].mean()))
        duration_table.add_row("Медианная", format_duration(trades["duration"].median()))
        duration_table.add_row("Максимальная", format_duration(trades["duration"].max()))

        console.print(duration_table)

    # Анализ по месяцам/дням недели
    if "entry_time" in trades.columns:
        trades["entry_time"] = pd.to_datetime(trades["entry_time"])
        trades["month"] = trades["entry_time"].dt.month
        trades["dow"] = trades["entry_time"].dt.day_name()

        monthly_pnl = trades.groupby("month")["pnl"].agg(["sum", "count"])
        print_info("\nПрибыль по месяцам:")
        for month, row in monthly_pnl.iterrows():
            print(f"  Месяц {month}: ${row['sum']:.2f} ({row['count']} сделок)")

    print_success("\n✓ Анализ завершён")


if __name__ == "__main__":
    backtest()
