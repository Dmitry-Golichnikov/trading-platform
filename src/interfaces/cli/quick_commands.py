"""Быстрые команды для частых операций."""

from __future__ import annotations

from pathlib import Path

import click

from src.interfaces.cli.utils import (
    console,
    create_table,
    handle_errors,
    print_header,
    print_info,
    print_success,
)


@click.group()
def quick() -> None:
    """Быстрые команды для частых операций."""
    pass


@quick.command("status")
@handle_errors
def system_status() -> None:
    """
    Показать статус системы и доступных ресурсов.

    Examples:
        $ trading-cli quick status
    """
    import psutil
    import torch

    from src.data.storage.catalog import DatasetCatalog
    from src.orchestration.mlflow_manager import MLflowManager

    print_header("📊 Статус системы")

    # Системные ресурсы
    system_table = create_table("Системные ресурсы", ["Ресурс", "Использование", "Доступно"])

    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    system_table.add_row("CPU", f"{cpu_percent}%", f"{cpu_count} cores")

    # Memory
    mem = psutil.virtual_memory()
    mem_used_gb = mem.used / (1024**3)
    mem_total_gb = mem.total / (1024**3)
    system_table.add_row("RAM", f"{mem.percent}%", f"{mem_total_gb - mem_used_gb:.1f}GB / {mem_total_gb:.1f}GB")

    # Disk
    disk = psutil.disk_usage(".")
    disk_used_gb = disk.used / (1024**3)
    disk_total_gb = disk.total / (1024**3)
    system_table.add_row("Disk", f"{disk.percent}%", f"{disk_total_gb - disk_used_gb:.1f}GB / {disk_total_gb:.1f}GB")

    # GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        system_table.add_row("GPU", "Available", f"{gpu_name} ({gpu_memory:.1f}GB)")
    else:
        system_table.add_row("GPU", "Not available", "—")

    console.print(system_table)

    # Данные
    print_info("\n📦 Данные:")
    catalog = DatasetCatalog()
    datasets = catalog.search()

    total_datasets = len(datasets)
    total_tickers = len(set(ds.ticker for ds in datasets))
    total_bars = sum(ds.total_bars for ds in datasets)

    data_table = create_table("", ["Метрика", "Значение"])
    data_table.add_row("Датасетов", str(total_datasets))
    data_table.add_row("Уникальных тикеров", str(total_tickers))
    data_table.add_row("Всего баров", f"{total_bars:,}")
    console.print(data_table)

    # Модели
    print_info("\n🤖 Модели:")
    try:
        mlflow_manager = MLflowManager()
        runs = mlflow_manager.search_runs(max_results=1000)
        total_models = len(runs)

        # Группировка по экспериментам
        experiments = set(run.info.experiment_id for run in runs)

        models_table = create_table("", ["Метрика", "Значение"])
        models_table.add_row("Всего моделей", str(total_models))
        models_table.add_row("Экспериментов", str(len(experiments)))
        console.print(models_table)
    except Exception:
        print_info("MLflow не доступен")

    # Артефакты
    print_info("\n📁 Артефакты:")
    artifacts_dir = Path("artifacts")
    if artifacts_dir.exists():
        total_size = sum(f.stat().st_size for f in artifacts_dir.rglob("*") if f.is_file())
        total_size_gb = total_size / (1024**3)

        artifacts_table = create_table("", ["Директория", "Файлов", "Размер"])

        for subdir in ["data", "features", "labels", "models", "backtests", "reports"]:
            path = artifacts_dir / subdir
            if path.exists():
                files_count = len(list(path.rglob("*")))
                size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                size_gb = size / (1024**3)
                artifacts_table.add_row(subdir, str(files_count), f"{size_gb:.2f}GB")

        console.print(artifacts_table)
        print_info(f"Всего артефактов: {total_size_gb:.2f}GB")
    else:
        print_info("Директория артефактов не найдена")


@quick.command("clean")
@click.option("--what", type=click.Choice(["cache", "logs", "temp", "all"]), default="cache", help="Что очистить")
@click.option("--force", is_flag=True, help="Не спрашивать подтверждение")
@handle_errors
def clean_artifacts(what: str, force: bool) -> None:
    """
    Очистить временные файлы и кэш.

    Examples:
        $ trading-cli quick clean --what cache
        $ trading-cli quick clean --what all --force
    """
    import shutil

    from src.interfaces.cli.utils import confirm

    print_header(f"🧹 Очистка: {what}")

    targets = []

    if what in ["cache", "all"]:
        targets.append(("Кэш признаков", Path("artifacts/features")))

    if what in ["logs", "all"]:
        targets.append(("Логи", Path("logs")))

    if what in ["temp", "all"]:
        targets.append(("Временные файлы", Path("tmp")))
        targets.append(("__pycache__", Path(".")))

    if not targets:
        print_info("Нечего очищать")
        return

    # Показать что будет удалено
    table = create_table("Будет удалено", ["Тип", "Путь", "Размер"])
    total_size = 0

    for name, path in targets:
        if path.exists():
            if name == "__pycache__":
                size = sum(f.stat().st_size for f in path.rglob("__pycache__") if f.is_file())
            else:
                size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            total_size += size
            size_mb = size / (1024**2)
            table.add_row(name, str(path), f"{size_mb:.2f}MB")

    console.print(table)
    print_info(f"Всего будет освобождено: {total_size / (1024**2):.2f}MB")

    # Подтверждение
    if not force:
        if not confirm("Продолжить?"):
            print_info("Отменено")
            return

    # Удаление
    deleted_count = 0
    for name, path in targets:
        if path.exists():
            try:
                if name == "__pycache__":
                    for pycache in path.rglob("__pycache__"):
                        shutil.rmtree(pycache)
                        deleted_count += 1
                else:
                    shutil.rmtree(path)
                    deleted_count += 1
                print_success(f"✓ {name} удалён")
            except Exception as e:
                console.print(f"[red]✗[/red] Ошибка удаления {name}: {e}")

    print_success(f"\n✓ Очистка завершена. Удалено: {deleted_count} элементов")


@quick.command("info")
@click.argument("item_type", type=click.Choice(["ticker", "model", "backtest"]))
@click.argument("item_id")
@handle_errors
def quick_info(item_type: str, item_id: str) -> None:
    """
    Быстро показать информацию.

    Examples:
        $ trading-cli quick info ticker SBER
        $ trading-cli quick info model abc123
        $ trading-cli quick info backtest bt_20231215
    """
    if item_type == "ticker":
        from src.data.storage.catalog import DatasetCatalog

        print_header(f"📊 Информация о тикере: {item_id}")

        catalog = DatasetCatalog()
        datasets = catalog.search(ticker=item_id)

        if not datasets:
            print_info(f"Данные для тикера {item_id} не найдены")
            return

        table = create_table(f"Датасеты {item_id}", ["Timeframe", "Start", "End", "Bars", "Source"])

        for ds in datasets:
            table.add_row(
                ds.timeframe, ds.start_date.isoformat(), ds.end_date.isoformat(), str(ds.total_bars), ds.source
            )

        console.print(table)

    elif item_type == "model":
        from src.orchestration.mlflow_manager import MLflowManager

        print_header(f"🤖 Информация о модели: {item_id}")

        mlflow_manager = MLflowManager()
        try:
            run = mlflow_manager.get_run(item_id)

            info_table = create_table("Информация", ["Поле", "Значение"])
            info_table.add_row("Run ID", run.info.run_id)
            info_table.add_row("Имя", run.data.tags.get("mlflow.runName", "N/A"))
            info_table.add_row("Статус", run.info.status)

            console.print(info_table)

            # Основные метрики
            if run.data.metrics:
                metrics_table = create_table("Метрики", ["Метрика", "Значение"])
                for key in ["accuracy", "roc_auc", "f1_score"]:
                    if key in run.data.metrics:
                        metrics_table.add_row(key, f"{run.data.metrics[key]:.4f}")
                console.print(metrics_table)

        except Exception as e:
            print_info(f"Модель не найдена: {e}")

    elif item_type == "backtest":
        import json

        print_header(f"📈 Информация о бэктесте: {item_id}")

        results_file = Path(f"artifacts/backtests/{item_id}_results.json")

        if not results_file.exists():
            print_info(f"Бэктест {item_id} не найден")
            return

        with open(results_file, "r") as f:
            results = json.load(f)

        table = create_table("Результаты", ["Метрика", "Значение"])
        table.add_row("Total Return", f"{results.get('total_return', 0) * 100:.2f}%")
        table.add_row("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.3f}")
        table.add_row("Max Drawdown", f"{results.get('max_drawdown', 0) * 100:.2f}%")
        table.add_row("Total Trades", str(results.get("total_trades", 0)))
        table.add_row("Win Rate", f"{results.get('win_rate', 0) * 100:.2f}%")

        console.print(table)


@quick.command("ls")
@click.argument("resource", type=click.Choice(["tickers", "models", "backtests", "features"]))
@click.option("--limit", "-l", type=int, default=10, help="Максимальное количество")
@handle_errors
def quick_list(resource: str, limit: int) -> None:
    """
    Быстро показать список ресурсов.

    Examples:
        $ trading-cli quick ls tickers
        $ trading-cli quick ls models -l 20
    """
    if resource == "tickers":
        from src.data.storage.catalog import DatasetCatalog

        print_header("📊 Доступные тикеры")

        catalog = DatasetCatalog()
        datasets = catalog.search()

        # Группировка по тикерам
        tickers_data: dict[str, list[str]] = {}
        for ds in datasets:
            if ds.ticker not in tickers_data:
                tickers_data[ds.ticker] = []
            tickers_data[ds.ticker].append(ds.timeframe)

        table = create_table(f"Тикеры (показано: {min(limit, len(tickers_data))})", ["Ticker", "Timeframes", "Count"])

        for ticker in sorted(tickers_data.keys())[:limit]:
            timeframes = ", ".join(sorted(tickers_data[ticker]))
            table.add_row(ticker, timeframes, str(len(tickers_data[ticker])))

        console.print(table)

    elif resource == "models":
        from src.orchestration.mlflow_manager import MLflowManager

        print_header("🤖 Последние модели")

        mlflow_manager = MLflowManager()
        runs = mlflow_manager.search_runs(max_results=limit)

        if not runs:
            print_info("Модели не найдены")
            return

        table = create_table(f"Модели (показано: {len(runs)})", ["ID", "Имя", "Метрика", "Дата"])

        for run in runs:
            run_id = run.info.run_id[:8]
            name = run.data.tags.get("mlflow.runName", "N/A")

            # Первая доступная метрика
            metric_str = "—"
            if run.data.metrics:
                first_metric = list(run.data.metrics.items())[0]
                metric_str = f"{first_metric[0]}: {first_metric[1]:.4f}"

            date_str = run.info.start_time

            table.add_row(run_id, name, metric_str, str(date_str))

        console.print(table)

    elif resource == "backtests":
        import json

        print_header("📈 Последние бэктесты")

        backtest_dir = Path("artifacts/backtests")
        if not backtest_dir.exists():
            print_info("Бэктесты не найдены")
            return

        results = []
        for f in backtest_dir.glob("*_results.json"):
            try:
                with open(f, "r") as file:
                    data = json.load(file)
                    results.append(
                        {
                            "id": f.stem.replace("_results", ""),
                            "sharpe": data.get("sharpe_ratio", 0),
                            "return": data.get("total_return", 0),
                            "date": f.stat().st_mtime,
                        }
                    )
            except Exception:
                pass

        results.sort(key=lambda x: x["date"], reverse=True)
        results = results[:limit]

        if not results:
            print_info("Бэктесты не найдены")
            return

        table = create_table(f"Бэктесты (показано: {len(results)})", ["ID", "Return", "Sharpe", "Date"])

        for bt in results:
            table.add_row(bt["id"][-20:], f"{bt['return'] * 100:.2f}%", f"{bt['sharpe']:.3f}", str(bt["date"]))

        console.print(table)


if __name__ == "__main__":
    quick()
