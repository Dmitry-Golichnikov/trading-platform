"""CLI команды для работы с моделями."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
import pandas as pd

from src.interfaces.cli.utils import (
    ProgressTracker,
    console,
    create_table,
    handle_errors,
    load_config,
    print_error,
    print_header,
    print_info,
    print_success,
)
from src.orchestration.mlflow_integration import MLflowManager


@click.group()
def models() -> None:
    """Команды для работы с моделями."""
    pass


@models.command("train")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Путь к конфигурации модели (YAML)",
)
@click.option(
    "--data",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Путь к данным для обучения (Parquet)",
)
@click.option(
    "--target",
    "-t",
    type=str,
    default="target",
    help="Имя колонки с таргетом",
)
@click.option(
    "--val-data",
    type=click.Path(exists=True, path_type=Path),
    help="Путь к валидационным данным (опционально)",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("artifacts/models"),
    help="Директория для сохранения модели",
)
@click.option(
    "--experiment-name",
    type=str,
    default="model-training",
    help="Имя эксперимента в MLflow",
)
@click.option(
    "--run-name",
    type=str,
    help="Имя запуска в MLflow",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "auto"]),
    default="auto",
    help="Устройство для обучения",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Подробный вывод",
)
@handle_errors
def train_model(
    config: Path,
    data: Path,
    target: str,
    val_data: Optional[Path],
    output_dir: Path,
    experiment_name: str,
    run_name: Optional[str],
    device: str,
    verbose: bool,
) -> None:
    """
    Обучить модель.

    Examples:
        $ trading-cli model train -c configs/models/lightgbm.yaml -d data/train.parquet
        $ trading-cli model train -c configs/models/lstm.yaml -d data/train.parquet --device cuda
    """
    from src.modeling.registry import ModelRegistry
    from src.modeling.trainer import ModelTrainer

    print_header("🎓 Обучение модели", f"Конфигурация: {config}")

    # Загрузка конфигурации
    print_info("Загрузка конфигурации...")
    model_config = load_config(config)

    # Загрузка данных
    print_info(f"Загрузка данных из {data}...")
    df = pd.read_parquet(data)
    print_success(f"Загружено {len(df):,} строк")

    # Проверка наличия таргета
    if target not in df.columns:
        print_error(f"Колонка таргета '{target}' не найдена в данных")
        raise click.Abort()

    # Разделение на признаки и таргет
    X = df.drop(columns=[target])
    y = df[target]

    # Загрузка валидационных данных если указаны
    X_val = None
    y_val = None
    if val_data:
        print_info(f"Загрузка валидационных данных из {val_data}...")
        val_df = pd.read_parquet(val_data)
        X_val = val_df.drop(columns=[target])
        y_val = val_df[target]
        print_success(f"Загружено {len(val_df):,} валидационных строк")

    # Создание модели
    model_type = model_config.get("type", "lightgbm")
    print_info(f"Создание модели типа: {model_type}")

    try:
        model_class = ModelRegistry.get_model(model_type)
        model_params = model_config.get("params", {})

        # Установка device для нейросетевых моделей
        if device != "auto" and "device" in model_params:
            model_params["device"] = device

        model = model_class(**model_params)
    except Exception as e:
        print_error(f"Ошибка создания модели: {e}")
        raise click.Abort()

    # Создание тренера
    trainer = ModelTrainer(
        model=model,
        experiment_name=experiment_name,
        verbose=verbose,
    )

    # Обучение модели
    with ProgressTracker("Обучение модели") as progress:
        try:
            training_result = trainer.train(
                X_train=X,
                y_train=y,
                X_val=X_val,
                y_val=y_val,
            )
            progress.update()
        except Exception as e:
            print_error(f"Ошибка при обучении: {e}")
            raise click.Abort()

    # Получение run_id из MLflow (если доступен)
    run_id = "unknown"
    if experiment_name:
        mlflow_manager = MLflowManager(experiment_name=experiment_name)
        mlflow_manager._setup_experiment()
        current_run = mlflow_manager.current_run
        if current_run:
            run_id = current_run.info.run_id

    # Сохранение модели
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{run_id}.pkl"
    training_result.model.save(model_path)

    # Вывод результатов
    print_success("✓ Обучение завершено")
    print_info(f"Model ID: {run_id}")
    print_info(f"Путь: {model_path}")

    # Вывод метрик
    if training_result.metrics:
        table = create_table("Метрики", ["Метрика", "Значение"])
        for metric_name, value in training_result.metrics.items():
            table.add_row(metric_name, f"{value:.4f}")
        console.print(table)


@models.command("list")
@click.option(
    "--experiment",
    "-e",
    type=str,
    help="Фильтр по имени эксперимента",
)
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
    default="created_at",
    help="Поле для сортировки",
)
@handle_errors
def list_models(experiment: Optional[str], limit: int, sort_by: str) -> None:
    """
    Показать список обученных моделей.

    Examples:
        $ trading-cli model list
        $ trading-cli model list -e model-training --limit 10
    """
    print_header("📋 Список моделей")

    # Получение списка моделей
    if experiment:
        mlflow_manager = MLflowManager(experiment_name=experiment)
        mlflow_manager._setup_experiment()
    else:
        mlflow_manager = MLflowManager()

    runs = mlflow_manager.search_runs(max_results=limit, order_by=[f"{sort_by} DESC"])

    if not runs:
        print_info("Модели не найдены")
        return

    # Вывод таблицы
    table = create_table(
        f"Модели (показано: {len(runs)})",
        ["Run ID", "Имя", "Модель", "Метрики", "Статус", "Дата"],
    )

    for run in runs:
        run_id = run.info.run_id[:8]
        run_name = run.data.tags.get("mlflow.runName", "N/A")
        model_type = run.data.params.get("model_type", "N/A")

        # Формируем строку с метриками
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in list(run.data.metrics.items())[:2]])

        status = run.info.status
        created_at = pd.Timestamp(run.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M")

        table.add_row(run_id, run_name, model_type, metrics_str, status, created_at)

    console.print(table)


@models.command("info")
@click.argument("model_id")
@click.option(
    "--show-params",
    is_flag=True,
    help="Показать параметры модели",
)
@click.option(
    "--show-metrics",
    is_flag=True,
    help="Показать все метрики",
)
@handle_errors
def model_info(model_id: str, show_params: bool, show_metrics: bool) -> None:
    """
    Показать информацию о модели.

    Examples:
        $ trading-cli model info abc123def
        $ trading-cli model info abc123def --show-params --show-metrics
    """
    print_header("📊 Информация о модели", f"Model ID: {model_id}")

    mlflow_manager = MLflowManager()

    # Поиск run
    try:
        run = mlflow_manager.get_run(model_id)
    except Exception as e:
        print_error(f"Модель не найдена: {e}")
        raise click.Abort()

    if run is None:
        print_error("Модель не найдена")
        raise click.Abort()

    # Основная информация
    info_table = create_table("Основная информация", ["Поле", "Значение"])
    info_table.add_row("Run ID", run.info.run_id)
    info_table.add_row("Имя", run.data.tags.get("mlflow.runName", "N/A"))
    info_table.add_row("Эксперимент", run.info.experiment_id)
    info_table.add_row("Статус", run.info.status)
    info_table.add_row(
        "Создан",
        pd.Timestamp(run.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M:%S"),
    )
    if run.info.end_time:
        duration = (run.info.end_time - run.info.start_time) / 1000
        info_table.add_row("Длительность", f"{duration:.1f}s")

    console.print(info_table)

    # Параметры
    if show_params and run.data.params:
        params_table = create_table("Параметры", ["Параметр", "Значение"])
        for key, value in sorted(run.data.params.items()):
            params_table.add_row(key, value)
        console.print(params_table)

    # Метрики
    if show_metrics and run.data.metrics:
        metrics_table = create_table("Метрики", ["Метрика", "Значение"])
        for key, value in sorted(run.data.metrics.items()):
            metrics_table.add_row(key, f"{value:.6f}")
        console.print(metrics_table)
    elif run.data.metrics:
        # Показываем только основные метрики
        metrics_table = create_table("Основные метрики", ["Метрика", "Значение"])
        main_metrics = ["accuracy", "roc_auc", "f1_score", "loss"]
        for metric in main_metrics:
            if metric in run.data.metrics:
                metrics_table.add_row(metric, f"{run.data.metrics[metric]:.6f}")
        console.print(metrics_table)


@models.command("evaluate")
@click.argument("model_id")
@click.option(
    "--data",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Путь к тестовым данным (Parquet)",
)
@click.option(
    "--target",
    "-t",
    type=str,
    default="target",
    help="Имя колонки с таргетом",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Путь для сохранения отчёта",
)
@handle_errors
def evaluate_model(model_id: str, data: Path, target: str, output: Optional[Path]) -> None:
    """
    Оценить модель на тестовых данных.

    Examples:
        $ trading-cli model evaluate abc123 -d data/test.parquet
        $ trading-cli model evaluate abc123 -d data/test.parquet -o report.html
    """
    from src.evaluation.evaluator import ModelEvaluator

    print_header("🎯 Оценка модели", f"Model ID: {model_id}")

    # Загрузка модели
    print_info("Загрузка модели...")
    mlflow_manager = MLflowManager()

    try:
        model = mlflow_manager.load_model(model_id)
    except Exception as e:
        print_error(f"Ошибка загрузки модели: {e}")
        raise click.Abort()

    if model is None:
        print_error("Модель не загружена")
        raise click.Abort()

    # Загрузка данных
    print_info(f"Загрузка данных из {data}...")
    df = pd.read_parquet(data)

    if target not in df.columns:
        print_error(f"Колонка таргета '{target}' не найдена")
        raise click.Abort()

    X = df.drop(columns=[target])
    y_true = df[target]

    # Оценка модели
    print_info("Вычисление метрик...")
    evaluator = ModelEvaluator(model)
    results = evaluator.evaluate(X, y_true)

    # Вывод результатов
    metrics_table = create_table("Метрики оценки", ["Метрика", "Значение"])
    for metric_name, value in results["metrics"].items():
        if isinstance(value, (int, float)):
            metrics_table.add_row(metric_name, f"{value:.4f}")
        else:
            metrics_table.add_row(metric_name, str(value))

    console.print(metrics_table)

    # Сохранение отчёта
    if output:
        evaluator.generate_report(X, y_true, output)
        print_success(f"Отчёт сохранён: {output}")


@models.command("predict")
@click.argument("model_id")
@click.option(
    "--data",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Путь к данным для прогноза (Parquet)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Путь для сохранения прогнозов",
)
@click.option(
    "--probabilities",
    is_flag=True,
    help="Сохранить вероятности (для классификации)",
)
@handle_errors
def predict(model_id: str, data: Path, output: Path, probabilities: bool) -> None:
    """
    Сделать прогноз с помощью модели.

    Examples:
        $ trading-cli model predict abc123 -d data/new.parquet -o predictions.parquet
        $ trading-cli model predict abc123 -d data/new.parquet -o predictions.parquet --probabilities
    """
    print_header("🔮 Прогнозирование", f"Model ID: {model_id}")

    # Загрузка модели
    print_info("Загрузка модели...")
    mlflow_manager = MLflowManager()

    try:
        model = mlflow_manager.load_model(model_id)
    except Exception as e:
        print_error(f"Ошибка загрузки модели: {e}")
        raise click.Abort()

    if model is None:
        print_error("Модель не загружена")
        raise click.Abort()

    # Загрузка данных
    print_info(f"Загрузка данных из {data}...")
    X = pd.read_parquet(data)
    print_success(f"Загружено {len(X):,} строк")

    # Прогноз
    print_info("Выполнение прогноза...")
    with ProgressTracker("Прогнозирование", total=len(X)) as progress:
        if probabilities and hasattr(model, "predict_proba"):
            predictions = model.predict_proba(X)
            result_df = pd.DataFrame(predictions, columns=["prob_0", "prob_1"])
        else:
            predictions = model.predict(X)
            result_df = pd.DataFrame({"prediction": predictions})

        progress.update(len(X))

    # Сохранение
    output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output, compression="snappy")
    print_success(f"Прогнозы сохранены: {output}")
    print_info(f"Всего прогнозов: {len(result_df):,}")


@models.command("compare")
@click.argument("model_ids", nargs=-1, required=True)
@click.option(
    "--metric",
    "-m",
    type=str,
    default="roc_auc",
    help="Метрика для сравнения",
)
@handle_errors
def compare_models(model_ids: tuple[str, ...], metric: str) -> None:
    """
    Сравнить несколько моделей.

    Examples:
        $ trading-cli model compare abc123 def456 ghi789
        $ trading-cli model compare abc123 def456 --metric f1_score
    """
    print_header("⚖️  Сравнение моделей", f"Метрика: {metric}")

    mlflow_manager = MLflowManager()

    # Получение информации о моделях
    models_data = []
    for model_id in model_ids:
        try:
            run = mlflow_manager.get_run(model_id)
            if run is None:
                print_error(f"Модель {model_id} не найдена")
                continue
            models_data.append(
                {
                    "id": model_id[:8],
                    "name": run.data.tags.get("mlflow.runName", "N/A"),
                    "metric": run.data.metrics.get(metric, 0.0),
                    "created": pd.Timestamp(run.info.start_time, unit="ms"),
                }
            )
        except Exception as e:
            print_error(f"Модель {model_id} не найдена: {e}")

    if not models_data:
        print_info("Нет данных для сравнения")
        return

    # Сортировка по метрике
    models_data.sort(key=lambda x: x["metric"], reverse=True)

    # Вывод таблицы
    table = create_table(f"Сравнение по {metric}", ["#", "Model ID", "Имя", metric, "Дата"])

    for i, model in enumerate(models_data, 1):
        rank = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else str(i)
        table.add_row(
            rank,
            model["id"],
            model["name"],
            f"{model['metric']:.4f}",
            model["created"].strftime("%Y-%m-%d"),
        )

    console.print(table)

    # Вывод победителя
    best = models_data[0]
    print_success(f"\n🏆 Лучшая модель: {best['id']} ({metric}={best['metric']:.4f})")


if __name__ == "__main__":
    models()
