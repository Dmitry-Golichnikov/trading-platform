"""CLI команды для оценки моделей."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import click
import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.evaluation import (
    DriftDetector,
    FeatureImportanceAnalyzer,
    ModelCalibrator,
    ModelEvaluationReport,
)


@click.group(name="evaluate")
def evaluate_group():
    """Команды для оценки моделей."""
    pass


@evaluate_group.command(name="model")
@click.option(
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="Путь к сохраненной модели",
)
@click.option(
    "--test-data",
    type=click.Path(exists=True),
    required=True,
    help="Путь к тестовым данным (Parquet/CSV)",
)
@click.option(
    "--train-data",
    type=click.Path(exists=True),
    help="Путь к тренировочным данным (для drift detection)",
)
@click.option(
    "--target-column",
    type=str,
    default="target",
    help="Название колонки с таргетом",
)
@click.option(
    "--task-type",
    type=click.Choice(["classification", "regression"]),
    default="classification",
    help="Тип задачи",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="artifacts/reports",
    help="Директория для сохранения отчёта",
)
@click.option(
    "--report-name",
    type=str,
    help="Название отчёта (по умолчанию model_evaluation_<timestamp>)",
)
@click.option(
    "--sections",
    type=str,
    default="metrics,calibration,importance,drift",
    help="Секции отчёта (через запятую)",
)
def evaluate_model(
    model_path: str,
    test_data: str,
    train_data: Optional[str],
    target_column: str,
    task_type: str,
    output_dir: str,
    report_name: Optional[str],
    sections: str,
):
    """Оценить модель и сгенерировать отчёт."""
    try:
        click.echo(f"Загрузка модели из {model_path}...")
        model = joblib.load(model_path)

        click.echo(f"Загрузка тестовых данных из {test_data}...")
        if test_data.endswith(".parquet"):
            df_test = pd.read_parquet(test_data)
        else:
            df_test = pd.read_csv(test_data)

        # Разделяем на X и y
        if target_column not in df_test.columns:
            click.echo(f"Ошибка: колонка '{target_column}' не найдена в данных", err=True)
            sys.exit(1)

        y_test = np.asarray(df_test[target_column].values)
        X_test = df_test.drop(columns=[target_column])
        feature_names = X_test.columns.tolist()

        # Загрузка train данных для drift
        X_train = None
        y_train = None
        if train_data:
            click.echo(f"Загрузка тренировочных данных из {train_data}...")
            if train_data.endswith(".parquet"):
                df_train = pd.read_parquet(train_data)
            else:
                df_train = pd.read_csv(train_data)

            y_train = np.asarray(df_train[target_column].values)
            X_train = df_train.drop(columns=[target_column])

        # Создаем отчёт
        click.echo("Генерация отчёта...")
        report = ModelEvaluationReport(
            model=model,
            task_type=task_type,
            model_name=Path(model_path).stem,
        )

        # Определяем секции
        include_sections = [s.strip() for s in sections.split(",")]

        # Генерируем отчёт
        from datetime import datetime

        if report_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"model_evaluation_{timestamp}"

        output_dir_path = Path(output_dir)
        html_path = output_dir_path / f"{report_name}.html"
        json_path = output_dir_path / f"{report_name}.json"

        report_data = report.generate(
            X_test=X_test,
            y_test=y_test,
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_names,
            output_path=html_path,
            include_sections=include_sections,
        )

        # Сохраняем JSON
        report.save_json(json_path)

        click.echo("\n✓ Отчёт успешно сгенерирован!")
        click.echo(f"  HTML: {html_path}")
        click.echo(f"  JSON: {json_path}")

        # Выводим основные метрики
        if "metrics" in report_data:
            click.echo("\nОсновные метрики:")
            metrics = report_data["metrics"]
            for key, value in list(metrics.items())[:10]:
                if isinstance(value, (int, float, np.integer, np.floating)):
                    click.echo(f"  {key}: {value:.4f}")

    except Exception as e:
        logger.error(f"Ошибка при оценке модели: {e}", exc_info=True)
        click.echo(f"Ошибка: {e}", err=True)
        sys.exit(1)


@evaluate_group.command(name="importance")
@click.option(
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="Путь к сохраненной модели",
)
@click.option(
    "--data",
    type=click.Path(exists=True),
    required=True,
    help="Путь к данным (Parquet/CSV)",
)
@click.option(
    "--target-column",
    type=str,
    default="target",
    help="Название колонки с таргетом",
)
@click.option(
    "--methods",
    type=str,
    default="tree,permutation",
    help="Методы для вычисления важности (через запятую: tree,permutation,shap)",
)
@click.option(
    "--top-n",
    type=int,
    default=20,
    help="Количество топ признаков для отображения",
)
@click.option(
    "--output",
    type=click.Path(),
    help="Путь для сохранения результатов (CSV)",
)
def feature_importance(
    model_path: str,
    data: str,
    target_column: str,
    methods: str,
    top_n: int,
    output: Optional[str],
):
    """Вычислить важность признаков."""
    try:
        click.echo(f"Загрузка модели из {model_path}...")
        model = joblib.load(model_path)

        click.echo(f"Загрузка данных из {data}...")
        if data.endswith(".parquet"):
            df = pd.read_parquet(data)
        else:
            df = pd.read_csv(data)

        # Разделяем на X и y
        if target_column not in df.columns:
            click.echo(f"Ошибка: колонка '{target_column}' не найдена в данных", err=True)
            sys.exit(1)

        y = np.asarray(df[target_column].values)
        X = df.drop(columns=[target_column])
        feature_names = X.columns.tolist()

        # Создаем анализатор
        analyzer = FeatureImportanceAnalyzer(model=model, X=X, y=y, feature_names=feature_names)

        # Вычисляем важность
        methods_list = [m.strip() for m in methods.split(",")]
        click.echo(f"Вычисление важности методами: {', '.join(methods_list)}...")

        importances = analyzer.compute_all_importances(methods=methods_list)

        # Выводим результаты
        for method, df_imp in importances.items():
            if not df_imp.empty:
                click.echo(f"\n{method.upper()} Feature Importance:")
                click.echo("-" * 60)

                # Топ N признаков
                df_top = df_imp.head(top_n)

                # Определяем колонку с важностью
                importance_col = None
                for col in ["importance", "importance_mean", "shap_importance"]:
                    if col in df_top.columns:
                        importance_col = col
                        break

                if importance_col:
                    for idx, row in df_top.iterrows():
                        feature = row["feature"]
                        importance = row[importance_col]
                        click.echo(f"  {feature:30s} {importance:.6f}")

                # Сохраняем в файл если указан
                if output:
                    output_path = Path(output)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    out_file = output_path.parent / f"{output_path.stem}_{method}{output_path.suffix}"
                    df_imp.to_csv(out_file, index=False)
                    click.echo(f"\n  Сохранено в: {out_file}")

    except Exception as e:
        logger.error(f"Ошибка при вычислении важности признаков: {e}", exc_info=True)
        click.echo(f"Ошибка: {e}", err=True)
        sys.exit(1)


@evaluate_group.command(name="calibrate")
@click.option(
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="Путь к сохраненной модели",
)
@click.option(
    "--data",
    type=click.Path(exists=True),
    required=True,
    help="Путь к данным для калибровки (Parquet/CSV)",
)
@click.option(
    "--target-column",
    type=str,
    default="target",
    help="Название колонки с таргетом",
)
@click.option(
    "--method",
    type=click.Choice(["isotonic", "platt", "beta"]),
    default="isotonic",
    help="Метод калибровки",
)
@click.option(
    "--output",
    type=click.Path(),
    help="Путь для сохранения калиброванной модели",
)
def calibrate_model(
    model_path: str,
    data: str,
    target_column: str,
    method: str,
    output: Optional[str],
):
    """Калибровать вероятности модели."""
    try:
        click.echo(f"Загрузка модели из {model_path}...")
        model = joblib.load(model_path)

        if not hasattr(model, "predict_proba"):
            click.echo("Ошибка: модель не поддерживает predict_proba", err=True)
            sys.exit(1)

        click.echo(f"Загрузка данных из {data}...")
        if data.endswith(".parquet"):
            df = pd.read_parquet(data)
        else:
            df = pd.read_csv(data)

        # Разделяем на X и y
        if target_column not in df.columns:
            click.echo(f"Ошибка: колонка '{target_column}' не найдена в данных", err=True)
            sys.exit(1)

        y = np.asarray(df[target_column].values)
        X = df.drop(columns=[target_column])

        # Получаем вероятности
        click.echo("Получение вероятностей...")
        y_proba = model.predict_proba(X)

        # Берем вероятности положительного класса
        if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba

        # Создаем калибратор
        click.echo(f"Калибровка методом {method}...")
        # method проверяется в ModelCalibrator
        calibrator = ModelCalibrator(method=method)  # type: ignore[arg-type]
        calibrator.fit(y_proba_pos, y)

        # Вычисляем метрики до и после калибровки
        from src.evaluation import CalibrationMetrics

        metrics_before = CalibrationMetrics.compute_all(y, y_proba_pos)
        y_proba_calibrated = calibrator.transform(y_proba_pos)
        metrics_after = CalibrationMetrics.compute_all(y, y_proba_calibrated)

        click.echo("\nМетрики калибровки:")
        click.echo("-" * 60)
        click.echo(f"{'Metric':<20} {'Before':>15} {'After':>15} {'Change':>15}")
        click.echo("-" * 60)

        for metric in metrics_before.keys():
            before = metrics_before[metric]
            after = metrics_after[metric]
            change = after - before
            change_str = f"{change:+.4f}"
            click.echo(f"{metric:<20} {before:>15.4f} {after:>15.4f} {change_str:>15}")

        # Сохраняем калибратор
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(calibrator, output_path)
            click.echo(f"\n✓ Калибратор сохранён в: {output_path}")
        else:
            # Сохраняем рядом с моделью
            model_path_obj = Path(model_path)
            calibrator_path = model_path_obj.parent / f"{model_path_obj.stem}_calibrator_{method}.pkl"
            joblib.dump(calibrator, calibrator_path)
            click.echo(f"\n✓ Калибратор сохранён в: {calibrator_path}")

    except Exception as e:
        logger.error(f"Ошибка при калибровке модели: {e}", exc_info=True)
        click.echo(f"Ошибка: {e}", err=True)
        sys.exit(1)


@evaluate_group.command(name="drift")
@click.option(
    "--reference-data",
    type=click.Path(exists=True),
    required=True,
    help="Путь к референсным данным (обычно train)",
)
@click.option(
    "--current-data",
    type=click.Path(exists=True),
    required=True,
    help="Путь к текущим данным (обычно test/production)",
)
@click.option(
    "--exclude-columns",
    type=str,
    help="Колонки для исключения (через запятую)",
)
@click.option(
    "--methods",
    type=str,
    default="psi,ks",
    help="Методы детекции дрейфа (через запятую: psi,ks,chi2,adversarial)",
)
@click.option(
    "--output",
    type=click.Path(),
    help="Путь для сохранения результатов (JSON)",
)
def detect_drift(
    reference_data: str,
    current_data: str,
    exclude_columns: Optional[str],
    methods: str,
    output: Optional[str],
):
    """Детектировать дрейф данных."""
    try:
        click.echo(f"Загрузка референсных данных из {reference_data}...")
        if reference_data.endswith(".parquet"):
            df_ref = pd.read_parquet(reference_data)
        else:
            df_ref = pd.read_csv(reference_data)

        click.echo(f"Загрузка текущих данных из {current_data}...")
        if current_data.endswith(".parquet"):
            df_curr = pd.read_parquet(current_data)
        else:
            df_curr = pd.read_csv(current_data)

        # Исключаем колонки
        if exclude_columns:
            exclude_list = [c.strip() for c in exclude_columns.split(",")]
            df_ref = df_ref.drop(columns=exclude_list, errors="ignore")
            df_curr = df_curr.drop(columns=exclude_list, errors="ignore")

        # Создаем детектор
        detector = DriftDetector(df_ref, df_curr)

        # Детекция
        methods_list = [m.strip() for m in methods.split(",")]
        click.echo(f"Детекция дрейфа методами: {', '.join(methods_list)}...")

        results = detector.detect_all(methods=methods_list)

        # Выводим результаты
        click.echo("\n" + "=" * 80)
        click.echo("РЕЗУЛЬТАТЫ ДЕТЕКЦИИ ДРЕЙФА")
        click.echo("=" * 80)

        # Summary
        summary = detector.summary()
        click.echo("\nСводка:")
        click.echo("-" * 60)
        for key, value in summary.items():
            click.echo(f"  {key}: {value}")

        # Детальные результаты по методам
        for method, result in results.items():
            click.echo(f"\n{method.upper()}:")
            click.echo("-" * 60)

            if isinstance(result, pd.DataFrame):
                # Топ 10 признаков с наибольшим дрейфом
                df_top = result.head(10)
                for idx, row in df_top.iterrows():
                    feature = row["feature"]
                    if "psi" in row:
                        value = row["psi"]
                        status = row["status"]
                        click.echo(f"  {feature:30s} PSI: {value:.4f} ({status})")
                    elif "ks_statistic" in row:
                        value = row["ks_statistic"]
                        p_value = row["p_value"]
                        drift = "✓" if row["drift_detected"] else "✗"
                        click.echo(f"  {feature:30s} KS: {value:.4f} (p={p_value:.4f}) {drift}")

            elif isinstance(result, dict) and "mean_roc_auc" in result:
                # Adversarial validation
                auc = result["mean_roc_auc"]
                interpretation = result["drift_interpretation"]
                click.echo(f"  AUC: {auc:.4f} ({interpretation})")

        # Сохраняем результаты
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Преобразуем DataFrames в dict для JSON
            results_json: Dict[str, Any] = {}
            for method, result in results.items():
                if isinstance(result, pd.DataFrame):
                    results_json[method] = result.to_dict(orient="records")
                else:
                    results_json[method] = result

            results_json["summary"] = summary

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results_json, f, indent=2, default=str)

            click.echo(f"\n✓ Результаты сохранены в: {output_path}")

    except Exception as e:
        logger.error(f"Ошибка при детекции дрейфа: {e}", exc_info=True)
        click.echo(f"Ошибка: {e}", err=True)
        sys.exit(1)


# Добавляем группу в главный CLI
def register_commands(cli):
    """Регистрация команд оценки в главном CLI."""
    cli.add_command(evaluate_group)
