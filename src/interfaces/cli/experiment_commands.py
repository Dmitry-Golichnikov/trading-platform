"""CLI commands for experiment management."""

import sys
from pathlib import Path

import click
import yaml

from src.orchestration import ExperimentManager, MLflowManager


@click.group()
def experiment():
    """Управление экспериментами."""
    pass


@experiment.command("create")
@click.option("--name", "-n", required=True, help="Имя эксперимента")
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Путь к конфигурации")
@click.option("--description", "-d", help="Описание эксперимента")
@click.option("--tag", "-t", multiple=True, help="Теги в формате key=value")
def create_experiment(name, config, description, tag):
    """Создать новый эксперимент."""
    try:
        # Load config
        with open(config, "r") as f:
            config_data = yaml.safe_load(f)

        # Parse tags
        tags = {}
        for tag_str in tag:
            if "=" in tag_str:
                key, value = tag_str.split("=", 1)
                tags[key] = value

        # Create experiment
        manager = ExperimentManager()
        exp_id = manager.create_experiment(
            name=name,
            config=config_data,
            description=description,
            tags=tags,
        )

        click.echo(f"✓ Создан эксперимент: {exp_id}")
        click.echo(f"  Имя: {name}")
        click.echo(f"  Конфигурация: {config}")

    except Exception as e:
        click.echo(f"✗ Ошибка при создании эксперимента: {e}", err=True)
        sys.exit(1)


@experiment.command("run")
@click.argument("experiment_id")
@click.option("--auto-register", is_flag=True, help="Автоматически регистрировать лучшую модель")
def run_experiment(experiment_id, auto_register):
    """Запустить эксперимент."""
    try:
        manager = ExperimentManager()

        click.echo(f"Запуск эксперимента: {experiment_id}")

        # Get experiment
        exp = manager.get_experiment(experiment_id)
        click.echo(f"  Имя: {exp['name']}")
        click.echo(f"  Статус: {exp['status']}")

        # Run experiment
        run_id = manager.run_experiment(
            experiment_id=experiment_id,
            auto_register=auto_register,
        )

        click.echo("✓ Эксперимент завершен")
        click.echo(f"  Run ID: {run_id}")

    except Exception as e:
        click.echo(f"✗ Ошибка при выполнении эксперимента: {e}", err=True)
        sys.exit(1)


@experiment.command("list")
@click.option("--status", "-s", help="Фильтр по статусу")
@click.option("--limit", "-l", type=int, help="Максимальное количество результатов")
def list_experiments(status, limit):
    """Список экспериментов."""
    try:
        manager = ExperimentManager()
        experiments = manager.list_experiments(status=status, limit=limit)

        if not experiments:
            click.echo("Нет экспериментов")
            return

        click.echo(f"\nНайдено экспериментов: {len(experiments)}\n")

        for exp in experiments:
            click.echo(f"ID: {exp['id']}")
            click.echo(f"  Имя: {exp['name']}")
            click.echo(f"  Статус: {exp['status']}")
            click.echo(f"  Создан: {exp['created_at']}")

            if exp.get("tags"):
                click.echo(f"  Теги: {exp['tags']}")

            if exp.get("runs"):
                click.echo(f"  Запуски: {len(exp['runs'])}")

            click.echo()

    except Exception as e:
        click.echo(f"✗ Ошибка при получении списка: {e}", err=True)
        sys.exit(1)


@experiment.command("status")
@click.argument("experiment_id")
def experiment_status(experiment_id):
    """Получить статус эксперимента."""
    try:
        manager = ExperimentManager()
        exp = manager.get_experiment(experiment_id)

        click.echo(f"\nЭксперимент: {exp['id']}")
        click.echo(f"  Имя: {exp['name']}")
        click.echo(f"  Статус: {exp['status']}")
        click.echo(f"  Создан: {exp['created_at']}")

        if exp.get("description"):
            click.echo(f"  Описание: {exp['description']}")

        if exp.get("tags"):
            click.echo("  Теги:")
            for key, value in exp["tags"].items():
                click.echo(f"    {key}: {value}")

        if exp.get("runs"):
            click.echo(f"\n  Запуски ({len(exp['runs'])}):")
            for run in exp["runs"]:
                click.echo(f"    - {run['run_id']} ({run['timestamp']})")

        click.echo()

    except Exception as e:
        click.echo(f"✗ Ошибка: {e}", err=True)
        sys.exit(1)


@experiment.command("compare")
@click.argument("experiment_ids", nargs=-1, required=True)
@click.option("--metric", "-m", multiple=True, help="Метрики для сравнения")
@click.option("--output", "-o", type=click.Path(), help="Сохранить результат в файл")
def compare_experiments(experiment_ids, metric, output):
    """Сравнить эксперименты."""
    try:
        manager = ExperimentManager()

        metrics = list(metric) if metric else None

        df = manager.compare_experiments(
            experiment_ids=list(experiment_ids),
            metrics=metrics,
        )

        if df.empty:
            click.echo("Нет данных для сравнения")
            return

        # Display comparison
        click.echo("\nСравнение экспериментов:\n")
        click.echo(df.to_string(index=False))

        # Save to file if requested
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.suffix == ".csv":
                df.to_csv(output_path, index=False)
            elif output_path.suffix == ".xlsx":
                df.to_excel(output_path, index=False)
            else:
                df.to_csv(output_path, index=False)

            click.echo(f"\n✓ Результат сохранен в: {output_path}")

    except Exception as e:
        click.echo(f"✗ Ошибка при сравнении: {e}", err=True)
        sys.exit(1)


@experiment.command("best")
@click.option("--metric", "-m", required=True, help="Метрика для выбора лучшего")
@click.option(
    "--direction", "-d", type=click.Choice(["maximize", "minimize"]), default="maximize", help="Направление оптимизации"
)
@click.option("--status", "-s", default="completed", help="Фильтр по статусу")
def get_best_experiment(metric, direction, status):
    """Получить лучший эксперимент по метрике."""
    try:
        manager = ExperimentManager()

        best = manager.get_best_experiment(
            metric=metric,
            direction=direction,
            status=status,
        )

        click.echo(f"\nЛучший эксперимент по {metric} ({direction}):\n")
        click.echo(f"ID: {best['id']}")
        click.echo(f"  Имя: {best['name']}")
        click.echo(f"  Статус: {best['status']}")
        click.echo(f"  Создан: {best['created_at']}")

        # Get metrics from MLflow
        if best.get("runs"):
            mlflow = MLflowManager()
            run_id = best["runs"][-1]["run_id"]
            run = mlflow.get_run(run_id)

            if run:
                click.echo("\n  Метрики:")
                for key, value in run.data.metrics.items():
                    click.echo(f"    {key}: {value:.4f}")

        click.echo()

    except Exception as e:
        click.echo(f"✗ Ошибка: {e}", err=True)
        sys.exit(1)


@experiment.command("delete")
@click.argument("experiment_id")
@click.option("--force", "-f", is_flag=True, help="Удалить без подтверждения")
def delete_experiment(experiment_id, force):
    """Удалить эксперимент."""
    try:
        manager = ExperimentManager()

        # Get experiment info
        exp = manager.get_experiment(experiment_id)

        if not force:
            click.echo("Вы уверены, что хотите удалить эксперимент?")
            click.echo(f"  ID: {exp['id']}")
            click.echo(f"  Имя: {exp['name']}")

            if not click.confirm("Продолжить?"):
                click.echo("Отменено")
                return

        manager.delete_experiment(experiment_id)
        click.echo(f"✓ Эксперимент удален: {experiment_id}")

    except Exception as e:
        click.echo(f"✗ Ошибка при удалении: {e}", err=True)
        sys.exit(1)


@experiment.command("export")
@click.option("--output", "-o", required=True, type=click.Path(), help="Путь к файлу для экспорта")
@click.option("--format", "-f", type=click.Choice(["csv", "json", "yaml"]), default="csv", help="Формат экспорта")
def export_experiments(output, format):
    """Экспортировать эксперименты."""
    try:
        manager = ExperimentManager()

        manager.export_experiments(output_path=output, format=format)

        click.echo(f"✓ Эксперименты экспортированы в: {output}")

    except Exception as e:
        click.echo(f"✗ Ошибка при экспорте: {e}", err=True)
        sys.exit(1)


@experiment.command("report")
@click.option("--output", "-o", type=click.Path(), help="Путь к файлу для сохранения отчета")
def generate_report(output):
    """Сгенерировать отчет по экспериментам."""
    try:
        manager = ExperimentManager()

        report = manager.generate_summary_report(output_path=output)

        if not output:
            click.echo(report)

        click.echo("✓ Отчет сгенерирован")

        if output:
            click.echo(f"  Сохранен в: {output}")

    except Exception as e:
        click.echo(f"✗ Ошибка при генерации отчета: {e}", err=True)
        sys.exit(1)


@experiment.command("cleanup")
@click.option("--force", "-f", is_flag=True, help="Удалить без подтверждения")
def cleanup_failed(force):
    """Удалить неудачные эксперименты."""
    try:
        if not force:
            if not click.confirm("Удалить все неудачные эксперименты?"):
                click.echo("Отменено")
                return

        manager = ExperimentManager()
        manager.cleanup_failed_experiments()

        click.echo("✓ Неудачные эксперименты удалены")

    except Exception as e:
        click.echo(f"✗ Ошибка: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    experiment()
