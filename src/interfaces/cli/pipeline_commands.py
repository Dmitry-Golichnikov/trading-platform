"""CLI команды для управления пайплайнами."""

import json
import logging
from pathlib import Path
from typing import Optional

import click
import yaml

from src.pipelines.full_pipeline import FullPipeline

logger = logging.getLogger(__name__)


@click.group(name="pipeline")
def pipeline_group():
    """Команды для управления пайплайнами."""
    pass


@pipeline_group.command(name="run")
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "--force",
    is_flag=True,
    help="Игнорировать кэш и пересчитать все шаги",
)
@click.option(
    "--no-checkpoints",
    is_flag=True,
    help="Отключить сохранение чекпоинтов",
)
def run_pipeline(config_file: str, force: bool, no_checkpoints: bool):
    """
    Запустить пайплайн.

    CONFIG_FILE: путь к YAML конфигурации пайплайна

    Примеры:
        python -m src.interfaces.cli pipeline run configs/pipelines/full_pipeline_example.yaml

        python -m src.interfaces.cli pipeline run configs/pipelines/training_only.yaml --force
    """
    click.echo(f"Загрузка конфигурации из {config_file}...")

    # Загрузить конфигурацию
    config_path = Path(config_file)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Создать пайплайн
    checkpoint_dir = config.get("checkpoints", {}).get("dir")
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)

    click.echo(f"Запуск пайплайна: {config.get('name', 'unknown')}")
    click.echo(f"Описание: {config.get('description', 'N/A')}")
    click.echo()

    pipeline = FullPipeline(
        config=config,
        checkpoint_dir=checkpoint_dir,
        enable_checkpoints=not no_checkpoints,
        force_rerun=force,
    )

    try:
        # Запустить
        result = pipeline.run()

        # Вывести результаты
        click.echo()
        click.echo("=" * 80)
        click.echo(click.style("✓ Пайплайн завершен успешно!", fg="green", bold=True))
        click.echo("=" * 80)
        click.echo()
        click.echo(f"Статус: {result.status}")
        click.echo(f"Длительность: {result.duration:.2f} сек")
        click.echo(f"Завершенных шагов: {sum(1 for s in result.steps if s.status == 'completed')}/{len(result.steps)}")

        if result.artifacts:
            click.echo()
            click.echo("Артефакты:")
            for key, value in result.artifacts.items():
                click.echo(f"  - {key}: {value}")

        if "report_path" in result.artifacts:
            click.echo()
            click.echo(f"Отчет сохранен в: {result.artifacts['report_path']}")

    except Exception as exc:
        click.echo()
        click.echo("=" * 80)
        click.echo(click.style("✗ Пайплайн завершен с ошибкой!", fg="red", bold=True))
        click.echo("=" * 80)
        click.echo()
        click.echo(f"Ошибка: {exc}")
        raise click.Abort()


@pipeline_group.command(name="resume")
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "--checkpoint-dir",
    type=click.Path(),
    help="Директория с чекпоинтами (по умолчанию из конфига)",
)
def resume_pipeline(config_file: str, checkpoint_dir: Optional[str]):
    """
    Возобновить пайплайн с последнего чекпоинта.

    CONFIG_FILE: путь к YAML конфигурации пайплайна

    Примеры:
        python -m src.interfaces.cli pipeline resume configs/pipelines/full_pipeline_example.yaml
    """
    click.echo(f"Загрузка конфигурации из {config_file}...")

    # Загрузить конфигурацию
    config_path = Path(config_file)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Определить директорию чекпоинтов
    if checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir)
    else:
        checkpoint_path = config.get("checkpoints", {}).get("dir")
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path or not checkpoint_path.exists():
        click.echo(click.style("✗ Чекпоинты не найдены!", fg="red"))
        return

    click.echo(f"Возобновление пайплайна с чекпоинтами из: {checkpoint_path}")

    # Создать и запустить пайплайн (force_rerun=False для использования чекпоинтов)
    pipeline = FullPipeline(
        config=config,
        checkpoint_dir=checkpoint_path,
        enable_checkpoints=True,
        force_rerun=False,
    )

    try:
        result = pipeline.run()

        click.echo()
        click.echo("=" * 80)
        click.echo(click.style("✓ Пайплайн завершен успешно!", fg="green", bold=True))
        click.echo("=" * 80)
        click.echo()
        click.echo(f"Статус: {result.status}")
        click.echo(f"Длительность: {result.duration:.2f} сек")
        click.echo(f"Завершенных шагов: {sum(1 for s in result.steps if s.status == 'completed')}/{len(result.steps)}")

    except Exception as exc:
        click.echo()
        click.echo(click.style(f"✗ Ошибка: {exc}", fg="red"))
        raise click.Abort()


@pipeline_group.command(name="status")
@click.argument("checkpoint_dir", type=click.Path(exists=True))
def pipeline_status(checkpoint_dir: str):
    """
    Показать статус выполнения пайплайна.

    CHECKPOINT_DIR: директория с чекпоинтами

    Примеры:
        python -m src.interfaces.cli pipeline status artifacts/checkpoints/full_pipeline
    """
    checkpoint_path = Path(checkpoint_dir)

    # Найти файлы состояния
    state_files = list(checkpoint_path.glob("state_*.json"))

    if not state_files:
        click.echo(click.style("✗ Файлы состояния не найдены", fg="red"))
        return

    # Взять последний файл
    latest_state = max(state_files, key=lambda p: p.stat().st_mtime)

    click.echo(f"Файл состояния: {latest_state.name}")
    click.echo()

    # Загрузить состояние
    with open(latest_state) as f:
        state = json.load(f)

    steps = state.get("steps", [])

    click.echo("Статус шагов:")
    click.echo()

    for step in steps:
        name = step["name"]
        status = step["status"]

        if status == "completed":
            icon = click.style("✓", fg="green")
        elif status == "running":
            icon = click.style("▶", fg="yellow")
        elif status == "failed":
            icon = click.style("✗", fg="red")
        else:
            icon = click.style("○", fg="white")

        click.echo(f"  {icon} {name:30s} [{status}]")

        if step.get("started_at"):
            click.echo(f"      Начато: {step['started_at']}")
        if step.get("completed_at"):
            click.echo(f"      Завершено: {step['completed_at']}")
        if step.get("error"):
            click.echo(click.style(f"      Ошибка: {step['error']}", fg="red"))

    click.echo()

    # Статистика
    total = len(steps)
    completed = sum(1 for s in steps if s["status"] == "completed")
    failed = sum(1 for s in steps if s["status"] == "failed")
    pending = sum(1 for s in steps if s["status"] == "pending")

    click.echo("Сводка:")
    click.echo(f"  Всего шагов: {total}")
    click.echo(f"  Завершено: {click.style(str(completed), fg='green')}")
    click.echo(f"  Ошибок: {click.style(str(failed), fg='red')}")
    click.echo(f"  Ожидает: {pending}")


@pipeline_group.command(name="clear")
@click.argument("checkpoint_dir", type=click.Path(exists=True))
@click.option(
    "--yes",
    is_flag=True,
    help="Пропустить подтверждение",
)
def clear_checkpoints(checkpoint_dir: str, yes: bool):
    """
    Удалить чекпоинты пайплайна.

    CHECKPOINT_DIR: директория с чекпоинтами

    Примеры:
        python -m src.interfaces.cli pipeline clear artifacts/checkpoints/full_pipeline
    """
    checkpoint_path = Path(checkpoint_dir)

    if not yes:
        click.confirm(
            f"Вы уверены, что хотите удалить все чекпоинты из {checkpoint_path}?",
            abort=True,
        )

    # Удалить директорию
    import shutil

    shutil.rmtree(checkpoint_path)

    click.echo(click.style(f"✓ Чекпоинты удалены из {checkpoint_path}", fg="green"))


@pipeline_group.command(name="list")
@click.option(
    "--config-dir",
    type=click.Path(exists=True),
    default="configs/pipelines",
    help="Директория с конфигурациями пайплайнов",
)
def list_pipelines(config_dir: str):
    """
    Показать список доступных конфигураций пайплайнов.

    Примеры:
        python -m src.interfaces.cli pipeline list
    """
    config_path = Path(config_dir)

    if not config_path.exists():
        click.echo(click.style(f"✗ Директория не найдена: {config_path}", fg="red"))
        return

    # Найти конфигурационные файлы
    config_files = list(config_path.glob("*.yaml")) + list(config_path.glob("*.yml"))

    if not config_files:
        click.echo("Конфигурации пайплайнов не найдены")
        return

    click.echo("Доступные конфигурации пайплайнов:")
    click.echo()

    for config_file in sorted(config_files):
        try:
            with open(config_file) as f:
                config = yaml.safe_load(f)

            name = config.get("name", config_file.stem)
            description = config.get("description", "N/A")

            click.echo(f"  • {click.style(name, fg='cyan', bold=True)}")
            click.echo(f"    Файл: {config_file.relative_to(Path.cwd())}")
            click.echo(f"    Описание: {description}")
            click.echo()
        except Exception as exc:
            click.echo(f"  • {config_file.name} (ошибка чтения: {exc})")
            click.echo()


if __name__ == "__main__":
    pipeline_group()
