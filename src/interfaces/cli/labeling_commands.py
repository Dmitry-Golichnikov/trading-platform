"""CLI команды для модуля разметки таргетов."""

import logging
from pathlib import Path
from typing import Optional

import click
import pandas as pd
import yaml

from src.labeling.metadata import LabelingMetadata
from src.labeling.pipeline import LabelingPipeline
from src.labeling.visualization import create_labeling_report

logger = logging.getLogger(__name__)


@click.group()
def labels():
    """Команды для работы с разметкой таргетов."""
    pass


@labels.command()
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    required=True,
    help="Путь к файлу с данными (parquet)",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Путь к конфигурации разметки (YAML)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="artifacts/labels",
    help="Директория для сохранения результатов",
)
@click.option("--dataset-id", type=str, default="default", help="ID датасета")
@click.option("--visualize/--no-visualize", default=True, help="Создать визуализацию")
def label_dataset(
    data_path: str,
    config: str,
    output_dir: str,
    dataset_id: str,
    visualize: bool,
):
    """
    Разметка датасета таргетами.

    Примеры:

        # Разметка с конфигурацией
        python -m src.interfaces.cli labels label-dataset \\
            --data-path data/SBER_1h.parquet \\
            --config configs/labeling/long_only.yaml

        # С визуализацией
        python -m src.interfaces.cli labels label-dataset \\
            --data-path data/SBER_1h.parquet \\
            --config configs/labeling/triple_barrier.yaml \\
            --visualize
    """
    click.echo(f"🏷️  Начинаем разметку датасета: {data_path}")

    try:
        # Загрузка данных
        click.echo(f"📂 Загрузка данных из {data_path}...")
        data = pd.read_parquet(data_path)
        click.echo(f"✓ Загружено {len(data)} записей")

        # Загрузка конфигурации
        click.echo(f"⚙️  Загрузка конфигурации из {config}...")
        with open(config, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        output_path = Path(output_dir)

        config_dict["dataset_id"] = dataset_id
        config_dict["output_dir"] = str(output_path)

        # Создание и запуск пайплайна
        click.echo("🚀 Создание пайплайна разметки...")
        pipeline = LabelingPipeline.from_config(config_dict, data)

        click.echo("⏳ Выполнение разметки...")
        labeled_data, metadata = pipeline.run(data, save_results=True)

        # Вывод статистики
        click.echo("\n" + "=" * 60)
        click.echo("📊 Результаты разметки:")
        click.echo("=" * 60)
        click.echo(metadata.get_summary())
        click.echo("=" * 60)

        # Визуализация
        if visualize:
            click.echo("\n📈 Создание визуализации...")
            viz_output_dir = output_path / pipeline.labeling_id / "visualizations"
            create_labeling_report(labeled_data, metadata=metadata.to_dict(), output_dir=viz_output_dir)
            click.echo(f"✓ Визуализация сохранена в: {viz_output_dir}")

        click.echo("\n✅ Разметка завершена успешно!")
        click.echo(f"📁 Результаты: {output_path / pipeline.labeling_id}")

    except Exception as e:
        click.echo(f"\n❌ Ошибка при разметке: {e}", err=True)
        logger.exception("Ошибка в label_dataset")
        raise click.Abort()


@labels.command()
@click.option(
    "--labeling-path",
    type=click.Path(exists=True),
    required=True,
    help="Путь к директории с результатами разметки",
)
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    help="Путь к оригинальным данным (опционально, для timeline)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    help="Директория для сохранения отчёта (по умолчанию в labeling_path)",
)
def analyze_labels(
    labeling_path: str,
    data_path: Optional[str],
    output_dir: Optional[str],
):
    """
    Анализ и визуализация результатов разметки.

    Примеры:

        # Анализ существующей разметки
        python -m src.interfaces.cli labels analyze-labels \\
            --labeling-path \\
            artifacts/labels/TripleBarrierLabeler_default_20231027_120000

        # С оригинальными данными для timeline
        python -m src.interfaces.cli labels analyze-labels \\
            --labeling-path \\
            artifacts/labels/TripleBarrierLabeler_default_20231027_120000 \\
            --data-path data/SBER_1h.parquet
    """
    click.echo(f"🔍 Анализ разметки: {labeling_path}")

    try:
        labeling_path_obj = Path(labeling_path)

        # Загрузка метаданных
        metadata_path = labeling_path_obj / "metadata.json"
        if not metadata_path.exists():
            click.echo(f"❌ Файл метаданных не найден: {metadata_path}", err=True)
            raise click.Abort()

        click.echo("📂 Загрузка метаданных...")
        metadata = LabelingMetadata.load(metadata_path)

        # Загрузка данных
        labels_path = labeling_path_obj / "labels.parquet"
        if not labels_path.exists():
            click.echo(f"❌ Файл с метками не найден: {labels_path}", err=True)
            raise click.Abort()

        click.echo("📂 Загрузка меток...")
        labeled_data = pd.read_parquet(labels_path)

        # Если есть оригинальные данные, объединяем для timeline
        if data_path:
            click.echo("📂 Загрузка оригинальных данных...")
            original_data = pd.read_parquet(data_path)
            labeled_data = original_data.join(labeled_data[["label"]], how="inner")

        # Вывод статистики
        click.echo("\n" + "=" * 60)
        click.echo("📊 Информация о разметке:")
        click.echo("=" * 60)
        click.echo(metadata.get_summary())
        click.echo("=" * 60)

        # Создание визуализации
        output_path = Path(output_dir) if output_dir else labeling_path_obj / "analysis"

        click.echo("\n📈 Создание отчёта...")
        create_labeling_report(labeled_data, metadata=metadata.to_dict(), output_dir=output_path)

        click.echo("✅ Анализ завершён!")
        click.echo(f"📁 Отчёт сохранён в: {output_path}")

    except Exception as e:
        click.echo(f"\n❌ Ошибка при анализе: {e}", err=True)
        logger.exception("Ошибка в analyze_labels")
        raise click.Abort()


@labels.command()
@click.option(
    "--labels-dir",
    type=click.Path(exists=True),
    default="artifacts/labels",
    help="Директория с разметками",
)
def list_labelings(labels_dir: str):
    """
    Список всех доступных разметок.

    Примеры:

        # Показать все разметки
        python -m src.interfaces.cli labels list-labelings

        # Из конкретной директории
        python -m src.interfaces.cli labels list-labelings \\
            --labels-dir my_labels/
    """
    click.echo(f"📋 Доступные разметки в {labels_dir}:\n")

    labels_path = Path(labels_dir)

    if not labels_path.exists():
        click.echo(f"❌ Директория не найдена: {labels_dir}", err=True)
        return

    # Поиск всех разметок
    labelings = []
    for item in labels_path.iterdir():
        if item.is_dir():
            metadata_path = item / "metadata.json"
            if metadata_path.exists():
                try:
                    metadata = LabelingMetadata.load(metadata_path)
                    labelings.append(
                        {
                            "id": metadata.labeling_id,
                            "method": metadata.method,
                            "dataset": metadata.dataset_id,
                            "samples": metadata.total_samples,
                            "distribution": metadata.class_distribution,
                            "created": metadata.created_at.strftime("%Y-%m-%d %H:%M"),
                            "path": item,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Ошибка чтения метаданных {metadata_path}: {e}")

    if not labelings:
        click.echo("Не найдено ни одной разметки.")
        return

    # Вывод таблицы
    click.echo(f"{'ID':<50} {'Method':<25} {'Dataset':<15} {'Samples':<10} {'Created':<20}")
    click.echo("=" * 130)

    for labeling in sorted(labelings, key=lambda x: str(x["created"]), reverse=True):
        click.echo(
            f"{labeling['id']:<50} "
            f"{labeling['method']:<25} "
            f"{labeling['dataset']:<15} "
            f"{labeling['samples']:<10} "
            f"{labeling['created']:<20}"
        )

    click.echo(f"\n📊 Всего разметок: {len(labelings)}")


@labels.command()
@click.option("--labeling-id", type=str, required=True, help="ID разметки для удаления")
@click.option(
    "--labels-dir",
    type=click.Path(exists=True),
    default="artifacts/labels",
    help="Директория с разметками",
)
@click.confirmation_option(prompt="Вы уверены что хотите удалить эту разметку?")
def delete_labeling(labeling_id: str, labels_dir: str):
    """
    Удаление разметки.

    Примеры:

        # Удалить конкретную разметку
        python -m src.interfaces.cli labels delete-labeling \\
            --labeling-id TripleBarrierLabeler_default_20231027_120000
    """
    import shutil

    labeling_path = Path(labels_dir) / labeling_id

    if not labeling_path.exists():
        click.echo(f"❌ Разметка не найдена: {labeling_id}", err=True)
        raise click.Abort()

    try:
        shutil.rmtree(labeling_path)
        click.echo(f"✅ Разметка удалена: {labeling_id}")
    except Exception as e:
        click.echo(f"❌ Ошибка при удалении: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    labels()
