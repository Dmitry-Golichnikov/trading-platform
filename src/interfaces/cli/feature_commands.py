"""CLI команды для работы с модулем генерации признаков."""

from pathlib import Path
from typing import Optional, Tuple

import click
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.data.storage.parquet_storage import ParquetStorage
from src.features import FeatureCache, FeatureGenerator

console = Console()


def _parse_dataset_id(dataset_id: str) -> Tuple[str, str]:
    """Разобрать идентификатор формата TICKER_TIMEFRAME."""
    if "_" not in dataset_id:
        raise ValueError("Идентификатор датасета должен быть в формате TICKER_TIMEFRAME")

    ticker, timeframe = dataset_id.split("_", 1)
    if not ticker or not timeframe:
        raise ValueError("Идентификатор датасета должен содержать тикер и таймфрейм")

    return ticker, timeframe


@click.group()
def features():
    """Команды для работы с признаками."""
    pass


@features.command("generate")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Путь к конфигурации признаков (YAML)",
)
@click.option(
    "--dataset",
    "-d",
    type=str,
    required=True,
    help="ID датасета или путь к Parquet файлу",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=False,
    help="Путь для сохранения признаков (если не указан, используется кэш)",
)
@click.option(
    "--use-cache/--no-cache",
    default=True,
    help="Использовать кэш",
)
@click.option(
    "--target-column",
    type=str,
    default=None,
    help="Колонка с таргетом (для feature selection)",
)
def generate_features(
    config: Path,
    dataset: str,
    output: Optional[Path],
    use_cache: bool,
    target_column: Optional[str],
):
    """
    Сгенерировать признаки на основе конфигурации.

    Examples:
        $ python -m src.interfaces.cli features generate
          -c configs/features/default.yaml -d SBER_1m
        $ python -m src.interfaces.cli features generate
          -c configs/features/minimal.yaml -d data.parquet -o features.parquet
    """
    try:
        # Инициализируем генератор
        console.print(f"[cyan]Загрузка конфигурации из {config}...[/cyan]")
        generator = FeatureGenerator(config, cache_enabled=use_cache)

        # Загружаем данные
        console.print(f"[cyan]Загрузка данных из {dataset}...[/cyan]")

        dataset_path = Path(dataset)
        if dataset_path.exists():
            # Загружаем из файла
            data = pd.read_parquet(dataset_path)
            dataset_id = dataset_path.stem
        else:
            # Загружаем из хранилища
            storage = ParquetStorage()
            ticker, timeframe = _parse_dataset_id(dataset)
            data = storage.load_dataset(ticker, timeframe)
            dataset_id = dataset

        console.print(f"[green]Загружено {len(data)} строк[/green]")

        # Загружаем таргет если указан
        target = None
        if target_column and target_column in data.columns:
            target = data[target_column]
            console.print(f"[cyan]Используется таргет из колонки {target_column}[/cyan]")

        # Генерируем признаки
        console.print("[cyan]Генерация признаков...[/cyan]")
        features = generator.generate(
            data=data,
            dataset_id=dataset_id,
            target=target,
            use_cache=use_cache,
        )

        console.print("[green]Сгенерировано " f"{features.shape[1]} признаков для {features.shape[0]} строк[/green]")

        # Сохраняем если указан output
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            features.to_parquet(output, compression="snappy")
            console.print(f"[green]Признаки сохранены в {output}[/green]")
        else:
            console.print("[yellow]Признаки сохранены в кэш " "(укажите --output для сохранения в файл)[/yellow]")

        # Показываем первые несколько признаков
        table = Table(title="Примеры признаков")
        table.add_column("Признак", style="cyan")
        table.add_column("Тип", style="magenta")
        table.add_column("NaN %", style="yellow")
        table.add_column("Min", style="green")
        table.add_column("Max", style="green")

        for col in features.columns[:10]:  # Показываем первые 10
            dtype = str(features[col].dtype)
            nan_pct = features[col].isna().mean() * 100
            min_val = features[col].min() if pd.api.types.is_numeric_dtype(features[col]) else "N/A"
            max_val = features[col].max() if pd.api.types.is_numeric_dtype(features[col]) else "N/A"

            table.add_row(
                col,
                dtype,
                f"{nan_pct:.1f}%",
                f"{min_val:.2f}" if isinstance(min_val, (int, float)) else str(min_val),
                f"{max_val:.2f}" if isinstance(max_val, (int, float)) else str(max_val),
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Ошибка: {e}[/red]")
        raise click.Abort()


@features.command("list")
@click.option(
    "--dataset",
    "-d",
    type=str,
    default=None,
    help="ID датасета (фильтр)",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Директория кэша (по умолчанию artifacts/features)",
)
def list_features(dataset: Optional[str], cache_dir: Optional[Path]):
    """
    Показать список кэшированных признаков.

    Examples:
        $ python -m src.interfaces.cli features list
        $ python -m src.interfaces.cli features list -d SBER_1m
    """
    try:
        # Инициализируем кэш
        cache_dir = cache_dir or Path("artifacts/features")
        cache = FeatureCache(cache_dir)

        # Получаем список
        cached = cache.list_cached(dataset_id=dataset)

        if cached.empty:
            console.print("[yellow]Кэшированные признаки не найдены[/yellow]")
            return

        # Показываем таблицу
        table = Table(title="Кэшированные признаки")
        table.add_column("Dataset ID", style="cyan")
        table.add_column("Config Hash", style="magenta")
        table.add_column("Признаков", style="green")
        table.add_column("Строк", style="green")
        table.add_column("Создано", style="yellow")

        for _, row in cached.iterrows():
            created_at = pd.to_datetime(row["created_at"]).strftime("%Y-%m-%d %H:%M")
            table.add_row(
                row["dataset_id"],
                row["config_hash"][:8],
                str(row["num_features"]),
                str(row["num_rows"]),
                created_at,
            )

        console.print(table)

        # Показываем статистику
        stats = cache.get_stats()
        console.print("\n[bold]Статистика кэша:[/bold]")
        console.print(f"  Всего записей: {stats['total_entries']}")
        console.print(f"  Датасетов: {stats['total_datasets']}")
        console.print(f"  Размер на диске: {stats['size_mb']:.2f} MB")

    except Exception as e:
        console.print(f"[red]Ошибка: {e}[/red]")
        raise click.Abort()


@features.command("clear-cache")
@click.option(
    "--dataset",
    "-d",
    type=str,
    default=None,
    help="ID датасета (если не указан, очищается весь кэш)",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Директория кэша (по умолчанию artifacts/features)",
)
@click.confirmation_option(prompt="Вы уверены что хотите очистить кэш?")
def clear_cache(dataset: Optional[str], cache_dir: Optional[Path]):
    """
    Очистить кэш признаков.

    Examples:
        $ python -m src.interfaces.cli features clear-cache
        $ python -m src.interfaces.cli features clear-cache -d SBER_1m
    """
    try:
        # Инициализируем кэш
        cache_dir = cache_dir or Path("artifacts/features")
        cache = FeatureCache(cache_dir)

        # Очищаем
        cache.invalidate(dataset_id=dataset)

        if dataset:
            console.print(f"[green]Кэш для датасета {dataset} очищен[/green]")
        else:
            console.print("[green]Весь кэш очищен[/green]")

    except Exception as e:
        console.print(f"[red]Ошибка: {e}[/red]")
        raise click.Abort()


@features.command("validate-config")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
def validate_config(config_path: Path):
    """
    Валидировать конфигурацию признаков.

    Examples:
        $ python -m src.interfaces.cli features validate-config
          configs/features/default.yaml
    """
    try:
        from src.features.config_parser import parse_feature_config

        console.print(f"[cyan]Валидация конфигурации {config_path}...[/cyan]")

        # Парсим и валидируем
        config = parse_feature_config(config_path)

        console.print("[green]✓ Конфигурация валидна[/green]")
        console.print(f"  Версия: {config.version}")
        console.print(f"  Признаков: {len(config.features)}")
        console.print(f"  Кэширование: {'включено' if config.cache_enabled else 'выключено'}")

        if config.selection:
            console.print(
                "  Feature selection: " f"{config.selection.method} " f"(top_k={config.selection.top_k or 'все'})"
            )

        # Показываем типы признаков
        feature_types: dict[str, int] = {}
        for f in config.features:
            feature_types[f.type] = feature_types.get(f.type, 0) + 1

        console.print("\n[bold]Типы признаков:[/bold]")
        for ftype, count in sorted(feature_types.items()):
            console.print(f"  {ftype}: {count}")

    except Exception as e:
        console.print(f"[red]✗ Ошибка валидации: {e}[/red]")
        raise click.Abort()


if __name__ == "__main__":
    features()
