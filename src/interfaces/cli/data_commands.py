"""CLI команды для работы с модулем данных."""

from __future__ import annotations

import asyncio
import os
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Sequence, cast

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from tqdm.auto import tqdm

from src.data.preprocessors.resampler import TimeframeResampler
from src.data.schemas import DatasetConfig
from src.data.storage.catalog import DatasetCatalog
from src.data.storage.parquet_storage import (
    ParquetStorage,
    SourceLiteral,
    TimeframeLiteral,
)
from src.data.storage.versioning import DataVersioning
from src.data.validators import IntegrityValidator, QualityValidator, SchemaValidator
from src.pipelines import DataPreparationPipeline

if TYPE_CHECKING:
    from src.pipelines.data_preparation import PipelineResult

console = Console()
load_dotenv()
DEFAULT_HISTORY_START = date(1990, 1, 1)


def _parse_optional_date(value: datetime | None, fallback: date) -> date:
    if value is None:
        return fallback
    return value.date()


def _parse_timeframes(values: tuple[str, ...] | None) -> tuple[str, ...]:
    return tuple(values) if values else ("1m", "5m", "15m", "1h", "4h", "1d")


def _format_int(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def _generate_report(results: Sequence["PipelineResult"]) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_dir = Path("artifacts/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"data_load_{timestamp}.md"

    total_tickers = len(results)
    successful = sum(1 for result in results if result.metadata)
    total_warnings = sum(len(result.warnings) for result in results)

    lines: list[str] = [
        f"# Отчёт по загрузке данных ({datetime.utcnow().isoformat()} UTC)",
        "",
        f"- Всего тикеров: {total_tickers}",
        f"- Успешно загружено: {successful}",
        f"- Без данных: {total_tickers - successful}",
        f"- Предупреждений: {total_warnings}",
        "",
    ]

    for result in results:
        lines.append(f"## {result.ticker}")

        if not result.metadata:
            lines.append("Данные не загружены.")
            if result.missing_years:
                lines.append(
                    f"Отсутствующие годы: {', '.join(map(str, result.missing_years))}"
                )
            lines.append("")
            continue

        missing_years = (
            ", ".join(map(str, result.missing_years)) if result.missing_years else "нет"
        )
        lines.append(f"- Пропущенные годы: {missing_years}")
        lines.append("")
        lines.append("| Таймфрейм | Начало | Конец | Баров | Пропуски | Источник |")
        lines.append("|-----------|--------|-------|-------|----------|----------|")

        for metadata in sorted(result.metadata, key=lambda item: item.timeframe):
            total_bars = _format_int(metadata.total_bars)
            missing_bars = _format_int(metadata.missing_bars)
            lines.append(
                "| {} | {} | {} | {} | {} | {} |".format(
                    metadata.timeframe,
                    metadata.start_date.isoformat(),
                    metadata.end_date.isoformat(),
                    total_bars,
                    missing_bars,
                    metadata.source,
                )
            )

        lines.append("")

        if result.warnings:
            lines.append("### Предупреждения")
            for warning in result.warnings:
                lines.append(f"- {warning}")
            lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


@click.group(help="Команды для работы с модулем данных")
def data() -> None:
    """Группа CLI команд."""


@data.command("load-data", help="Загрузить данные и сохранить в хранилище")
@click.option(
    "--ticker", "tickers", multiple=True, help="Тикеры для загрузки (можно несколько)"
)
@click.option(
    "--tickers-file", type=click.Path(path_type=Path), help="Файл со списком тикеров"
)
@click.option(
    "--timeframe",
    default="1m",
    show_default=True,
    type=click.Choice(["1m", "5m", "15m", "1h", "4h", "1d"]),
    help="Целевой таймфрейм",
)
@click.option(
    "--from-date", type=click.DateTime(["%Y-%m-%d"]), help="Дата начала (YYYY-MM-DD)"
)
@click.option(
    "--to-date", type=click.DateTime(["%Y-%m-%d"]), help="Дата окончания (YYYY-MM-DD)"
)
@click.option(
    "--source-type",
    default="api",
    show_default=True,
    type=click.Choice(["local", "api"]),
    help="Источник данных",
)
@click.option(
    "--file-path",
    type=click.Path(path_type=Path),
    help="Путь к локальному файлу (для local источника)",
)
@click.option(
    "--resample-from",
    type=click.Choice(["1m", "5m", "15m", "1h", "4h"]),
    help="Таймфрейм для ресэмплинга",
)
@click.option(
    "--target-timeframe",
    "target_timeframes",
    multiple=True,
    type=click.Choice(["1m", "5m", "15m", "1h", "4h", "1d"]),
    help="Таймфреймы для сохранения",
)
@click.option(
    "--update-latest/--no-update-latest",
    default=True,
    show_default=True,
    help="Обновлять текущий год",
)
@click.option(
    "--backfill-missing/--no-backfill-missing",
    default=True,
    show_default=True,
    help="Докачивать недостающие годы",
)
@click.option(
    "--concurrency",
    default=8,
    show_default=True,
    help="Количество параллельных загрузок (1 — последовательно)",
)
@click.option(
    "--token",
    help=(
        "Токен доступа к Tinkoff Invest API (при отсутствии берётся из "
        "TINKOFF_API_TOKEN/TINKOFF_INVEST_TOKEN)"
    ),
)
def load_data(
    tickers: tuple[str, ...],
    tickers_file: Optional[Path],
    timeframe: str,
    from_date: datetime,
    to_date: datetime,
    source_type: str,
    file_path: Optional[Path],
    resample_from: Optional[str],
    target_timeframes: tuple[str, ...],
    update_latest: bool,
    backfill_missing: bool,
    concurrency: int,
    token: Optional[str],
) -> None:
    """Обработать данные согласно конфигурации."""

    ticker_value: Optional[str | list[str]]
    default_tickers_file = Path("tickers_all.txt")
    effective_file = tickers_file

    if not tickers:
        if effective_file is None and default_tickers_file.exists():
            effective_file = default_tickers_file
        if effective_file is None:
            raise click.BadParameter(
                "Не указаны тикеры. Передайте --ticker/--tickers-file или "
                "создайте tickers_all.txt"
            )
        ticker_value = None
    elif len(tickers) == 1:
        ticker_value = tickers[0]
    else:
        ticker_value = list(tickers)

    today = datetime.utcnow().date()
    default_start = DEFAULT_HISTORY_START
    from_date_value = _parse_optional_date(from_date, fallback=default_start)
    to_date_value = _parse_optional_date(to_date, fallback=today)

    if from_date_value > to_date_value:
        raise click.BadParameter("Дата начала не может быть позже даты окончания")

    default_targets = ("1m", "5m", "15m", "1h", "4h", "1d")
    target_tf = (
        _parse_timeframes(target_timeframes) if target_timeframes else default_targets
    )

    token_value = (
        token or os.getenv("TINKOFF_API_TOKEN") or os.getenv("TINKOFF_INVEST_TOKEN")
    )

    timeframe_value = cast(TimeframeLiteral, timeframe)
    source_type_value = cast(Literal["local", "api"], source_type)
    resample_value = cast(Literal["1m", "5m", "15m", "1h", "4h"] | None, resample_from)

    config = DatasetConfig(
        ticker=ticker_value,
        tickers_file=effective_file,
        timeframe=timeframe_value,
        from_date=from_date_value,
        to_date=to_date_value,
        source_type=source_type_value,
        file_path=file_path,
        resample_from=resample_value,
        update_latest_year=update_latest,
        backfill_missing=backfill_missing,
        api_token=token_value,
    )

    async def _run() -> None:
        progress_state: dict[str, int] = {}
        overall = tqdm(
            desc="Loading",
            unit="year",
            total=0,
            dynamic_ncols=True,
        )

        def progress_callback(
            ticker: str, stage: str, completed: int, total: int
        ) -> None:
            total_value = total if total else 1

            if stage == "start":
                progress_state[ticker] = 0
                overall.total += total_value
                overall.refresh()
                overall.set_postfix_str(ticker)
                return

            if stage == "progress":
                prev = progress_state.get(ticker, 0)
                delta = max(0, completed - prev)
                if delta:
                    overall.update(delta)
                    progress_state[ticker] = completed
                    overall.set_postfix_str(ticker)
                return

            if stage == "done":
                if ticker not in progress_state:
                    overall.total += total_value
                    overall.refresh()
                prev = progress_state.pop(ticker, 0)
                delta = max(0, total_value - prev)
                if delta:
                    overall.update(delta)
                overall.set_postfix_str(ticker)

        try:
            pipeline = DataPreparationPipeline(
                config,
                target_timeframes=target_tf,
                concurrency=concurrency,
                progress_callback=progress_callback,
            )
            results = await pipeline.run()
        finally:
            overall.close()

        table = Table(title="Data Preparation Results")
        table.add_column("Ticker")
        table.add_column("Timeframes")
        table.add_column("Missing Years")

        for result in results:
            timeframes = (
                ", ".join(sorted({metadata.timeframe for metadata in result.metadata}))
                or "—"
            )
            missing = (
                ", ".join(map(str, result.missing_years))
                if result.missing_years
                else "—"
            )
            table.add_row(result.ticker, timeframes, missing)

        console.print(table)

        report_path = _generate_report(results)
        console.print(f"[green]Report saved:[/green] {report_path.as_posix()}")

    asyncio.run(_run())


@data.command("list-datasets", help="Список доступных датасетов")
@click.option("--ticker", help="Фильтр по тикеру")
@click.option("--timeframe", help="Фильтр по таймфрейму")
@click.option("--source", help="Фильтр по источнику")
def list_datasets(
    ticker: Optional[str], timeframe: Optional[str], source: Optional[str]
) -> None:
    catalog = DatasetCatalog()
    datasets = catalog.search(
        ticker=ticker,
        timeframe=timeframe,
        source=source,
    )

    if not datasets:
        console.print("[yellow]Датасеты не найдены[/yellow]")
        return

    table = Table(title="Datasets")
    table.add_column("Ticker")
    table.add_column("Timeframe")
    table.add_column("Start")
    table.add_column("End")
    table.add_column("Bars")
    table.add_column("Hash")

    for ds in datasets:
        table.add_row(
            ds.ticker,
            ds.timeframe,
            ds.start_date.isoformat(),
            ds.end_date.isoformat(),
            str(ds.total_bars),
            ds.hash[:8],
        )

    console.print(table)


@data.command("validate-dataset", help="Проверить сохранённый датасет")
@click.option("--ticker", required=True, help="Тикер")
@click.option("--timeframe", required=True, help="Таймфрейм")
def validate_dataset(ticker: str, timeframe: str) -> None:
    storage = ParquetStorage()
    data = storage.load_dataset(ticker, timeframe)

    schema_validator = SchemaValidator()
    integrity_validator = IntegrityValidator()
    quality_validator = QualityValidator()

    schema = schema_validator.validate_all(data)
    integrity = integrity_validator.validate_all(data, timeframe)
    quality_price = quality_validator.detect_price_anomalies(data)
    quality_volume = quality_validator.check_volume_sanity(data)
    quality_spread = quality_validator.check_spread(data)

    table = Table(title=f"Validation: {ticker}/{timeframe}")
    table.add_column("Metric")
    table.add_column("Value")

    table.add_row("Bars", str(len(data)))
    table.add_row("Schema valid", str(schema.is_valid))
    table.add_row("Integrity valid", str(integrity.is_valid))
    table.add_row("Warnings", str(len(schema.warnings + integrity.warnings)))
    table.add_row("Price anomalies", str(quality_price.statistics.get("anomalies", 0)))
    table.add_row("Zero volumes", str(quality_volume.statistics.get("zero_volumes", 0)))
    table.add_row(
        "Large spreads", str(quality_spread.statistics.get("large_spreads", 0))
    )

    if schema.errors or integrity.errors:
        console.print("[red]Ошибки валидации:[/red]", schema.errors + integrity.errors)
    if schema.warnings or integrity.warnings:
        console.print(
            "[yellow]Предупреждения:[/yellow]", schema.warnings + integrity.warnings
        )

    console.print(table)


@data.command("resample-dataset", help="Ресэмплировать существующий датасет")
@click.option("--ticker", required=True)
@click.option("--source-timeframe", default="1m", show_default=True)
@click.option("--target-timeframe", required=True)
def resample_dataset(ticker: str, source_timeframe: str, target_timeframe: str) -> None:
    storage = ParquetStorage()
    data = storage.load_dataset(ticker, source_timeframe)

    resampler = TimeframeResampler()
    resampled = resampler.resample(data, source_timeframe, target_timeframe)

    pipeline_source: SourceLiteral = "manual"
    target_tf_literal = cast(TimeframeLiteral, target_timeframe)
    metadata = storage.save_dataset(
        resampled, ticker, target_tf_literal, source=pipeline_source
    )
    DatasetCatalog().add_dataset(metadata)
    DataVersioning().save_version(
        resampled, ticker, target_timeframe, description="CLI resample"
    )

    console.print(
        (
            "[green]Ресэмплинг завершён:[/green] {}/{} ({} bars)".format(
                ticker,
                target_timeframe,
                metadata.total_bars,
            )
        )
    )


@data.command("dataset-info", help="Информация о датасете")
@click.option("--ticker", required=True)
@click.option("--timeframe", help="Если не указан, показать все таймфреймы")
def dataset_info(ticker: str, timeframe: Optional[str]) -> None:
    storage = ParquetStorage()

    if timeframe:
        metadata = storage.get_metadata(ticker, timeframe)
        console.print(metadata.model_dump())
        return

    table = Table(title=f"Datasets for {ticker}")
    table.add_column("Timeframe")
    table.add_column("Start")
    table.add_column("End")
    table.add_column("Bars")

    catalog = DatasetCatalog()
    datasets = catalog.search(ticker=ticker)
    if not datasets:
        console.print("[yellow]Датасеты не найдены[/yellow]")
        return

    for ds in datasets:
        table.add_row(
            ds.timeframe,
            ds.start_date.isoformat(),
            ds.end_date.isoformat(),
            str(ds.total_bars),
        )

    console.print(table)


@data.command("export-dataset", help="Экспортировать датасет в указанный формат")
@click.option("--ticker", required=True, help="Тикер")
@click.option("--timeframe", required=True, help="Таймфрейм")
@click.option(
    "--format",
    "export_format",
    type=click.Choice(["csv", "parquet", "json"]),
    default="csv",
    show_default=True,
    help="Формат экспорта",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    help=(
        "Путь к выходному файлу (если не указан, "
        "использовать {ticker}_{timeframe}.{format})"
    ),
)
@click.option(
    "--from-date",
    type=click.DateTime(["%Y-%m-%d"]),
    help="Начальная дата (если не указана, экспортировать весь датасет)",
)
@click.option(
    "--to-date",
    type=click.DateTime(["%Y-%m-%d"]),
    help="Конечная дата (если не указана, экспортировать весь датасет)",
)
@click.option(
    "--compress/--no-compress",
    default=False,
    show_default=True,
    help="Сжать выходной файл (gzip)",
)
def export_dataset(
    ticker: str,
    timeframe: str,
    export_format: str,
    output: Optional[Path],
    from_date: Optional[datetime],
    to_date: Optional[datetime],
    compress: bool,
) -> None:
    """Экспортировать датасет в указанный формат."""
    from src.common.exceptions import StorageError

    storage = ParquetStorage()

    # Загрузить датасет
    try:
        data = storage.load_dataset(
            ticker,
            timeframe,
            from_date=from_date,
            to_date=to_date,
        )
    except StorageError as e:
        console.print(f"[red]Ошибка загрузки датасета:[/red] {e}")
        return

    if data.empty:
        console.print("[yellow]Датасет пустой[/yellow]")
        return

    # Определить имя выходного файла
    if output is None:
        suffix = f".{export_format}"
        if compress and export_format == "csv":
            suffix += ".gz"
        output = Path(f"{ticker}_{timeframe}{suffix}")

    # Экспортировать в указанный формат
    try:
        if export_format == "csv":
            if compress:
                data.to_csv(output, index=False, compression="gzip")
            else:
                data.to_csv(output, index=False)
        elif export_format == "parquet":
            # mypy требует Literal для compression
            if compress:
                data.to_parquet(str(output), compression="gzip", index=False)
            else:
                data.to_parquet(str(output), compression="snappy", index=False)
        elif export_format == "json":
            if compress:
                import gzip

                with gzip.open(output, "wt", encoding="utf-8") as f:
                    data.to_json(f, orient="records", date_format="iso", indent=2)
            else:
                data.to_json(output, orient="records", date_format="iso", indent=2)

        file_size = output.stat().st_size / (1024 * 1024)  # MB
        console.print(
            f"[green]Экспорт завершён:[/green] {output.as_posix()} "
            f"({len(data)} bars, {file_size:.2f} MB)"
        )

    except Exception as e:
        console.print(f"[red]Ошибка экспорта:[/red] {e}")
