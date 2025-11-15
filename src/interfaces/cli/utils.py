"""Утилиты для CLI интерфейса."""

from __future__ import annotations

import functools
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.theme import Theme

# Создаем глобальную консоль с кастомной темой
custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "highlight": "bold magenta",
    }
)
console = Console(theme=custom_theme)

F = TypeVar("F", bound=Callable[..., Any])


def format_number(value: int | float, precision: int = 2) -> str:
    """Форматировать число с разделителями тысяч."""
    if isinstance(value, int):
        return f"{value:,}".replace(",", " ")
    return f"{value:,.{precision}f}".replace(",", " ")


def format_percentage(value: float, precision: int = 2) -> str:
    """Форматировать процент."""
    return f"{value:.{precision}f}%"


def format_bytes(size: float) -> str:
    """Форматировать размер в байтах."""
    size_float = float(size)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_float < 1024.0:
            return f"{size_float:.2f} {unit}"
        size_float /= 1024.0
    return f"{size_float:.2f} PB"


def format_duration(seconds: float) -> str:
    """Форматировать длительность."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_timestamp(dt: datetime) -> str:
    """Форматировать timestamp для отображения."""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def print_success(message: str) -> None:
    """Вывести сообщение об успехе."""
    console.print(f"[success]✓[/success] {message}")


def print_info(message: str) -> None:
    """Вывести информационное сообщение."""
    console.print(f"[info]ℹ[/info] {message}")


def print_warning(message: str) -> None:
    """Вывести предупреждение."""
    console.print(f"[warning]⚠[/warning] {message}")


def print_error(message: str) -> None:
    """Вывести ошибку."""
    console.print(f"[error]✗[/error] {message}")


def print_header(title: str, subtitle: Optional[str] = None) -> None:
    """Вывести заголовок."""
    text = title
    if subtitle:
        text = f"{title}\n{subtitle}"
    console.print(Panel(text, style="bold cyan", expand=False))


def print_separator(char: str = "─", length: int = 80) -> None:
    """Вывести разделитель."""
    console.print(char * length, style="dim")


def create_table(title: str, columns: list[str], show_header: bool = True) -> Table:
    """Создать таблицу с заданными колонками."""
    table = Table(title=title, show_header=show_header, header_style="bold magenta")
    for col in columns:
        table.add_column(col)
    return table


def create_progress_bar() -> Progress:
    """Создать progress bar с кастомным стилем."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def confirm(message: str, default: bool = False) -> bool:
    """Запросить подтверждение у пользователя."""
    return click.confirm(message, default=default)


def validate_path(path: str | Path, must_exist: bool = True) -> Path:
    """Валидировать путь к файлу/директории."""
    path_obj = Path(path)
    if must_exist and not path_obj.exists():
        raise click.BadParameter(f"Путь не существует: {path}")
    return path_obj


def validate_ticker(ticker: str) -> str:
    """Валидировать тикер."""
    ticker = ticker.upper().strip()
    if not ticker:
        raise click.BadParameter("Тикер не может быть пустым")
    if not ticker.isalnum():
        raise click.BadParameter(f"Недопустимый тикер: {ticker}")
    return ticker


def validate_timeframe(timeframe: str) -> str:
    """Валидировать таймфрейм."""
    valid_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
    if timeframe not in valid_timeframes:
        raise click.BadParameter(
            f"Недопустимый таймфрейм: {timeframe}. " f"Допустимые значения: {', '.join(valid_timeframes)}"
        )
    return timeframe


def validate_date(date_str: str) -> datetime:
    """Валидировать строку даты."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise click.BadParameter(f"Недопустимый формат даты: {date_str}. " "Ожидается формат: YYYY-MM-DD")


def parse_key_value_pairs(items: tuple[str, ...]) -> dict[str, str]:
    """Парсить пары ключ=значение."""
    result = {}
    for item in items:
        if "=" not in item:
            raise click.BadParameter(f"Недопустимый формат: {item}. " "Ожидается формат: key=value")
        key, value = item.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def handle_keyboard_interrupt(func: F) -> F:
    """Декоратор для обработки KeyboardInterrupt."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            print_warning("\nОперация прервана пользователем")
            sys.exit(130)  # Standard exit code for SIGINT

    return wrapper  # type: ignore


def handle_errors(func: F) -> F:
    """Декоратор для обработки ошибок с красивым выводом."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            print_warning("\nОперация прервана пользователем")
            sys.exit(130)
        except click.Abort:
            print_error("Операция отменена")
            sys.exit(1)
        except click.ClickException:
            raise  # Let Click handle its own exceptions
        except Exception as e:
            print_error(f"Произошла ошибка: {e}")
            if "--verbose" in sys.argv or "--debug" in sys.argv:
                console.print_exception()
            sys.exit(1)

    return wrapper  # type: ignore


def timed_operation(description: str) -> Callable[[F], F]:
    """Декоратор для замера времени выполнения операции."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            print_info(f"{description}...")
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                print_success(f"{description} завершено за {format_duration(elapsed)}")
                return result
            except Exception:
                elapsed = time.time() - start_time
                print_error(f"{description} не удалось (время: {format_duration(elapsed)})")
                raise

        return wrapper  # type: ignore

    return decorator


def load_tickers_from_file(file_path: Path) -> list[str]:
    """Загрузить список тикеров из файла."""
    if not file_path.exists():
        raise click.BadParameter(f"Файл не найден: {file_path}")

    tickers = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ticker = validate_ticker(line)
                tickers.append(ticker)

    if not tickers:
        raise click.BadParameter(f"Файл не содержит тикеров: {file_path}")

    return tickers


def save_output(data: Any, output_path: Path, format: str = "auto") -> None:
    """Сохранить данные в файл с автоопределением формата."""
    import json

    import pandas as pd
    import yaml

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "auto":
        format = output_path.suffix.lstrip(".")

    if format in ["yaml", "yml"]:
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    elif format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    elif format == "csv":
        if isinstance(data, pd.DataFrame):
            data.to_csv(output_path, index=False)
        else:
            raise ValueError("CSV format requires pandas DataFrame")
    elif format == "parquet":
        if isinstance(data, pd.DataFrame):
            data.to_parquet(output_path, compression="snappy")
        else:
            raise ValueError("Parquet format requires pandas DataFrame")
    else:
        raise ValueError(f"Неподдерживаемый формат: {format}")

    print_success(f"Данные сохранены: {output_path}")


def load_config(config_path: Path) -> dict[str, Any]:
    """Загрузить конфигурацию из YAML/JSON файла."""
    import json

    import yaml

    if not config_path.exists():
        raise click.BadParameter(f"Конфигурация не найдена: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.suffix in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        elif config_path.suffix == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Неподдерживаемый формат конфигурации: {config_path.suffix}")


def print_dict_as_table(data: dict[str, Any], title: str = "Информация") -> None:
    """Вывести словарь в виде таблицы."""
    table = Table(title=title, show_header=False)
    table.add_column("Ключ", style="cyan")
    table.add_column("Значение", style="green")

    for key, value in data.items():
        table.add_row(str(key), str(value))

    console.print(table)


class ProgressTracker:
    """Класс для отслеживания прогресса операций."""

    def __init__(self, description: str, total: Optional[int] = None):
        self.progress = create_progress_bar()
        self.task_id: Optional[TaskID] = None
        self.description = description
        self.total = total

    def __enter__(self) -> "ProgressTracker":
        self.progress.start()
        self.task_id = self.progress.add_task(self.description, total=self.total)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.progress.stop()

    def update(self, advance: int = 1, description: Optional[str] = None) -> None:
        """Обновить прогресс."""
        if self.task_id is not None:
            kwargs: dict[str, Any] = {"advance": advance}
            if description:
                kwargs["description"] = description
            self.progress.update(self.task_id, **kwargs)

    def set_total(self, total: int) -> None:
        """Установить общее количество."""
        if self.task_id is not None:
            self.progress.update(self.task_id, total=total)
