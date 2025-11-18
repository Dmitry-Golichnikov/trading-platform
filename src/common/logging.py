"""Система логирования."""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
    format_type: str = "json",
) -> None:
    """
    Настраивает систему логирования.

    Args:
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
        log_file: Путь к файлу логов (опционально)
        rotation: Правило ротации логов
        retention: Правило хранения логов
        format_type: Формат логов (json или text)
    """
    # Удаляем дефолтный handler
    logger.remove()

    # Формат для логов
    if format_type == "json":
        log_format = (
            "{"
            '"time": "{time:YYYY-MM-DD HH:mm:ss.SSS}", '
            '"level": "{level}", '
            '"message": "{message}", '
            '"file": "{file}", '
            '"line": {line}'
            "}"
        )
    else:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:"
            "<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

    # Добавляем handler для stderr
    logger.add(
        sys.stderr,
        format=log_format,
        level=level,
        colorize=format_type != "json",
    )

    # Добавляем handler для файла если указан
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format=log_format,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

    logger.info(f"Логирование настроено: level={level}, format={format_type}")
