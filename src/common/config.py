"""Система управления конфигурациями."""

from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel


class Config(BaseModel):
    """Базовый класс конфигурации."""

    pass


def load_env() -> None:
    """Загружает переменные окружения из .env файла."""
    load_dotenv()


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """
    Загружает YAML конфигурацию из файла.

    Args:
        config_path: Путь к YAML файлу

    Returns:
        Словарь с конфигурацией

    Raises:
        FileNotFoundError: Если файл не найден
        yaml.YAMLError: Если файл содержит невалидный YAML
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def merge_configs(
    base_config: Dict[str, Any], override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Объединяет две конфигурации с приоритетом для override_config.

    Args:
        base_config: Базовая конфигурация
        override_config: Конфигурация для переопределения

    Returns:
        Объединенная конфигурация
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged
