"""CLI интерфейсы проекта."""

from src.interfaces.cli.data_commands import cli, data
from src.interfaces.cli.labeling_commands import labels

__all__ = ["cli", "data", "labels"]
