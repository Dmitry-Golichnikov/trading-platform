"""Точка входа для CLI: python -m src.interfaces.cli ..."""

import click

from src.interfaces.cli.data_commands import cli as data_cli
from src.interfaces.cli.feature_commands import features as features_cli
from src.interfaces.cli.labeling_commands import labels as labels_cli


@click.group()
def cli():
    """Торговая платформа - CLI интерфейс."""
    pass


# Добавляем группы команд
cli.add_command(data_cli, name="data")
cli.add_command(features_cli, name="features")
cli.add_command(labels_cli, name="labels")


if __name__ == "__main__":  # pragma: no cover
    cli()
