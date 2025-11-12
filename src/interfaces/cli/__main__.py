"""Точка входа для CLI: python -m src.interfaces.cli ..."""

import click

from src.interfaces.cli.data_commands import cli as data_cli
from src.interfaces.cli.evaluation_commands import evaluate_group
from src.interfaces.cli.feature_commands import features as features_cli
from src.interfaces.cli.hyperopt_commands import hyperopt_group
from src.interfaces.cli.labeling_commands import labels as labels_cli


@click.group()
def cli():
    """Торговая платформа - CLI интерфейс."""
    pass


# Добавляем группы команд
cli.add_command(data_cli, name="data")
cli.add_command(features_cli, name="features")
cli.add_command(labels_cli, name="labels")
cli.add_command(hyperopt_group, name="hyperopt")
cli.add_command(evaluate_group, name="evaluate")


if __name__ == "__main__":  # pragma: no cover
    cli()
