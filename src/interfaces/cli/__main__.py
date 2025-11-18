"""Точка входа для CLI: python -m src.interfaces.cli ..."""

import sys

import click
from rich.console import Console

from src.interfaces.cli.backtest_commands import backtest
from src.interfaces.cli.data_commands import cli as data_cli
from src.interfaces.cli.evaluation_commands import evaluate_group
from src.interfaces.cli.experiment_commands import experiment
from src.interfaces.cli.feature_commands import features as features_cli
from src.interfaces.cli.hyperopt_commands import hyperopt_group
from src.interfaces.cli.labeling_commands import labels as labels_cli
from src.interfaces.cli.model_commands import models
from src.interfaces.cli.pipeline_commands import pipeline_group
from src.interfaces.cli.quick_commands import quick

console = Console()


def _debug_callback(ctx: click.Context, param: click.Parameter | None, value: bool) -> bool:
    """Печатать сообщение даже если команда не выполняется (например, при --help)."""
    if value:
        console.print("[yellow]DEBUG режим включен[/yellow]")
    return value


@click.group()
@click.version_option(version="0.1.0", prog_name="trading-cli")
@click.option("--verbose", "-v", is_flag=True, help="Подробный вывод")
@click.option("--debug", is_flag=True, callback=_debug_callback, is_eager=True, help="Режим отладки")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, debug: bool) -> None:
    """
    🚀 Торговая платформа - CLI интерфейс

    Модульная платформа для разработки и тестирования торговых моделей.

    Основные команды:

      data       - Работа с данными (загрузка, валидация, экспорт)
      features   - Генерация и управление признаками
      labels     - Разметка таргетов
      models     - Обучение и оценка моделей
      backtest   - Бэктестинг стратегий
      hyperopt   - Оптимизация гиперпараметров
      experiment - Управление экспериментами
      pipeline   - Запуск пайплайнов
      evaluate   - Оценка моделей
      quick      - Быстрые команды для частых операций

    Примеры:

      $ trading-cli data load --ticker SBER --from 2020-01-01
      $ trading-cli features generate -c configs/features/default.yaml -d SBER_1m
      $ trading-cli model train -c configs/models/lightgbm.yaml -d data/train.parquet
      $ trading-cli backtest run -s strategy.yaml -m abc123 -d data/test.parquet

    Для получения помощи по конкретной команде:

      $ trading-cli COMMAND --help
    """
    # Сохраняем verbose и debug в контексте
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["debug"] = debug

    if debug:
        console.print("[yellow]DEBUG режим включен[/yellow]")


# Добавляем группы команд
cli.add_command(data_cli, name="data")
cli.add_command(features_cli, name="features")
cli.add_command(labels_cli, name="labels")
cli.add_command(models, name="model")
cli.add_command(backtest, name="backtest")
cli.add_command(hyperopt_group, name="hyperopt")
cli.add_command(evaluate_group, name="evaluate")
cli.add_command(pipeline_group, name="pipeline")
cli.add_command(experiment, name="experiment")
cli.add_command(quick, name="quick")


@cli.command("version")
def version() -> None:
    """Показать версию приложения."""
    console.print("[bold green]Trading Platform CLI v0.1.0[/bold green]")
    console.print("Python:", sys.version.split()[0])


@cli.command("autocomplete")
@click.option(
    "--shell",
    type=click.Choice(["bash", "zsh", "fish"]),
    required=True,
    help="Shell для генерации",
)
def autocomplete(shell: str) -> None:
    """
    Генерировать скрипт автодополнения для shell.

    Установка:

      Bash:
        $ trading-cli autocomplete --shell bash >> ~/.bashrc
        $ source ~/.bashrc

      Zsh:
        $ trading-cli autocomplete --shell zsh >> ~/.zshrc
        $ source ~/.zshrc

      Fish:
        $ trading-cli autocomplete --shell fish > ~/.config/fish/completions/trading-cli.fish
    """
    if shell == "bash":
        script = """
# trading-cli bash completion
_trading_cli_completion() {
    local IFS=$'\\n'
    COMPREPLY=( $( env COMP_WORDS="${COMP_WORDS[*]}" \\
                   COMP_CWORD=$COMP_CWORD \\
                   _TRADING_CLI_COMPLETE=complete $1 ) )
    return 0
}

complete -F _trading_cli_completion -o default trading-cli
"""
    elif shell == "zsh":
        script = """
# trading-cli zsh completion
#compdef trading-cli

_trading_cli_completion() {
    eval $(env _TRADING_CLI_COMPLETE=complete-zsh trading-cli)
}

if [[ "$(basename -- ${(%):-%x})" != "_trading_cli_completion" ]]; then
    compdef _trading_cli_completion trading-cli
fi
"""
    elif shell == "fish":
        script = """
# trading-cli fish completion
function __fish_trading_cli_complete
    set -lx _TRADING_CLI_COMPLETE complete-fish
    trading-cli
end

complete --command trading-cli --no-files --arguments '(__fish_trading_cli_complete)'
"""
    else:
        console.print(f"[red]Неподдерживаемый shell: {shell}[/red]")
        return

    console.print(script)


if __name__ == "__main__":  # pragma: no cover
    cli(obj={})
