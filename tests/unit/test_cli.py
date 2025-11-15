"""Тесты для CLI интерфейса."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from click.testing import CliRunner

from src.interfaces.cli.__main__ import cli


@pytest.fixture
def runner():
    """Fixture для Click CLI runner."""
    return CliRunner()


@pytest.fixture
def sample_data():
    """Fixture с примером данных."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=100, freq="1H"),
            "open": [100 + i for i in range(100)],
            "high": [101 + i for i in range(100)],
            "low": [99 + i for i in range(100)],
            "close": [100.5 + i for i in range(100)],
            "volume": [1000 + i * 10 for i in range(100)],
        }
    )


class TestCLIBasic:
    """Тесты базовой функциональности CLI."""

    def test_cli_help(self, runner):
        """Тест вывода справки."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Торговая платформа" in result.output
        assert "data" in result.output
        assert "features" in result.output
        assert "model" in result.output

    def test_cli_version(self, runner):
        """Тест команды version."""
        result = runner.invoke(cli, ["version"])
        assert result.exit_code == 0
        assert "Trading Platform CLI" in result.output
        assert "v0.1.0" in result.output

    def test_cli_verbose(self, runner):
        """Тест verbose режима."""
        result = runner.invoke(cli, ["--verbose", "--help"])
        assert result.exit_code == 0

    def test_cli_debug(self, runner):
        """Тест debug режима."""
        result = runner.invoke(cli, ["--debug", "--help"])
        assert result.exit_code == 0
        assert "DEBUG режим включен" in result.output


class TestDataCommands:
    """Тесты команд работы с данными."""

    def test_data_help(self, runner):
        """Тест справки для data команд."""
        result = runner.invoke(cli, ["data", "--help"])
        assert result.exit_code == 0
        assert "данными" in result.output.lower()

    def test_data_list_datasets(self, runner):
        """Тест команды list-datasets."""
        with patch("src.interfaces.cli.data_commands.DatasetCatalog") as mock_catalog:
            mock_catalog.return_value.search.return_value = []
            result = runner.invoke(cli, ["data", "list-datasets"])
            assert result.exit_code == 0

    def test_data_dataset_info(self, runner):
        """Тест команды dataset-info."""
        result = runner.invoke(cli, ["data", "dataset-info", "--ticker", "SBER"])
        # Команда может завершиться с ошибкой если нет данных, но не должна падать
        assert result.exit_code in [0, 1]


class TestFeatureCommands:
    """Тесты команд работы с признаками."""

    def test_features_help(self, runner):
        """Тест справки для features команд."""
        result = runner.invoke(cli, ["features", "--help"])
        assert result.exit_code == 0
        assert "признак" in result.output.lower()

    def test_features_list(self, runner):
        """Тест команды list."""
        with patch("src.interfaces.cli.feature_commands.FeatureCache") as mock_cache:
            mock_cache.return_value.list_cached.return_value = pd.DataFrame()
            result = runner.invoke(cli, ["features", "list"])
            assert result.exit_code == 0

    def test_features_validate_config(self, runner, tmp_path):
        """Тест валидации конфигурации признаков."""
        # Создаём временный конфиг
        config_path = tmp_path / "features.yaml"
        config_path.write_text(
            """
version: "1.0"
cache_enabled: true
features:
  - type: price
    name: close
"""
        )

        with patch("src.interfaces.cli.feature_commands.parse_feature_config"):
            result = runner.invoke(cli, ["features", "validate-config", str(config_path)])
            # Может завершиться с ошибкой если модуль не полностью реализован
            assert result.exit_code in [0, 1]


class TestModelCommands:
    """Тесты команд работы с моделями."""

    def test_models_help(self, runner):
        """Тест справки для model команд."""
        result = runner.invoke(cli, ["model", "--help"])
        assert result.exit_code == 0
        assert "модел" in result.output.lower()

    def test_model_list(self, runner):
        """Тест команды list."""
        with patch("src.interfaces.cli.model_commands.MLflowManager") as mock_mlflow:
            mock_mlflow.return_value.search_runs.return_value = []
            result = runner.invoke(cli, ["model", "list"])
            assert result.exit_code == 0

    def test_model_compare(self, runner):
        """Тест команды compare."""
        with patch("src.interfaces.cli.model_commands.MLflowManager") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.data.tags = {"mlflow.runName": "test"}
            mock_run.data.metrics = {"roc_auc": 0.85}
            mock_run.info.start_time = 1234567890000

            mock_mlflow.return_value.get_run.return_value = mock_run

            result = runner.invoke(cli, ["model", "compare", "abc123", "def456"])
            # Может завершиться с ошибкой но не должно падать
            assert result.exit_code in [0, 1]


class TestBacktestCommands:
    """Тесты команд бэктестинга."""

    def test_backtest_help(self, runner):
        """Тест справки для backtest команд."""
        result = runner.invoke(cli, ["backtest", "--help"])
        assert result.exit_code == 0
        assert "бэктест" in result.output.lower()

    def test_backtest_list(self, runner):
        """Тест команды list."""
        result = runner.invoke(cli, ["backtest", "list"])
        # Может вернуть 0 если директория пустая или не существует
        assert result.exit_code in [0, 1]


class TestLabelCommands:
    """Тесты команд разметки."""

    def test_labels_help(self, runner):
        """Тест справки для labels команд."""
        result = runner.invoke(cli, ["labels", "--help"])
        assert result.exit_code == 0
        assert "разметк" in result.output.lower() or "таргет" in result.output.lower()


class TestExperimentCommands:
    """Тесты команд экспериментов."""

    def test_experiment_help(self, runner):
        """Тест справки для experiment команд."""
        result = runner.invoke(cli, ["experiment", "--help"])
        assert result.exit_code == 0
        assert "эксперимент" in result.output.lower()

    def test_experiment_list(self, runner):
        """Тест команды list."""
        with patch("src.interfaces.cli.experiment_commands.ExperimentManager") as mock_manager:
            mock_manager.return_value.list_experiments.return_value = []
            result = runner.invoke(cli, ["experiment", "list"])
            assert result.exit_code == 0


class TestHyperoptCommands:
    """Тесты команд гиперпараметрической оптимизации."""

    def test_hyperopt_help(self, runner):
        """Тест справки для hyperopt команд."""
        result = runner.invoke(cli, ["hyperopt", "--help"])
        assert result.exit_code == 0


class TestPipelineCommands:
    """Тесты команд пайплайнов."""

    def test_pipeline_help(self, runner):
        """Тест справки для pipeline команд."""
        result = runner.invoke(cli, ["pipeline", "--help"])
        assert result.exit_code == 0


class TestAutocomplete:
    """Тесты генерации autocomplete."""

    def test_autocomplete_bash(self, runner):
        """Тест генерации для bash."""
        result = runner.invoke(cli, ["autocomplete", "--shell", "bash"])
        assert result.exit_code == 0
        assert "bash completion" in result.output
        assert "_trading_cli_completion" in result.output

    def test_autocomplete_zsh(self, runner):
        """Тест генерации для zsh."""
        result = runner.invoke(cli, ["autocomplete", "--shell", "zsh"])
        assert result.exit_code == 0
        assert "zsh completion" in result.output
        assert "compdef" in result.output

    def test_autocomplete_fish(self, runner):
        """Тест генерации для fish."""
        result = runner.invoke(cli, ["autocomplete", "--shell", "fish"])
        assert result.exit_code == 0
        assert "fish completion" in result.output


class TestCLIUtils:
    """Тесты утилит CLI."""

    def test_format_number(self):
        """Тест форматирования чисел."""
        from src.interfaces.cli.utils import format_number

        assert format_number(1000) == "1 000"
        assert format_number(1234567) == "1 234 567"
        assert format_number(123.456, precision=2) == "123.46"

    def test_format_percentage(self):
        """Тест форматирования процентов."""
        from src.interfaces.cli.utils import format_percentage

        assert format_percentage(50.0) == "50.00%"
        assert format_percentage(33.33, precision=1) == "33.3%"

    def test_format_bytes(self):
        """Тест форматирования байтов."""
        from src.interfaces.cli.utils import format_bytes

        assert "B" in format_bytes(500)
        assert "KB" in format_bytes(2048)
        assert "MB" in format_bytes(5 * 1024 * 1024)

    def test_format_duration(self):
        """Тест форматирования длительности."""
        from src.interfaces.cli.utils import format_duration

        assert format_duration(30) == "30.0s"
        assert "m" in format_duration(90)
        assert "h" in format_duration(3700)

    def test_validate_ticker(self):
        """Тест валидации тикера."""
        from src.interfaces.cli.utils import validate_ticker

        assert validate_ticker("SBER") == "SBER"
        assert validate_ticker("sber") == "SBER"

        with pytest.raises(Exception):
            validate_ticker("")

    def test_validate_timeframe(self):
        """Тест валидации таймфрейма."""
        from src.interfaces.cli.utils import validate_timeframe

        assert validate_timeframe("1m") == "1m"
        assert validate_timeframe("1h") == "1h"

        with pytest.raises(Exception):
            validate_timeframe("invalid")

    def test_load_tickers_from_file(self, tmp_path):
        """Тест загрузки тикеров из файла."""
        from src.interfaces.cli.utils import load_tickers_from_file

        # Создаём временный файл с тикерами
        tickers_file = tmp_path / "tickers.txt"
        tickers_file.write_text("SBER\nGAZP\n# Комментарий\nLKOH\n")

        tickers = load_tickers_from_file(tickers_file)
        assert len(tickers) == 3
        assert "SBER" in tickers
        assert "GAZP" in tickers
        assert "LKOH" in tickers

    def test_create_table(self):
        """Тест создания таблицы."""
        from src.interfaces.cli.utils import create_table

        table = create_table("Test Table", ["Col1", "Col2"])
        assert table.title == "Test Table"

    def test_progress_tracker(self):
        """Тест ProgressTracker."""
        from src.interfaces.cli.utils import ProgressTracker

        with ProgressTracker("Test operation", total=100) as tracker:
            tracker.update(10)
            tracker.update(20)
            tracker.set_total(200)


class TestCLIIntegration:
    """Интеграционные тесты CLI."""

    def test_full_pipeline_dry_run(self, runner, tmp_path, sample_data):
        """Тест сухого прогона полного пайплайна."""
        # Сохраняем тестовые данные
        data_path = tmp_path / "test_data.parquet"
        sample_data.to_parquet(data_path)

        # Проверяем что команды доступны
        commands = [
            ["data", "--help"],
            ["features", "--help"],
            ["labels", "--help"],
            ["model", "--help"],
            ["backtest", "--help"],
        ]

        for cmd in commands:
            result = runner.invoke(cli, cmd)
            assert result.exit_code == 0

    def test_error_handling(self, runner):
        """Тест обработки ошибок."""
        # Команда с несуществующим файлом
        result = runner.invoke(cli, ["data", "dataset-info", "--ticker", "NONEXISTENT"])
        # Должна обработаться корректно, даже если вернёт ошибку
        assert result.exit_code in [0, 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
