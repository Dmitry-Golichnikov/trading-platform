"""Тесты для модуля logging."""

import sys
from pathlib import Path
from unittest import mock

import pytest
from loguru import logger

from src.common.logging import setup_logging


def test_setup_logging_default_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Проверяет настройку логгера для stderr и JSON формата."""

    with (
        mock.patch.object(logger, "remove") as mock_remove,
        mock.patch.object(
            logger,
            "add",
        ) as mock_add,
    ):
        setup_logging(level="INFO", format_type="json")

    assert mock_remove.called
    add_args, add_kwargs = mock_add.call_args_list[0]
    assert add_args[0] is sys.stderr
    assert add_kwargs["format"].startswith("{")
    assert add_kwargs["level"] == "INFO"
    assert add_kwargs["colorize"] is False


@mock.patch("src.common.logging.logger.add")
@mock.patch("src.common.logging.logger.remove")
def test_setup_logging_with_file(
    mock_remove: mock.Mock, mock_add: mock.Mock, tmp_path: Path
) -> None:
    """Проверяет что при указании файла добавляется файловый handler."""

    log_file = tmp_path / "logs" / "app.log"

    setup_logging(
        level="INFO",
        log_file=log_file,
        format_type="text",
        rotation="1 MB",
        retention="3 days",
    )

    # первый вызов add: stderr
    stderr_args, stderr_kwargs = mock_add.call_args_list[0]
    assert stderr_args[0] is sys.stderr
    assert stderr_kwargs["colorize"] is True

    # второй вызов add: файл
    file_args, file_kwargs = mock_add.call_args_list[1]
    assert file_args[0] == log_file
    assert file_kwargs["rotation"] == "1 MB"
    assert file_kwargs["retention"] == "3 days"
    assert file_kwargs["compression"] == "zip"
