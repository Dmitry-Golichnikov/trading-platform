"""Конфигурация pytest и общие fixtures для тестов."""

import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Создает временную директорию для тестов."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def artifacts_dir(temp_dir: Path) -> Path:
    """Создает временную директорию для артефактов."""
    artifacts = temp_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    return artifacts


@pytest.fixture
def config_dir(temp_dir: Path) -> Path:
    """Создает временную директорию для конфигураций."""
    configs = temp_dir / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    return configs


@pytest.fixture(autouse=True)
def set_test_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Устанавливает тестовое окружение."""
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("GPU_ENABLED", "false")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///tmp/mlruns")


@pytest.fixture
def sample_ohlcv_data() -> dict:
    """Возвращает примерные OHLCV данные для тестов."""
    return {
        "timestamp": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "volume": [1000, 1500],
    }
