"""Тесты для модуля config."""

from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from src.common import config as config_module
from src.common.config import load_yaml_config, merge_configs


def test_load_yaml_config(temp_dir: Path) -> None:
    """Тест загрузки YAML конфигурации."""
    # Создаем тестовый конфиг
    config_file = temp_dir / "test_config.yaml"
    test_config = {
        "param1": "value1",
        "param2": 42,
        "nested": {"key": "value"},
    }

    with open(config_file, "w") as f:
        yaml.dump(test_config, f)

    # Загружаем и проверяем
    loaded_config = load_yaml_config(config_file)
    assert loaded_config == test_config


def test_load_yaml_config_file_not_found() -> None:
    """Тест загрузки несуществующего файла."""
    with pytest.raises(FileNotFoundError):
        load_yaml_config(Path("nonexistent.yaml"))


def test_load_yaml_config_invalid_yaml(temp_dir: Path) -> None:
    """Тест ошибки при некорректном YAML."""

    config_file = temp_dir / "invalid.yaml"
    config_file.write_text("invalid: [unclosed", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        load_yaml_config(config_file)


def test_merge_configs() -> None:
    """Тест объединения конфигураций."""
    base_config: Dict[str, Any] = {
        "a": 1,
        "b": 2,
        "nested": {"x": 10, "y": 20},
    }

    override_config: Dict[str, Any] = {
        "b": 3,
        "c": 4,
        "nested": {"y": 30, "z": 40},
    }

    merged = merge_configs(base_config, override_config)

    assert merged["a"] == 1  # из base
    assert merged["b"] == 3  # переопределено
    assert merged["c"] == 4  # добавлено
    assert merged["nested"]["x"] == 10  # из base
    assert merged["nested"]["y"] == 30  # переопределено
    assert merged["nested"]["z"] == 40  # добавлено


def test_merge_configs_empty() -> None:
    """Тест объединения с пустой конфигурацией."""
    base_config = {"a": 1, "b": 2}
    empty_config: Dict[str, Any] = {}

    merged = merge_configs(base_config, empty_config)
    assert merged == base_config

    merged = merge_configs(empty_config, base_config)
    assert merged == base_config


def test_load_env_calls_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Проверяет что load_env вызывает load_dotenv."""

    called = {"value": False}

    def fake_load_dotenv() -> None:
        called["value"] = True

    monkeypatch.setattr(config_module, "load_dotenv", fake_load_dotenv)

    config_module.load_env()

    assert called["value"] is True
