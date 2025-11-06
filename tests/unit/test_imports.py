"""Тесты импорта всех базовых модулей."""

import importlib
import sys
from typing import List

import pytest


def test_common_modules_import() -> None:
    """Проверяет что все модули из common импортируются без ошибок."""
    common_modules = [
        "src.common.config",
        "src.common.logging",
        "src.common.validation",
        "src.common.interfaces",
    ]

    for module_name in common_modules:
        try:
            module = importlib.import_module(module_name)
            assert module is not None
        except ImportError as e:
            pytest.fail(f"Не удалось импортировать модуль {module_name}: {e}")


def test_main_modules_import() -> None:
    """Проверяет что основные модули проекта импортируются."""
    main_modules = [
        "src.data",
        "src.features",
        "src.labeling",
        "src.modeling",
        "src.evaluation",
        "src.backtesting",
        "src.pipelines",
        "src.orchestration",
    ]

    for module_name in main_modules:
        try:
            module = importlib.import_module(module_name)
            assert module is not None
        except ImportError as e:
            pytest.fail(f"Не удалось импортировать модуль {module_name}: {e}")


@pytest.mark.parametrize(
    "module_name,expected_exports",
    [
        ("src.common.config", ["load_yaml_config", "merge_configs", "load_env"]),
        ("src.common.logging", ["setup_logging"]),
        (
            "src.common.validation",
            ["OHLCVData", "validate_ohlcv_dataframe", "check_look_ahead"],
        ),
        ("src.common.interfaces", ["DataLoader", "Model", "FeatureCalculator"]),
    ],
)
def test_module_exports(module_name: str, expected_exports: List[str]) -> None:
    """Проверяет что модули экспортируют ожидаемые функции/классы."""
    module = importlib.import_module(module_name)

    for export_name in expected_exports:
        assert hasattr(module, export_name), f"Модуль {module_name} не содержит {export_name}"


def test_src_package_exists() -> None:
    """Проверяет что пакет src существует и импортируется."""
    import src

    assert src is not None
    assert hasattr(src, "__path__")


def test_no_circular_imports() -> None:
    """Проверяет отсутствие циклических импортов."""
    # Импортируем все модули в определенном порядке
    modules_to_import = [
        "src.common.interfaces",
        "src.common.validation",
        "src.common.config",
        "src.common.logging",
    ]

    imported_modules = []
    for module_name in modules_to_import:
        try:
            module = importlib.import_module(module_name)
            imported_modules.append(module)
        except ImportError as e:
            if "circular import" in str(e).lower():
                pytest.fail(f"Обнаружен циклический импорт в модуле {module_name}")
            raise

    assert len(imported_modules) == len(modules_to_import)


def test_python_version() -> None:
    """Проверяет что используется Python 3.10+."""
    assert sys.version_info >= (
        3,
        10,
    ), f"Требуется Python 3.10+, текущая версия: {sys.version}"
