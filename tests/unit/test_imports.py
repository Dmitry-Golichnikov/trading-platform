"""Тесты для проверки импортов основных модулей."""


def test_import_src() -> None:
    """Проверяет что основной пакет src импортируется."""
    import src

    assert src.__version__ == "0.1.0"


def test_import_common_modules() -> None:
    """Проверяет что модули common импортируются."""
    from src.common import config, interfaces, logging, validation

    assert config is not None
    assert interfaces is not None
    assert logging is not None
    assert validation is not None


def test_import_all_main_modules() -> None:
    """Проверяет что все основные модули импортируются."""
    import src.backtesting
    import src.data
    import src.evaluation
    import src.features
    import src.interfaces
    import src.labeling
    import src.modeling
    import src.orchestration
    import src.pipelines

    assert src.data is not None
    assert src.features is not None
    assert src.labeling is not None
    assert src.modeling is not None
    assert src.evaluation is not None
    assert src.backtesting is not None
    assert src.pipelines is not None
    assert src.orchestration is not None
    assert src.interfaces is not None


def test_import_interface_modules() -> None:
    """Проверяет что интерфейсы импортируются."""
    import src.interfaces.cli
    import src.interfaces.gui

    assert src.interfaces.cli is not None
    assert src.interfaces.gui is not None
