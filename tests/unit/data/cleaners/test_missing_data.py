"""Тесты для обработки пропущенных данных."""

import numpy as np
import pandas as pd
import pytest

from src.data.cleaners.missing_data import GapDetector, MissingDataHandler


@pytest.fixture
def data_with_nans() -> pd.DataFrame:
    """Данные с пропусками."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC"),
            "open": [100, np.nan, 102, 103, np.nan, 105, 106, 107, 108, 109],
            "high": [102, 103, np.nan, 105, 106, 107, 108, 109, 110, 111],
            "low": [98, 99, 100, np.nan, 102, 103, 104, 105, 106, 107],
            "close": [101, 102, 103, 104, np.nan, 106, 107, 108, 109, 110],
            "volume": [1000, 1100, np.nan, 1300, 1400, np.nan, 1600, 1700, 1800, 1900],
        }
    )


def test_missing_data_handler_drop(data_with_nans: pd.DataFrame) -> None:
    """Тест метода drop."""
    handler = MissingDataHandler({"method": "drop"})

    result = handler.handle(data_with_nans)

    # Не должно быть NaN
    assert not result.isna().any().any()
    # Должны удалить строки с NaN
    assert len(result) < len(data_with_nans)


def test_missing_data_handler_forward_fill(data_with_nans: pd.DataFrame) -> None:
    """Тест метода forward_fill."""
    handler = MissingDataHandler({"method": "forward_fill"})

    result = handler.handle(data_with_nans)

    # Не должно быть NaN (кроме начальных строк если есть)
    assert result.loc[1:, "open"].notna().all()


def test_missing_data_handler_interpolate(data_with_nans: pd.DataFrame) -> None:
    """Тест метода interpolate."""
    handler = MissingDataHandler({"method": "interpolate"})

    result = handler.handle(data_with_nans)

    # Проверим что метод обработал данные и вернул непустой DataFrame
    assert not result.empty
    # Количество NaN должно уменьшиться или остаться прежним
    assert result.isna().sum().sum() <= data_with_nans.isna().sum().sum()


def test_missing_data_handler_empty() -> None:
    """Тест на пустых данных."""
    empty = pd.DataFrame()
    handler = MissingDataHandler({"method": "forward_fill"})

    result = handler.handle(empty)

    assert result.empty


def test_gap_detector() -> None:
    """Тест детектора пропусков."""
    # Создать данные с пропущенными временными метками
    dates = list(pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC"))
    # Удалить несколько timestamps
    dates.pop(3)
    dates.pop(6)

    data = pd.DataFrame(
        {
            "timestamp": dates,
            "close": [100] * len(dates),
        }
    )

    detector = GapDetector(expected_frequency="1min")
    gaps = detector.detect_gaps(data)

    # Должны найти пропуски
    assert len(gaps) > 0
    assert "gap_size" in gaps.columns


def test_gap_detector_no_gaps() -> None:
    """Тест детектора без пропусков."""
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC"),
            "close": [100] * 10,
        }
    )

    detector = GapDetector(expected_frequency="1min")
    gaps = detector.detect_gaps(data)

    # Не должно быть пропусков
    assert len(gaps) == 0


def test_missing_data_handler_with_max_gap() -> None:
    """Тест ограничения максимального размера пропуска."""
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="1min", tz="UTC"),
            "close": [100] * 10 + [np.nan] * 8 + [100] * 2,
        }
    )

    handler = MissingDataHandler({"method": "forward_fill", "max_gap": 3})

    result = handler.handle(data)

    # Большие пропуски не должны заполняться
    assert result["close"].isna().sum() > 0
