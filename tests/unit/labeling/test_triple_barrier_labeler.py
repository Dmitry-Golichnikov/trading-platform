"""Тесты для TripleBarrierLabeler."""

import numpy as np
import pandas as pd
import pytest

from src.labeling.methods.triple_barrier import TripleBarrierLabeler


@pytest.fixture
def sample_data():
    """Создание тестовых данных."""
    dates = pd.date_range("2023-01-01", periods=100, freq="1H")
    np.random.seed(42)

    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

    data = pd.DataFrame(
        {
            "open": close_prices - np.random.rand(100) * 0.5,
            "high": close_prices + np.random.rand(100) * 1.5,
            "low": close_prices - np.random.rand(100) * 1.5,
            "close": close_prices,
            "volume": np.random.randint(1000, 10000, 100),
        },
        index=dates,
    )

    return data


def test_triple_barrier_initialization():
    """Тест инициализации TripleBarrierLabeler."""
    labeler = TripleBarrierLabeler(upper_barrier=0.02, lower_barrier=0.02, time_barrier=20)

    assert labeler.params["upper_barrier"] == 0.02
    assert labeler.params["lower_barrier"] == 0.02
    assert labeler.params["time_barrier"] == 20


def test_triple_barrier_invalid_params():
    """Тест валидации параметров."""
    with pytest.raises(ValueError):
        TripleBarrierLabeler(upper_barrier=-0.02)

    with pytest.raises(ValueError):
        TripleBarrierLabeler(time_barrier=-10)


def test_triple_barrier_basic_labeling(sample_data):
    """Тест базовой разметки."""
    labeler = TripleBarrierLabeler(upper_barrier=0.02, lower_barrier=0.02, time_barrier=20, direction="long+short")

    result = labeler.label(sample_data)

    # Проверяем наличие нужных колонок
    assert "label" in result.columns
    assert "barrier_hit" in result.columns
    assert "holding_period" in result.columns
    assert "realized_return" in result.columns

    # Проверяем типы
    assert result["label"].dtype == int
    assert result["holding_period"].dtype == int

    # Проверяем что barrier_hit имеет допустимые значения
    valid_barriers = {"upper", "lower", "time", "no_data", ""}
    assert set(result["barrier_hit"].unique()).issubset(valid_barriers)


def test_triple_barrier_atr_barriers(sample_data):
    """Тест ATR-based барьеров."""
    labeler = TripleBarrierLabeler(
        upper_barrier="atr",
        lower_barrier="atr",
        time_barrier=20,
        atr_window=14,
        atr_multiplier=2.0,
    )

    result = labeler.label(sample_data)

    assert "label" in result.columns
    assert result["label"].dtype == int


def test_triple_barrier_long_only(sample_data):
    """Тест long-only режима."""
    labeler = TripleBarrierLabeler(upper_barrier=0.02, lower_barrier=0.01, time_barrier=20, direction="long")

    result = labeler.label(sample_data)

    # В long-only не должно быть явных short меток в смысле long+short
    # Но могут быть -1 для неудачных long сделок
    assert set(result["label"].unique()).issubset({-1, 0, 1})


def test_triple_barrier_with_commissions(sample_data):
    """Тест учёта комиссий."""
    labeler_no_comm = TripleBarrierLabeler(
        upper_barrier=0.02,
        lower_barrier=0.02,
        time_barrier=20,
        include_commissions=False,
    )

    labeler_with_comm = TripleBarrierLabeler(
        upper_barrier=0.02,
        lower_barrier=0.02,
        time_barrier=20,
        include_commissions=True,
        commission_pct=0.001,
    )

    result_no_comm = labeler_no_comm.label(sample_data)
    result_with_comm = labeler_with_comm.label(sample_data)

    # С комиссиями realized_return должен быть меньше
    # (для profitable сделок)
    mask = result_no_comm["realized_return"] > 0
    if mask.any():
        assert (result_with_comm.loc[mask, "realized_return"] < result_no_comm.loc[mask, "realized_return"]).any()


def test_triple_barrier_min_return(sample_data):
    """Тест фильтрации по min_return."""
    labeler = TripleBarrierLabeler(
        upper_barrier=0.02,
        lower_barrier=0.02,
        time_barrier=20,
        min_return=0.01,  # Минимум 1%
    )

    result = labeler.label(sample_data)

    # Метки с |return| < min_return должны быть 0
    mask_non_zero = result["label"] != 0
    if mask_non_zero.any():
        assert (result.loc[mask_non_zero, "realized_return"].abs() >= 0.01).all()


def test_triple_barrier_holding_periods(sample_data):
    """Тест периодов удержания."""
    labeler = TripleBarrierLabeler(upper_barrier=0.02, lower_barrier=0.02, time_barrier=20)

    result = labeler.label(sample_data)

    # Holding period не должен превышать time_barrier
    assert result["holding_period"].max() <= 20

    # Holding period должен быть >= 0
    assert result["holding_period"].min() >= 0
