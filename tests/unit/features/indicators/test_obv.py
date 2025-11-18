"""
Тесты для OBV индикатора.
"""

from src.features.indicators.volume.obv import OBV


def test_obv_output(sample_ohlcv_data):
    """Тест что OBV возвращает правильную колонку."""
    obv = OBV()
    result = obv.calculate(sample_ohlcv_data)

    assert "OBV" in result.columns


def test_obv_uptrend(simple_uptrend_data):
    """Тест OBV на восходящем тренде."""
    obv = OBV()
    result = obv.calculate(simple_uptrend_data)

    obv_values = result["OBV"].dropna()

    # OBV должен расти на восходящем тренде
    assert obv_values.iloc[-1] > obv_values.iloc[0]


def test_obv_downtrend(simple_downtrend_data):
    """Тест OBV на нисходящем тренде."""
    obv = OBV()
    result = obv.calculate(simple_downtrend_data)

    obv_values = result["OBV"].dropna()

    # OBV должен падать на нисходящем тренде
    assert obv_values.iloc[-1] < obv_values.iloc[0]


def test_obv_required_columns():
    """Тест получения необходимых колонок."""
    obv = OBV()
    assert set(obv.get_required_columns()) == {"close", "volume"}


def test_obv_lookback_period():
    """Тест периода разогрева."""
    obv = OBV()
    assert obv.get_lookback_period() == 1
