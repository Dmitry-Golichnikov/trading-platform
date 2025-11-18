"""
Тесты для TimeframeResampler.
"""

import pandas as pd
import pytest

from src.common.exceptions import PreprocessingError
from src.data.preprocessors.resampler import TimeframeResampler


class TestTimeframeResampler:
    """Тесты для ресэмплера."""

    @pytest.fixture
    def sample_1m_data(self) -> pd.DataFrame:
        """Создать тестовые минутные данные."""
        timestamps = pd.date_range("2020-01-01", periods=10, freq="1min")
        data = pd.DataFrame(
            {
                "timestamp": timestamps,
                "ticker": "TEST",
                "open": [
                    100.0,
                    101.0,
                    102.0,
                    103.0,
                    104.0,
                    105.0,
                    106.0,
                    107.0,
                    108.0,
                    109.0,
                ],
                "high": [
                    101.0,
                    102.0,
                    103.0,
                    104.0,
                    105.0,
                    106.0,
                    107.0,
                    108.0,
                    109.0,
                    110.0,
                ],
                "low": [
                    99.0,
                    100.0,
                    101.0,
                    102.0,
                    103.0,
                    104.0,
                    105.0,
                    106.0,
                    107.0,
                    108.0,
                ],
                "close": [
                    100.5,
                    101.5,
                    102.5,
                    103.5,
                    104.5,
                    105.5,
                    106.5,
                    107.5,
                    108.5,
                    109.5,
                ],
                "volume": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            }
        )
        return data

    def test_resample_1m_to_5m(self, sample_1m_data: pd.DataFrame) -> None:
        """Тест ресэмплинга 1m → 5m."""
        resampler = TimeframeResampler()
        result = resampler.resample(sample_1m_data, "1m", "5m")

        # Должно получиться 2 бара (10 минут / 5 минут)
        assert len(result) == 2

        # Проверить агрегацию первого бара (0-4 минуты)
        first_bar = result.iloc[0]
        assert first_bar["open"] == 100.0  # first
        assert first_bar["high"] == 105.0  # max
        assert first_bar["low"] == 99.0  # min
        assert first_bar["close"] == 104.5  # last
        assert first_bar["volume"] == 1500  # sum(100+200+300+400+500)

    def test_resample_1m_to_1h(self, sample_1m_data: pd.DataFrame) -> None:
        """Тест ресэмплинга 1m → 1h."""
        resampler = TimeframeResampler()
        result = resampler.resample(sample_1m_data, "1m", "1h")

        # Должно получиться 1 бар (10 минут < 1 час)
        assert len(result) == 1

        # Проверить агрегацию
        bar = result.iloc[0]
        assert bar["open"] == 100.0
        assert bar["high"] == 110.0
        assert bar["low"] == 99.0
        assert bar["close"] == 109.5
        assert bar["volume"] == 5500

    def test_invalid_timeframe_order(self, sample_1m_data: pd.DataFrame) -> None:
        """Тест ошибки при попытке ресэмплить в меньший таймфрейм."""
        resampler = TimeframeResampler()

        with pytest.raises(PreprocessingError):
            resampler.resample(sample_1m_data, "1h", "1m")

    def test_resample_multiple_timeframes(self, sample_1m_data: pd.DataFrame) -> None:
        """Тест ресэмплинга в несколько таймфреймов."""
        resampler = TimeframeResampler()
        results = resampler.resample_multiple_timeframes(sample_1m_data, source_tf="1m", target_tfs=["5m", "1h"])

        assert "1m" in results
        assert "5m" in results
        assert "1h" in results

        # Оригинальные данные должны остаться нетронутыми
        assert len(results["1m"]) == 10
        # 5m агрегация
        assert len(results["5m"]) == 2
        # 1h агрегация
        assert len(results["1h"]) == 1

    def test_ticker_preserved(self, sample_1m_data: pd.DataFrame) -> None:
        """Тест что ticker сохраняется."""
        resampler = TimeframeResampler()
        result = resampler.resample(sample_1m_data, "1m", "5m")

        assert "ticker" in result.columns
        assert result["ticker"].iloc[0] == "TEST"
