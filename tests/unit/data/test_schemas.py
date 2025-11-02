"""
Тесты для Pydantic схем модуля данных.
"""

from datetime import date, datetime
from pathlib import Path

import pytest

from src.data.schemas import DatasetConfig, OHLCVBar


class TestOHLCVBar:
    """Тесты для OHLCVBar схемы."""

    def test_valid_bar(self) -> None:
        """Тест валидного бара."""
        bar = OHLCVBar(
            timestamp=datetime(2020, 1, 1, 12, 0, tzinfo=None).replace(
                tzinfo=datetime.now().astimezone().tzinfo
            ),
            ticker="SBER",
            open=100.0,
            high=105.0,
            low=99.0,
            close=102.0,
            volume=1000,
        )

        assert bar.ticker == "SBER"
        assert bar.volume == 1000

    def test_high_validation(self) -> None:
        """Тест валидации high >= max(open, close)."""
        with pytest.raises(ValueError, match="high"):
            OHLCVBar(
                timestamp=datetime(
                    2020, 1, 1, tzinfo=datetime.now().astimezone().tzinfo
                ),
                ticker="SBER",
                open=100.0,
                high=99.0,  # Invalid: high < open
                low=98.0,
                close=101.0,
                volume=1000,
            )

    def test_low_validation(self) -> None:
        """Тест валидации low <= min(open, close)."""
        with pytest.raises(ValueError, match="low"):
            OHLCVBar(
                timestamp=datetime(
                    2020, 1, 1, tzinfo=datetime.now().astimezone().tzinfo
                ),
                ticker="SBER",
                open=100.0,
                high=105.0,
                low=101.0,  # Invalid: low > open
                close=102.0,
                volume=1000,
            )

    def test_negative_volume(self) -> None:
        """Тест на отрицательный volume."""
        with pytest.raises(ValueError):
            OHLCVBar(
                timestamp=datetime(
                    2020, 1, 1, tzinfo=datetime.now().astimezone().tzinfo
                ),
                ticker="SBER",
                open=100.0,
                high=105.0,
                low=99.0,
                close=102.0,
                volume=-100,  # Invalid
            )

    def test_timezone_aware_required(self) -> None:
        """Тест что timestamp должен быть timezone-aware."""
        with pytest.raises(ValueError, match="timezone"):
            OHLCVBar(
                timestamp=datetime(2020, 1, 1, 12, 0),  # Naive datetime
                ticker="SBER",
                open=100.0,
                high=105.0,
                low=99.0,
                close=102.0,
                volume=1000,
            )


class TestDatasetConfig:
    """Тесты для DatasetConfig схемы."""

    def test_valid_config(self) -> None:
        """Тест валидной конфигурации."""
        config = DatasetConfig(
            ticker="SBER",
            timeframe="1m",
            from_date=date(2020, 1, 1),
            to_date=date(2023, 12, 31),
            source_type="local",
            file_path=Path("data.parquet"),
        )

        assert config.ticker == "SBER"
        assert config.update_latest_year is True
        assert config.backfill_missing is True

    def test_requires_ticker_or_file(self) -> None:
        """Тест что нужен ticker или tickers_file."""
        with pytest.raises(ValueError, match="ticker"):
            DatasetConfig(
                ticker=None,  # Missing
                tickers_file=None,  # Missing
                timeframe="1m",
                from_date=date(2020, 1, 1),
                to_date=date(2023, 12, 31),
                source_type="local",
            )

    def test_local_requires_file_path(self) -> None:
        """Тест что source_type='local' требует file_path."""
        with pytest.raises(ValueError, match="file_path"):
            DatasetConfig(
                ticker="SBER",
                timeframe="1m",
                from_date=date(2020, 1, 1),
                to_date=date(2023, 12, 31),
                source_type="local",
                file_path=None,  # Missing
            )

    def test_date_validation(self) -> None:
        """Тест валидации дат."""
        with pytest.raises(ValueError, match="to_date"):
            DatasetConfig(
                ticker="SBER",
                timeframe="1m",
                from_date=date(2023, 1, 1),
                to_date=date(2020, 1, 1),  # Before from_date
                source_type="api",
            )

    def test_resample_from_validation(self) -> None:
        """Тест валидации resample_from."""
        with pytest.raises(ValueError, match="resample_from"):
            DatasetConfig(
                ticker="SBER",
                timeframe="1m",
                from_date=date(2020, 1, 1),
                to_date=date(2023, 12, 31),
                source_type="api",
                resample_from="5m",  # Cannot resample from larger timeframe
            )
