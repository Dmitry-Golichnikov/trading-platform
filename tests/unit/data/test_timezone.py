"""Тесты для модуля timezone."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.common.exceptions import PreprocessingError
from src.data.preprocessors.timezone import (
    convert_to_utc,
    handle_dst_transition,
    localize_timestamp,
    validate_timezone_aware,
)


class TestTimezoneUtils:
    """Тесты для утилит работы с таймзонами."""

    def test_convert_to_utc_naive(self) -> None:
        """Тест конвертации naive timestamps в UTC."""
        data = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01 10:00:00", "2024-01-01 11:00:00"]),
                "value": [1, 2],
            }
        )

        result = convert_to_utc(data)

        assert validate_timezone_aware(result)
        assert str(result["timestamp"].dt.tz) == "UTC"

    def test_convert_to_utc_with_tz(self) -> None:
        """Тест конвертации timezone-aware timestamps в UTC."""
        data = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01 10:00:00", "2024-01-01 11:00:00"]).tz_localize(
                    "Europe/Moscow"
                ),
                "value": [1, 2],
            }
        )

        result = convert_to_utc(data)

        assert str(result["timestamp"].dt.tz) == "UTC"
        # Московское время UTC+3, должно быть конвертировано
        assert result["timestamp"].iloc[0].hour == 7  # 10 - 3 = 7

    def test_convert_to_utc_missing_column(self) -> None:
        """Тест конвертации без колонки timestamp."""
        data = pd.DataFrame({"value": [1, 2]})

        with pytest.raises(PreprocessingError, match="must have 'timestamp' column"):
            convert_to_utc(data)

    def test_validate_timezone_aware_true(self) -> None:
        """Тест валидации timezone-aware timestamps."""
        data = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01 10:00:00"], utc=True),
            }
        )

        assert validate_timezone_aware(data) is True

    def test_validate_timezone_aware_false(self) -> None:
        """Тест валидации naive timestamps."""
        data = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01 10:00:00"]),
            }
        )

        assert validate_timezone_aware(data) is False

    def test_validate_timezone_aware_missing_column(self) -> None:
        """Тест валидации без колонки timestamp."""
        data = pd.DataFrame({"value": [1, 2]})

        assert validate_timezone_aware(data) is False

    def test_localize_timestamp_naive(self) -> None:
        """Тест локализации naive timestamp."""
        ts = datetime(2024, 1, 1, 10, 0, 0)
        result = localize_timestamp(ts, "Europe/Moscow")

        assert result.tzinfo is not None
        assert result.hour == 10

    def test_localize_timestamp_aware(self) -> None:
        """Тест конвертации timezone-aware timestamp."""
        ts = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        result = localize_timestamp(ts, "Europe/Moscow")

        assert result.tzinfo is not None
        # UTC 10:00 -> Moscow 13:00 (UTC+3)
        assert result.hour == 13

    def test_handle_dst_transition_utc(self) -> None:
        """Тест обработки DST для UTC данных (не требуется)."""
        data = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2024-03-31 01:00:00",
                        "2024-03-31 02:00:00",
                        "2024-03-31 03:00:00",
                    ],
                    utc=True,
                ),
                "ticker": ["TEST"] * 3,
                "open": [100.0, 101.0, 102.0],
                "high": [100.5, 101.5, 102.5],
                "low": [99.5, 100.5, 101.5],
                "close": [100.2, 101.2, 102.2],
                "volume": [1000, 1100, 1200],
            }
        )

        result = handle_dst_transition(data)

        # Для UTC данных изменений не должно быть
        assert len(result) == len(data)
        pd.testing.assert_frame_equal(result, data)

    def test_handle_dst_transition_duplicates(self) -> None:
        """Тест обработки дублирующихся timestamp (fall back)."""
        # Имитируем переход на зимнее время: создаём дубликаты напрямую в UTC
        # чтобы избежать AmbiguousTimeError
        ts1 = pd.Timestamp("2024-10-27 00:00:00", tz="UTC")
        ts2 = pd.Timestamp("2024-10-27 01:00:00", tz="UTC")
        ts3 = pd.Timestamp("2024-10-27 01:00:00", tz="UTC")  # Дубликат
        ts4 = pd.Timestamp("2024-10-27 02:00:00", tz="UTC")

        data = pd.DataFrame(
            {
                "timestamp": [ts1, ts2, ts3, ts4],
                "ticker": ["TEST"] * 4,
                "open": [100.0, 101.0, 101.5, 102.0],
                "high": [100.5, 101.5, 102.0, 102.5],
                "low": [99.5, 100.5, 101.0, 101.5],
                "close": [100.2, 101.2, 101.7, 102.2],
                "volume": [1000, 1100, 1150, 1200],
            }
        )

        # Конвертируем в Europe/Berlin для имитации DST
        data["timestamp"] = data["timestamp"].dt.tz_convert("Europe/Berlin")

        result = handle_dst_transition(data)

        # Дубликаты должны быть объединены
        assert len(result) == 3

        # Проверить усреднённые значения для дубликата
        dup_timestamp = ts2.tz_convert("Europe/Berlin")
        dup_row = result[result["timestamp"] == dup_timestamp]
        assert len(dup_row) == 1

        # close должен быть усреднён
        expected_close = (101.2 + 101.7) / 2
        assert abs(dup_row.iloc[0]["close"] - expected_close) < 0.01

        # volume должен быть суммой
        expected_volume = 1100 + 1150
        assert dup_row.iloc[0]["volume"] == expected_volume

    def test_handle_dst_transition_naive_timestamps(self) -> None:
        """Тест обработки DST с naive timestamps (должна быть ошибка)."""
        data = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-03-31 02:00:00"]),
                "ticker": ["TEST"],
                "open": [100.0],
                "high": [100.5],
                "low": [99.5],
                "close": [100.2],
                "volume": [1000],
            }
        )

        with pytest.raises(PreprocessingError, match="Timestamps must be timezone-aware"):
            handle_dst_transition(data)

    def test_handle_dst_transition_empty_data(self) -> None:
        """Тест обработки DST для пустого DataFrame."""
        data = pd.DataFrame(
            {
                "timestamp": pd.to_datetime([]),
                "ticker": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            }
        )

        result = handle_dst_transition(data)
        assert len(result) == 0

    def test_handle_dst_transition_missing_timestamp(self) -> None:
        """Тест обработки DST без колонки timestamp."""
        data = pd.DataFrame({"value": [1, 2, 3]})

        with pytest.raises(PreprocessingError, match="must have 'timestamp' column"):
            handle_dst_transition(data)
