"""Тесты для модуля merger."""

from numbers import Integral, Real

import pandas as pd
import pytest

from src.common.exceptions import PreprocessingError
from src.data.preprocessors.merger import DataMerger


def _ensure_float(value: object) -> float:
    assert isinstance(value, Real)
    return float(value)


def _ensure_int(value: object) -> int:
    assert isinstance(value, Integral)
    return int(value)


class TestDataMerger:
    """Тесты для DataMerger."""

    @pytest.fixture
    def sample_data_1(self) -> pd.DataFrame:
        """Первый источник данных."""
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 10:00:00",
                        "2024-01-01 10:01:00",
                        "2024-01-01 10:02:00",
                    ],
                    utc=True,
                ),
                "ticker": ["TEST"] * 3,
                "open": [100.0, 101.0, 102.0],
                "high": [100.5, 101.5, 102.5],
                "low": [99.5, 100.5, 101.5],
                "close": [100.2, 101.2, 102.2],
                "volume": [1000, 1100, 1200],
                "source": ["local"] * 3,
            }
        )

    @pytest.fixture
    def sample_data_2(self) -> pd.DataFrame:
        """Второй источник данных (частично перекрывается)."""
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 10:01:00",  # Дубликат
                        "2024-01-01 10:02:00",  # Дубликат
                        "2024-01-01 10:03:00",  # Новый
                    ],
                    utc=True,
                ),
                "ticker": ["TEST"] * 3,
                "open": [101.5, 102.5, 103.0],
                "high": [102.0, 103.0, 103.5],
                "low": [101.0, 102.0, 102.5],
                "close": [101.8, 102.8, 103.2],
                "volume": [1150, 1250, 1300],
                "source": ["tinkoff"] * 3,
            }
        )

    def test_merge_single_source(self, sample_data_1: pd.DataFrame) -> None:
        """Тест объединения одного источника."""
        merger = DataMerger()
        result = merger.merge_sources([sample_data_1])

        assert len(result) == len(sample_data_1)
        pd.testing.assert_frame_equal(result, sample_data_1)

    def test_merge_latest_strategy(
        self, sample_data_1: pd.DataFrame, sample_data_2: pd.DataFrame
    ) -> None:
        """Тест стратегии 'latest'."""
        merger = DataMerger()
        result = merger.merge_sources([sample_data_1, sample_data_2], strategy="latest")

        # Должно быть 4 уникальных timestamp
        assert len(result) == 4
        assert result["timestamp"].is_monotonic_increasing

        # Последние значения для дубликатов должны быть из sample_data_2
        row_10_01 = result[
            result["timestamp"] == pd.Timestamp("2024-01-01 10:01:00", tz="UTC")
        ]
        close_value = _ensure_float(row_10_01.iloc[0]["close"])
        assert close_value == pytest.approx(101.8)

    def test_merge_priority_strategy(
        self, sample_data_1: pd.DataFrame, sample_data_2: pd.DataFrame
    ) -> None:
        """Тест стратегии 'priority'."""
        merger = DataMerger()

        # Приоритет: tinkoff > local
        result = merger.merge_sources(
            [sample_data_1, sample_data_2],
            strategy="priority",
            priority_order=["tinkoff", "local"],
        )

        assert len(result) == 4

        # Для дубликатов должны быть значения из tinkoff (высокий приоритет)
        row_10_01 = result[
            result["timestamp"] == pd.Timestamp("2024-01-01 10:01:00", tz="UTC")
        ]
        close_value = _ensure_float(row_10_01.iloc[0]["close"])
        assert close_value == pytest.approx(101.8)
        assert row_10_01.iloc[0]["source"] == "tinkoff"

    def test_merge_average_strategy(
        self, sample_data_1: pd.DataFrame, sample_data_2: pd.DataFrame
    ) -> None:
        """Тест стратегии 'average'."""
        merger = DataMerger()
        result = merger.merge_sources(
            [sample_data_1, sample_data_2], strategy="average"
        )

        assert len(result) == 4

        # Для дубликатов значения должны быть усреднены
        row_10_01 = result[
            result["timestamp"] == pd.Timestamp("2024-01-01 10:01:00", tz="UTC")
        ]
        expected_close = (101.2 + 101.8) / 2
        close_value = _ensure_float(row_10_01.iloc[0]["close"])
        assert close_value == pytest.approx(expected_close, abs=0.01)

        # Volume должен быть суммой
        expected_volume = 1100 + 1150
        volume_value = _ensure_int(row_10_01.iloc[0]["volume"])
        assert volume_value == expected_volume

    def test_fill_gaps_forward_fill(self, sample_data_1: pd.DataFrame) -> None:
        """Тест заполнения пропусков forward fill."""
        # Создать пропуски
        data = sample_data_1.copy()
        data.loc[1, "close"] = None
        data.loc[1, "volume"] = None

        merger = DataMerger()
        result = merger.fill_gaps(data, method="forward_fill")

        # Пропуски должны быть заполнены предыдущими значениями
        result_close = _ensure_float(result.loc[1, "close"])
        original_close = _ensure_float(data.loc[0, "close"])
        assert result_close == pytest.approx(original_close)
        result_volume = _ensure_float(result.loc[1, "volume"])
        original_volume = _ensure_float(data.loc[0, "volume"])
        assert result_volume == pytest.approx(original_volume)

    def test_fill_gaps_interpolate(self, sample_data_1: pd.DataFrame) -> None:
        """Тест заполнения пропусков интерполяцией."""
        data = sample_data_1.copy()
        data.loc[1, "close"] = None

        merger = DataMerger()
        result = merger.fill_gaps(data, method="interpolate")

        # Значение должно быть интерполировано
        first_close = _ensure_float(data.loc[0, "close"])
        third_close = _ensure_float(data.loc[2, "close"])
        expected = (first_close + third_close) / 2
        interpolated_close = _ensure_float(result.loc[1, "close"])
        assert interpolated_close == pytest.approx(expected, abs=0.01)

    def test_fill_gaps_drop(self, sample_data_1: pd.DataFrame) -> None:
        """Тест удаления строк с пропусками."""
        data = sample_data_1.copy()
        data.loc[1, "close"] = None

        merger = DataMerger()
        result = merger.fill_gaps(data, method="drop")

        assert len(result) == 2  # Одна строка удалена
        assert not result.isna().any().any()

    def test_validate_after_merge(self, sample_data_1: pd.DataFrame) -> None:
        """Тест валидации после объединения."""
        merger = DataMerger()

        # Валидные данные не должны вызывать исключение
        merger.validate_after_merge(sample_data_1, "1m")

        # Невалидные данные должны вызвать исключение
        invalid_data = sample_data_1.copy()
        invalid_data.loc[0, "high"] = 50.0  # high < open
        invalid_data.loc[0, "low"] = 150.0  # low > close

        with pytest.raises(PreprocessingError):
            merger.validate_after_merge(invalid_data, "1m")

    def test_merge_empty_sources(self) -> None:
        """Тест объединения пустого списка источников."""
        merger = DataMerger()

        with pytest.raises(PreprocessingError, match="No sources provided"):
            merger.merge_sources([])

    def test_merge_priority_without_priority_order(
        self, sample_data_1: pd.DataFrame
    ) -> None:
        """Тест стратегии priority без указания порядка приоритета."""
        merger = DataMerger()

        with pytest.raises(
            PreprocessingError, match="priority_order must be specified"
        ):
            merger.merge_sources([sample_data_1], strategy="priority")

    def test_resolve_conflicts(
        self, sample_data_1: pd.DataFrame, sample_data_2: pd.DataFrame
    ) -> None:
        """Тест разрешения конфликтов."""
        merger = DataMerger()

        combined = pd.concat([sample_data_1, sample_data_2], ignore_index=True)
        result = merger.resolve_conflicts(combined, priority_order=["local", "tinkoff"])

        assert len(result) == 4
        # local имеет высокий приоритет, должны остаться его значения для дубликатов
        row_10_01 = result[
            result["timestamp"] == pd.Timestamp("2024-01-01 10:01:00", tz="UTC")
        ]
        assert row_10_01.iloc[0]["source"] == "local"
