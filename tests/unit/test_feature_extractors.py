"""Unit тесты для feature extractors."""

import numpy as np
import pandas as pd
import pytest

from src.features.extractors.calendar import CalendarExtractor
from src.features.extractors.price import PriceExtractor
from src.features.extractors.ticker import TickerExtractor
from src.features.extractors.volume import VolumeExtractor


@pytest.fixture
def sample_ohlc_data():
    """Создать тестовые OHLC данные."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1H")
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "open": 100 + np.cumsum(np.random.randn(100) * 0.1),
            "high": 100 + np.cumsum(np.random.randn(100) * 0.1) + 0.5,
            "low": 100 + np.cumsum(np.random.randn(100) * 0.1) - 0.5,
            "close": 100 + np.cumsum(np.random.randn(100) * 0.1),
            "volume": np.random.randint(1000, 10000, 100),
        },
        index=dates,
    )

    # Обеспечиваем high >= open, close и low <= open, close
    data["high"] = data[["open", "high", "close"]].max(axis=1)
    data["low"] = data[["open", "low", "close"]].min(axis=1)

    return data


class TestPriceExtractor:
    """Тесты для PriceExtractor."""

    def test_extract_returns(self, sample_ohlc_data):
        """Тест извлечения простых доходностей."""
        extractor = PriceExtractor(["returns"])
        features = extractor.extract(sample_ohlc_data)

        assert "returns" in features.columns
        assert len(features) == len(sample_ohlc_data)
        # Первое значение должно быть NaN
        assert pd.isna(features["returns"].iloc[0])
        # Остальные должны быть числами
        assert features["returns"].iloc[1:].notna().all()

    def test_extract_log_returns(self, sample_ohlc_data):
        """Тест извлечения логарифмических доходностей."""
        extractor = PriceExtractor(["log_returns"])
        features = extractor.extract(sample_ohlc_data)

        assert "log_returns" in features.columns
        # Лог-доходности должны быть близки к простым для малых изменений
        simple_returns = sample_ohlc_data["close"].pct_change()
        np.testing.assert_array_almost_equal(
            features["log_returns"].iloc[1:10],
            simple_returns.iloc[1:10],
            decimal=2,
        )

    def test_extract_high_low_ratio(self, sample_ohlc_data):
        """Тест извлечения high/low ratio."""
        extractor = PriceExtractor(["high_low_ratio"])
        features = extractor.extract(sample_ohlc_data)

        assert "high_low_ratio" in features.columns
        # Все значения должны быть >= 1
        assert (features["high_low_ratio"] >= 1).all()

    def test_extract_body_size(self, sample_ohlc_data):
        """Тест извлечения размера тела свечи."""
        extractor = PriceExtractor(["body_size"])
        features = extractor.extract(sample_ohlc_data)

        assert "body_size" in features.columns
        # Все значения должны быть между 0 и 1
        assert (features["body_size"] >= 0).all()
        assert (features["body_size"] <= 1).all()

    def test_extract_price_position(self, sample_ohlc_data):
        """Тест извлечения позиции цены."""
        extractor = PriceExtractor(["price_position"])
        features = extractor.extract(sample_ohlc_data)

        assert "price_position" in features.columns
        # Все значения должны быть между 0 и 1
        assert (features["price_position"] >= 0).all()
        assert (features["price_position"] <= 1).all()

    def test_invalid_feature(self):
        """Тест с невалидным признаком."""
        with pytest.raises(ValueError):
            PriceExtractor(["invalid_feature"])

    def test_multiple_features(self, sample_ohlc_data):
        """Тест извлечения нескольких признаков."""
        extractor = PriceExtractor(["returns", "high_low_ratio", "body_size"])
        features = extractor.extract(sample_ohlc_data)

        assert len(features.columns) == 3
        assert all(col in features.columns for col in ["returns", "high_low_ratio", "body_size"])


class TestVolumeExtractor:
    """Тесты для VolumeExtractor."""

    def test_extract_volume_change(self, sample_ohlc_data):
        """Тест извлечения изменения объёма."""
        extractor = VolumeExtractor(["volume_change"])
        features = extractor.extract(sample_ohlc_data)

        assert "volume_change" in features.columns
        assert pd.isna(features["volume_change"].iloc[0])

    def test_extract_volume_ma_ratio(self, sample_ohlc_data):
        """Тест извлечения отношения объёма к MA."""
        extractor = VolumeExtractor(["volume_ma_ratio"])
        features = extractor.extract(sample_ohlc_data)

        assert "volume_ma_ratio" in features.columns
        # Все значения должны быть положительными
        assert (features["volume_ma_ratio"] > 0).all()

    def test_extract_money_volume(self, sample_ohlc_data):
        """Тест извлечения денежного объёма."""
        extractor = VolumeExtractor(["money_volume"])
        features = extractor.extract(sample_ohlc_data)

        assert "money_volume" in features.columns
        assert (features["money_volume"] > 0).all()

    def test_invalid_feature(self):
        """Тест с невалидным признаком."""
        with pytest.raises(ValueError):
            VolumeExtractor(["invalid_feature"])


class TestCalendarExtractor:
    """Тесты для CalendarExtractor."""

    def test_extract_hour(self, sample_ohlc_data):
        """Тест извлечения часа."""
        extractor = CalendarExtractor(["hour"])
        features = extractor.extract(sample_ohlc_data)

        assert "hour" in features.columns
        # Часы должны быть между 0 и 23
        assert (features["hour"] >= 0).all()
        assert (features["hour"] <= 23).all()

    def test_extract_day_of_week(self, sample_ohlc_data):
        """Тест извлечения дня недели."""
        extractor = CalendarExtractor(["day_of_week"])
        features = extractor.extract(sample_ohlc_data)

        assert "day_of_week" in features.columns
        # Дни недели должны быть между 0 и 6
        assert (features["day_of_week"] >= 0).all()
        assert (features["day_of_week"] <= 6).all()

    def test_extract_month(self, sample_ohlc_data):
        """Тест извлечения месяца."""
        extractor = CalendarExtractor(["month"])
        features = extractor.extract(sample_ohlc_data)

        assert "month" in features.columns
        # Месяцы должны быть между 1 и 12
        assert (features["month"] >= 1).all()
        assert (features["month"] <= 12).all()

    def test_extract_is_month_start(self, sample_ohlc_data):
        """Тест извлечения флага начала месяца."""
        extractor = CalendarExtractor(["is_month_start"])
        features = extractor.extract(sample_ohlc_data)

        assert "is_month_start" in features.columns
        # Должны быть только 0 и 1
        assert features["is_month_start"].isin([0, 1]).all()

    def test_non_datetime_index(self):
        """Тест с неправильным типом индекса."""
        data = pd.DataFrame({"close": [100, 101, 102]})
        extractor = CalendarExtractor(["hour"])

        with pytest.raises(ValueError):
            extractor.extract(data)

    def test_invalid_feature(self):
        """Тест с невалидным признаком."""
        with pytest.raises(ValueError):
            CalendarExtractor(["invalid_feature"])


class TestTickerExtractor:
    """Тесты для TickerExtractor."""

    @pytest.fixture
    def sample_ticker_data(self, sample_ohlc_data):
        """Добавить колонку ticker к данным."""
        data = sample_ohlc_data.copy()
        data["ticker"] = np.random.choice(["SBER", "GAZP", "LKOH"], len(data))
        return data

    def test_label_encoding(self, sample_ticker_data):
        """Тест label encoding."""
        extractor = TickerExtractor(encoding="label")
        features = extractor.extract(sample_ticker_data)

        assert "ticker_encoded" in features.columns
        # Закодированные значения должны быть целыми числами
        assert features["ticker_encoded"].dtype in [np.int32, np.int64]

    def test_onehot_encoding(self, sample_ticker_data):
        """Тест one-hot encoding."""
        extractor = TickerExtractor(encoding="onehot")
        features = extractor.extract(sample_ticker_data)

        # Должны появиться колонки для каждого уникального тикера
        ticker_columns = [col for col in features.columns if col.startswith("ticker_")]
        assert len(ticker_columns) > 0

    def test_missing_ticker_column(self, sample_ohlc_data):
        """Тест с отсутствующей колонкой ticker."""
        extractor = TickerExtractor(encoding="label")

        with pytest.raises(ValueError):
            extractor.extract(sample_ohlc_data)
