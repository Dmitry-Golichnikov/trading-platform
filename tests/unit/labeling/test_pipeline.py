"""Тесты для LabelingPipeline."""

import numpy as np
import pandas as pd
import pytest

from src.labeling.filters.sequence_filter import SequenceFilter
from src.labeling.filters.smoothing import SmoothingFilter
from src.labeling.methods.horizon import HorizonLabeler
from src.labeling.methods.triple_barrier import TripleBarrierLabeler
from src.labeling.pipeline import LabelingPipeline


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


def test_pipeline_initialization():
    """Тест инициализации пайплайна."""
    labeler = HorizonLabeler(horizon=20)
    filters = [SmoothingFilter(method="median", window=3)]

    pipeline = LabelingPipeline(labeler=labeler, filters=filters, dataset_id="test_dataset")

    assert pipeline.labeler == labeler
    assert len(pipeline.filters) == 1
    assert pipeline.dataset_id == "test_dataset"
    assert pipeline.labeling_id is not None


def test_pipeline_run_basic(sample_data, tmp_path):
    """Тест базового запуска пайплайна."""
    labeler = HorizonLabeler(horizon=10, direction="long+short", threshold_pct=0.01)

    pipeline = LabelingPipeline(labeler=labeler, filters=[], dataset_id="test", output_dir=tmp_path)

    labeled_data, metadata = pipeline.run(sample_data, save_results=False)

    # Проверяем результат
    assert "label" in labeled_data.columns
    assert len(labeled_data) == len(sample_data)

    # Проверяем метаданные
    assert metadata.labeling_id == pipeline.labeling_id
    assert metadata.method == "HorizonLabeler"
    assert metadata.total_samples == len(sample_data)


def test_pipeline_with_filters(sample_data, tmp_path):
    """Тест пайплайна с фильтрами."""
    labeler = TripleBarrierLabeler(upper_barrier=0.02, lower_barrier=0.02, time_barrier=20)

    filters = [SmoothingFilter(method="median", window=3), SequenceFilter(min_length=2)]

    pipeline = LabelingPipeline(labeler=labeler, filters=filters, dataset_id="test", output_dir=tmp_path)

    labeled_data, metadata = pipeline.run(sample_data, save_results=False)

    # Проверяем что фильтры применены
    assert len(metadata.filters_applied) == 2
    assert "SmoothingFilter" in metadata.filters_applied
    assert "SequenceFilter" in metadata.filters_applied


def test_pipeline_save_results(sample_data, tmp_path):
    """Тест сохранения результатов."""
    labeler = HorizonLabeler(horizon=10)

    pipeline = LabelingPipeline(labeler=labeler, filters=[], dataset_id="test", output_dir=tmp_path)

    labeled_data, metadata = pipeline.run(sample_data, save_results=True)

    # Проверяем что файлы созданы
    output_path = tmp_path / pipeline.labeling_id
    assert output_path.exists()

    data_path = output_path / "labels.parquet"
    metadata_path = output_path / "metadata.json"

    assert data_path.exists()
    assert metadata_path.exists()


def test_pipeline_from_config(sample_data):
    """Тест создания пайплайна из конфигурации."""
    config = {
        "method": "horizon",
        "params": {"horizon": 20, "direction": "long+short", "threshold_pct": 0.01},
        "filters": [
            {"type": "smoothing", "params": {"method": "median", "window": 3}},
            {"type": "sequence", "params": {"min_length": 2}},
        ],
        "dataset_id": "test",
        "output_dir": "artifacts/labels",
    }

    pipeline = LabelingPipeline.from_config(config, sample_data)

    assert isinstance(pipeline.labeler, HorizonLabeler)
    assert len(pipeline.filters) == 2
    assert pipeline.dataset_id == "test"


def test_pipeline_from_config_triple_barrier(sample_data):
    """Тест создания Triple Barrier пайплайна из конфигурации."""
    config = {
        "method": "triple_barrier",
        "params": {
            "upper_barrier": 0.02,
            "lower_barrier": 0.02,
            "time_barrier": 20,
            "direction": "long+short",
        },
        "filters": [],
        "dataset_id": "test",
    }

    pipeline = LabelingPipeline.from_config(config, sample_data)

    assert isinstance(pipeline.labeler, TripleBarrierLabeler)


def test_pipeline_validation_error_no_label():
    """Тест ошибки валидации - отсутствие колонки label."""

    # Создаём фейковый labeler который не добавляет колонку label
    class BadLabeler:
        def __init__(self):
            pass

        def label(self, data):
            return data

        def get_params(self):
            return {}

        def __class__(self):
            return type("BadLabeler", (), {})

    dates = pd.date_range("2023-01-01", periods=10, freq="1H")
    data = pd.DataFrame(
        {
            "open": [100] * 10,
            "high": [101] * 10,
            "low": [99] * 10,
            "close": [100] * 10,
        },
        index=dates,
    )

    pipeline = LabelingPipeline(labeler=BadLabeler(), filters=[], dataset_id="test")

    with pytest.raises(ValueError):
        pipeline.run(data, save_results=False)


def test_pipeline_metadata_statistics(sample_data, tmp_path):
    """Тест статистики в метаданных."""
    labeler = TripleBarrierLabeler(upper_barrier=0.02, lower_barrier=0.02, time_barrier=20)

    pipeline = LabelingPipeline(labeler=labeler, filters=[], dataset_id="test", output_dir=tmp_path)

    labeled_data, metadata = pipeline.run(sample_data, save_results=False)

    # Проверяем наличие статистики
    assert "initial_distribution" in metadata.statistics

    # Для Triple Barrier должна быть статистика по holding_period
    if "holding_period" in labeled_data.columns:
        assert "avg_holding_period" in metadata.statistics
