"""Тесты для протоколов из src.common.interfaces."""

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.common.interfaces import DataLoader, FeatureCalculator, Model


class DummyDataLoader:
    """Простейший загрузчик данных."""

    def load(
        self,
        source: str,
        start_date: str | None = None,
        end_date: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "source": [source],
                "start": [start_date],
                "end": [end_date],
                "kwargs": [kwargs],
            }
        )


class DummyModel:
    """Минимальная реализация модели."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._trained = True

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series([0] * len(X), index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                0: [0.5] * len(X),
                1: [0.5] * len(X),
            },
            index=X.index,
        )

    def save(self, path: Path) -> None:
        path.write_text("model", encoding="utf-8")

    def load(self, path: Path) -> None:
        path.read_text(encoding="utf-8")


class DummyFeatureCalculator:
    """Простейший расчётчик признаков."""

    def calculate(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        result = df.copy()
        for key, value in config.items():
            result[key] = value
        return result


def test_dataloader_protocol_runtime_check() -> None:
    loader: DataLoader = DummyDataLoader()

    df = loader.load(
        "ticker",
        start_date="2024-01-01",
        end_date="2024-01-02",
        limit=10,
    )
    assert list(df.columns) == [
        "source",
        "start",
        "end",
        "kwargs",
    ]
    assert df.loc[0, "source"] == "ticker"
    assert df.loc[0, "kwargs"] == {
        "limit": 10,
    }


def test_model_protocol_runtime_check(tmp_path: Path) -> None:
    model: Model = DummyModel()

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])

    model.fit(X, y)
    preds = model.predict(X)
    proba = model.predict_proba(X)

    assert len(preds) == len(X)
    assert (proba.sum(axis=1) - 1).abs().max() < 1e-9

    path = tmp_path / "model.txt"
    model.save(path)
    assert path.exists()
    model.load(path)


def test_feature_calculator_protocol_runtime_check() -> None:
    calculator: FeatureCalculator = DummyFeatureCalculator()

    df = pd.DataFrame(
        {
            "x": [1, 2],
        }
    )
    config = {"feature": 42}

    result = calculator.calculate(df, config)
    assert "feature" in result.columns
    assert (result["feature"] == 42).all()
