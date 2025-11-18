"""Тесты для загрузчика TinkoffDataLoader."""

from __future__ import annotations

import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import httpx
import pytest

from src.data.loaders.tinkoff import InstrumentInfo, TinkoffDataLoader

UTC = timezone.utc


class MockResponse:
    def __init__(self, status_code: int, content: bytes, text: str = "") -> None:
        self.status_code = status_code
        self.content = content
        self.text = text


class MockHttpClient:
    def __init__(self, responses: list[MockResponse]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(self, url: str, params: dict[str, str]) -> MockResponse:
        self.calls.append((url, params))
        if not self._responses:
            raise AssertionError("No mock response configured")
        return self._responses.pop(0)

    def close(self) -> None:  # pragma: no cover - nothing to close
        pass


def _create_archive(rows: list[str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        payload = "".join(rows).encode("utf-8")
        zf.writestr("2020-01-01.csv", payload)
    return buffer.getvalue()


def _instrument_info(ticker: str) -> InstrumentInfo:
    return InstrumentInfo(
        ticker=ticker,
        figi="TESTFIGI",
        instrument_id="TESTFIGI",
        first_1min_candle=datetime(2020, 1, 1, tzinfo=UTC),
    )


@pytest.mark.parametrize(
    "rows",
    [
        [
            "FIGI;2020-01-01T00:00:00Z;100;101;99;100.5;10;\n",
            "FIGI;2020-01-01T00:01:00Z;101;102;100;101.5;12;\n",
        ]
    ],
)
def test_tinkoff_loader_download_and_parse(tmp_path: Path, rows: list[str]) -> None:
    archive_bytes = _create_archive(rows)
    response = MockResponse(200, archive_bytes)
    http_client = MockHttpClient([response])

    loader = TinkoffDataLoader(
        token="TEST",
        downloads_dir=tmp_path / "downloads",
        extracted_dir=tmp_path / "extracted",
        instrument_resolver=_instrument_info,
    )

    # Подменяем http клиент
    loader._http_client = cast(httpx.Client, http_client)

    from_dt = datetime(2020, 1, 1, tzinfo=UTC)
    to_dt = datetime(2020, 1, 1, 23, 59, 0, tzinfo=UTC)

    df = loader.load("SBER", from_dt, to_dt, timeframe="1m")

    assert not df.empty
    assert list(df.columns) == [
        "timestamp",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert df["ticker"].unique().tolist() == ["SBER"]
    assert len(http_client.calls) == 1
    assert (tmp_path / "downloads" / "SBER" / "2020.zip").exists()
    assert (tmp_path / "extracted" / "SBER" / "2020").exists()


def test_tinkoff_loader_uses_cache(tmp_path: Path) -> None:
    rows = ["FIGI;2020-01-01T00:00:00Z;100;101;99;100.5;10;\n"]
    archive_bytes = _create_archive(rows)
    response = MockResponse(200, archive_bytes)
    http_client = MockHttpClient([response])

    loader = TinkoffDataLoader(
        token="TEST",
        downloads_dir=tmp_path / "downloads",
        extracted_dir=tmp_path / "extracted",
        instrument_resolver=_instrument_info,
        cache_ttl_hours=999,
    )
    loader._http_client = cast(httpx.Client, http_client)

    from_dt = datetime(2020, 1, 1, tzinfo=UTC)
    to_dt = datetime(2020, 1, 2, tzinfo=UTC)

    df1 = loader.load("SBER", from_dt, to_dt)
    assert len(df1) == 1
    # Второй вызов должен использовать кэш и не обращаться к API
    df2 = loader.load("SBER", from_dt, to_dt)
    assert len(df2) == 1
    assert len(http_client.calls) == 1
