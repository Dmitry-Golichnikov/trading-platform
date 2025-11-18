"""
Pydantic схемы для модуля данных.

Модуль содержит схемы для валидации и сериализации данных OHLCV,
метаданных датасетов и конфигураций загрузки.
"""

from datetime import date, datetime
from pathlib import Path
from typing import Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class OHLCVBar(BaseModel):
    """
    Схема для одного OHLCV бара.

    Attributes:
        timestamp: Временная метка с timezone
        ticker: Тикер инструмента
        open: Цена открытия
        high: Максимальная цена
        low: Минимальная цена
        close: Цена закрытия
        volume: Объем торгов
    """

    timestamp: datetime = Field(..., description="Временная метка с timezone")
    ticker: str = Field(..., description="Тикер инструмента")
    open: float = Field(..., gt=0, description="Цена открытия")
    high: float = Field(..., gt=0, description="Максимальная цена")
    low: float = Field(..., gt=0, description="Минимальная цена")
    close: float = Field(..., gt=0, description="Цена закрытия")
    volume: int = Field(..., ge=0, description="Объем торгов (неотрицательный)")

    @field_validator("timestamp")
    @classmethod
    def validate_timezone_aware(cls, v: datetime) -> datetime:
        """Проверить что timestamp имеет информацию о таймзоне."""
        if v.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return v

    @model_validator(mode="after")
    def validate_ohlc_relationships(self) -> "OHLCVBar":
        """Проверить корректность соотношений OHLC."""
        if self.high < max(self.open, self.close):
            raise ValueError(f"high ({self.high}) must be >= max(open, close) " f"({max(self.open, self.close)})")
        if self.low > min(self.open, self.close):
            raise ValueError(f"low ({self.low}) must be <= min(open, close) " f"({min(self.open, self.close)})")
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2020-01-03T07:04:00Z",
                "ticker": "SBER",
                "open": 13.37,
                "high": 13.40,
                "low": 13.35,
                "close": 13.38,
                "volume": 100,
            }
        }


class DatasetMetadata(BaseModel):
    """
    Метаданные датасета.

    Attributes:
        dataset_id: Уникальный идентификатор датасета
        ticker: Тикер инструмента
        timeframe: Временной интервал ('1m', '5m', '15m', '1h', '4h', '1d')
        start_date: Дата начала данных
        end_date: Дата окончания данных
        source: Источник данных
        timezone: Таймзона данных
        total_bars: Общее количество баров
        missing_bars: Количество пропущенных баров
        hash: SHA256 хэш данных для версионирования
        created_at: Дата создания датасета
        schema_version: Версия схемы данных
    """

    dataset_id: UUID = Field(default_factory=uuid4, description="UUID датасета")
    ticker: str = Field(..., description="Тикер инструмента")
    timeframe: Literal["1m", "5m", "15m", "1h", "4h", "1d"] = Field(..., description="Временной интервал")
    start_date: datetime = Field(..., description="Дата начала данных")
    end_date: datetime = Field(..., description="Дата окончания данных")
    source: Literal["local", "tinkoff", "manual"] = Field(..., description="Источник данных")
    timezone: str = Field(default="UTC", description="Таймзона данных")
    total_bars: int = Field(..., ge=0, description="Общее количество баров")
    missing_bars: int = Field(default=0, ge=0, description="Количество пропусков")
    hash: str = Field(..., description="SHA256 хэш данных")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Дата создания")
    schema_version: str = Field(default="1.0", description="Версия схемы")

    @model_validator(mode="after")
    def validate_dates(self) -> "DatasetMetadata":
        """Проверить что end_date >= start_date."""
        if self.end_date < self.start_date:
            raise ValueError("end_date must be >= start_date")
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "SBER",
                "timeframe": "1m",
                "start_date": "2020-01-01T00:00:00Z",
                "end_date": "2020-12-31T23:59:00Z",
                "source": "tinkoff",
                "total_bars": 100000,
                "missing_bars": 42,
                "hash": "abc123...",
            }
        }


class DatasetConfig(BaseModel):
    """
    Конфигурация загрузки данных.

    Attributes:
        ticker: Тикер или список тикеров
        tickers_file: Путь к файлу со списком тикеров (опционально)
        timeframe: Целевой временной интервал
        from_date: Дата начала загрузки
        to_date: Дата окончания загрузки
        source_type: Тип источника данных
        file_path: Путь к локальному файлу (для source_type='local')
        resample_from: Загрузить из другого таймфрейма и ресэмплировать
        timezone: Таймзона данных
        validate_data: Выполнять валидацию данных
        update_latest_year: Обновлять данные текущего года
        backfill_missing: Автоматически докачивать недостающие архивы
    """

    ticker: Optional[str | list[str]] = Field(default=None, description="Тикер или список тикеров")
    tickers_file: Optional[Path] = Field(default=None, description="Путь к файлу со списком тикеров")
    timeframe: Literal["1m", "5m", "15m", "1h", "4h", "1d"] = Field(..., description="Целевой временной интервал")
    from_date: date = Field(..., description="Дата начала загрузки")
    to_date: date = Field(..., description="Дата окончания загрузки")
    source_type: Literal["local", "api"] = Field(..., description="Тип источника данных")
    file_path: Optional[Path] = Field(default=None, description="Путь к локальному файлу (для source_type='local')")
    resample_from: Optional[Literal["1m", "5m", "15m", "1h", "4h"]] = Field(
        default=None, description="Загрузить из меньшего таймфрейма и ресэмплировать"
    )
    timezone: str = Field(default="UTC", description="Таймзона данных")
    validate_data: bool = Field(default=True, description="Выполнять валидацию данных")
    update_latest_year: bool = Field(default=True, description="Обновлять данные текущего года")
    backfill_missing: bool = Field(default=True, description="Автоматически докачивать недостающие архивы")
    api_token: Optional[str] = Field(
        default=None,
        description=(
            "Токен доступа Tinkoff Invest API (если не указан, используется "
            "переменная окружения TINKOFF_INVEST_TOKEN)"
        ),
    )

    @model_validator(mode="after")
    def validate_config(self) -> "DatasetConfig":
        """Валидация конфигурации."""
        # Должен быть указан либо ticker, либо tickers_file
        if self.ticker is None and self.tickers_file is None:
            raise ValueError("Either 'ticker' or 'tickers_file' must be specified")

        # Для локального источника нужен file_path
        if self.source_type == "local" and self.file_path is None:
            raise ValueError("'file_path' is required when source_type='local'")

        # Проверка дат
        if self.to_date < self.from_date:
            raise ValueError("to_date must be >= from_date")

        # Проверка resample_from
        if self.resample_from:
            timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
            from_idx = timeframes.index(self.resample_from)
            to_idx = timeframes.index(self.timeframe)
            if from_idx >= to_idx:
                raise ValueError(
                    f"resample_from ({self.resample_from}) must be smaller " f"than timeframe ({self.timeframe})"
                )

        if self.source_type == "api":
            import os

            env_token = os.getenv("TINKOFF_API_TOKEN") or os.getenv("TINKOFF_INVEST_TOKEN")
            if not self.api_token and not env_token:
                raise ValueError(
                    "API token is required for source_type='api'. "
                    "Укажите DatasetConfig.api_token или переменную окружения "
                    "TINKOFF_API_TOKEN/TINKOFF_INVEST_TOKEN."
                )

        return self

    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "SBER",
                "timeframe": "5m",
                "from_date": "2020-01-01",
                "to_date": "2023-12-31",
                "source_type": "api",
                "resample_from": "1m",
                "validate_data": True,
                "update_latest_year": True,
                "backfill_missing": True,
                "api_token": "tinkoff-token",
            }
        }
