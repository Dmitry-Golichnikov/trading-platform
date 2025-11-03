"""Парсинг и валидация конфигураций признаков."""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator


class IndicatorFeatureConfig(BaseModel):
    """Конфигурация признака на основе индикатора."""

    type: Literal["indicator"] = "indicator"
    name: str = Field(..., description="Название индикатора из реестра")
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Параметры индикатора"
    )
    columns: Optional[List[str]] = Field(None, description="Колонки для применения")
    prefix: Optional[str] = Field(None, description="Префикс для названий признаков")


class PriceFeatureConfig(BaseModel):
    """Конфигурация ценовых признаков."""

    type: Literal["price"] = "price"
    features: List[str] = Field(..., description="Список ценовых признаков")

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: List[str]) -> List[str]:
        """Валидация списка признаков."""
        allowed = {
            "returns",
            "log_returns",
            "high_low_ratio",
            "close_open_ratio",
            "body_size",
            "upper_wick",
            "lower_wick",
            "price_position",
        }
        invalid = set(v) - allowed
        if invalid:
            raise ValueError(f"Неизвестные ценовые признаки: {invalid}")
        return v


class VolumeFeatureConfig(BaseModel):
    """Конфигурация объёмных признаков."""

    type: Literal["volume"] = "volume"
    features: List[str] = Field(..., description="Список объёмных признаков")

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: List[str]) -> List[str]:
        """Валидация списка признаков."""
        allowed = {
            "volume_change",
            "volume_ma_ratio",
            "money_volume",
            "relative_volume",
            "volume_volatility",
        }
        invalid = set(v) - allowed
        if invalid:
            raise ValueError(f"Неизвестные объёмные признаки: {invalid}")
        return v


class CalendarFeatureConfig(BaseModel):
    """Конфигурация календарных признаков."""

    type: Literal["calendar"] = "calendar"
    features: List[str] = Field(..., description="Список календарных признаков")

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: List[str]) -> List[str]:
        """Валидация списка признаков."""
        allowed = {
            "hour",
            "day_of_week",
            "month",
            "is_month_start",
            "is_month_end",
            "trading_day_of_month",
            "time_since_open",
            "time_to_close",
        }
        invalid = set(v) - allowed
        if invalid:
            raise ValueError(f"Неизвестные календарные признаки: {invalid}")
        return v


class TickerFeatureConfig(BaseModel):
    """Конфигурация тикер-специфичных признаков."""

    type: Literal["ticker"] = "ticker"
    encoding: Literal["label", "onehot", "target"] = Field(
        "label", description="Тип кодирования тикера"
    )
    sector: bool = Field(False, description="Добавить сектор (если есть данные)")
    industry: bool = Field(False, description="Добавить индустрию (если есть данные)")


class RollingFeatureConfig(BaseModel):
    """Конфигурация rolling статистик."""

    type: Literal["rolling"] = "rolling"
    window: int = Field(..., gt=0, description="Размер окна")
    functions: List[str] = Field(..., description="Список функций")
    columns: List[str] = Field(..., description="Колонки для применения")
    min_periods: Optional[int] = Field(
        None, description="Минимальное количество наблюдений"
    )

    @field_validator("functions")
    @classmethod
    def validate_functions(cls, v: List[str]) -> List[str]:
        """Валидация списка функций."""
        allowed = {
            "mean",
            "std",
            "min",
            "max",
            "median",
            "sum",
            "skew",
            "kurt",
            "quantile",
        }
        invalid = set(v) - allowed
        if invalid:
            raise ValueError(f"Неизвестные функции: {invalid}")
        return v


class LagsFeatureConfig(BaseModel):
    """Конфигурация лагов."""

    type: Literal["lags"] = "lags"
    lags: List[int] = Field(..., description="Список лагов")
    columns: List[str] = Field(..., description="Колонки для применения")

    @field_validator("lags")
    @classmethod
    def validate_lags(cls, v: List[int]) -> List[int]:
        """Валидация лагов."""
        if any(lag <= 0 for lag in v):
            raise ValueError("Все лаги должны быть положительными")
        return v


class DifferencesFeatureConfig(BaseModel):
    """Конфигурация разностей."""

    type: Literal["differences"] = "differences"
    periods: List[int] = Field(..., description="Периоды для diff")
    columns: List[str] = Field(..., description="Колонки для применения")
    method: Literal["diff", "pct_change"] = Field(
        "diff", description="Метод вычисления"
    )

    @field_validator("periods")
    @classmethod
    def validate_periods(cls, v: List[int]) -> List[int]:
        """Валидация периодов."""
        if any(p <= 0 for p in v):
            raise ValueError("Все периоды должны быть положительными")
        return v


class RatiosFeatureConfig(BaseModel):
    """Конфигурация соотношений."""

    type: Literal["ratios"] = "ratios"
    pairs: List[tuple[str, str]] = Field(
        ..., description="Пары колонок (числитель, знаменатель)"
    )


class HigherTimeframeFeatureConfig(BaseModel):
    """Конфигурация признаков из старших таймфреймов."""

    type: Literal["higher_timeframe"] = "higher_timeframe"
    source_tf: str = Field(
        ..., description="Исходный таймфрейм (например, '1h', '4h', '1d')"
    )
    indicators: List[str] = Field(..., description="Список индикаторов для вычисления")
    alignment: Literal["forward_fill", "backward_fill", "interpolate"] = Field(
        "forward_fill", description="Метод выравнивания"
    )


# Union всех типов конфигураций признаков
FeatureConfigItem = Union[
    IndicatorFeatureConfig,
    PriceFeatureConfig,
    VolumeFeatureConfig,
    CalendarFeatureConfig,
    TickerFeatureConfig,
    RollingFeatureConfig,
    LagsFeatureConfig,
    DifferencesFeatureConfig,
    RatiosFeatureConfig,
    HigherTimeframeFeatureConfig,
]


class FeatureSelectionConfig(BaseModel):
    """Конфигурация отбора признаков."""

    enabled: bool = Field(True, description="Включить отбор признаков")
    method: Literal[
        "variance_threshold",
        "correlation",
        "mutual_info",
        "chi2",
        "rfe",
        "tree_importance",
        "l1",
        "shap",
    ] = Field("variance_threshold", description="Метод отбора")
    params: Dict[str, Any] = Field(default_factory=dict, description="Параметры метода")
    top_k: Optional[int] = Field(
        None, description="Количество топ признаков для отбора"
    )


class FeatureConfig(BaseModel):
    """Полная конфигурация генерации признаков."""

    version: str = Field("1.0", description="Версия конфига")
    features: List[FeatureConfigItem] = Field(..., description="Список признаков")
    selection: Optional[FeatureSelectionConfig] = Field(
        None, description="Конфигурация отбора"
    )
    cache_enabled: bool = Field(True, description="Включить кэширование")


def parse_feature_config(config_path: Union[str, Path, dict]) -> FeatureConfig:
    """
    Парсинг конфигурации признаков из YAML файла или словаря.

    Args:
        config_path: Путь к YAML файлу или словарь конфигурации

    Returns:
        Валидированная конфигурация
    """
    if isinstance(config_path, dict):
        config_dict = config_path
    else:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

    return FeatureConfig(**config_dict)


def validate_feature_config(config: dict) -> tuple[bool, Optional[str]]:
    """
    Валидация конфигурации признаков.

    Args:
        config: Словарь конфигурации

    Returns:
        Кортеж (is_valid, error_message)
    """
    try:
        FeatureConfig(**config)
        return True, None
    except Exception as e:
        return False, str(e)
