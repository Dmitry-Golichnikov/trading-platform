"""Генератор признаков - главный оркестратор."""

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.features.cache import FeatureCache
from src.features.config_parser import (
    FeatureConfig,
    FeatureConfigItem,
    FeatureSelectionConfig,
    parse_feature_config,
)
from src.features.extractors.calendar import CalendarExtractor
from src.features.extractors.higher_tf import HigherTimeframeExtractor
from src.features.extractors.price import PriceExtractor
from src.features.extractors.ticker import TickerExtractor
from src.features.extractors.volume import VolumeExtractor
from src.features.indicators.registry import IndicatorRegistry
from src.features.selectors.embedded_methods import EmbeddedSelector
from src.features.selectors.filter_methods import FilterSelector
from src.features.selectors.wrapper_methods import WrapperSelector
from src.features.transformers.differences import DifferencesTransformer
from src.features.transformers.lags import LagsTransformer
from src.features.transformers.ratios import RatiosTransformer
from src.features.transformers.rolling import RollingTransformer

logger = logging.getLogger(__name__)


class FeatureGenerator:
    """Генератор признаков на основе конфигурации."""

    def __init__(
        self,
        config: Union[str, Path, dict, FeatureConfig],
        cache_enabled: bool = True,
        cache_dir: Optional[Path] = None,
    ) -> None:
        """Инициализировать генератор."""

        if isinstance(config, FeatureConfig):
            self.config = config
        else:
            self.config = parse_feature_config(config)

        self.cache_enabled = cache_enabled and self.config.cache_enabled
        self.cache: Optional[FeatureCache] = None
        if self.cache_enabled:
            cache_path = cache_dir or Path("artifacts/features")
            self.cache = FeatureCache(cache_path)

        self.indicator_registry = IndicatorRegistry()

        logger.info(
            "FeatureGenerator инициализирован: %d конфигураций",
            len(self.config.features),
        )

    def generate(
        self,
        data: pd.DataFrame,
        dataset_id: Optional[str] = None,
        target: Optional[pd.Series] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Сгенерировать признаки на основе конфигурации."""

        if self.cache_enabled and use_cache and dataset_id and self.cache:
            cached = self.cache.get(dataset_id, self.config.model_dump())
            if cached is not None:
                cached = cached.copy()
                target_freq = getattr(data.index, "freq", None)
                if target_freq is not None and isinstance(cached.index, pd.DatetimeIndex):
                    cached.index = pd.DatetimeIndex(cached.index, freq=target_freq)
                logger.info(
                    "Признаки загружены из кэша для датасета %s",
                    dataset_id,
                )
                return cached

        logger.info("Генерация признаков...")

        features_collection: list[pd.DataFrame] = []
        for feature_config in self.config.features:
            try:
                generated = self._generate_single_feature(data, feature_config)
            except Exception as exc:  # pragma: no cover - логирование ошибок
                logger.error(
                    "Ошибка генерации признака %s: %s",
                    getattr(feature_config, "type", "unknown"),
                    exc,
                )
                continue

            if generated is not None and not generated.empty:
                features_collection.append(generated)

        if features_collection:
            result = pd.concat(features_collection, axis=1)
        else:
            result = pd.DataFrame(index=data.index)

        logger.info("Сгенерировано %d признаков", result.shape[1])

        selection_config = self.config.selection
        if selection_config is not None and selection_config.enabled and target is not None:
            result = self._apply_selection(result, target, selection_config)
            logger.info("После отбора осталось %d признаков", result.shape[1])

        if self.cache_enabled and dataset_id and self.cache:
            metadata = {
                "num_features": result.shape[1],
                "feature_names": result.columns.tolist(),
            }
            self.cache.save(dataset_id, self.config.model_dump(), result, metadata)
            logger.info("Признаки сохранены в кэш для датасета %s", dataset_id)

        return result

    def _generate_single_feature(self, data: pd.DataFrame, feature_config: FeatureConfigItem) -> Optional[pd.DataFrame]:
        """Сгенерировать один тип признаков."""

        feature_type = feature_config.type
        if feature_type == "indicator":
            return self._generate_indicator(data, feature_config)
        if feature_type == "price":
            return self._generate_price(data, feature_config)
        if feature_type == "volume":
            return self._generate_volume(data, feature_config)
        if feature_type == "calendar":
            return self._generate_calendar(data, feature_config)
        if feature_type == "ticker":
            return self._generate_ticker(data, feature_config)
        if feature_type == "rolling":
            return self._generate_rolling(data, feature_config)
        if feature_type == "lags":
            return self._generate_lags(data, feature_config)
        if feature_type == "differences":
            return self._generate_differences(data, feature_config)
        if feature_type == "ratios":
            return self._generate_ratios(data, feature_config)
        if feature_type == "higher_timeframe":
            return self._generate_higher_timeframe(data, feature_config)

        logger.warning("Неизвестный тип признака: %s", feature_type)
        return None

    def _generate_indicator(self, data: pd.DataFrame, config) -> pd.DataFrame:
        base_params = dict(config.params or {})
        if "period" in base_params and "window" not in base_params:
            base_params["window"] = base_params.pop("period")
        if "length" in base_params and "window" not in base_params:
            base_params["window"] = base_params.pop("length")

        columns = config.columns or []
        results: list[pd.DataFrame] = []

        if columns:
            for column in columns:
                params = base_params.copy()
                params.setdefault("column", column)
                indicator = self.indicator_registry.get(config.name, **params)
                result = indicator.calculate(data)
                formatted = self._format_indicator_result(
                    result,
                    prefix=config.prefix,
                    suffix=column,
                )
                results.append(formatted)
        else:
            indicator = self.indicator_registry.get(config.name, **base_params)
            result = indicator.calculate(data)
            formatted = self._format_indicator_result(result, prefix=config.prefix)
            results.append(formatted)

        return pd.concat(results, axis=1)

    @staticmethod
    def _format_indicator_result(
        result: Union[pd.Series, pd.DataFrame],
        *,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ) -> pd.DataFrame:
        if isinstance(result, pd.DataFrame):
            df = result.copy()
        else:
            df = result.to_frame()

        if prefix:
            df.columns = [f"{prefix}_{col}" for col in df.columns]

        if suffix:
            df.columns = [f"{col}_{suffix}" for col in df.columns]

        return df

    def _generate_price(self, data: pd.DataFrame, config) -> pd.DataFrame:
        return PriceExtractor(config.features).extract(data)

    def _generate_volume(self, data: pd.DataFrame, config) -> pd.DataFrame:
        return VolumeExtractor(config.features).extract(data)

    def _generate_calendar(self, data: pd.DataFrame, config) -> pd.DataFrame:
        return CalendarExtractor(config.features).extract(data)

    def _generate_ticker(self, data: pd.DataFrame, config) -> pd.DataFrame:
        extractor = TickerExtractor(
            encoding=config.encoding,
            sector=config.sector,
            industry=config.industry,
        )
        return extractor.extract(data)

    def _generate_rolling(self, data: pd.DataFrame, config) -> pd.DataFrame:
        transformer = RollingTransformer(
            window=config.window,
            functions=config.functions,
            columns=config.columns,
            min_periods=config.min_periods,
        )
        return transformer.transform(data)

    def _generate_lags(self, data: pd.DataFrame, config) -> pd.DataFrame:
        return LagsTransformer(config.lags, config.columns).transform(data)

    def _generate_differences(self, data: pd.DataFrame, config) -> pd.DataFrame:
        transformer = DifferencesTransformer(
            periods=config.periods,
            columns=config.columns,
            method=config.method,
        )
        return transformer.transform(data)

    def _generate_ratios(self, data: pd.DataFrame, config) -> pd.DataFrame:
        return RatiosTransformer(config.pairs).transform(data)

    def _generate_higher_timeframe(self, data: pd.DataFrame, config) -> pd.DataFrame:
        extractor = HigherTimeframeExtractor(
            source_tf=config.source_tf,
            indicators=config.indicators,
            alignment=config.alignment,
        )
        return extractor.extract(data)

    def _apply_selection(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        selection_config: FeatureSelectionConfig,
    ) -> pd.DataFrame:
        selector: Union[FilterSelector, WrapperSelector, EmbeddedSelector]
        method = selection_config.method

        if method in {"variance_threshold", "correlation", "mutual_info", "chi2"}:
            selector = FilterSelector(
                method=method,
                top_k=selection_config.top_k,
                **selection_config.params,
            )
        elif method in {"rfe", "forward", "backward"}:
            selector = WrapperSelector(
                method=method,
                n_features_to_select=selection_config.top_k,
                **selection_config.params,
            )
        elif method in {"tree_importance", "l1", "shap"}:
            selector = EmbeddedSelector(
                method=method,
                top_k=selection_config.top_k,
                **selection_config.params,
            )
        else:
            logger.warning("Неизвестный метод отбора: %s", method)
            return features

        selected = selector.fit_transform(features, target)
        logger.info(
            "Feature selection (%s): %d -> %d",
            method,
            features.shape[1],
            selected.shape[1],
        )
        return selected

    def get_feature_names(self) -> list[str]:  # pragma: no cover - заглушка
        logger.warning("get_feature_names() не полностью реализован")
        return []

    def invalidate_cache(self, dataset_id: Optional[str] = None) -> None:
        if self.cache:
            self.cache.invalidate(dataset_id)
            logger.info("Кэш инвалидирован для %s", dataset_id or "всех датасетов")
