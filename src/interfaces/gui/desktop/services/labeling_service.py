from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.labeling.metadata import LabelingMetadata
from src.labeling.pipeline import LabelingPipeline

logger = logging.getLogger(__name__)

AVAILABLE_METHODS = {
    "triple_barrier": "Triple Barrier",
    "horizon": "Fixed/Adaptive Horizon",
    "regression": "Regression Targets",
    "custom": "Custom Rules",
}


@dataclass(slots=True)
class LabelingResult:
    data: pd.DataFrame
    metadata: LabelingMetadata


class LabelingService:
    """Обертка над LabelingPipeline для GUI."""

    def __init__(self) -> None:
        self._last_result: Optional[LabelingResult] = None

    def list_methods(self) -> dict[str, str]:
        return AVAILABLE_METHODS

    def run_labeling(self, data: pd.DataFrame, config: dict) -> LabelingResult:
        if "method" not in config:
            raise ValueError("В конфигурации должен быть указан метод разметки")

        pipeline = LabelingPipeline.from_config(config, data)
        labeled, metadata = pipeline.run(data, save_results=config.get("save_results", True))

        result = LabelingResult(data=labeled, metadata=metadata)
        self._last_result = result
        logger.info("Разметка %s завершена, всего %d записей", metadata.method, len(labeled))
        return result

    @property
    def last_result(self) -> Optional[LabelingResult]:
        return self._last_result
