from __future__ import annotations

import itertools
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from .backtest_service import BacktestJobResult, BacktestService
from .runner import PipelineProgressUpdate
from .training_service import TrainingJobResult, TrainingService


@dataclass(slots=True)
class ExperimentItem:
    name: str
    overrides: dict[str, Any]


@dataclass(slots=True)
class ExperimentPlan:
    base_training_config: dict[str, Any]
    base_backtest_config: dict[str, Any]
    datasets: list[ExperimentItem]
    feature_sets: list[ExperimentItem]
    labeling_setups: list[ExperimentItem]
    models: list[ExperimentItem]
    strategies: list[ExperimentItem]


@dataclass(slots=True)
class ExperimentProgress:
    total: int
    index: int
    status: str
    message: str
    combination: dict[str, str]


@dataclass(slots=True)
class ExperimentResult:
    combination: dict[str, str]
    training: TrainingJobResult
    backtest: BacktestJobResult


class ExperimentService:
    """Batch-эксперименты поверх сервисов обучения и бэктеста."""

    def __init__(self, training_service: TrainingService, backtest_service: BacktestService) -> None:
        self.training_service = training_service
        self.backtest_service = backtest_service

    def run_plan(
        self,
        plan: ExperimentPlan,
        *,
        progress_callback: Optional[Callable[[ExperimentProgress], None]] = None,
        pipeline_callback: Optional[Callable[[PipelineProgressUpdate], None]] = None,
    ) -> list[ExperimentResult]:
        dimensions: Iterable[tuple[ExperimentItem, ...]] = itertools.product(
            plan.datasets,
            plan.feature_sets,
            plan.labeling_setups,
            plan.models,
            plan.strategies,
        )

        combos = list(dimensions)
        total = len(combos)
        results: list[ExperimentResult] = []

        for idx, combo in enumerate(combos, start=1):
            dataset, feature_set, labeling, model, strategy = combo
            combo_names = {
                "dataset": dataset.name,
                "features": feature_set.name,
                "labeling": labeling.name,
                "model": model.name,
                "strategy": strategy.name,
            }

            if progress_callback:
                progress_callback(
                    ExperimentProgress(
                        total=total,
                        index=idx,
                        status="running",
                        message="Запуск эксперимента",
                        combination=combo_names,
                    )
                )

            training_config = self._compose_config(
                plan.base_training_config,
                dataset.overrides,
                feature_set.overrides,
                labeling.overrides,
                model.overrides,
            )

            training_result = self.training_service.run_training(training_config, progress_callback=pipeline_callback)

            backtest_config = self._compose_config(
                plan.base_backtest_config,
                dataset.overrides,
                strategy.overrides,
            )

            backtest_result = self.backtest_service.run_backtest(backtest_config, progress_callback=pipeline_callback)

            results.append(
                ExperimentResult(
                    combination=combo_names,
                    training=training_result,
                    backtest=backtest_result,
                )
            )

            if progress_callback:
                progress_callback(
                    ExperimentProgress(
                        total=total,
                        index=idx,
                        status="completed",
                        message="Эксперимент завершен",
                        combination=combo_names,
                    )
                )

        return results

    def _compose_config(self, base: dict[str, Any], *overrides: dict[str, Any]) -> dict[str, Any]:
        config = deepcopy(base)
        for override in overrides:
            self._deep_merge(config, override)
        return config

    def _deep_merge(self, target: dict[str, Any], source: dict[str, Any]) -> None:
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
