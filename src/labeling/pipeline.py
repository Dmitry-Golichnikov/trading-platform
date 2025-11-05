"""Пайплайн генерации и обработки таргетов."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.labeling.base import BaseLabeler
from src.labeling.filters.danger_zones import DangerZonesFilter
from src.labeling.filters.majority_vote import MajorityVoteFilter
from src.labeling.filters.sequence_filter import SequenceFilter
from src.labeling.filters.smoothing import SmoothingFilter
from src.labeling.metadata import LabelingMetadata, generate_labeling_id

logger = logging.getLogger(__name__)


class LabelingPipeline:
    """
    Пайплайн генерации и обработки таргетов.

    Этапы:
    1. Генерация сырых labels
    2. Применение постфильтров
    3. Балансировка (опционально)
    4. Валидация
    5. Сохранение + метаданные
    """

    def __init__(
        self,
        labeler: BaseLabeler,
        filters: Optional[List] = None,
        dataset_id: str = "unknown",
        output_dir: Optional[Path] = None,
    ):
        """
        Инициализация пайплайна.

        Args:
            labeler: Объект разметчика (BaseLabeler)
            filters: Список постфильтров
            dataset_id: ID датасета
            output_dir: Директория для сохранения результатов
        """
        self.labeler = labeler
        self.filters = filters or []
        self.dataset_id = dataset_id
        self.output_dir = output_dir or Path("artifacts/labels")

        # Генерация уникального ID
        method_name = self.labeler.__class__.__name__
        self.labeling_id = generate_labeling_id(method_name, dataset_id)

        logger.info(f"Инициализирован LabelingPipeline: {self.labeling_id}")

    def run(
        self,
        data: pd.DataFrame,
        save_results: bool = True,
    ) -> Tuple[pd.DataFrame, LabelingMetadata]:
        """
        Запуск пайплайна разметки.

        Args:
            data: DataFrame с OHLCV данными
            save_results: Сохранять результаты на диск

        Returns:
            Кортеж (labeled_data, metadata)
        """
        logger.info(f"Запуск пайплайна разметки для {len(data)} сэмплов")

        # 1. Генерация сырых labels
        logger.info("Этап 1/5: Генерация сырых labels")
        labeled_data = self.labeler.label(data)

        if "label" not in labeled_data.columns:
            raise ValueError("Labeler output must contain 'label' column")

        initial_distribution = labeled_data["label"].value_counts().to_dict()
        logger.info(f"Начальное распределение: {initial_distribution}")

        # 2. Применение постфильтров
        logger.info("Этап 2/5: Применение постфильтров")
        filters_applied = []

        for i, filter_obj in enumerate(self.filters):
            filter_name = filter_obj.__class__.__name__
            logger.info(
                f"  Применение фильтра {i+1}/{len(self.filters)}: {filter_name}"
            )

            try:
                if isinstance(filter_obj, DangerZonesFilter):
                    # DangerZonesFilter требует полные данные
                    labeled_data["label"] = filter_obj.apply(
                        labeled_data["label"], labeled_data
                    )
                else:
                    labeled_data["label"] = filter_obj.apply(labeled_data["label"])

                filters_applied.append(filter_name)
            except Exception as e:
                logger.error(f"Ошибка в фильтре {filter_name}: {e}")

        final_distribution = labeled_data["label"].value_counts().to_dict()
        logger.info(f"Распределение после фильтров: {final_distribution}")

        # 3. Балансировка (через веса, не через sampling)
        logger.info("Этап 3/5: Вычисление весов для балансировки")
        # Веса можно вычислить здесь и добавить в результат
        # Но сам sampling лучше делать на этапе обучения

        # 4. Валидация
        logger.info("Этап 4/5: Валидация результатов")
        self._validate_labels(labeled_data)

        # 5. Создание метаданных
        logger.info("Этап 5/5: Создание метаданных")
        metadata = self._create_metadata(
            labeled_data,
            initial_distribution,
            final_distribution,
            filters_applied,
        )

        # Сохранение результатов
        if save_results:
            self._save_results(labeled_data, metadata)

        logger.info("Пайплайн разметки завершён успешно")

        return labeled_data, metadata

    def _validate_labels(self, data: pd.DataFrame) -> None:
        """
        Валидация результатов разметки.

        Args:
            data: DataFrame с разметкой

        Raises:
            ValueError: Если валидация не прошла
        """
        # Проверка наличия колонки label
        if "label" not in data.columns:
            raise ValueError("Отсутствует колонка 'label'")

        # Проверка на NaN
        if data["label"].isna().any():
            nan_count = data["label"].isna().sum()
            logger.warning(f"Найдено {nan_count} NaN значений в labels")

        # Проверка распределения классов
        distribution = data["label"].value_counts()

        if len(distribution) == 0:
            raise ValueError("Не найдено ни одной метки")

        # Проверка на минимальное количество примеров каждого класса
        min_samples = 10
        for cls, count in distribution.items():
            if count < min_samples:
                logger.warning(
                    f"Класс {cls} имеет только {count} примеров "
                    f"(минимум рекомендуется {min_samples})"
                )

        logger.info("Валидация прошла успешно")

    def _create_metadata(
        self,
        data: pd.DataFrame,
        initial_distribution: Dict[int, int],
        final_distribution: Dict[int, int],
        filters_applied: List[str],
    ) -> LabelingMetadata:
        """
        Создание метаданных разметки.

        Args:
            data: DataFrame с разметкой
            initial_distribution: Начальное распределение
            final_distribution: Финальное распределение
            filters_applied: Список применённых фильтров

        Returns:
            Объект LabelingMetadata
        """
        metadata = LabelingMetadata(
            labeling_id=self.labeling_id,
            method=self.labeler.__class__.__name__,
            config=self.labeler.get_params(),
            dataset_id=self.dataset_id,
            total_samples=len(data),
            class_distribution=final_distribution,
            filters_applied=filters_applied,
        )

        # Добавление дополнительной статистики
        metadata.add_statistics("initial_distribution", initial_distribution)

        # Если есть дополнительные колонки, добавим их статистику
        if "future_return" in data.columns:
            metadata.add_statistics(
                "future_return_mean", float(data["future_return"].mean())
            )
            metadata.add_statistics(
                "future_return_std", float(data["future_return"].std())
            )

        if "holding_period" in data.columns:
            metadata.add_statistics(
                "avg_holding_period", float(data["holding_period"].mean())
            )

        return metadata

    def _save_results(
        self,
        data: pd.DataFrame,
        metadata: LabelingMetadata,
    ) -> None:
        """
        Сохранение результатов разметки.

        Args:
            data: DataFrame с разметкой
            metadata: Метаданные
        """
        # Создание директории
        output_path = self.output_dir / self.labeling_id
        output_path.mkdir(parents=True, exist_ok=True)

        # Сохранение данных
        data_path = output_path / "labels.parquet"
        data.to_parquet(data_path, index=True)
        logger.info(f"Данные сохранены: {data_path}")

        # Сохранение метаданных
        metadata_path = output_path / "metadata.json"
        metadata.save(metadata_path)

        logger.info(f"Результаты сохранены в: {output_path}")

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], data: pd.DataFrame
    ) -> "LabelingPipeline":
        """
        Создание пайплайна из конфигурации.

        Args:
            config: Словарь с конфигурацией
            data: DataFrame с данными (для некоторых фильтров)

        Returns:
            Объект LabelingPipeline
        """
        # Импортируем методы разметки
        from src.labeling.methods import (
            CustomRulesLabeler,
            HorizonLabeler,
            RegressionTargetsLabeler,
            TripleBarrierLabeler,
        )

        # Создание labeler
        method = config["method"]
        params = config.get("params", {})

        labeler: BaseLabeler
        if method == "horizon":
            labeler = HorizonLabeler(**params)
        elif method == "triple_barrier":
            labeler = TripleBarrierLabeler(**params)
        elif method == "regression":
            labeler = RegressionTargetsLabeler(**params)
        elif method == "custom":
            labeler = CustomRulesLabeler(**params)
        else:
            raise ValueError(f"Неизвестный метод разметки: {method}")

        # Создание фильтров
        filters: List[object] = []
        for filter_config in config.get("filters", []):
            filter_type = filter_config["type"]
            filter_params = filter_config.get("params", {})

            if filter_type == "smoothing":
                filters.append(SmoothingFilter(**filter_params))
            elif filter_type == "sequence":
                filters.append(SequenceFilter(**filter_params))
            elif filter_type == "majority_vote":
                filters.append(MajorityVoteFilter(**filter_params))
            elif filter_type == "danger_zones":
                filters.append(DangerZonesFilter(**filter_params))
            else:
                logger.warning(f"Неизвестный тип фильтра: {filter_type}")

        # Создание пайплайна
        pipeline = cls(
            labeler=labeler,
            filters=filters,
            dataset_id=config.get("dataset_id", "unknown"),
            output_dir=Path(config.get("output_dir", "artifacts/labels")),
        )

        return pipeline
