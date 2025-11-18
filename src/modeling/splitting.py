"""
Data Splitting для временных рядов.

Методы разделения данных с учётом временной природы.
"""

import logging
from datetime import timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Класс для разделения данных train/val/test.

    Поддерживает различные стратегии разделения с учётом временных рядов.
    """

    @staticmethod
    def split_sequential(
        data: pd.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Последовательное разделение данных (для временных рядов).

        Разделяет данные в хронологическом порядке:
        [train | val | test]

        Args:
            data: Исходные данные (должны быть отсортированы по времени)
            train_size: Доля train выборки
            val_size: Доля validation выборки
            test_size: Доля test выборки

        Returns:
            (train_df, val_df, test_df)

        Raises:
            ValueError: Если размеры не суммируются в 1.0

        Примеры:
            >>> train, val, test = DataSplitter.split_sequential(df)
        """
        # Проверяем размеры
        total = train_size + val_size + test_size
        if not np.isclose(total, 1.0):
            raise ValueError(f"Сумма размеров должна быть 1.0, получено: {total:.4f}")

        n = len(data)

        # Вычисляем индексы разделения
        train_end = int(n * train_size)
        val_end = train_end + int(n * val_size)

        # Разделяем
        train_df = data.iloc[:train_end].copy()
        val_df = data.iloc[train_end:val_end].copy()
        test_df = data.iloc[val_end:].copy()

        logger.info(f"Sequential split: train={len(train_df)}, " f"val={len(val_df)}, test={len(test_df)}")

        return train_df, val_df, test_df

    @staticmethod
    def split_walk_forward(
        data: pd.DataFrame,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        min_train_size: Optional[int] = None,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Walk-forward splitting (для кросс-валидации временных рядов).

        Создаёт несколько последовательных сплитов с увеличивающимся train размером.

        Схема:
            Split 1: [train_1 | test_1] ...
            Split 2: [train_1 + test_1 | test_2] ...
            Split 3: [train_1 + test_1 + test_2 | test_3] ...
            ...

        Args:
            data: Исходные данные
            n_splits: Количество сплитов
            test_size: Размер test выборки в каждом сплите (в строках)
            min_train_size: Минимальный размер train выборки

        Returns:
            Список кортежей (train_df, test_df) для каждого сплита

        Примеры:
            >>> splits = DataSplitter.split_walk_forward(df, n_splits=5)
            >>> for train, test in splits:
            >>>     model.fit(train)
            >>>     score = model.evaluate(test)
        """
        n = len(data)

        if test_size is None:
            test_size = n // (n_splits + 1)

        if min_train_size is None:
            min_train_size = n // (n_splits + 1)

        # Проверяем что хватает данных
        required_size = min_train_size + test_size * n_splits
        if required_size > n:
            raise ValueError(f"Недостаточно данных для {n_splits} сплитов. " f"Требуется {required_size}, доступно {n}")

        splits = []

        for i in range(n_splits):
            # Train: от начала до текущей позиции
            train_end = min_train_size + i * test_size

            # Test: следующий блок
            test_start = train_end
            test_end = test_start + test_size

            if test_end > n:
                break

            train_df = data.iloc[:train_end].copy()
            test_df = data.iloc[test_start:test_end].copy()

            splits.append((train_df, test_df))

        logger.info(f"Walk-forward split: {len(splits)} splits, " f"test_size={test_size}")

        return splits

    @staticmethod
    def split_purged(
        data: pd.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        embargo_td: timedelta = timedelta(days=1),
        timestamp_col: str = "timestamp",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Purged split с embargo для предотвращения data leakage.

        Удаляет "эмбарго" период между train и val, val и test для
        предотвращения утечки информации из-за overlap в признаках.

        Args:
            data: Исходные данные
            train_size: Доля train выборки
            val_size: Доля val выборки
            test_size: Доля test выборки
            embargo_td: Период embargo (удаляется между сплитами)
            timestamp_col: Имя колонки с timestamp

        Returns:
            (train_df, val_df, test_df)

        Примеры:
            >>> train, val, test = DataSplitter.split_purged(
            >>>     df,
            >>>     embargo_td=timedelta(hours=24)
            >>> )
        """
        if timestamp_col not in data.columns:
            raise ValueError(f"Колонка '{timestamp_col}' не найдена в данных")

        # Проверяем размеры
        total = train_size + val_size + test_size
        if not np.isclose(total, 1.0):
            raise ValueError(f"Сумма размеров должна быть 1.0, получено: {total:.4f}")

        n = len(data)

        # Базовые индексы
        train_end_idx = int(n * train_size)
        val_end_idx = train_end_idx + int(n * val_size)

        # Получаем временные метки
        train_end_time = data[timestamp_col].iloc[train_end_idx]
        val_end_time = data[timestamp_col].iloc[val_end_idx]

        # Применяем embargo
        train_embargo_cutoff = train_end_time + embargo_td
        val_embargo_cutoff = val_end_time + embargo_td

        # Разделяем с учётом embargo
        train_df = data[data[timestamp_col] < train_end_time].copy()
        val_df = data[(data[timestamp_col] >= train_embargo_cutoff) & (data[timestamp_col] < val_end_time)].copy()
        test_df = data[data[timestamp_col] >= val_embargo_cutoff].copy()

        # Считаем сколько удалили
        removed = n - (len(train_df) + len(val_df) + len(test_df))

        logger.info(
            f"Purged split: train={len(train_df)}, val={len(val_df)}, "
            f"test={len(test_df)}, removed={removed} (embargo={embargo_td})"
        )

        return train_df, val_df, test_df

    @staticmethod
    def split_by_date(
        data: pd.DataFrame,
        train_end_date: str,
        val_end_date: Optional[str] = None,
        timestamp_col: str = "timestamp",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Разделение по конкретным датам.

        Args:
            data: Исходные данные
            train_end_date: Дата окончания train (формат: 'YYYY-MM-DD')
            val_end_date: Дата окончания val (если None - train/test)
            timestamp_col: Имя колонки с timestamp

        Returns:
            (train_df, val_df, test_df) или (train_df, empty_df, test_df)

        Примеры:
            >>> train, val, test = DataSplitter.split_by_date(
            >>>     df,
            >>>     train_end_date='2023-01-01',
            >>>     val_end_date='2023-06-01'
            >>> )
        """
        if timestamp_col not in data.columns:
            raise ValueError(f"Колонка '{timestamp_col}' не найдена")

        # Конвертируем даты
        train_end = pd.to_datetime(train_end_date)

        if val_end_date:
            val_end = pd.to_datetime(val_end_date)

            train_df = data[data[timestamp_col] < train_end].copy()
            val_df = data[(data[timestamp_col] >= train_end) & (data[timestamp_col] < val_end)].copy()
            test_df = data[data[timestamp_col] >= val_end].copy()
        else:
            train_df = data[data[timestamp_col] < train_end].copy()
            val_df = pd.DataFrame()
            test_df = data[data[timestamp_col] >= train_end].copy()

        logger.info(f"Split by date: train={len(train_df)}, val={len(val_df)}, " f"test={len(test_df)}")

        return train_df, val_df, test_df

    @staticmethod
    def validate_split(
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        timestamp_col: str = "timestamp",
    ) -> dict:
        """
        Проверить корректность разделения.

        Проверяет:
        - Нет пересечений
        - Правильный порядок по времени
        - Достаточно данных

        Args:
            train: Train выборка
            val: Validation выборка
            test: Test выборка
            timestamp_col: Имя колонки с timestamp

        Returns:
            Словарь с результатами проверки
        """
        issues = []

        # Проверка размеров
        if len(train) == 0:
            issues.append("Train выборка пустая")
        if len(test) == 0:
            issues.append("Test выборка пустая")

        # Проверка временного порядка
        if timestamp_col in train.columns and timestamp_col in test.columns:
            if len(train) > 0 and len(test) > 0:
                train_max = train[timestamp_col].max()
                test_min = test[timestamp_col].min()

                if train_max >= test_min:
                    issues.append(
                        f"Временное пересечение: train заканчивается {train_max}, " f"test начинается {test_min}"
                    )

            if len(val) > 0:
                val_min = val[timestamp_col].min()
                val_max = val[timestamp_col].max()

                if len(train) > 0 and train_max >= val_min:
                    issues.append("Временное пересечение между train и val")

                if len(test) > 0 and val_max >= test_min:
                    issues.append("Временное пересечение между val и test")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
            "total_size": len(train) + len(val) + len(test),
        }
