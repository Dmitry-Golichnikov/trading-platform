"""
PyTorch Dataset для sequence данных временных рядов.

Реализует подготовку последовательностей с использованием sliding window.
"""

from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    """
    PyTorch Dataset для временных рядов с sliding window.

    Создаёт последовательности фиксированной длины из временных рядов
    с использованием sliding window подхода.

    Параметры:
        X: Массив признаков, shape (n_samples, n_features)
        y: Массив таргетов, shape (n_samples,)
        seq_length: Длина последовательности
        stride: Шаг окна (по умолчанию 1)
        predict_horizon: Горизонт предсказания (по умолчанию 0 - следующий шаг)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_length: int,
        stride: int = 1,
        predict_horizon: int = 0,
    ):
        if len(X) != len(y):
            raise ValueError(f"X и y должны иметь одинаковую длину: {len(X)} != {len(y)}")

        if seq_length < 1:
            raise ValueError(f"seq_length должен быть >= 1, получено {seq_length}")

        if stride < 1:
            raise ValueError(f"stride должен быть >= 1, получено {stride}")

        if predict_horizon < 0:
            raise ValueError(f"predict_horizon должен быть >= 0, получено {predict_horizon}")

        self.X = X
        self.y = y
        self.seq_length = seq_length
        self.stride = stride
        self.predict_horizon = predict_horizon

        # Вычисляем количество возможных последовательностей
        # Нужно seq_length точек для входа + predict_horizon для таргета
        self.max_start_idx = len(X) - seq_length - predict_horizon

        if self.max_start_idx < 0:
            raise ValueError(
                f"Недостаточно данных для создания последовательностей: "
                f"len(X)={len(X)}, seq_length={seq_length}, predict_horizon={predict_horizon}"
            )

        # Индексы начальных позиций окон
        self.valid_indices = list(range(0, self.max_start_idx + 1, stride))

    def __len__(self) -> int:
        """Количество последовательностей в датасете."""
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Получить последовательность и таргет.

        Args:
            idx: Индекс последовательности

        Returns:
            (X_seq, y_target) - последовательность признаков и таргет
        """
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.seq_length
        target_idx = end_idx + self.predict_horizon

        X_seq = self.X[start_idx:end_idx]  # shape: (seq_length, n_features)
        y_target = self.y[target_idx]  # shape: (,) или (n_targets,)

        return torch.FloatTensor(X_seq), torch.FloatTensor([y_target])

    def get_sequence_info(self) -> dict:
        """Получить информацию о датасете."""
        return {
            "total_samples": len(self.X),
            "n_features": self.X.shape[1] if len(self.X.shape) > 1 else 1,
            "seq_length": self.seq_length,
            "stride": self.stride,
            "predict_horizon": self.predict_horizon,
            "n_sequences": len(self),
            "max_start_idx": self.max_start_idx,
        }


def create_sequence_dataloader(
    X: pd.DataFrame,
    y: pd.Series,
    seq_length: int,
    batch_size: int = 32,
    stride: int = 1,
    predict_horizon: int = 0,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """
    Создать PyTorch DataLoader для sequence данных.

    Параметры:
        X: DataFrame с признаками
        y: Series с таргетами
        seq_length: Длина последовательности
        batch_size: Размер батча
        stride: Шаг sliding window
        predict_horizon: Горизонт предсказания
        shuffle: Перемешивать ли данные
        num_workers: Количество worker процессов для загрузки данных
        drop_last: Отбросить последний неполный батч

    Returns:
        DataLoader для обучения
    """
    # Конвертируем в numpy arrays
    X_np = X.values if isinstance(X, pd.DataFrame) else X
    y_np = y.values if isinstance(y, pd.Series) else y

    # Убеждаемся, что это numpy arrays
    if not isinstance(X_np, np.ndarray):
        X_np = np.array(X_np)
    if not isinstance(y_np, np.ndarray):
        y_np = np.array(y_np)

    # Создаём датасет
    dataset = SequenceDataset(
        X=X_np,
        y=y_np,
        seq_length=seq_length,
        stride=stride,
        predict_horizon=predict_horizon,
    )

    # Создаём DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),  # Ускорение на GPU
    )

    return dataloader


class MultiHorizonSequenceDataset(Dataset):
    """
    Dataset для предсказания нескольких горизонтов одновременно.

    Полезно для многошагового прогнозирования.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_length: int,
        horizons: list[int],
        stride: int = 1,
    ):
        self.X = X
        self.y = y
        self.seq_length = seq_length
        self.horizons = sorted(horizons)  # Сортируем для консистентности
        self.stride = stride

        # Максимальный горизонт определяет, сколько данных нам нужно
        max_horizon = max(self.horizons)
        self.max_start_idx = len(X) - seq_length - max_horizon

        if self.max_start_idx < 0:
            raise ValueError("Недостаточно данных для создания последовательностей")

        self.valid_indices = list(range(0, self.max_start_idx + 1, stride))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Возвращает:
            X_seq: shape (seq_length, n_features)
            y_targets: shape (n_horizons,) - таргеты для каждого горизонта
        """
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.seq_length

        X_seq = self.X[start_idx:end_idx]

        # Собираем таргеты для всех горизонтов
        y_targets = np.array([self.y[end_idx + h] for h in self.horizons])

        return torch.FloatTensor(X_seq), torch.FloatTensor(y_targets)
