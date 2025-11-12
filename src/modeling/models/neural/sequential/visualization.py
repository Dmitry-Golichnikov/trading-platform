"""
Визуализация для sequential моделей.

Включает:
- Training curves
- Attention weights
- Predictions vs ground truth
- Sequence patterns
"""

from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

sns.set_style("whitegrid")


def plot_training_history(
    history: Union[dict, pd.DataFrame],
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 4),
) -> None:
    """
    Визуализировать историю обучения.

    Args:
        history: История обучения (dict или DataFrame)
        save_path: Путь для сохранения графика
        figsize: Размер фигуры
    """
    if isinstance(history, dict):
        history = pd.DataFrame(history)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss curves
    if "train_loss" in history.columns:
        axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
    if "val_loss" in history.columns:
        axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Learning rate
    if "learning_rate" in history.columns:
        axes[1].plot(history["learning_rate"], linewidth=2, color="green")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Learning Rate")
        axes[1].set_title("Learning Rate Schedule")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"График сохранён: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_attention_weights(
    attention_weights: torch.Tensor,
    seq_labels: Optional[list] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
    cmap: str = "YlOrRd",
) -> None:
    """
    Визуализировать attention weights.

    Args:
        attention_weights: Attention веса, shape (seq_len,) или (batch, seq_len)
        seq_labels: Метки для временных шагов
        save_path: Путь для сохранения
        figsize: Размер фигуры
        cmap: Цветовая схема
    """
    # Конвертируем в numpy
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()

    # Если batch, берём первый пример
    if len(attention_weights.shape) > 1:
        attention_weights = attention_weights[0]

    seq_len = len(attention_weights)

    if seq_labels is None:
        seq_labels = [f"t-{seq_len - i - 1}" for i in range(seq_len)]

    plt.figure(figsize=figsize)

    # Heatmap
    plt.subplot(2, 1, 1)
    sns.heatmap(
        attention_weights.reshape(1, -1),
        cmap=cmap,
        cbar=True,
        xticklabels=seq_labels,
        yticklabels=["Attention"],
    )
    plt.title("Attention Weights Heatmap")

    # Bar plot
    plt.subplot(2, 1, 2)
    plt.bar(range(seq_len), attention_weights, color="steelblue", alpha=0.7)
    plt.xlabel("Timestep")
    plt.ylabel("Attention Weight")
    plt.title("Attention Weights Distribution")
    plt.xticks(range(0, seq_len, max(1, seq_len // 10)))
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"График сохранён: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: Optional[pd.DatetimeIndex] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 6),
    title: str = "Predictions vs Ground Truth",
) -> None:
    """
    Визуализировать предсказания vs истинные значения.

    Args:
        y_true: Истинные значения
        y_pred: Предсказания
        timestamps: Временные метки
        save_path: Путь для сохранения
        figsize: Размер фигуры
        title: Заголовок
    """
    n_samples = min(len(y_true), 500)  # Ограничиваем для читаемости

    if timestamps is None:
        timestamps_array: np.ndarray = np.arange(n_samples)
    else:
        if hasattr(timestamps, "values"):
            timestamps_array = np.array(timestamps[:n_samples].values)
        else:
            timestamps_array = np.array(timestamps[:n_samples])

    y_true = y_true[:n_samples]
    y_pred = y_pred[:n_samples]

    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # Time series plot
    axes[0].plot(timestamps_array, y_true, label="Ground Truth", linewidth=1.5, alpha=0.7)
    axes[0].plot(timestamps_array, y_pred, label="Predictions", linewidth=1.5, alpha=0.7)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Value")
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Error plot
    error = y_pred - y_true
    axes[1].plot(timestamps_array, error, color="red", linewidth=1, alpha=0.7)
    axes[1].axhline(y=0, color="black", linestyle="--", linewidth=1)
    axes[1].fill_between(timestamps_array, 0, error, alpha=0.3, color="red")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Error")
    axes[1].set_title("Prediction Error")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"График сохранён: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_scatter_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
    figsize: tuple = (8, 8),
) -> None:
    """
    Scatter plot предсказаний vs истинных значений.

    Args:
        y_true: Истинные значения
        y_pred: Предсказания
        save_path: Путь для сохранения
        figsize: Размер фигуры
    """
    plt.figure(figsize=figsize)

    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5, s=20)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")

    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.title("Predictions vs Ground Truth")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")

    # Добавляем метрики
    from sklearn.metrics import mean_squared_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    textstr = f"MSE: {mse:.4f}\nR²: {r2:.4f}"
    plt.text(
        0.05,
        0.95,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"График сохранён: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_sequence_heatmap(
    sequences: np.ndarray,
    n_samples: int = 10,
    feature_names: Optional[list] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 8),
) -> None:
    """
    Heatmap нескольких последовательностей.

    Args:
        sequences: Последовательности, shape (n_samples, seq_len, n_features)
        n_samples: Количество примеров для отображения
        feature_names: Названия признаков
        save_path: Путь для сохранения
        figsize: Размер фигуры
    """
    sequences = sequences[:n_samples]
    n_samples, seq_len, n_features = sequences.shape

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n_features)]

    fig, axes = plt.subplots(n_samples, 1, figsize=figsize, sharex=True)

    if n_samples == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        sns.heatmap(
            sequences[i].T,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            cbar=i == 0,
            yticklabels=feature_names if i == 0 else False,
            xticklabels=False,
        )
        ax.set_ylabel(f"Sample {i + 1}")

    axes[-1].set_xlabel("Timestep")
    plt.suptitle("Sequence Features Heatmap", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"График сохранён: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix_over_time(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window_size: int = 100,
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 6),
) -> None:
    """
    Confusion matrix в скользящем окне (для бинарной классификации).

    Args:
        y_true: Истинные метки
        y_pred: Предсказанные метки
        window_size: Размер окна
        save_path: Путь для сохранения
        figsize: Размер фигуры
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    n_windows = len(y_true) // window_size

    metrics: Dict[str, list[float]] = {
        "accuracy": [],
        "precision": [],
        "recall": [],
    }

    for i in range(n_windows):
        start = i * window_size
        end = start + window_size

        y_true_window = y_true[start:end]
        y_pred_window = y_pred[start:end]

        metrics["accuracy"].append(accuracy_score(y_true_window, y_pred_window))
        metrics["precision"].append(precision_score(y_true_window, y_pred_window, zero_division=0))
        metrics["recall"].append(recall_score(y_true_window, y_pred_window, zero_division=0))

    plt.figure(figsize=figsize)

    x = np.arange(n_windows) * window_size

    plt.plot(x, metrics["accuracy"], label="Accuracy", linewidth=2, marker="o")
    plt.plot(x, metrics["precision"], label="Precision", linewidth=2, marker="s")
    plt.plot(x, metrics["recall"], label="Recall", linewidth=2, marker="^")

    plt.xlabel("Sample")
    plt.ylabel("Metric Value")
    plt.title(f"Classification Metrics over Time (window={window_size})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"График сохранён: {save_path}")
    else:
        plt.show()

    plt.close()
