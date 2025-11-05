"""Визуализация результатов разметки."""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Настройка стиля
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def plot_label_distribution(
    labels: pd.Series,
    title: str = "Label Distribution",
    save_path: Optional[Path] = None,
) -> None:
    """
    Визуализация распределения меток.

    Args:
        labels: Series с метками
        title: Заголовок графика
        save_path: Путь для сохранения графика
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    distribution = labels.value_counts().sort_index()
    axes[0].bar(distribution.index, distribution.values, color="steelblue", alpha=0.7)
    axes[0].set_xlabel("Label")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{title} - Counts")
    axes[0].grid(axis="y", alpha=0.3)

    # Добавляем значения на столбцы
    for i, (label, count) in enumerate(distribution.items()):
        axes[0].text(label, count, str(count), ha="center", va="bottom")

    # Pie chart
    colors = ["#ff6b6b", "#95e1d3", "#4ecdc4"]
    axes[1].pie(
        distribution.values,
        labels=[f"Label {label_value}" for label_value in distribution.index],
        autopct="%1.1f%%",
        colors=colors[: len(distribution)],
        startangle=90,
    )
    axes[1].set_title(f"{title} - Proportions")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"График сохранён: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_holding_periods(
    data: pd.DataFrame,
    title: str = "Holding Periods Distribution",
    save_path: Optional[Path] = None,
) -> None:
    """
    Визуализация распределения периодов удержания.

    Args:
        data: DataFrame с колонкой holding_period
        title: Заголовок графика
        save_path: Путь для сохранения графика
    """
    if "holding_period" not in data.columns:
        logger.warning("Колонка 'holding_period' отсутствует")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    holding_periods = data["holding_period"].dropna()

    # Histogram
    axes[0].hist(
        holding_periods, bins=50, color="steelblue", alpha=0.7, edgecolor="black"
    )
    axes[0].axvline(
        holding_periods.mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {holding_periods.mean():.1f}",
    )
    axes[0].axvline(
        holding_periods.median(),
        color="green",
        linestyle="--",
        label=f"Median: {holding_periods.median():.1f}",
    )
    axes[0].set_xlabel("Holding Period (bars)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"{title}")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Box plot по классам
    if "label" in data.columns:
        data_with_labels = data[data["holding_period"].notna()].copy()
        labels_unique = sorted(data_with_labels["label"].unique())

        box_data = [
            data_with_labels[data_with_labels["label"] == label][
                "holding_period"
            ].values
            for label in labels_unique
        ]

        bp = axes[1].boxplot(
            box_data,
            labels=[f"Label {label_value}" for label_value in labels_unique],
            patch_artist=True,
        )

        # Окрашиваем box plots
        colors = ["#ff6b6b", "#95e1d3", "#4ecdc4"]
        for patch, color in zip(bp["boxes"], colors[: len(labels_unique)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        axes[1].set_xlabel("Label")
        axes[1].set_ylabel("Holding Period (bars)")
        axes[1].set_title("Holding Periods by Label")
        axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"График сохранён: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_returns_by_label(
    data: pd.DataFrame,
    return_column: str = "future_return",
    title: str = "Returns Distribution by Label",
    save_path: Optional[Path] = None,
) -> None:
    """
    Визуализация распределения returns по классам.

    Args:
        data: DataFrame с колонками label и return
        return_column: Название колонки с returns
        title: Заголовок графика
        save_path: Путь для сохранения графика
    """
    if return_column not in data.columns or "label" not in data.columns:
        logger.warning("Необходимые колонки отсутствуют")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    data_clean = data[[return_column, "label"]].dropna()
    labels_unique = sorted(data_clean["label"].unique())

    # 1. Violin plot
    data_for_violin = []
    labels_for_violin = []
    for label in labels_unique:
        returns = data_clean[data_clean["label"] == label][return_column].values
        data_for_violin.extend(returns)
        labels_for_violin.extend([f"Label {label}"] * len(returns))

    df_violin = pd.DataFrame({"return": data_for_violin, "label": labels_for_violin})

    sns.violinplot(data=df_violin, x="label", y="return", ax=axes[0, 0])
    axes[0, 0].set_title("Violin Plot")
    axes[0, 0].axhline(0, color="red", linestyle="--", alpha=0.5)
    axes[0, 0].grid(axis="y", alpha=0.3)

    # 2. Box plot
    box_data = [
        data_clean[data_clean["label"] == label][return_column].values
        for label in labels_unique
    ]

    bp = axes[0, 1].boxplot(
        box_data,
        labels=[f"Label {label_value}" for label_value in labels_unique],
        patch_artist=True,
    )

    colors = ["#ff6b6b", "#95e1d3", "#4ecdc4"]
    for patch, color in zip(bp["boxes"], colors[: len(labels_unique)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[0, 1].set_title("Box Plot")
    axes[0, 1].axhline(0, color="red", linestyle="--", alpha=0.5)
    axes[0, 1].set_ylabel("Return")
    axes[0, 1].grid(axis="y", alpha=0.3)

    # 3. Histograms
    for label in labels_unique:
        returns = data_clean[data_clean["label"] == label][return_column].values
        axes[1, 0].hist(returns, bins=50, alpha=0.5, label=f"Label {label}")

    axes[1, 0].axvline(0, color="red", linestyle="--", alpha=0.5)
    axes[1, 0].set_xlabel("Return")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Histograms Overlay")
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.3)

    # 4. Statistics table
    stats = []
    for label in labels_unique:
        returns = data_clean[data_clean["label"] == label][return_column].values
        stats.append(
            {
                "Label": label,
                "Count": len(returns),
                "Mean": f"{returns.mean():.4f}",
                "Std": f"{returns.std():.4f}",
                "Min": f"{returns.min():.4f}",
                "Max": f"{returns.max():.4f}",
            }
        )

    stats_df = pd.DataFrame(stats)
    axes[1, 1].axis("off")

    table = axes[1, 1].table(
        cellText=stats_df.values,
        colLabels=stats_df.columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.12] * len(stats_df.columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Окрашиваем header
    for i in range(len(stats_df.columns)):
        table[(0, i)].set_facecolor("#4ecdc4")
        table[(0, i)].set_text_props(weight="bold", color="white")

    axes[1, 1].set_title("Statistics")

    plt.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"График сохранён: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_labels_timeline(
    data: pd.DataFrame,
    window: int = 1000,
    save_path: Optional[Path] = None,
) -> None:
    """
    Визуализация меток на временной шкале.

    Args:
        data: DataFrame с индексом DatetimeIndex и колонкой label
        window: Размер окна для отображения (последние N баров)
        save_path: Путь для сохранения графика
    """
    if "label" not in data.columns:
        logger.warning("Колонка 'label' отсутствует")
        return

    # Берём последние window баров
    data_window = data.iloc[-window:]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # 1. Цена и метки
    if "close" in data_window.columns:
        axes[0].plot(
            data_window.index,
            data_window["close"],
            color="black",
            linewidth=1,
            label="Close Price",
        )

        # Отмечаем метки цветом
        for label in data_window["label"].unique():
            mask = data_window["label"] == label
            if label == 1:
                color, marker = "green", "^"
                label_name = "Long"
            elif label == -1:
                color, marker = "red", "v"
                label_name = "Short"
            else:
                color, marker = "gray", "o"
                label_name = "Neutral"

            axes[0].scatter(
                data_window[mask].index,
                data_window[mask]["close"],
                color=color,
                marker=marker,
                s=30,
                alpha=0.6,
                label=label_name,
            )

        axes[0].set_ylabel("Price")
        axes[0].set_title("Price with Labels")
        axes[0].legend(loc="upper left")
        axes[0].grid(alpha=0.3)

    # 2. Timeline меток
    colors_map = {-1: "red", 0: "gray", 1: "green"}
    colors = [colors_map.get(label, "blue") for label in data_window["label"]]

    axes[1].scatter(data_window.index, data_window["label"], c=colors, s=10, alpha=0.6)
    axes[1].set_ylabel("Label")
    axes[1].set_xlabel("Time")
    axes[1].set_title("Labels Timeline")
    axes[1].set_yticks([-1, 0, 1])
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"График сохранён: {save_path}")
    else:
        plt.show()

    plt.close()


def create_labeling_report(
    data: pd.DataFrame,
    metadata: Optional[dict] = None,
    output_dir: Path = Path("artifacts/reports"),
) -> None:
    """
    Создание полного отчёта по разметке.

    Args:
        data: DataFrame с разметкой
        metadata: Словарь с метаданными
        output_dir: Директория для сохранения отчёта
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Создание отчёта по разметке...")

    # 1. Label distribution
    plot_label_distribution(
        data["label"], save_path=output_dir / "label_distribution.png"
    )

    # 2. Holding periods (если есть)
    if "holding_period" in data.columns:
        plot_holding_periods(data, save_path=output_dir / "holding_periods.png")

    # 3. Returns by label (если есть)
    if "future_return" in data.columns or "realized_return" in data.columns:
        return_col = (
            "future_return" if "future_return" in data.columns else "realized_return"
        )
        plot_returns_by_label(
            data,
            return_column=return_col,
            save_path=output_dir / "returns_by_label.png",
        )

    # 4. Timeline
    if isinstance(data.index, pd.DatetimeIndex):
        plot_labels_timeline(data, save_path=output_dir / "labels_timeline.png")

    logger.info(f"Отчёт сохранён в: {output_dir}")
