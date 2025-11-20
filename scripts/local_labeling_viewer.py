#!/usr/bin/env python
"""
Локальный просмотрщик результатов разметки без веб-интерфейса.

Скрипт позволяет:
  * загрузить parquet-файл с метками (`artifacts/labeling/<id>/labels.parquet`);
  * просмотреть таблицу целиком в TK-интерфейсе;
  * отобразить линейный график цены с отмеченными метками (long/short).

Пример запуска:
    python scripts/local_labeling_viewer.py --artifacts-dir artifacts --labeling-id my_dataset__triple_barrier
"""

from __future__ import annotations

import argparse
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


def discover_labeling_sets(labeling_root: Path) -> List[str]:
    """Найти доступные наборы разметки (папки с labels.parquet)."""
    if not labeling_root.exists():
        return []
    sets = []
    for path in sorted(labeling_root.iterdir()):
        if (path / "labels.parquet").exists():
            sets.append(path.name)
    return sets


class LabelingViewer(tk.Tk):
    """Простой локальный интерфейс просмотра разметки."""

    def __init__(self, artifacts_dir: Path, initial_id: Optional[str], price_column: Optional[str] = None):
        super().__init__()
        self.title("Labeling Viewer (offline)")
        self.geometry("1400x900")

        self.artifacts_dir = artifacts_dir
        self.labeling_root = artifacts_dir / "labeling"
        self.price_column = price_column

        self.labeling_sets: List[str] = discover_labeling_sets(self.labeling_root)
        self.current_df: Optional[pd.DataFrame] = None

        self.labeling_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Нет загруженных данных")

        self._build_ui()

        default_id = initial_id or (self.labeling_sets[0] if self.labeling_sets else "")
        if default_id:
            self.labeling_var.set(default_id)
            self.load_data()

    # UI helpers -----------------------------------------------------------------
    def _build_ui(self) -> None:
        top_frame = ttk.Frame(self)
        top_frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(top_frame, text="Labeling set:").pack(side="left")

        self.combo = ttk.Combobox(
            top_frame,
            textvariable=self.labeling_var,
            values=self.labeling_sets,
            state="readonly",
            width=60,
        )
        self.combo.pack(side="left", padx=5)
        self.combo.bind("<<ComboboxSelected>>", lambda _: self.load_data())

        ttk.Button(top_frame, text="Обновить список", command=self.refresh_sets).pack(side="left", padx=5)
        ttk.Button(top_frame, text="Перезагрузить", command=self.load_data).pack(side="left")

        ttk.Label(top_frame, textvariable=self.status_var).pack(side="right")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Таблица
        self.table_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.table_tab, text="Таблица")
        self.table_container = ttk.Frame(self.table_tab)
        self.table_container.pack(fill="both", expand=True)

        # График
        self.chart_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.chart_tab, text="График")

        self.chart_controls = ttk.Frame(self.chart_tab)
        self.chart_controls.pack(fill="x", padx=5, pady=5)

        self.show_price_var = tk.BooleanVar(value=True)
        self.show_long_var = tk.BooleanVar(value=True)
        self.show_short_var = tk.BooleanVar(value=True)
        self.chart_mode_var = tk.StringVar(value="line")

        ttk.Checkbutton(
            self.chart_controls,
            text="Показать цену",
            variable=self.show_price_var,
            command=self._update_chart,
        ).pack(side="left", padx=5)
        ttk.Checkbutton(
            self.chart_controls,
            text="Показать LONG",
            variable=self.show_long_var,
            command=self._update_chart,
        ).pack(side="left", padx=5)
        ttk.Checkbutton(
            self.chart_controls,
            text="Показать SHORT",
            variable=self.show_short_var,
            command=self._update_chart,
        ).pack(side="left", padx=5)

        mode_frame = ttk.Frame(self.chart_controls)
        mode_frame.pack(side="left", padx=15)
        ttk.Label(mode_frame, text="Режим:").pack(side="left", padx=(0, 5))
        ttk.Radiobutton(
            mode_frame,
            text="Линия",
            value="line",
            variable=self.chart_mode_var,
            command=self._update_chart,
        ).pack(side="left")
        ttk.Radiobutton(
            mode_frame,
            text="Свечи",
            value="candles",
            variable=self.chart_mode_var,
            command=self._update_chart,
        ).pack(side="left")

        self.figure = Figure(figsize=(10, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.chart_tab)
        self.canvas.draw()
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.chart_tab)
        self.toolbar.update()

    # Actions --------------------------------------------------------------------
    def refresh_sets(self) -> None:
        """Пересканировать директорию artifacts/labeling."""
        self.labeling_sets = discover_labeling_sets(self.labeling_root)
        self.combo["values"] = self.labeling_sets
        if self.labeling_sets and not self.labeling_var.get():
            self.labeling_var.set(self.labeling_sets[0])

    def load_data(self) -> None:
        """Загрузить parquet выбранного набора."""
        labeling_id = self.labeling_var.get()
        if not labeling_id:
            messagebox.showinfo("Выбор набора", "Выберите набор разметки.")
            return

        data_path = self.labeling_root / labeling_id / "labels.parquet"
        if not data_path.exists():
            messagebox.showerror("Файл не найден", f"Не удалось найти {data_path}")
            return

        try:
            df = pd.read_parquet(data_path).reset_index(drop=True)
        except Exception as exc:  # pragma: no cover - UI helper
            messagebox.showerror("Ошибка чтения", f"Не удалось загрузить {data_path}:\n{exc}")
            return

        df = df.replace([np.inf, -np.inf], np.nan)
        datetime_cols: List[str] = list(df.select_dtypes(include=["datetime64[ns]"]).columns)
        datetime_cols += [col for col in df.select_dtypes(include=["datetimetz"]).columns if col not in datetime_cols]
        for col in datetime_cols:
            df[col] = pd.to_datetime(df[col])
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        self.current_df = df
        self.status_var.set(f"{labeling_id} | строк: {len(df):,}".replace(",", " "))

        self._populate_table()
        self._update_chart()

    # Table ----------------------------------------------------------------------
    def _populate_table(self) -> None:
        for child in self.table_container.winfo_children():
            child.destroy()

        if self.current_df is None or self.current_df.empty:
            ttk.Label(self.table_container, text="Нет данных для отображения").pack(expand=True)
            return

        columns = list(self.current_df.columns)
        tree = ttk.Treeview(self.table_container, columns=columns, show="headings", height=25)
        tree.pack(side="left", fill="both", expand=True)

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, stretch=True, anchor="center")

        y_scroll = ttk.Scrollbar(self.table_container, orient="vertical", command=tree.yview)
        x_scroll = ttk.Scrollbar(self.table_container, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        y_scroll.pack(side="right", fill="y")
        x_scroll.pack(side="bottom", fill="x")

        for _, row in self.current_df.iterrows():
            values = [self._format_value(row[col]) for col in columns]
            tree.insert("", "end", values=values)

    @staticmethod
    def _format_value(value):
        if pd.isna(value):
            return ""
        if isinstance(value, (pd.Timestamp, datetime)):
            return value.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(value, (float, np.floating)):
            return f"{value:.6f}" if abs(value) < 1 else f"{value:.4f}"
        return str(value)

    # Chart ----------------------------------------------------------------------
    def _update_chart(self) -> None:
        self.ax.clear()
        if self.current_df is None or self.current_df.empty:
            self.canvas.draw()
            return

        df = self.current_df
        show_price = self.show_price_var.get()
        show_long = self.show_long_var.get()
        show_short = self.show_short_var.get()
        chart_mode = self.chart_mode_var.get()

        price_col = self._resolve_price_column(df)
        timestamps = df["timestamp"] if "timestamp" in df.columns else df.index
        x_line = timestamps

        if price_col is None:
            self.ax.text(0.5, 0.5, "Колонка цены не найдена", ha="center", va="center")
            self.canvas.draw()
            return
        price = pd.to_numeric(df[price_col], errors="coerce")
        scatter_x = None

        if chart_mode == "candles":
            required_cols = {"open", "high", "low", "close"}
            if not required_cols.issubset(df.columns):
                self.ax.text(
                    0.5,
                    0.5,
                    "Для свечей нужны колонки open/high/low/close",
                    ha="center",
                    va="center",
                    wrap=True,
                )
                self.canvas.draw()
                return
            x_numeric, bar_width = self._prepare_candlestick_x(timestamps)
            scatter_x = x_numeric
            if show_price:
                self._draw_candles(x_numeric, df, bar_width)
        else:
            if show_price:
                self.ax.plot(x_line, price, label=f"Цена ({price_col})", color="#1976d2", linewidth=1.5)
            scatter_x = np.asarray(x_line)

        if "label" in df.columns:
            long_mask = df["label"] == 1
            short_mask = df["label"] == -1
            x_values = np.asarray(scatter_x)

            if show_long and long_mask.any():
                self.ax.scatter(
                    x_values[long_mask.to_numpy()],
                    price[long_mask] * 0.995,
                    marker="^",
                    color="#00c853",
                    s=40,
                    label="Long (1)",
                )
            if show_short and short_mask.any():
                self.ax.scatter(
                    x_values[short_mask.to_numpy()],
                    price[short_mask] * 1.005,
                    marker="v",
                    color="#d50000",
                    s=40,
                    label="Short (-1)",
                )

        self.ax.set_title("Динамика цены и меток")
        self.ax.set_xlabel("Время" if "timestamp" in df.columns else "Индекс")
        self.ax.set_ylabel("Цена")
        self.ax.legend(loc="upper right")
        self.ax.grid(True, which="major", alpha=0.2)
        self.figure.autofmt_xdate()
        self.canvas.draw()

    def _prepare_candlestick_x(self, timestamps) -> Tuple[np.ndarray, float]:
        values = np.asarray(timestamps)
        if np.issubdtype(values.dtype, np.datetime64):
            numeric = mdates.date2num(pd.to_datetime(values))
            self.ax.xaxis_date()
        else:
            numeric = values.astype(float) if np.issubdtype(values.dtype, np.number) else np.arange(len(values))

        if len(numeric) > 1:
            steps = np.diff(numeric)
            step = np.median(steps[steps > 0]) if np.any(steps > 0) else 1.0
            width = step * 0.7
        else:
            width = 0.6
        return numeric, width

    def _draw_candles(self, x_numeric: np.ndarray, df: pd.DataFrame, width: float) -> None:
        opens = pd.to_numeric(df["open"], errors="coerce")
        highs = pd.to_numeric(df["high"], errors="coerce")
        lows = pd.to_numeric(df["low"], errors="coerce")
        closes = pd.to_numeric(df["close"], errors="coerce")

        for xi, o, h, l, c in zip(x_numeric, opens, highs, lows, closes):
            if np.isnan([o, h, l, c]).any():
                continue
            color = "#00c853" if c >= o else "#d50000"
            self.ax.vlines(xi, l, h, color=color, linewidth=0.8, alpha=0.9)
            body_bottom = min(o, c)
            body_height = max(abs(c - o), 1e-4)
            rect = Rectangle(
                (xi - width / 2, body_bottom),
                width,
                body_height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.5,
                alpha=0.8,
            )
            self.ax.add_patch(rect)

    def _resolve_price_column(self, df: pd.DataFrame) -> Optional[str]:
        if self.price_column and self.price_column in df.columns:
            return self.price_column

        preferred = ["close", "price", "typical_price", "open"]
        for col in preferred:
            if col in df.columns:
                return col

        numeric_cols = df.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
        return numeric_cols[0] if len(numeric_cols) > 0 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Локальный просмотрщик результатов разметки")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"), help="Корневая директория артефактов")
    parser.add_argument(
        "--labeling-id", type=str, help="ID набора разметки. Если не указан, выбирается первый доступный."
    )
    parser.add_argument("--price-column", type=str, help="Колонка цены для графика (по умолчанию auto)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    viewer = LabelingViewer(args.artifacts_dir, args.labeling_id, price_column=args.price_column)
    viewer.mainloop()


if __name__ == "__main__":
    main()
