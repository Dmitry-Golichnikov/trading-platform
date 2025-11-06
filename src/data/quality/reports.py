"""
Генерация отчётов о качестве данных.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.cleaners.missing_data import GapDetector
from src.data.quality.metrics import DataQualityMetrics

logger = logging.getLogger(__name__)


class QualityReport:
    """
    Генератор отчётов о качестве данных.

    Создаёт детальные отчёты в форматах JSON и HTML.
    """

    def __init__(self) -> None:
        """Инициализировать генератор отчётов."""
        self.metrics_calculator = DataQualityMetrics()
        self.gap_detector = GapDetector()

    def generate_report(self, data: pd.DataFrame, output_path: Path, format: str = "json") -> None:
        """
        Сгенерировать отчёт о качестве.

        Args:
            data: Данные для анализа
            output_path: Путь для сохранения отчёта
            format: Формат отчёта ('json' или 'html')
        """
        logger.info(f"Generating quality report for {len(data)} rows")

        report_data = self._collect_report_data(data)

        if format == "json":
            self._save_json_report(report_data, output_path)
        elif format == "html":
            self._save_html_report(report_data, output_path)
        else:
            raise ValueError(f"Unknown report format: {format}")

        logger.info(f"Quality report saved to {output_path}")

    def _collect_report_data(self, data: pd.DataFrame) -> dict[str, Any]:
        """Собрать данные для отчёта."""
        # Основная статистика
        basic_stats = {
            "total_rows": len(data),
            "total_columns": len(data.columns),
            "date_range": self._get_date_range(data),
            "ticker": (data["ticker"].iloc[0] if "ticker" in data.columns else None),
        }

        # Метрики качества
        quality_metrics = self.metrics_calculator.get_all_metrics(data)

        # Анализ пропусков
        gap_analysis = {}
        if "timestamp" in data.columns:
            gap_analysis = self.gap_detector.analyze_gaps(data)

        # Описательная статистика
        descriptive_stats = {}
        numeric_cols = data.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            descriptive_stats[col] = {
                "mean": float(data[col].mean()),
                "std": float(data[col].std()),
                "min": float(data[col].min()),
                "max": float(data[col].max()),
                "median": float(data[col].median()),
            }

        # Пропущенные значения
        missing_values = data.isna().sum().to_dict()

        return {
            "generated_at": datetime.now().isoformat(),
            "basic_stats": basic_stats,
            "quality_metrics": quality_metrics,
            "gap_analysis": gap_analysis,
            "descriptive_stats": descriptive_stats,
            "missing_values": {k: int(v) for k, v in missing_values.items() if v > 0},
        }

    def _get_date_range(self, data: pd.DataFrame) -> dict[str, str] | None:
        """Получить диапазон дат."""
        if "timestamp" not in data.columns:
            return None

        timestamps = pd.to_datetime(data["timestamp"])
        return {"start": str(timestamps.min()), "end": str(timestamps.max())}

    def _save_json_report(self, report_data: dict[str, Any], output_path: Path) -> None:
        """Сохранить отчёт в JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                report_data,
                f,
                indent=2,
                ensure_ascii=False,
                default=self._json_default,
            )

    @staticmethod
    def _json_default(obj: Any) -> Any:
        """Сериализация нестандартных типов для JSON."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        return str(obj)

    def _save_html_report(self, report_data: dict[str, Any], output_path: Path) -> None:
        """Сохранить отчёт в HTML."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        html_content = self._generate_html(report_data)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    def _generate_html(self, report_data: dict[str, Any]) -> str:
        """Генерировать HTML контент."""
        basic_stats = report_data["basic_stats"]
        metrics = report_data["quality_metrics"]
        gap_html = self._format_gap_analysis(report_data.get("gap_analysis", {}))
        missing_html = self._format_missing_values(report_data.get("missing_values", {}))

        # Базовый HTML template
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Data Quality Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .metric {{ background-color: #f2f2f2; padding: 10px; margin: 10px 0; }}
        .score {{ font-size: 24px; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Data Quality Report</h1>
    <p>Generated: {report_data['generated_at']}</p>

    <h2>Basic Statistics</h2>
    <div class="metric">
        <strong>Total Rows:</strong> {basic_stats['total_rows']}<br>
        <strong>Total Columns:</strong> {basic_stats['total_columns']}<br>
        {self._format_date_range(basic_stats.get('date_range'))}
    </div>

    <h2>Quality Metrics</h2>
    <div class="metric">
        <div class="score">Overall Score: {metrics['quality_score']}/100</div>
        <table>
            <tr>
                <th>Metric</th>
                <th>Score</th>
            </tr>
            <tr>
                <td>Completeness</td>
                <td>{metrics['completeness']}%</td>
            </tr>
            <tr>
                <td>Validity</td>
                <td>{metrics['validity']}%</td>
            </tr>
            <tr>
                <td>Consistency</td>
                <td>{metrics['consistency']}%</td>
            </tr>
            <tr>
                <td>Uniqueness</td>
                <td>{metrics['uniqueness']}%</td>
            </tr>
        </table>
    </div>

    {gap_html}
    {missing_html}

</body>
</html>
"""
        return html

    def _format_date_range(self, date_range: dict | None) -> str:
        """Форматировать диапазон дат."""
        if not date_range:
            return ""

        start = date_range["start"]
        end = date_range["end"]
        return "<strong>Date Range:</strong> " f"{start} to {end}"

    def _format_gap_analysis(self, gap_analysis: dict) -> str:
        """Форматировать анализ пропусков."""
        if not gap_analysis or gap_analysis.get("total_gaps", 0) == 0:
            return "<h2>Gap Analysis</h2><p>No gaps detected</p>"

        total_gaps = gap_analysis["total_gaps"]
        total_missing = gap_analysis["total_missing_bars"]
        max_gap = gap_analysis["max_gap_size"]
        avg_gap = gap_analysis["avg_gap_size"]

        return (
            "<h2>Gap Analysis</h2>\n"
            '<div class="metric">\n'
            f"    <strong>Total Gaps:</strong> {total_gaps}<br>\n"
            f"    <strong>Total Missing Bars:</strong> {total_missing}<br>\n"
            f"    <strong>Max Gap Size:</strong> {max_gap} bars<br>\n"
            f"    <strong>Average Gap Size:</strong> {avg_gap:.2f} bars\n"
            "</div>"
        )

    def _format_missing_values(self, missing_values: dict) -> str:
        """Форматировать информацию о пропущенных значениях."""
        if not missing_values:
            return "<h2>Missing Values</h2><p>No missing values</p>"

        rows = "".join([f"<tr><td>{col}</td><td>{count}</td></tr>" for col, count in missing_values.items()])

        return (
            "<h2>Missing Values</h2>\n"
            "<table>\n"
            "    <tr>\n"
            "        <th>Column</th>\n"
            "        <th>Missing Count</th>\n"
            "    </tr>\n"
            f"    {rows}\n"
            "</table>"
        )


class ComparisonReport:
    """Сравнение качества нескольких датасетов."""

    def __init__(self) -> None:
        """Инициализировать генератор сравнений."""
        self.metrics_calculator = DataQualityMetrics()

    def compare_datasets(self, datasets: dict[str, pd.DataFrame], output_path: Path) -> None:
        """
        Сравнить датасеты.

        Args:
            datasets: Словарь {имя: DataFrame}
            output_path: Путь для сохранения отчёта
        """
        logger.info(f"Comparing {len(datasets)} datasets")

        comparison_data = {}
        for name, data in datasets.items():
            comparison_data[name] = self.metrics_calculator.get_all_metrics(data)

        # Сохранить как JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_at": datetime.now().isoformat(),
                    "datasets": comparison_data,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(f"Comparison report saved to {output_path}")
