"""Генерация отчётов по оценке моделей."""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not installed. HTML reports will be limited.")

from .calibration import CalibrationMetrics, ModelCalibrator
from .drift_detection import DriftDetector
from .feature_importance import FeatureImportanceAnalyzer
from .metrics import MetricsCalculator


class ModelEvaluationReport:
    """Комплексный отчёт оценки модели."""

    def __init__(
        self,
        model: Any,
        task_type: str = "classification",
        model_name: Optional[str] = None,
    ):
        """
        Инициализация генератора отчётов.

        Args:
            model: Обученная модель
            task_type: Тип задачи ('classification' или 'regression')
            model_name: Название модели
        """
        self.model = model
        self.task_type = task_type
        self.model_name = model_name or type(model).__name__
        self.report_data: Dict[str, Any] = {}

    def generate(
        self,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: np.ndarray,
        X_train: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_train: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        output_path: Optional[Union[str, Path]] = None,
        include_sections: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Сгенерировать полный отчёт.

        Args:
            X_test: Тестовые признаки
            y_test: Тестовые таргеты
            X_train: Тренировочные признаки (для drift detection)
            y_train: Тренировочные таргеты
            feature_names: Названия признаков
            output_path: Путь для сохранения HTML отчёта
            include_sections: Список секций для включения

        Returns:
            Dict с данными отчёта
        """
        if include_sections is None:
            include_sections = ["metrics", "calibration", "importance", "drift"]

        self.report_data = {
            "model_name": self.model_name,
            "task_type": self.task_type,
            "timestamp": datetime.now().isoformat(),
            "n_test_samples": len(y_test),
        }

        # Получаем предсказания
        y_pred = self.model.predict(X_test)

        # Вероятности для классификации
        y_proba = None
        if self.task_type == "classification" and hasattr(self.model, "predict_proba"):
            y_proba = self.model.predict_proba(X_test)

        # 1. Метрики
        if "metrics" in include_sections:
            self.report_data["metrics"] = self._compute_metrics(y_test, y_pred, y_proba)

        # 2. Калибровка (только для классификации)
        if "calibration" in include_sections and self.task_type == "classification":
            if y_proba is not None:
                self.report_data["calibration"] = self._calibration_analysis(y_test, y_proba)

        # 3. Важность признаков
        if "importance" in include_sections:
            self.report_data["feature_importance"] = self._feature_importance(X_test, y_test, feature_names)

        # 4. Drift detection
        if "drift" in include_sections and X_train is not None:
            self.report_data["drift"] = self._drift_analysis(X_train, X_test, feature_names)

        # 5. Визуализации
        self.report_data["visualizations"] = self._generate_plots(y_test, y_pred, y_proba, include_sections)

        # Сохраняем HTML отчёт
        if output_path:
            self._save_html_report(output_path)

        return self.report_data

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Вычислить метрики модели."""
        n_features = None
        if hasattr(self.model, "n_features_in_"):
            n_features = self.model.n_features_in_

        metrics = MetricsCalculator.compute_metrics(
            y_true,
            y_pred,
            task_type=self.task_type,
            y_proba=y_proba,
            n_features=n_features,
        )

        return metrics

    def _calibration_analysis(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """Анализ калибровки модели."""
        # Берем вероятности положительного класса
        if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba

        # Метрики калибровки
        calibration_metrics = CalibrationMetrics.compute_all(y_true, y_proba_pos)

        # Данные для calibration curve
        calibrator = ModelCalibrator(method="isotonic")
        prob_true, prob_pred = calibrator.calibration_curve_data(y_proba_pos, y_true)

        return {
            "metrics": calibration_metrics,
            "curve_data": {
                "prob_true": prob_true.tolist(),
                "prob_pred": prob_pred.tolist(),
            },
        }

    def _feature_importance(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Анализ важности признаков."""
        try:
            analyzer = FeatureImportanceAnalyzer(self.model, X, y, feature_names=feature_names)

            # Вычисляем важность разными методами
            importances = analyzer.compute_all_importances(methods=["tree", "permutation"])

            # Преобразуем DataFrames в dict для JSON
            importance_dict = {}
            for method, df in importances.items():
                if not df.empty:
                    importance_dict[method] = df.to_dict(orient="records")

            return importance_dict

        except Exception as e:
            warnings.warn(f"Failed to compute feature importance: {e}")
            return {}

    def _drift_analysis(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        X_test: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Анализ дрейфа данных."""
        try:
            # Преобразуем в DataFrame если нужно
            if not isinstance(X_train, pd.DataFrame):
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
                X_train = pd.DataFrame(X_train, columns=feature_names)

            if not isinstance(X_test, pd.DataFrame):
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
                X_test = pd.DataFrame(X_test, columns=feature_names)

            detector = DriftDetector(X_train, X_test, feature_names=feature_names)

            # Детекция всеми методами
            drift_results = detector.detect_all(methods=["psi", "ks"])

            # Преобразуем в JSON-serializable формат
            drift_dict: Dict[str, Any] = {}
            for method, result in drift_results.items():
                if isinstance(result, pd.DataFrame):
                    drift_dict[method] = result.to_dict(orient="records")
                else:
                    drift_dict[method] = result

            # Добавляем summary
            drift_dict["summary"] = detector.summary()

            return drift_dict

        except Exception as e:
            warnings.warn(f"Failed to compute drift analysis: {e}")
            return {}

    def _generate_plots(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        include_sections: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Генерировать графики."""
        if not PLOTLY_AVAILABLE:
            return {}

        plots = {}

        if self.task_type == "classification":
            # Confusion Matrix
            plots["confusion_matrix"] = self._plot_confusion_matrix(y_true, y_pred)

            # ROC Curve
            if y_proba is not None and include_sections and "calibration" in include_sections:
                plots["roc_curve"] = self._plot_roc_curve(y_true, y_proba)

                # Calibration Curve
                plots["calibration_curve"] = self._plot_calibration_curve(y_true, y_proba)

        elif self.task_type == "regression":
            # Predicted vs Actual
            plots["predicted_vs_actual"] = self._plot_predicted_vs_actual(y_true, y_pred)

            # Residuals
            plots["residuals"] = self._plot_residuals(y_true, y_pred)

        # Feature Importance
        if "feature_importance" in self.report_data:
            plots["feature_importance"] = self._plot_feature_importance()

        # Drift
        if "drift" in self.report_data and "psi" in self.report_data["drift"]:
            plots["drift_psi"] = self._plot_drift_psi()

        return plots

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Построить confusion matrix."""
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=[f"Predicted {i}" for i in range(len(cm))],
                y=[f"Actual {i}" for i in range(len(cm))],
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
            )
        )

        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
        )

        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def _plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray) -> str:
        """Построить ROC curve."""
        from sklearn.metrics import auc, roc_curve

        # Берем вероятности положительного класса
        if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba

        fpr, tpr, _ = roc_curve(y_true, y_proba_pos)
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC curve (AUC = {roc_auc:.3f})",
                line=dict(color="darkorange", width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random",
                line=dict(color="navy", width=2, dash="dash"),
            )
        )

        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            showlegend=True,
        )

        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def _plot_calibration_curve(self, y_true: np.ndarray, y_proba: np.ndarray) -> str:
        """Построить calibration curve."""
        if "calibration" not in self.report_data:
            return ""

        curve_data = self.report_data["calibration"]["curve_data"]
        prob_true = curve_data["prob_true"]
        prob_pred = curve_data["prob_pred"]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=prob_pred,
                y=prob_true,
                mode="lines+markers",
                name="Calibration curve",
                line=dict(color="darkorange", width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Perfect calibration",
                line=dict(color="navy", width=2, dash="dash"),
            )
        )

        fig.update_layout(
            title="Calibration Curve",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives",
            showlegend=True,
        )

        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def _plot_predicted_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Построить predicted vs actual для регрессии."""
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=y_true,
                y=y_pred,
                mode="markers",
                name="Predictions",
                marker=dict(color="blue", opacity=0.5),
            )
        )

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())

        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Perfect prediction",
                line=dict(color="red", width=2, dash="dash"),
            )
        )

        fig.update_layout(
            title="Predicted vs Actual",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            showlegend=True,
        )

        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def _plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Построить residuals plot."""
        residuals = y_true - y_pred

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode="markers",
                name="Residuals",
                marker=dict(color="blue", opacity=0.5),
            )
        )

        # Zero line
        fig.add_trace(
            go.Scatter(
                x=[y_pred.min(), y_pred.max()],
                y=[0, 0],
                mode="lines",
                name="Zero",
                line=dict(color="red", width=2, dash="dash"),
            )
        )

        fig.update_layout(
            title="Residuals Plot",
            xaxis_title="Predicted Values",
            yaxis_title="Residuals",
            showlegend=True,
        )

        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def _plot_feature_importance(self) -> str:
        """Построить feature importance."""
        if "feature_importance" not in self.report_data:
            return ""

        importance_data = self.report_data["feature_importance"]

        # Используем первый доступный метод
        method = None
        df_data = None

        for m in ["tree", "permutation", "shap"]:
            if m in importance_data and importance_data[m]:
                method = m
                df_data = pd.DataFrame(importance_data[m])
                break

        if df_data is None or df_data.empty:
            return ""

        # Топ 20 признаков
        df_data = df_data.head(20)

        # Определяем колонку с важностью
        importance_col = None
        for col in ["importance", "importance_mean", "shap_importance"]:
            if col in df_data.columns:
                importance_col = col
                break

        if importance_col is None:
            return ""

        fig = go.Figure(
            go.Bar(
                x=df_data[importance_col],
                y=df_data["feature"],
                orientation="h",
            )
        )

        fig.update_layout(
            title=f"Feature Importance ({method})",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=max(400, len(df_data) * 20),
        )

        # Reverse y-axis для удобства
        fig.update_yaxes(autorange="reversed")

        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def _plot_drift_psi(self) -> str:
        """Построить PSI drift plot."""
        if "drift" not in self.report_data or "psi" not in self.report_data["drift"]:
            return ""

        psi_data = pd.DataFrame(self.report_data["drift"]["psi"])

        if psi_data.empty:
            return ""

        # Топ 20 признаков с наибольшим дрейфом
        psi_data = psi_data.head(20)

        # Цвета по статусу
        color_map = {
            "no_drift": "green",
            "minor_drift": "orange",
            "major_drift": "red",
        }
        colors = psi_data["status"].map(color_map)

        fig = go.Figure(
            go.Bar(
                x=psi_data["psi"],
                y=psi_data["feature"],
                orientation="h",
                marker=dict(color=colors),
            )
        )

        # Добавляем линии порогов
        fig.add_vline(x=0.1, line_dash="dash", line_color="orange", annotation_text="Minor drift")
        fig.add_vline(x=0.2, line_dash="dash", line_color="red", annotation_text="Major drift")

        fig.update_layout(
            title="Population Stability Index (PSI)",
            xaxis_title="PSI",
            yaxis_title="Feature",
            height=max(400, len(psi_data) * 20),
        )

        # Reverse y-axis
        fig.update_yaxes(autorange="reversed")

        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def _save_html_report(self, output_path: Union[str, Path]) -> None:
        """Сохранить HTML отчёт."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        html_content = self._generate_html()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"Report saved to: {output_path}")

    def _generate_html(self) -> str:
        """Сгенерировать HTML контент."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Model Evaluation Report - {self.model_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 5px;
        }}
        .info-box {{
            background-color: #e8f5e9;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .metrics-table th, .metrics-table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        .metrics-table th {{
            background-color: #4CAF50;
            color: white;
        }}
        .metrics-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .plot-container {{
            margin: 20px 0;
        }}
        .timestamp {{
            color: #888;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Model Evaluation Report</h1>

        <div class="info-box">
            <strong>Model:</strong> {self.model_name}<br>
            <strong>Task Type:</strong> {self.task_type}<br>
            <strong>Test Samples:</strong> {self.report_data.get('n_test_samples', 'N/A')}<br>
            <span class="timestamp">Generated: {self.report_data.get('timestamp', '')}</span>
        </div>
"""

        # Метрики
        if "metrics" in self.report_data:
            html += self._metrics_html()

        # Калибровка
        if "calibration" in self.report_data:
            html += self._calibration_html()

        # Визуализации
        if "visualizations" in self.report_data:
            html += self._visualizations_html()

        # Feature Importance
        if "feature_importance" in self.report_data:
            html += self._feature_importance_html()

        # Drift
        if "drift" in self.report_data:
            html += self._drift_html()

        html += """
    </div>
</body>
</html>
"""

        return html

    def _metrics_html(self) -> str:
        """HTML для метрик."""
        metrics = self.report_data["metrics"]

        # Фильтруем только скалярные метрики
        scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float, np.integer, np.floating))}

        html = "<h2>Metrics</h2>\n"
        html += '<table class="metrics-table">\n'
        html += "<tr><th>Metric</th><th>Value</th></tr>\n"

        for metric, value in scalar_metrics.items():
            html += f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>\n"

        html += "</table>\n"

        return html

    def _calibration_html(self) -> str:
        """HTML для калибровки."""
        calibration = self.report_data["calibration"]
        metrics = calibration.get("metrics", {})

        html = "<h2>Calibration</h2>\n"
        html += '<table class="metrics-table">\n'
        html += "<tr><th>Metric</th><th>Value</th></tr>\n"

        for metric, value in metrics.items():
            html += f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>\n"

        html += "</table>\n"

        return html

    def _visualizations_html(self) -> str:
        """HTML для визуализаций."""
        plots = self.report_data["visualizations"]

        if not plots:
            return ""

        html = "<h2>Visualizations</h2>\n"

        for plot_name, plot_html in plots.items():
            if plot_html:
                html += f'<div class="plot-container">\n{plot_html}\n</div>\n'

        return html

    def _feature_importance_html(self) -> str:
        """HTML для важности признаков."""
        return ""  # Визуализация уже в plots

    def _drift_html(self) -> str:
        """HTML для дрейфа."""
        drift = self.report_data["drift"]

        if "summary" not in drift:
            return ""

        summary = drift["summary"]

        html = "<h2>Drift Analysis</h2>\n"
        html += '<table class="metrics-table">\n'
        html += "<tr><th>Metric</th><th>Value</th></tr>\n"

        for metric, value in summary.items():
            html += f"<tr><td>{metric}</td><td>{value}</td></tr>\n"

        html += "</table>\n"

        return html

    def save_json(self, output_path: Union[str, Path]) -> None:
        """
        Сохранить отчёт в JSON формате.

        Args:
            output_path: Путь для сохранения
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Удаляем HTML визуализации из JSON
        report_json = self.report_data.copy()
        if "visualizations" in report_json:
            del report_json["visualizations"]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_json, f, indent=2, default=str)

        print(f"JSON report saved to: {output_path}")
