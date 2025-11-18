"""Метрики для оценки моделей."""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


class ClassificationMetrics:
    """Метрики для задач классификации."""

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Accuracy - доля правильных предсказаний."""
        return float(accuracy_score(y_true, y_pred))

    @staticmethod
    def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Balanced Accuracy - средняя чувствительность по классам."""
        return float(balanced_accuracy_score(y_true, y_pred))

    @staticmethod
    def precision(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = "binary",
        zero_division: float = 0.0,
    ) -> float:
        """
        Precision - доля истинно положительных среди всех положительных предсказаний.

        Args:
            y_true: Истинные метки
            y_pred: Предсказанные метки
            average: Способ усреднения ('binary', 'micro', 'macro', 'weighted')
            zero_division: Значение при делении на ноль
        """
        return float(precision_score(y_true, y_pred, average=average, zero_division=zero_division))

    @staticmethod
    def recall(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = "binary",
        zero_division: float = 0.0,
    ) -> float:
        """
        Recall - доля найденных положительных примеров.

        Args:
            y_true: Истинные метки
            y_pred: Предсказанные метки
            average: Способ усреднения ('binary', 'micro', 'macro', 'weighted')
            zero_division: Значение при делении на ноль
        """
        return float(recall_score(y_true, y_pred, average=average, zero_division=zero_division))

    @staticmethod
    def f1(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = "binary",
        zero_division: float = 0.0,
    ) -> float:
        """
        F1-score - гармоническое среднее precision и recall.

        Args:
            y_true: Истинные метки
            y_pred: Предсказанные метки
            average: Способ усреднения ('binary', 'micro', 'macro', 'weighted')
            zero_division: Значение при делении на ноль
        """
        return float(f1_score(y_true, y_pred, average=average, zero_division=zero_division))

    @staticmethod
    def roc_auc(y_true: np.ndarray, y_proba: np.ndarray, multi_class: str = "ovr") -> float:
        """
        ROC AUC - площадь под ROC-кривой.

        Args:
            y_true: Истинные метки
            y_proba: Вероятности классов
            multi_class: Стратегия для мультиклассовой классификации ('ovr', 'ovo')
        """
        try:
            if len(y_proba.shape) == 1 or y_proba.shape[1] == 1:
                # Бинарная классификация
                return float(roc_auc_score(y_true, y_proba))
            else:
                # Мультиклассовая
                return float(roc_auc_score(y_true, y_proba, multi_class=multi_class))
        except ValueError:
            # Если только один класс в y_true
            return 0.5

    @staticmethod
    def pr_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """
        PR AUC - площадь под Precision-Recall кривой.

        Args:
            y_true: Истинные метки
            y_proba: Вероятности положительного класса
        """
        try:
            return float(average_precision_score(y_true, y_proba))
        except ValueError:
            return 0.0

    @staticmethod
    def mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Matthews Correlation Coefficient - корреляция между предсказаниями и реальностью.

        Значения от -1 (полная противоположность) до +1 (идеальное предсказание).
        """
        return float(matthews_corrcoef(y_true, y_pred))

    @staticmethod
    def confusion_matrix_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Union[np.ndarray, int]]:
        """
        Confusion Matrix и основные компоненты.

        Returns:
            Dict с ключами: matrix, TP, TN, FP, FN
        """
        cm = confusion_matrix(y_true, y_pred)

        # Для бинарной классификации извлекаем компоненты
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return {
                "matrix": cm,
                "TP": int(tp),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
            }
        else:
            return {"matrix": cm}

    @staticmethod
    def classification_report_dict(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[list] = None,
        zero_division: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Подробный отчёт по классификации.

        Returns:
            Dict с метриками для каждого класса
        """
        report = classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            output_dict=True,
            zero_division=zero_division,
        )
        return report

    @staticmethod
    def compute_all(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        average: str = "binary",
        multi_class: str = "ovr",
    ) -> Dict[str, Any]:
        """
        Вычислить все метрики классификации.

        Args:
            y_true: Истинные метки
            y_pred: Предсказанные метки
            y_proba: Вероятности (опционально, для ROC-AUC, PR-AUC)
            average: Способ усреднения для мультиклассовой классификации
            multi_class: Стратегия для мультиклассовой классификации

        Returns:
            Dict со всеми метриками
        """
        metrics = {
            "accuracy": ClassificationMetrics.accuracy(y_true, y_pred),
            "balanced_accuracy": ClassificationMetrics.balanced_accuracy(y_true, y_pred),
            "precision": ClassificationMetrics.precision(y_true, y_pred, average=average),
            "recall": ClassificationMetrics.recall(y_true, y_pred, average=average),
            "f1": ClassificationMetrics.f1(y_true, y_pred, average=average),
            "mcc": ClassificationMetrics.mcc(y_true, y_pred),
        }

        # Confusion matrix
        cm_dict = ClassificationMetrics.confusion_matrix_dict(y_true, y_pred)
        # Обновляем только скалярные значения
        for key, value in cm_dict.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                metrics[key] = float(value)
            elif key == "matrix":
                metrics[key] = value  # type: ignore[assignment]

        # Метрики, требующие вероятностей
        if y_proba is not None:
            if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                # Берем вероятности положительного класса
                y_proba_pos = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba
            else:
                y_proba_pos = y_proba

            metrics["roc_auc"] = ClassificationMetrics.roc_auc(y_true, y_proba_pos, multi_class=multi_class)

            # PR-AUC только для бинарной классификации
            if len(np.unique(y_true)) == 2:
                if len(y_proba_pos.shape) > 1:
                    y_proba_pos = y_proba_pos[:, 1] if y_proba_pos.shape[1] >= 2 else y_proba_pos[:, 0]
                metrics["pr_auc"] = ClassificationMetrics.pr_auc(y_true, y_proba_pos)

        return metrics


class RegressionMetrics:
    """Метрики для задач регрессии."""

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error - средняя квадратичная ошибка."""
        return float(mean_squared_error(y_true, y_pred))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error - корень из средней квадратичной ошибки."""
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error - средняя абсолютная ошибка."""
        return float(mean_absolute_error(y_true, y_pred))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
        """
        Mean Absolute Percentage Error - средняя абсолютная процентная ошибка.

        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения
            epsilon: Малое значение для избежания деления на ноль
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100)

    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R² - коэффициент детерминации."""
        return float(r2_score(y_true, y_pred))

    @staticmethod
    def adjusted_r2(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
        """
        Adjusted R² - скорректированный коэффициент детерминации.

        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения
            n_features: Количество признаков в модели
        """
        n = len(y_true)
        r2 = RegressionMetrics.r2(y_true, y_pred)

        if n <= n_features + 1:
            return r2

        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        return float(adj_r2)

    @staticmethod
    def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.5) -> float:
        """
        Quantile Loss - функция потерь для квантильной регрессии.

        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения
            quantile: Квантиль (от 0 до 1)
        """
        errors = y_true - y_pred
        loss = np.maximum(quantile * errors, (quantile - 1) * errors)
        return float(np.mean(loss))

    @staticmethod
    def compute_all(y_true: np.ndarray, y_pred: np.ndarray, n_features: Optional[int] = None) -> Dict[str, float]:
        """
        Вычислить все метрики регрессии.

        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения
            n_features: Количество признаков (для adjusted R²)

        Returns:
            Dict со всеми метриками
        """
        metrics = {
            "mse": RegressionMetrics.mse(y_true, y_pred),
            "rmse": RegressionMetrics.rmse(y_true, y_pred),
            "mae": RegressionMetrics.mae(y_true, y_pred),
            "mape": RegressionMetrics.mape(y_true, y_pred),
            "r2": RegressionMetrics.r2(y_true, y_pred),
        }

        if n_features is not None:
            metrics["adjusted_r2"] = RegressionMetrics.adjusted_r2(y_true, y_pred, n_features)

        # Квантильные потери
        for q in [0.25, 0.5, 0.75]:
            metrics[f"quantile_loss_{q}"] = RegressionMetrics.quantile_loss(y_true, y_pred, quantile=q)

        return metrics


class MetricsCalculator:
    """Универсальный калькулятор метрик."""

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: str = "classification",
        y_proba: Optional[np.ndarray] = None,
        n_features: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Вычислить метрики для указанного типа задачи.

        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения
            task_type: Тип задачи ('classification' или 'regression')
            y_proba: Вероятности (для классификации)
            n_features: Количество признаков (для регрессии)
            **kwargs: Дополнительные параметры

        Returns:
            Dict с метриками
        """
        if task_type == "classification":
            return ClassificationMetrics.compute_all(y_true, y_pred, y_proba=y_proba, **kwargs)
        elif task_type == "regression":
            return RegressionMetrics.compute_all(y_true, y_pred, n_features=n_features)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    @staticmethod
    def metrics_to_dataframe(metrics: Dict[str, Any]) -> pd.DataFrame:
        """
        Преобразовать метрики в DataFrame для удобного просмотра.

        Args:
            metrics: Dict с метриками

        Returns:
            DataFrame с метриками
        """
        # Фильтруем только скалярные значения
        scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float, np.integer, np.floating))}

        df = pd.DataFrame([scalar_metrics]).T
        df.columns = ["value"]
        df.index.name = "metric"
        return df
