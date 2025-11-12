"""Калибровка вероятностей моделей."""

import warnings
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class BetaCalibrator:
    """Калибровка с использованием бета-распределения."""

    def __init__(self):
        """Инициализация калибратора."""
        self.params: Optional[Dict[str, float]] = None

    def fit(self, y_proba: np.ndarray, y_true: np.ndarray) -> "BetaCalibrator":
        """
        Обучить калибратор.

        Args:
            y_proba: Некалиброванные вероятности
            y_true: Истинные метки

        Returns:
            self
        """
        # Преобразуем вероятности чтобы избежать 0 и 1
        eps = 1e-15
        y_proba = np.clip(y_proba, eps, 1 - eps)

        # Оптимизируем параметры a, b, c
        def loss(params):
            a, b, c = params
            calibrated = self._transform(y_proba, a, b, c)
            # Log loss
            return -np.mean(y_true * np.log(calibrated) + (1 - y_true) * np.log(1 - calibrated))

        # Начальные значения
        x0 = [1.0, 0.0, 1.0]

        # Оптимизация
        result = minimize(loss, x0, method="Nelder-Mead")

        self.params = {"a": result.x[0], "b": result.x[1], "c": result.x[2]}
        return self

    def transform(self, y_proba: np.ndarray) -> np.ndarray:
        """
        Калибровать вероятности.

        Args:
            y_proba: Некалиброванные вероятности

        Returns:
            Калиброванные вероятности
        """
        if self.params is None:
            raise ValueError("Calibrator must be fitted before transform")

        return self._transform(y_proba, self.params["a"], self.params["b"], self.params["c"])

    @staticmethod
    def _transform(y_proba: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """Применить бета-трансформацию."""
        eps = 1e-15
        y_proba = np.clip(y_proba, eps, 1 - eps)
        calibrated = 1 / (1 + np.exp(-(a * np.log(y_proba / (1 - y_proba)) + b))) ** c
        return np.clip(calibrated, eps, 1 - eps)


class ModelCalibrator:
    """Калибровка вероятностей моделей."""

    METHODS = ["isotonic", "platt", "beta"]

    def __init__(
        self,
        method: Literal["isotonic", "platt", "beta"] = "isotonic",
        n_bins: int = 10,
    ):
        """
        Инициализация калибратора.

        Args:
            method: Метод калибровки ('isotonic', 'platt', 'beta')
            n_bins: Количество bins для оценки калибровки
        """
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Available: {self.METHODS}")

        self.method = method
        self.n_bins = n_bins
        self.calibrator: Optional[Any] = None

    def fit(self, y_proba: np.ndarray, y_true: np.ndarray) -> "ModelCalibrator":
        """
        Обучить калибратор.

        Args:
            y_proba: Некалиброванные вероятности
            y_true: Истинные метки

        Returns:
            self
        """
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(y_proba, y_true)

        elif self.method == "platt":
            # Platt scaling - логистическая регрессия
            self.calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
            self.calibrator.fit(y_proba.reshape(-1, 1), y_true)

        elif self.method == "beta":
            self.calibrator = BetaCalibrator()
            self.calibrator.fit(y_proba, y_true)

        return self

    def transform(self, y_proba: np.ndarray) -> np.ndarray:
        """
        Калибровать вероятности.

        Args:
            y_proba: Некалиброванные вероятности

        Returns:
            Калиброванные вероятности
        """
        if self.calibrator is None:
            raise ValueError("Calibrator must be fitted before transform")

        if self.method == "isotonic":
            return self.calibrator.predict(y_proba)

        elif self.method == "platt":
            return self.calibrator.predict_proba(y_proba.reshape(-1, 1))[:, 1]

        elif self.method == "beta":
            return self.calibrator.transform(y_proba)

        return y_proba

    def fit_transform(self, y_proba: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Обучить и применить калибровку.

        Args:
            y_proba: Некалиброванные вероятности
            y_true: Истинные метки

        Returns:
            Калиброванные вероятности
        """
        return self.fit(y_proba, y_true).transform(y_proba)

    def calibration_curve_data(self, y_proba: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Получить данные для calibration curve.

        Args:
            y_proba: Вероятности
            y_true: Истинные метки

        Returns:
            Tuple (prob_true, prob_pred) для построения графика
        """
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=self.n_bins, strategy="uniform")
        return prob_true, prob_pred


class CalibrationMetrics:
    """Метрики для оценки качества калибровки."""

    @staticmethod
    def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """
        Brier Score - средний квадрат разности между вероятностями и реальностью.

        Args:
            y_true: Истинные метки
            y_proba: Предсказанные вероятности

        Returns:
            Brier score (чем меньше, тем лучше)
        """
        return float(np.mean((y_proba - y_true) ** 2))

    @staticmethod
    def expected_calibration_error(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
        """
        Expected Calibration Error (ECE) - средневзвешенная разница между
        уверенностью и точностью в каждом bin.

        Args:
            y_true: Истинные метки
            y_proba: Предсказанные вероятности
            n_bins: Количество bins

        Returns:
            ECE (чем меньше, тем лучше)
        """
        # Создаем bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Находим примеры в текущем bin
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                # Средняя уверенность в bin
                avg_confidence_in_bin = np.mean(y_proba[in_bin])
                # Средняя точность в bin
                avg_accuracy_in_bin = np.mean(y_true[in_bin])
                # Добавляем взвешенный вклад
                ece += np.abs(avg_confidence_in_bin - avg_accuracy_in_bin) * prop_in_bin

        return float(ece)

    @staticmethod
    def maximum_calibration_error(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
        """
        Maximum Calibration Error (MCE) - максимальная разница между
        уверенностью и точностью среди всех bins.

        Args:
            y_true: Истинные метки
            y_proba: Предсказанные вероятности
            n_bins: Количество bins

        Returns:
            MCE (чем меньше, тем лучше)
        """
        # Создаем bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        max_error = 0.0

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Находим примеры в текущем bin
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)

            if np.sum(in_bin) > 0:
                # Средняя уверенность в bin
                avg_confidence_in_bin = np.mean(y_proba[in_bin])
                # Средняя точность в bin
                avg_accuracy_in_bin = np.mean(y_true[in_bin])
                # Обновляем максимальную ошибку
                error = np.abs(avg_confidence_in_bin - avg_accuracy_in_bin)
                max_error = max(max_error, error)

        return float(max_error)

    @staticmethod
    def compute_all(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
        """
        Вычислить все метрики калибровки.

        Args:
            y_true: Истинные метки
            y_proba: Предсказанные вероятности
            n_bins: Количество bins для ECE и MCE

        Returns:
            Dict с метриками калибровки
        """
        return {
            "brier_score": CalibrationMetrics.brier_score(y_true, y_proba),
            "ece": CalibrationMetrics.expected_calibration_error(y_true, y_proba, n_bins),
            "mce": CalibrationMetrics.maximum_calibration_error(y_true, y_proba, n_bins),
        }


def compare_calibration_methods(
    y_proba: np.ndarray, y_true: np.ndarray, methods: Optional[list] = None
) -> Dict[str, Dict[str, float]]:
    """
    Сравнить различные методы калибровки.

    Args:
        y_proba: Некалиброванные вероятности
        y_true: Истинные метки
        methods: Список методов для сравнения (по умолчанию все)

    Returns:
        Dict с результатами для каждого метода
    """
    if methods is None:
        methods = ModelCalibrator.METHODS

    results = {}

    # Метрики до калибровки
    results["uncalibrated"] = CalibrationMetrics.compute_all(y_true, y_proba)

    # Калибровка разными методами
    for method in methods:
        try:
            # method проверяется на валидность в ModelCalibrator
            calibrator = ModelCalibrator(method=method)  # type: ignore[arg-type]
            y_proba_calibrated = calibrator.fit_transform(y_proba, y_true)
            results[method] = CalibrationMetrics.compute_all(y_true, y_proba_calibrated)
        except Exception as e:
            warnings.warn(f"Failed to calibrate with method {method}: {e}")
            results[method] = {"error": str(e)}  # type: ignore[dict-item]

    return results
