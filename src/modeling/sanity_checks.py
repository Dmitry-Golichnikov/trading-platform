"""
Sanity Checks перед обучением.

Проверки данных и конфигурации для предотвращения типичных проблем.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SanityCheckResult:
    """
    Результат проверки.

    Attributes:
        passed: Проверка пройдена
        warnings: Список предупреждений
        errors: Список ошибок
        info: Дополнительная информация
    """

    passed: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    info: Dict[str, "Any"] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Строковое представление."""
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"SanityCheck({status}, "
            f"warnings={len(self.warnings)}, "
            f"errors={len(self.errors)})"
        )

    def summary(self) -> str:
        """Полная сводка по проверке."""
        lines = [
            "=" * 60,
            f"Sanity Check Result: {'PASSED ✓' if self.passed else 'FAILED ✗'}",
            "=" * 60,
        ]

        if self.errors:
            lines.append("\nОШИБКИ:")
            for err in self.errors:
                lines.append(f"  ✗ {err}")

        if self.warnings:
            lines.append("\nПРЕДУПРЕЖДЕНИЯ:")
            for warn in self.warnings:
                lines.append(f"  ⚠ {warn}")

        if self.info:
            lines.append("\nИНФОРМАЦИЯ:")
            for key, value in self.info.items():
                lines.append(f"  • {key}: {value}")

        lines.append("=" * 60)
        return "\n".join(lines)


class ModelSanityChecker:
    """
    Проверки перед обучением модели.

    Включает проверки:
    - Распределение таргета
    - Look-ahead bias
    - Data leakage
    - Качество признаков
    """

    def __init__(self, verbose: bool = True):
        """
        Args:
            verbose: Выводить результаты проверок в лог
        """
        self.verbose = verbose

    def check_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> SanityCheckResult:
        """
        Выполнить все проверки.

        Args:
            X_train: Train признаки
            y_train: Train таргет
            X_val: Validation признаки (опционально)
            y_val: Validation таргет (опционально)

        Returns:
            Результат всех проверок
        """
        result = SanityCheckResult()

        # Проверка таргета
        target_check = self.check_target_distribution(y_train)
        result.warnings.extend(target_check.warnings)
        result.errors.extend(target_check.errors)
        result.info.update(target_check.info)

        # Проверка признаков
        feature_check = self.check_feature_quality(X_train)
        result.warnings.extend(feature_check.warnings)
        result.errors.extend(feature_check.errors)
        result.info.update(feature_check.info)

        # Проверка data leakage
        if X_val is not None and y_val is not None:
            leakage_check = self.check_data_leakage(X_train, X_val, y_train, y_val)
            result.warnings.extend(leakage_check.warnings)
            result.errors.extend(leakage_check.errors)
            result.info.update(leakage_check.info)

        # Определяем passed
        result.passed = len(result.errors) == 0

        if self.verbose:
            logger.info(result.summary())

        return result

    def check_target_distribution(self, y: pd.Series) -> SanityCheckResult:
        """
        Проверить распределение таргета.

        Args:
            y: Таргет

        Returns:
            Результат проверки
        """
        result = SanityCheckResult()

        # Базовая статистика
        n_samples = len(y)
        n_unique = y.nunique()
        n_missing = y.isna().sum()

        result.info["target_samples"] = n_samples
        result.info["target_unique_values"] = n_unique
        result.info["target_missing"] = n_missing

        # Проверка на пустоту
        if n_samples == 0:
            result.errors.append("Таргет пустой (0 примеров)")
            result.passed = False
            return result

        # Проверка на слишком мало примеров
        if n_samples < 100:
            result.warnings.append(
                f"Слишком мало примеров для обучения: {n_samples} < 100"
            )

        # Проверка пропусков
        if n_missing > 0:
            missing_pct = n_missing / n_samples * 100
            result.errors.append(
                f"Таргет содержит пропуски: {n_missing} ({missing_pct:.1f}%)"
            )
            result.passed = False

        # Проверка на constant target
        if n_unique == 1:
            result.errors.append(
                "Таргет константный (все значения одинаковые). Обучение невозможно."
            )
            result.passed = False
            return result

        # Для классификации: проверка баланса классов
        if n_unique <= 10:  # Предполагаем классификацию
            value_counts = y.value_counts()
            class_distribution = (value_counts / n_samples * 100).to_dict()
            result.info["class_distribution"] = class_distribution

            # Проверка дисбаланса
            min_class_pct = value_counts.min() / n_samples * 100
            if min_class_pct < 5:
                result.warnings.append(
                    "Сильный дисбаланс классов: минимальный класс = "
                    f"{min_class_pct:.2f}%. Рекомендуется использовать "
                    "class_weight или resampling."
                )

            # Проверка на слишком мало примеров минимального класса
            min_class_samples = value_counts.min()
            if min_class_samples < 10:
                result.warnings.append(
                    f"Слишком мало примеров минимального класса: {min_class_samples}. "
                    f"Рекомендуется минимум 10-20 примеров на класс."
                )

        return result

    def check_feature_quality(self, X: pd.DataFrame) -> SanityCheckResult:
        """
        Проверить качество признаков.

        Args:
            X: Признаки

        Returns:
            Результат проверки
        """
        result = SanityCheckResult()

        n_samples, n_features = X.shape
        result.info["n_samples"] = n_samples
        result.info["n_features"] = n_features

        # Проверка на пустоту
        if n_samples == 0 or n_features == 0:
            result.errors.append(f"Данные пустые: shape={X.shape}")
            result.passed = False
            return result

        # Проверка пропусков
        missing_per_col = X.isna().sum()
        high_missing_cols = missing_per_col[missing_per_col / n_samples > 0.5]

        if len(high_missing_cols) > 0:
            result.warnings.append(
                f"Признаки с >50% пропусков: {list(high_missing_cols.index)}"
            )

        result.info["features_with_missing"] = (missing_per_col > 0).sum()

        # Проверка на constant features
        constant_features = []
        for col in X.columns:
            if X[col].nunique() == 1:
                constant_features.append(col)

        if constant_features:
            result.warnings.append(
                f"Константные признаки (удалить): {constant_features[:10]}"
                + ("..." if len(constant_features) > 10 else "")
            )
            result.info["constant_features"] = len(constant_features)

        # Проверка на дубликаты признаков
        # (это дорогая операция, делаем только если признаков немного)
        if n_features < 100:
            duplicates = self._find_duplicate_features(X)
            if duplicates:
                result.warnings.append(
                    f"Найдено {len(duplicates)} дубликатов признаков"
                )

        # Проверка на бесконечные значения
        inf_cols = []
        for col in X.select_dtypes(include=[np.number]).columns:
            if np.isinf(X[col]).any():
                inf_cols.append(col)

        if inf_cols:
            result.errors.append(f"Признаки с бесконечными значениями: {inf_cols[:10]}")
            result.passed = False

        return result

    def check_look_ahead(
        self, X: pd.DataFrame, y: pd.Series, timestamp_col: Optional[str] = None
    ) -> SanityCheckResult:
        """
        Проверить look-ahead bias (простая эвристика).

        Args:
            X: Признаки
            y: Таргет
            timestamp_col: Колонка с timestamp (если есть)

        Returns:
            Результат проверки
        """
        result = SanityCheckResult()

        # Проверяем что таргет не является признаком
        if "target" in X.columns or "label" in X.columns:
            result.errors.append(
                "Таргет содержится в признаках! Удалите столбцы 'target' или 'label'."
            )
            result.passed = False

        # Проверяем корреляцию с таргетом (слишком высокая может указывать на leakage)
        if y.dtype in [np.int64, np.float64]:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                correlations = X[numeric_cols].corrwith(y).abs()
                suspicious = correlations[correlations > 0.99]

                if len(suspicious) > 0:
                    result.warnings.append(
                        f"Подозрительно высокая корреляция с таргетом (>0.99): "
                        f"{list(suspicious.index)}"
                    )

        return result

    def check_data_leakage(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
    ) -> SanityCheckResult:
        """
        Проверить утечку данных между train и val.

        Args:
            X_train: Train признаки
            X_val: Val признаки
            y_train: Train таргет
            y_val: Val таргет

        Returns:
            Результат проверки
        """
        result = SanityCheckResult()

        # Проверка на дубликаты строк
        train_hashes = pd.util.hash_pandas_object(X_train, index=False)
        val_hashes = pd.util.hash_pandas_object(X_val, index=False)

        duplicates = set(train_hashes) & set(val_hashes)

        if duplicates:
            n_dup = len(duplicates)
            dup_pct = n_dup / len(X_val) * 100
            result.warnings.append(
                f"Найдено {n_dup} ({dup_pct:.2f}%) дубликатов между train и val"
            )

        # Проверка временного overlap (если есть timestamp)
        if "timestamp" in X_train.columns and "timestamp" in X_val.columns:
            train_max = X_train["timestamp"].max()
            val_min = X_val["timestamp"].min()

            if train_max >= val_min:
                result.warnings.append(
                    f"Временное пересечение: train заканчивается {train_max}, "
                    f"val начинается {val_min}"
                )

        return result

    def _find_duplicate_features(self, X: pd.DataFrame) -> List[tuple]:
        """
        Найти дубликаты признаков.

        Returns:
            Список пар дубликатов
        """
        duplicates = []
        cols = list(X.columns)

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if X[cols[i]].equals(X[cols[j]]):
                    duplicates.append((cols[i], cols[j]))

        return duplicates
