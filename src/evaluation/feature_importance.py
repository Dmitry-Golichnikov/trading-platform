"""Анализ важности признаков."""

import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence, permutation_importance


class TreeImportance:
    """Извлечение важности признаков из tree-based моделей."""

    @staticmethod
    def get_feature_importance(
        model: Any,
        feature_names: Optional[List[str]] = None,
        importance_type: str = "gain",
    ) -> pd.DataFrame:
        """
        Получить важность признаков из tree-based модели.

        Args:
            model: Обученная модель (LightGBM, XGBoost, CatBoost, etc.)
            feature_names: Названия признаков
            importance_type: Тип важности ('gain', 'split', 'weight')

        Returns:
            DataFrame с важностью признаков
        """
        # Определяем тип модели и извлекаем важность
        importance = None
        model_type = type(model).__name__

        try:
            # LightGBM
            if hasattr(model, "booster_"):
                # sklearn wrapper
                if importance_type == "gain":
                    importance = model.booster_.feature_importance(importance_type="gain")
                else:
                    importance = model.booster_.feature_importance(importance_type="split")
            elif hasattr(model, "feature_importances_"):
                # Общий sklearn интерфейс
                importance = model.feature_importances_
            elif hasattr(model, "get_feature_importance"):
                # CatBoost
                importance = model.get_feature_importance()
            else:
                raise AttributeError(f"Cannot extract importance from {model_type}")

            # Создаем DataFrame
            if feature_names is None:
                n_features = len(importance)
                feature_names = [f"feature_{i}" for i in range(n_features)]

            df = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values(
                "importance", ascending=False
            )

            # Нормализуем
            df["importance_normalized"] = df["importance"] / df["importance"].sum()

            return df

        except Exception as e:
            warnings.warn(f"Failed to extract feature importance: {e}")
            return pd.DataFrame()


class PermutationImportance:
    """Permutation importance - важность через перемешивание признаков."""

    @staticmethod
    def compute(
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_repeats: int = 10,
        random_state: Optional[int] = None,
        scoring: Optional[Union[str, Callable]] = None,
    ) -> pd.DataFrame:
        """
        Вычислить permutation importance.

        Args:
            model: Обученная модель
            X: Признаки
            y: Целевая переменная
            feature_names: Названия признаков
            n_repeats: Количество повторений перемешивания
            random_state: Random seed
            scoring: Метрика для оценки

        Returns:
            DataFrame с важностью признаков
        """
        # Вычисляем permutation importance
        result = permutation_importance(
            model,
            X,
            y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=scoring,
        )

        # Формируем DataFrame
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns.tolist()
            else:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)

        return df


class SHAPImportance:
    """SHAP values для интерпретации моделей."""

    @staticmethod
    def compute(
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        max_samples: Optional[int] = 100,
    ) -> pd.DataFrame:
        """
        Вычислить SHAP values.

        Args:
            model: Обученная модель
            X: Признаки
            feature_names: Названия признаков
            max_samples: Максимальное количество сэмплов для расчета

        Returns:
            DataFrame с важностью признаков
        """
        try:
            import shap
        except ImportError:
            warnings.warn("SHAP not installed. Install with: pip install shap")
            return pd.DataFrame()

        # Ограничиваем количество сэмплов
        if max_samples and len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices] if isinstance(X, np.ndarray) else X.iloc[indices]
        else:
            X_sample = X

        try:
            # Создаем explainer
            model_type = type(model).__name__

            if "LightGBM" in model_type or "XGBoost" in model_type or "CatBoost" in model_type:
                # Tree explainer для tree-based моделей
                explainer = shap.TreeExplainer(model)
            else:
                # Общий explainer
                explainer = shap.Explainer(model, X_sample)

            # Вычисляем SHAP values
            shap_values = explainer.shap_values(X_sample)

            # Если multi-output, берем первый выход
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # Средняя абсолютная важность
            importance = np.abs(shap_values).mean(axis=0)

            # Формируем DataFrame
            if feature_names is None:
                if isinstance(X, pd.DataFrame):
                    feature_names = X.columns.tolist()
                else:
                    feature_names = [f"feature_{i}" for i in range(X.shape[1])]

            df = pd.DataFrame({"feature": feature_names, "shap_importance": importance}).sort_values(
                "shap_importance", ascending=False
            )

            return df

        except Exception as e:
            warnings.warn(f"Failed to compute SHAP values: {e}")
            return pd.DataFrame()


class PartialDependence:
    """Partial Dependence Plots."""

    @staticmethod
    def compute(
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        features: Union[int, str, List[Union[int, str]]],
        feature_names: Optional[List[str]] = None,
        grid_resolution: int = 50,
    ) -> Dict[str, Any]:
        """
        Вычислить partial dependence.

        Args:
            model: Обученная модель
            X: Признаки
            features: Индексы или названия признаков для анализа
            feature_names: Названия всех признаков
            grid_resolution: Количество точек в сетке

        Returns:
            Dict с результатами
        """
        try:
            # Преобразуем названия в индексы
            if isinstance(features, (str, int)):
                features = [features]

            if feature_names is None and isinstance(X, pd.DataFrame):
                feature_names = X.columns.tolist()

            feature_indices = []
            for feat in features:
                if isinstance(feat, str):
                    if feature_names:
                        feature_indices.append(feature_names.index(feat))
                    else:
                        raise ValueError("feature_names required when features are strings")
                else:
                    feature_indices.append(feat)

            # Вычисляем partial dependence
            pd_result = partial_dependence(
                model,
                X,
                features=feature_indices,
                grid_resolution=grid_resolution,
            )

            result = {
                "average": pd_result["average"],
                "grid_values": pd_result["grid_values"],
                "features": features,
            }

            return result

        except Exception as e:
            warnings.warn(f"Failed to compute partial dependence: {e}")
            return {}


class FeatureImportanceAnalyzer:
    """Комплексный анализ важности признаков."""

    def __init__(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Инициализация анализатора.

        Args:
            model: Обученная модель
            X: Признаки
            y: Целевая переменная
            feature_names: Названия признаков
        """
        self.model = model
        self.X = X
        self.y = y

        # Определяем имена признаков
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                self.feature_names = X.columns.tolist()
            else:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            self.feature_names = feature_names

    def compute_all_importances(
        self,
        methods: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        """
        Вычислить важность признаков всеми доступными методами.

        Args:
            methods: Список методов ('tree', 'permutation', 'shap')
            **kwargs: Дополнительные параметры для методов

        Returns:
            Dict с результатами для каждого метода
        """
        if methods is None:
            methods = ["tree", "permutation", "shap"]

        results = {}

        # Tree-based importance
        if "tree" in methods:
            try:
                results["tree"] = TreeImportance.get_feature_importance(self.model, self.feature_names)
            except Exception as e:
                warnings.warn(f"Failed to compute tree importance: {e}")

        # Permutation importance
        if "permutation" in methods:
            try:
                results["permutation"] = PermutationImportance.compute(
                    self.model,
                    self.X,
                    self.y,
                    self.feature_names,
                    n_repeats=kwargs.get("n_repeats", 10),
                    random_state=kwargs.get("random_state", None),
                )
            except Exception as e:
                warnings.warn(f"Failed to compute permutation importance: {e}")

        # SHAP importance
        if "shap" in methods:
            try:
                results["shap"] = SHAPImportance.compute(
                    self.model,
                    self.X,
                    self.feature_names,
                    max_samples=kwargs.get("max_samples", 100),
                )
            except Exception as e:
                warnings.warn(f"Failed to compute SHAP importance: {e}")

        return results

    def get_top_features(self, n_top: int = 20, method: str = "tree") -> List[str]:
        """
        Получить топ признаков по важности.

        Args:
            n_top: Количество топ признаков
            method: Метод для определения важности

        Returns:
            Список названий топ признаков
        """
        importances = self.compute_all_importances(methods=[method])

        if method not in importances or importances[method].empty:
            return []

        df = importances[method]
        return df.head(n_top)["feature"].tolist()

    def compare_methods(self) -> pd.DataFrame:
        """
        Сравнить ранжирование признаков разными методами.

        Returns:
            DataFrame с рангами признаков по разным методам
        """
        all_importances = self.compute_all_importances()

        # Создаем DataFrame с рангами
        ranks = pd.DataFrame({"feature": self.feature_names})

        for method, df in all_importances.items():
            if not df.empty:
                # Добавляем ранги
                df_sorted = df.sort_values(df.columns[1], ascending=False).reset_index(drop=True)
                df_sorted[f"{method}_rank"] = range(1, len(df_sorted) + 1)

                # Merge с основным DataFrame
                ranks = ranks.merge(df_sorted[["feature", f"{method}_rank"]], on="feature", how="left")

        # Вычисляем средний ранг
        rank_columns = [col for col in ranks.columns if col.endswith("_rank")]
        if rank_columns:
            ranks["mean_rank"] = ranks[rank_columns].mean(axis=1)
            ranks = ranks.sort_values("mean_rank")

        return ranks
