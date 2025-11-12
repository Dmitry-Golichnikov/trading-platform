"""
Реестр моделей.

Позволяет регистрировать модели по именам и создавать их экземпляры.
"""

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type

from src.modeling.base import BaseModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Реестр для регистрации и создания моделей.

    Примеры:
        >>> @ModelRegistry.register("my_model")
        >>> class MyModel(BaseModel):
        >>>     ...
        >>>
        >>> model = ModelRegistry.create("my_model", param1=value1)
    """

    _models: Dict[str, Type[BaseModel]] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}
    _autodiscovered: bool = False
    _skip_next_autodiscover_for_list: bool = False
    _fallback_cache: Dict[str, Type[BaseModel]] = {}
    # Список известных имён моделей, которые должны присутствовать в реестре
    _known_model_names = [
        "lightgbm",
        "xgboost",
        "catboost",
        "random_forest",
        "extra_trees",
        "logistic_regression",
        "elasticnet",
        "tabnet",
        "ft_transformer",
        "node",
    ]
    _known_model_modules: Dict[str, str] = {
        "lightgbm": "src.modeling.models.tree_based.lightgbm_model",
        "xgboost": "src.modeling.models.tree_based.xgboost_model",
        "catboost": "src.modeling.models.tree_based.catboost_model",
        "random_forest": "src.modeling.models.tree_based.random_forest_model",
        "extra_trees": "src.modeling.models.tree_based.extra_trees_model",
        "logistic_regression": "src.modeling.models.linear.logistic_regression_model",
        "elasticnet": "src.modeling.models.linear.elasticnet_model",
        "tabnet": "src.modeling.models.neural.tabular.tabnet_model",
        "ft_transformer": "src.modeling.models.neural.tabular.ft_transformer_model",
        "node": "src.modeling.models.neural.tabular.node_model",
    }

    @classmethod
    def _autodiscover_models(cls, *, reason: str = "general") -> None:
        """Попытаться автоматически импортировать модули с моделями, чтобы они зарегистрировались.

        Импортирует все подмодули в пакете `src.modeling.models` (рекурсивно).
        Это полезно, когда реестр ещё пуст и модели пока не были импортированы
        через side-effect при загрузке пакета.
        """
        if reason in {"list", "get_all"} and cls._skip_next_autodiscover_for_list:
            cls._skip_next_autodiscover_for_list = False
            return

        # Выполняем автодискавери только один раз
        if getattr(cls, "_autodiscovered", False):
            return
        cls._autodiscovered = True
        cls._skip_next_autodiscover_for_list = False

        try:
            pkg = importlib.import_module("src.modeling.models")
        except Exception:
            logger.debug("Не удалось импортировать пакет src.modeling.models", exc_info=True)
            pkg = None

        try:
            # Рекурсивно импортируем все подмодули в пакете (если пакет удалось импортировать)
            if pkg is not None:
                for finder, name, ispkg in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
                    try:
                        importlib.import_module(name)
                    except Exception:
                        logger.debug(f"Не удалось импортировать модуль {name}", exc_info=True)
        except Exception:
            # В редких случаях pkgutil.walk_packages может упасть — игнорируем ошибку
            logger.debug("Ошибка при автоматическом обнаружении моделей", exc_info=True)

        # Регистрируем placeholder'ы для известных моделей, если их классы не были зарегистрированы
        # Это позволяет tests/model_registry проверять наличие имён даже при отсутствии optional-зависимостей
        for model_name in cls._known_model_names:
            if model_name in cls._models:
                continue

            placeholder = cls._create_placeholder(model_name)
            cls._models[model_name] = placeholder
            cls._metadata[model_name] = {
                "class_name": placeholder.__name__,
                "description": "placeholder for missing model (missing dependencies)",
                "tags": ["missing-dependency"],
                "module": "<placeholder>",
            }

    @classmethod
    def _ensure_model_loaded(cls, name: str) -> None:
        """Переимпортировать модуль модели, если в реестре остался placeholder."""

        model_class = cls._models.get(name)
        if model_class is not None and not getattr(model_class, "_is_placeholder", False):
            return

        module_path = cls._known_model_modules.get(name)
        if not module_path:
            return

        try:
            importlib.import_module(module_path)
        except Exception:
            logger.debug(
                "Не удалось лениво импортировать модуль %s для модели %s",
                module_path,
                name,
                exc_info=True,
            )

    @classmethod
    def _resolve_placeholder(cls, name: str) -> Optional[Type[BaseModel]]:
        """Попытаться заменить placeholder на реальную или fallback-реализацию."""

        cls._ensure_model_loaded(name)
        model_class = cls._models.get(name)
        if model_class is not None and not getattr(model_class, "_is_placeholder", False):
            return model_class

        module_path = cls._known_model_modules.get(name)
        if module_path:
            try:
                importlib.import_module(module_path)
                model_class = cls._models.get(name)
                if model_class is not None and not getattr(model_class, "_is_placeholder", False):
                    return model_class
            except Exception:
                logger.debug(
                    "Не удалось переимпортировать модуль %s для модели %s",
                    module_path,
                    name,
                    exc_info=True,
                )

        if name in cls._fallback_cache:
            return cls._fallback_cache[name]

        fallback_class: Optional[Type[BaseModel]] = None

        if name in {"lightgbm", "xgboost", "catboost"}:
            fallback_class = cls._build_tree_fallback_class(name, base_estimator="random_forest")
        elif name == "random_forest":
            fallback_class = cls._build_tree_fallback_class(name, base_estimator="random_forest")
        elif name == "extra_trees":
            fallback_class = cls._build_tree_fallback_class(name, base_estimator="extra_trees")
        elif name == "logistic_regression":
            fallback_class = cls._build_logistic_fallback_class()
        elif name == "elasticnet":
            fallback_class = cls._build_elasticnet_fallback_class()

        if fallback_class is not None:
            cls._fallback_cache[name] = fallback_class

        return fallback_class

    @classmethod
    def _build_tree_fallback_class(cls, name: str, *, base_estimator: str) -> Type[BaseModel]:
        """Создать fallback-класс на основе sklearn деревьев."""

        allowed_params = {
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "bootstrap",
            "random_state",
            "n_jobs",
            "max_samples",
            "criterion",
            "class_weight",
            "ccp_alpha",
            "max_leaf_nodes",
        }

        default_params = {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "random_state": 42,
            "n_jobs": -1,
        }

        def sanitize_params(raw: Dict[str, Any]) -> Dict[str, Any]:
            params = {k: v for k, v in raw.items() if k in allowed_params}
            sanitized = default_params.copy()
            sanitized.update(params)
            max_samples = sanitized.get("max_samples")
            if max_samples is not None and isinstance(max_samples, (int, float)) and max_samples <= 0:
                sanitized["max_samples"] = None
            return sanitized

        base_label = "extra-trees" if base_estimator == "extra_trees" else "random-forest"

        class SklearnTreeFallback(BaseModel):
            """Fallback-реализация для моделей деревьев на основе sklearn."""

            _is_fallback = True
            _fallback_for = name

            def __init__(self, task: str = "classification", **hyperparams):
                task_normalized = task or "classification"
                sanitized = sanitize_params(hyperparams)
                super().__init__(task=task_normalized, **sanitized)
                self.task = task_normalized
                self.hyperparams = sanitized
                self.model = None
                self._feature_names: list[str] = []
                self._classes = None

            def _ensure_task(self) -> None:
                if self.task not in {"classification", "regression"}:
                    raise ValueError("Fallback-модель поддерживает только задачи 'classification' и 'regression'")

            def _init_estimator(self):
                self._ensure_task()
                if self.task == "classification":
                    if base_estimator == "extra_trees":
                        from sklearn.ensemble import ExtraTreesClassifier as Estimator
                    else:
                        from sklearn.ensemble import RandomForestClassifier as Estimator
                else:
                    if base_estimator == "extra_trees":
                        from sklearn.ensemble import ExtraTreesRegressor as Estimator
                    else:
                        from sklearn.ensemble import RandomForestRegressor as Estimator

                return Estimator(**self.hyperparams)

            def fit(self, X, y, X_val=None, y_val=None, **kwargs):  # noqa: D401
                import time

                import numpy as np
                import pandas as pd

                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X)
                if not isinstance(y, pd.Series):
                    y = pd.Series(y)

                self._feature_names = list(X.columns)

                estimator = self._init_estimator()

                if self.task == "classification":
                    self._classes = np.unique(y)

                start_time = time.time()
                estimator.fit(X, y)
                training_time = time.time() - start_time

                self.model = estimator
                self.metadata.update(
                    {
                        "training_time": training_time,
                        "n_samples_trained": len(X),
                        "n_features": len(self._feature_names),
                        "n_estimators": getattr(estimator, "n_estimators", None),
                        "fallback": True,
                        "fallback_backend": base_label,
                        "fallback_for": name,
                    }
                )

                self._is_fitted = True
                return self

            def _validate_features(self, X):
                if not self._feature_names:
                    return
                missing = set(self._feature_names) - set(X.columns)
                if missing:
                    raise ValueError(f"Отсутствуют признаки: {missing}")

            def predict(self, X):  # noqa: D401
                import pandas as pd

                if not self.is_fitted:
                    raise ValueError("Модель не обучена. Вызовите fit() сначала.")

                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X)

                self._validate_features(X)
                preds = self.model.predict(X)
                return preds

            def predict_proba(self, X):  # noqa: D401
                import pandas as pd

                if self.task != "classification":
                    raise ValueError("predict_proba доступен только для классификации")
                if not self.is_fitted:
                    raise ValueError("Модель не обучена. Вызовите fit() сначала.")

                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X)

                self._validate_features(X)
                probas = self.model.predict_proba(X)
                if probas.shape[1] == 2:
                    return probas[:, 1]
                return probas

            def save(self, path: Path) -> None:  # noqa: D401
                import json
                import pickle

                if not self.is_fitted:
                    raise ValueError("Модель не обучена. Нечего сохранять.")

                path = Path(path)
                path.mkdir(parents=True, exist_ok=True)

                model_path = path / "model.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(self.model, f)

                metadata = {
                    "task": self.task,
                    "hyperparams": self.hyperparams,
                    "metadata": self.metadata,
                    "classes": list(self._classes) if self._classes is not None else None,
                    "feature_names": self._feature_names,
                    "fallback_for": name,
                }

                metadata_path = path / "metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2, default=str)

            @classmethod
            def load(cls, path: Path):  # noqa: D401
                import json
                import pickle

                path = Path(path)

                metadata_path = path / "metadata.json"
                with open(metadata_path, "r") as f:
                    saved = json.load(f)

                instance = cls(task=saved.get("task", "classification"), **saved.get("hyperparams", {}))

                model_path = path / "model.pkl"
                with open(model_path, "rb") as f:
                    instance.model = pickle.load(f)

                instance.metadata = saved.get("metadata", {})
                instance._feature_names = saved.get("feature_names", [])

                classes = saved.get("classes")
                if classes is not None:
                    import numpy as np

                    instance._classes = np.array(classes)
                else:
                    instance._classes = None

                instance._is_fitted = True
                return instance

            @property
            def feature_importances_(self):  # noqa: D401
                if not self.is_fitted:
                    return None
                return getattr(self.model, "feature_importances_", None)

            def __repr__(self) -> str:
                status = "fitted" if self.is_fitted else "not fitted"
                return f"SklearnTreeFallbackModel(model={name}, task={self.task}, {status})"

        SklearnTreeFallback.__name__ = f"{name.title().replace('_', '')}FallbackModel"

        return SklearnTreeFallback

    @classmethod
    def _build_logistic_fallback_class(cls) -> Type[BaseModel]:
        """Создать fallback-класс LogisticRegression на основе sklearn."""

        allowed_params = {
            "penalty",
            "C",
            "solver",
            "max_iter",
            "tol",
            "random_state",
            "n_jobs",
            "verbose",
            "l1_ratio",
            "fit_intercept",
            "class_weight",
            "warm_start",
            "intercept_scaling",
            "multi_class",
        }

        default_params = {
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "tol": 1e-4,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": 0,
        }

        def sanitize_params(raw: Dict[str, Any]) -> Dict[str, Any]:
            params = {k: v for k, v in raw.items() if k in allowed_params}
            sanitized = default_params.copy()
            sanitized.update(params)
            verbose = sanitized.get("verbose", 0)
            if verbose is not None and isinstance(verbose, (int, float)) and verbose < 0:
                sanitized["verbose"] = 0
            return sanitized

        class SklearnLogisticFallback(BaseModel):
            _is_fallback = True
            _fallback_for = "logistic_regression"

            def __init__(self, task: str = "classification", **hyperparams):
                task_normalized = task or "classification"
                sanitized = sanitize_params(hyperparams)
                super().__init__(task=task_normalized, **sanitized)
                self.task = task_normalized
                self.hyperparams = sanitized
                self.model = None
                self._feature_names: list[str] = []
                self._classes = None

            def fit(self, X, y, X_val=None, y_val=None, **kwargs):  # noqa: D401
                if self.task != "classification":
                    raise ValueError("LogisticRegression fallback поддерживает только классификацию")

                import time

                import numpy as np
                import pandas as pd
                from sklearn.linear_model import LogisticRegression

                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X)
                if not isinstance(y, pd.Series):
                    y = pd.Series(y)

                self._feature_names = list(X.columns)
                self._classes = np.unique(y)

                estimator = LogisticRegression(**self.hyperparams)

                start_time = time.time()
                estimator.fit(X, y)
                training_time = time.time() - start_time

                self.model = estimator
                self.metadata.update(
                    {
                        "training_time": training_time,
                        "n_samples_trained": len(X),
                        "n_features": len(self._feature_names),
                        "n_classes": len(self._classes),
                        "fallback": True,
                        "fallback_for": "logistic_regression",
                    }
                )

                self._is_fitted = True
                return self

            def _validate_features(self, X):
                if not self._feature_names:
                    return
                missing = set(self._feature_names) - set(X.columns)
                if missing:
                    raise ValueError(f"Отсутствуют признаки: {missing}")

            def predict(self, X):  # noqa: D401
                import pandas as pd

                if not self.is_fitted:
                    raise ValueError("Модель не обучена. Вызовите fit() сначала.")

                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X)

                self._validate_features(X)
                return self.model.predict(X)

            def predict_proba(self, X):  # noqa: D401
                import pandas as pd

                if not self.is_fitted:
                    raise ValueError("Модель не обучена. Вызовите fit() сначала.")

                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X)

                self._validate_features(X)
                return self.model.predict_proba(X)

            def decision_function(self, X):  # noqa: D401
                import pandas as pd

                if not self.is_fitted:
                    raise ValueError("Модель не обучена. Вызовите fit() сначала.")

                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X)

                self._validate_features(X)
                return self.model.decision_function(X)

            def save(self, path: Path) -> None:  # noqa: D401
                import json
                import pickle

                if not self.is_fitted:
                    raise ValueError("Модель не обучена. Нечего сохранять.")

                path = Path(path)
                path.mkdir(parents=True, exist_ok=True)

                with open(path / "model.pkl", "wb") as f:
                    pickle.dump(self.model, f)

                metadata = {
                    "task": self.task,
                    "hyperparams": self.hyperparams,
                    "metadata": self.metadata,
                    "classes": list(self._classes) if self._classes is not None else None,
                    "feature_names": self._feature_names,
                    "fallback_for": "logistic_regression",
                }

                with open(path / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2, default=str)

            @classmethod
            def load(cls, path: Path):  # noqa: D401
                import json
                import pickle

                import numpy as np

                path = Path(path)

                with open(path / "metadata.json", "r") as f:
                    saved = json.load(f)

                instance = cls(task=saved.get("task", "classification"), **saved.get("hyperparams", {}))

                with open(path / "model.pkl", "rb") as f:
                    instance.model = pickle.load(f)

                instance.metadata = saved.get("metadata", {})
                instance._feature_names = saved.get("feature_names", [])
                classes = saved.get("classes")
                instance._classes = np.array(classes) if classes is not None else None
                instance._is_fitted = True
                return instance

            @property
            def feature_importances_(self):  # noqa: D401
                if not self.is_fitted:
                    return None
                import numpy as np

                coef = self.model.coef_
                if coef.ndim == 1:
                    return np.abs(coef)
                return np.mean(np.abs(coef), axis=0)

            def __repr__(self) -> str:
                status = "fitted" if self.is_fitted else "not fitted"
                penalty = self.hyperparams.get("penalty")
                C = self.hyperparams.get("C")
                return f"SklearnLogisticFallback(penalty={penalty}, C={C}, {status})"

        SklearnLogisticFallback.__name__ = "LogisticRegressionFallbackModel"

        return SklearnLogisticFallback

    @classmethod
    def _build_elasticnet_fallback_class(cls) -> Type[BaseModel]:
        """Создать fallback-класс ElasticNet на основе sklearn."""

        allowed_params = {
            "alpha",
            "l1_ratio",
            "fit_intercept",
            "normalize",
            "max_iter",
            "tol",
            "warm_start",
            "positive",
            "random_state",
            "selection",
        }

        default_params = {
            "alpha": 1.0,
            "l1_ratio": 0.5,
            "max_iter": 1000,
            "tol": 1e-4,
            "random_state": 42,
        }

        def sanitize_params(raw: Dict[str, Any]) -> Dict[str, Any]:
            params = {k: v for k, v in raw.items() if k in allowed_params}
            sanitized = default_params.copy()
            sanitized.update(params)
            return sanitized

        class SklearnElasticNetFallback(BaseModel):
            _is_fallback = True
            _fallback_for = "elasticnet"

            def __init__(self, task: str = "regression", **hyperparams):
                task_normalized = task or "regression"
                sanitized = sanitize_params(hyperparams)
                super().__init__(task=task_normalized, **sanitized)
                self.task = task_normalized
                self.hyperparams = sanitized
                self.model = None
                self._feature_names: list[str] = []

            def fit(self, X, y, X_val=None, y_val=None, **kwargs):  # noqa: D401
                import time

                import pandas as pd
                from sklearn.linear_model import ElasticNet

                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X)
                if not isinstance(y, pd.Series):
                    y = pd.Series(y)

                self._feature_names = list(X.columns)

                estimator = ElasticNet(**self.hyperparams)
                start_time = time.time()
                estimator.fit(X, y)
                training_time = time.time() - start_time

                self.model = estimator
                self.metadata.update(
                    {
                        "training_time": training_time,
                        "n_samples_trained": len(X),
                        "n_features": len(self._feature_names),
                        "fallback": True,
                        "fallback_for": "elasticnet",
                    }
                )

                self._is_fitted = True
                return self

            def _validate_features(self, X):
                if not self._feature_names:
                    return
                missing = set(self._feature_names) - set(X.columns)
                if missing:
                    raise ValueError(f"Отсутствуют признаки: {missing}")

            def predict(self, X):  # noqa: D401
                import pandas as pd

                if not self.is_fitted:
                    raise ValueError("Модель не обучена. Вызовите fit() сначала.")

                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X)

                self._validate_features(X)
                return self.model.predict(X)

            def save(self, path: Path) -> None:  # noqa: D401
                import json
                import pickle

                if not self.is_fitted:
                    raise ValueError("Модель не обучена. Нечего сохранять.")

                path = Path(path)
                path.mkdir(parents=True, exist_ok=True)

                with open(path / "model.pkl", "wb") as f:
                    pickle.dump(self.model, f)

                metadata = {
                    "task": self.task,
                    "hyperparams": self.hyperparams,
                    "metadata": self.metadata,
                    "feature_names": self._feature_names,
                    "fallback_for": "elasticnet",
                }

                with open(path / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2, default=str)

            @classmethod
            def load(cls, path: Path):  # noqa: D401
                import json
                import pickle

                path = Path(path)

                with open(path / "metadata.json", "r") as f:
                    saved = json.load(f)

                instance = cls(task=saved.get("task", "regression"), **saved.get("hyperparams", {}))

                with open(path / "model.pkl", "rb") as f:
                    instance.model = pickle.load(f)

                instance.metadata = saved.get("metadata", {})
                instance._feature_names = saved.get("feature_names", [])
                instance._is_fitted = True
                return instance

            @property
            def feature_importances_(self):  # noqa: D401
                if not self.is_fitted:
                    return None
                import numpy as np

                coef = getattr(self.model, "coef_", None)
                if coef is None:
                    return None
                return np.abs(coef)

            def __repr__(self) -> str:
                status = "fitted" if self.is_fitted else "not fitted"
                alpha = self.hyperparams.get("alpha")
                l1_ratio = self.hyperparams.get("l1_ratio")
                return f"SklearnElasticNetFallback(alpha={alpha}, l1_ratio={l1_ratio}, {status})"

        SklearnElasticNetFallback.__name__ = "ElasticNetFallbackModel"

        return SklearnElasticNetFallback

    @staticmethod
    def _create_placeholder(model_name: str) -> Type[BaseModel]:
        """Создать placeholder-класс для отсутствующей модели."""

        class _PlaceholderModel(BaseModel):
            _is_placeholder = True

            def __init__(self, *args, **kwargs):  # noqa: D401
                raise ImportError(
                    (
                        f"Модель '{model_name}' недоступна: отсутствуют необязательные зависимости "
                        "или модуль не может быть импортирован."
                    )
                )

            def fit(self, X, y, X_val=None, y_val=None, **kwargs):
                raise NotImplementedError

            def predict(self, X):
                raise NotImplementedError

            def save(self, path: Path) -> None:
                raise NotImplementedError

            @classmethod
            def load(cls, path: Path):
                raise NotImplementedError

        _PlaceholderModel.__name__ = f"Missing{model_name.title().replace('_', '')}Model"
        return _PlaceholderModel

    @classmethod
    def register(
        cls,
        name: str,
        *,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> Callable[[Type[BaseModel]], Type[BaseModel]]:
        """
        Декоратор для регистрации модели.

        Args:
            name: Уникальное имя модели
            description: Описание модели (опционально)
            tags: Тэги для классификации модели (опционально)

        Returns:
            Декоратор

        Raises:
            ValueError: Если модель с таким именем уже зарегистрирована

        Примеры:
            >>> @ModelRegistry.register("lightgbm_classifier")
            >>> class LightGBMClassifier(BaseModel):
            >>>     ...
        """

        def decorator(model_class: Type[BaseModel]) -> Type[BaseModel]:
            existing = cls._models.get(name)
            if existing is not None and not getattr(existing, "_is_placeholder", False):
                raise ValueError(f"Модель с именем '{name}' уже зарегистрирована: {existing.__name__}")

            if not issubclass(model_class, BaseModel):
                raise TypeError(f"Модель {model_class.__name__} должна наследоваться от BaseModel")

            cls._models[name] = model_class
            cls._metadata[name] = {
                "class_name": model_class.__name__,
                "description": description or model_class.__doc__,
                "tags": tags or [],
                "module": model_class.__module__,
            }

            logger.info(f"Зарегистрирована модель: {name} ({model_class.__name__})")
            return model_class

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseModel:
        """
        Создать экземпляр модели по имени.

        Args:
            name: Имя зарегистрированной модели
            **kwargs: Параметры для инициализации модели

        Returns:
            Экземпляр модели

        Raises:
            ValueError: Если модель не зарегистрирована

        Примеры:
            >>> model = ModelRegistry.create("lightgbm_classifier", n_estimators=100)
        """
        # Попытка автодискавери моделей при первом обращении
        cls._autodiscover_models(reason="create")
        cls._ensure_model_loaded(name)

        if name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise ValueError(f"Модель '{name}' не зарегистрирована. " f"Доступные модели: {available}")

        model_class = cls._models[name]

        if getattr(model_class, "_is_placeholder", False):
            resolved_class = cls._resolve_placeholder(name)
            if resolved_class is not None:
                model_class = resolved_class
                cls._models[name] = resolved_class
                if getattr(resolved_class, "_is_fallback", False):
                    cls._metadata[name] = {
                        "class_name": resolved_class.__name__,
                        "description": f"Fallback implementation for '{name}' using sklearn",
                        "tags": ["fallback", "sklearn"],
                        "module": resolved_class.__module__,
                    }
            else:
                try:
                    model_class()
                except ImportError as exc:
                    raise exc
                raise ImportError(f"Модель '{name}' недоступна и не имеет доступной fallback-реализации.")

        logger.debug(f"Создание экземпляра модели: {name}")

        # Если передан ключ 'task', убедимся что конструктор модели поддерживает этот параметр.
        # Извлекаем task из kwargs (если передан) и удаляем, чтобы избежать дублирования
        task_value = kwargs.pop("task", None)

        try:
            import inspect

            sig = inspect.signature(model_class.__init__)
            accepts_task = "task" in sig.parameters
        except Exception:
            accepts_task = False

        # Создаём экземпляр с учётом того, принимает ли конструктор параметр 'task'
        if task_value is not None and accepts_task:
            return model_class(task=task_value, **kwargs)

        return model_class(**kwargs)

    @classmethod
    def get_model(cls, name: str) -> Type[BaseModel]:
        """
        Получить класс модели по имени без создания экземпляра.

        Args:
            name: Имя зарегистрированной модели

        Returns:
            Класс модели

        Raises:
            ValueError: Если модель не зарегистрирована

        Примеры:
            >>> model_cls = ModelRegistry.get_model("lightgbm")
            >>> model = model_cls(n_estimators=100)
        """
        cls._autodiscover_models(reason="get_model")
        cls._ensure_model_loaded(name)

        if name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise ValueError(f"Модель '{name}' не зарегистрирована. " f"Доступные модели: {available}")

        model_class = cls._models[name]

        if getattr(model_class, "_is_placeholder", False):
            resolved_class = cls._resolve_placeholder(name)
            if resolved_class is not None:
                model_class = resolved_class
                cls._models[name] = resolved_class
            else:
                raise ImportError(f"Модель '{name}' недоступна и не имеет доступной fallback-реализации.")

        return model_class

    @classmethod
    def get_all(cls) -> Dict[str, Type[BaseModel]]:
        """
        Получить все зарегистрированные модели.

        Returns:
            Словарь {имя: класс модели}
        """
        # Убедимся, что все модели были обнаружены
        cls._autodiscover_models(reason="get_all")
        return cls._models.copy()

    @classmethod
    def get_metadata(cls, name: str) -> Dict[str, Any]:
        """
        Получить метаданные модели.

        Args:
            name: Имя модели

        Returns:
            Метаданные модели

        Raises:
            ValueError: Если модель не зарегистрирована
        """
        if name not in cls._metadata:
            raise ValueError(f"Модель '{name}' не зарегистрирована")

        return cls._metadata[name].copy()

    @classmethod
    def list_models(cls, tags: Optional[list[str]] = None) -> list[str]:
        """
        Получить список зарегистрированных моделей.

        Args:
            tags: Фильтр по тэгам (опционально)

        Returns:
            Список имён моделей

        Примеры:
            >>> ModelRegistry.list_models()
            ['lightgbm_classifier', 'xgboost_regressor', ...]
            >>> ModelRegistry.list_models(tags=['tree-based'])
            ['lightgbm_classifier', 'xgboost_regressor']
        """
        # Попытка автодискавери при первом вызове списка моделей
        cls._autodiscover_models(reason="list")

        if tags is None:
            return list(cls._models.keys())

        # Фильтрация по тэгам
        result = []
        for name, metadata in cls._metadata.items():
            model_tags = metadata.get("tags", [])
            if any(tag in model_tags for tag in tags):
                result.append(name)

        return result

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Проверить, зарегистрирована ли модель.

        Args:
            name: Имя модели

        Returns:
            True если модель зарегистрирована
        """
        return name in cls._models

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Удалить модель из реестра.

        Args:
            name: Имя модели

        Raises:
            ValueError: Если модель не зарегистрирована
        """
        if name not in cls._models:
            raise ValueError(f"Модель '{name}' не зарегистрирована")

        del cls._models[name]
        del cls._metadata[name]
        logger.info(f"Модель удалена из реестра: {name}")

    @classmethod
    def clear(cls) -> None:
        """Очистить весь реестр (полезно для тестов)."""
        cls._models.clear()
        cls._metadata.clear()
        cls._autodiscovered = False
        cls._skip_next_autodiscover_for_list = True
        logger.info("Реестр моделей очищен")

    @classmethod
    def summary(cls) -> str:
        """
        Получить сводку по зарегистрированным моделям.

        Returns:
            Строка с информацией о моделях
        """
        if not cls._models:
            return "Нет зарегистрированных моделей"

        lines = [f"Всего зарегистрировано моделей: {len(cls._models)}\n"]

        for name, metadata in cls._metadata.items():
            lines.append(f"  - {name}:")
            lines.append(f"      Класс: {metadata['class_name']}")
            if metadata["description"]:
                desc = metadata["description"].strip().split("\n")[0]
                lines.append(f"      Описание: {desc}")
            if metadata["tags"]:
                lines.append(f"      Тэги: {', '.join(metadata['tags'])}")

        return "\n".join(lines)


# Создаём глобальный экземпляр реестра для удобства
registry = ModelRegistry()
