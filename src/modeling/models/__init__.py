"""Модели машинного обучения."""

import logging
from importlib import import_module
from typing import Iterable

logger = logging.getLogger(__name__)

__all__: list[str] = []


def _safe_import(module_path: str, symbols: Iterable[str]) -> None:
    """Импортировать модуль и выбранные символы, не прерываясь при ошибках."""

    try:
        module = import_module(module_path)
    except Exception:  # noqa: BLE001 - хотим логировать и продолжать
        logger.debug("Не удалось импортировать модуль %s", module_path, exc_info=True)
        return

    for name in symbols:
        try:
            obj = getattr(module, name)
        except AttributeError:
            logger.debug("Модуль %s не содержит объект %s", module_path, name, exc_info=True)
            continue

        globals()[name] = obj
        if name not in __all__:
            __all__.append(name)


_safe_import(
    "src.modeling.models.tree_based.lightgbm_model",
    ["LightGBMModel"],
)
_safe_import(
    "src.modeling.models.tree_based.xgboost_model",
    ["XGBoostModel"],
)
_safe_import(
    "src.modeling.models.tree_based.catboost_model",
    ["CatBoostModel"],
)
_safe_import(
    "src.modeling.models.tree_based.random_forest_model",
    ["RandomForestModel"],
)
_safe_import(
    "src.modeling.models.tree_based.extra_trees_model",
    ["ExtraTreesModel"],
)

_safe_import(
    "src.modeling.models.linear.logistic_regression_model",
    ["LogisticRegressionModel"],
)
_safe_import(
    "src.modeling.models.linear.elasticnet_model",
    ["ElasticNetModel"],
)

_safe_import(
    "src.modeling.models.neural.tabular.tabnet_model",
    ["TabNetModel"],
)
_safe_import(
    "src.modeling.models.neural.tabular.ft_transformer_model",
    ["FTTransformerModel"],
)
_safe_import(
    "src.modeling.models.neural.tabular.node_model",
    ["NODEModel"],
)
