from typing import Any, Dict, List

import pandas as pd


class TrainingService:
    def get_available_models(self) -> List[str]:
        return ["LightGBM", "CatBoost"]

    def get_default_params(self, model: str) -> Dict[str, Any]:
        if model == "LightGBM":
            return {"n_estimators": 100, "learning_rate": 0.1, "max_depth": -1}
        elif model == "CatBoost":
            return {"iterations": 100, "learning_rate": 0.1, "depth": 6}
        return {}

    def prepare_data(self, dataset_data: pd.DataFrame, features_data: pd.DataFrame, labels_data: pd.DataFrame):
        # Merge logic: assuming index alignment for simplicity in prototype
        # In real app, need robust join on timestamp

        # Mock split
        X = features_data
        y = labels_data["label"]

        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_val = y.iloc[split_idx:]

        return X_train, y_train, X_val, y_val
