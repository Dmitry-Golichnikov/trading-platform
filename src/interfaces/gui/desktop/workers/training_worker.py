from typing import Any, Dict

import pandas as pd
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from src.interfaces.gui.desktop.workers.callbacks import QtProgressCallback
from src.modeling.models.tree_based.catboost_model import CatBoostModel

# Import models
from src.modeling.models.tree_based.lightgbm_model import LightGBMModel
from src.modeling.trainer import ModelTrainer

# Add others as needed


class WorkerSignals(QObject):
    epoch_finished = pyqtSignal(int, dict)
    training_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)


class TrainingWorker(QThread):
    def __init__(
        self,
        model_type: str,
        params: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ):
        super().__init__()
        self.signals = WorkerSignals()
        self.model_type = model_type
        self.params = params
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.trainer = None

    def run(self):
        try:
            # Initialize Model
            if self.model_type == "LightGBM":
                model = LightGBMModel(**self.params)
            elif self.model_type == "CatBoost":
                model = CatBoostModel(**self.params)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            # Initialize Trainer
            self.trainer = ModelTrainer(model=model, verbose=True)

            # Setup Callback
            qt_callback = QtProgressCallback(self.signals)

            # Train
            self.trainer.train(
                self.X_train,
                self.y_train,
                self.X_val,
                self.y_val,
                callbacks=[qt_callback],
                epochs=self.params.get("n_estimators", 100),  # Estimate epochs for progress
            )

        except Exception as e:
            self.signals.error_occurred.emit(str(e))
