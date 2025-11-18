"""
Model Service

Business logic for model operations.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.interfaces.gui.backend.api.models import ModelInfo, ModelMetrics


class ModelService:
    """Service for model operations"""

    def __init__(self, artifacts_dir: Path = Path("artifacts")):
        self.artifacts_dir = artifacts_dir
        self.models_dir = artifacts_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def list_models(
        self, model_type: Optional[str] = None, experiment_id: Optional[str] = None, limit: int = 100
    ) -> List[ModelInfo]:
        """List available models"""
        models = []

        # Scan models directory
        for model_file in self.models_dir.glob("*"):
            if model_file.is_file():
                try:
                    # Try to load metadata
                    metadata_file = model_file.parent / f"{model_file.stem}_metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                    else:
                        metadata = {}

                    # Filter by type
                    if model_type and metadata.get("type") != model_type:
                        continue

                    # Filter by experiment
                    if experiment_id and metadata.get("experiment_id") != experiment_id:
                        continue

                    models.append(
                        ModelInfo(
                            id=model_file.stem,
                            name=metadata.get("name", model_file.stem),
                            type=metadata.get("type", "unknown"),
                            experiment_id=metadata.get("experiment_id"),
                            metrics=metadata.get("metrics", {}),
                            hyperparameters=metadata.get("hyperparameters", {}),
                            feature_importance=metadata.get("feature_importance"),
                            created_at=datetime.fromtimestamp(model_file.stat().st_mtime),
                            size_mb=model_file.stat().st_size / (1024 * 1024),
                            is_deployed=metadata.get("is_deployed", False),
                        )
                    )
                except Exception as e:
                    print(f"Error processing model {model_file}: {e}")
                    continue

        # Sort by creation date (newest first)
        models.sort(key=lambda m: m.created_at, reverse=True)

        return models[:limit]

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get model by ID"""
        # Find model file
        model_file = None
        for ext in [".pkl", ".pt", ".pth", ".cbm", ".lgb", ".xgb"]:
            candidate = self.models_dir / f"{model_id}{ext}"
            if candidate.exists():
                model_file = candidate
                break

        if not model_file:
            return None

        # Load metadata
        metadata_file = self.models_dir / f"{model_id}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        return ModelInfo(
            id=model_id,
            name=metadata.get("name", model_id),
            type=metadata.get("type", "unknown"),
            experiment_id=metadata.get("experiment_id"),
            metrics=metadata.get("metrics", {}),
            hyperparameters=metadata.get("hyperparameters", {}),
            feature_importance=metadata.get("feature_importance"),
            created_at=datetime.fromtimestamp(model_file.stat().st_mtime),
            size_mb=model_file.stat().st_size / (1024 * 1024),
            is_deployed=metadata.get("is_deployed", False),
        )

    def get_model_metrics(self, model_id: str) -> Optional[ModelMetrics]:
        """Get detailed model metrics"""
        # Load metadata
        metadata_file = self.models_dir / f"{model_id}_metadata.json"
        if not metadata_file.exists():
            return None

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        return ModelMetrics(
            model_id=model_id,
            train_metrics=metadata.get("train_metrics", {}),
            val_metrics=metadata.get("val_metrics", {}),
            test_metrics=metadata.get("test_metrics"),
            feature_importance=metadata.get("feature_importance"),
            confusion_matrix=metadata.get("confusion_matrix"),
            calibration_data=metadata.get("calibration_data"),
        )

    def delete_model(self, model_id: str) -> bool:
        """Delete model"""
        deleted = False

        # Delete model file
        for ext in [".pkl", ".pt", ".pth", ".cbm", ".lgb", ".xgb"]:
            model_file = self.models_dir / f"{model_id}{ext}"
            if model_file.exists():
                model_file.unlink()
                deleted = True

        # Delete metadata
        metadata_file = self.models_dir / f"{model_id}_metadata.json"
        if metadata_file.exists():
            metadata_file.unlink()

        return deleted

    def deploy_model(self, model_id: str) -> bool:
        """Mark model as deployed"""
        metadata_file = self.models_dir / f"{model_id}_metadata.json"
        if not metadata_file.exists():
            return False

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        metadata["is_deployed"] = True
        metadata["deployed_at"] = datetime.now().isoformat()

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return True
