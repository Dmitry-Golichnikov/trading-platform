"""
Labeling Service

Business logic for labeling operations.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.interfaces.gui.backend.api.models import LabelingSetInfo, LabelingMethod, TaskStatus
from src.interfaces.gui.backend.services.dataset_service import DatasetService


class LabelingService:
    """Service for labeling operations"""

    def __init__(self, artifacts_dir: Path = Path("artifacts")):
        self.artifacts_dir = artifacts_dir
        self.labeling_dir = artifacts_dir / "labeling"
        self.labeling_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_service = DatasetService(artifacts_dir)

    def build_labeling_set_id(self, dataset_id: str, name: str) -> str:
        """Build labeling set ID from dataset ID and name"""
        return f"{dataset_id}__{name}"

    def _labeling_set_dir(self, labeling_set_id: str) -> Path:
        """Get directory for labeling set"""
        return self.labeling_dir / labeling_set_id

    def _metadata_file(self, labeling_set_id: str) -> Path:
        """Get metadata file path for labeling set"""
        return self._labeling_set_dir(labeling_set_id) / "metadata.json"

    def _data_file(self, labeling_set_id: str) -> Path:
        """Get data file path for labeling set"""
        return self._labeling_set_dir(labeling_set_id) / "labels.parquet"

    def _calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """Calculate hash of configuration"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _save_metadata(self, labeling_set_id: str, metadata: Dict[str, Any]) -> None:
        """Save metadata to file"""
        metadata_file = self._metadata_file(labeling_set_id)
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

    def _load_metadata(self, labeling_set_id: str) -> Optional[Dict[str, Any]]:
        """Load metadata from file"""
        metadata_file = self._metadata_file(labeling_set_id)
        if not metadata_file.exists():
            return None
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load metadata for {labeling_set_id}: {e}")
            return None

    def list_labeling_sets(
        self,
        dataset_id: Optional[str] = None,
        method: Optional[LabelingMethod] = None,
    ) -> List[LabelingSetInfo]:
        """List all labeling sets"""
        labeling_sets = []

        if not self.labeling_dir.exists():
            return labeling_sets

        for set_dir in sorted(self.labeling_dir.iterdir()):
            if not set_dir.is_dir():
                continue

            metadata = self._load_metadata(set_dir.name)
            if not metadata:
                continue

            # Apply filters
            if dataset_id and metadata.get("dataset_id") != dataset_id:
                continue
            if method and metadata.get("method") != method:
                continue

            try:
                labeling_set = LabelingSetInfo(
                    id=set_dir.name,
                    name=metadata.get("name", "Unknown"),
                    dataset_id=metadata.get("dataset_id", ""),
                    method=LabelingMethod(metadata.get("method", "horizon")),
                    config=metadata.get("config", {}),
                    num_samples=metadata.get("num_samples", 0),
                    class_distribution=metadata.get("class_distribution"),
                    created_at=datetime.fromisoformat(metadata.get("created_at", datetime.now().isoformat())),
                    updated_at=(
                        datetime.fromisoformat(metadata["updated_at"])
                        if metadata.get("updated_at")
                        else None
                    ),
                    status=metadata.get("status", "unknown"),
                    description=metadata.get("description"),
                    feature_set_id=metadata.get("feature_set_id"),
                )
                labeling_sets.append(labeling_set)
            except Exception as e:
                print(f"Failed to parse labeling set {set_dir.name}: {e}")
                continue

        return labeling_sets

    def get_labeling_set(self, labeling_set_id: str) -> Optional[LabelingSetInfo]:
        """Get specific labeling set"""
        metadata = self._load_metadata(labeling_set_id)
        if not metadata:
            return None

        try:
            return LabelingSetInfo(
                id=labeling_set_id,
                name=metadata.get("name", "Unknown"),
                dataset_id=metadata.get("dataset_id", ""),
                method=LabelingMethod(metadata.get("method", "horizon")),
                config=metadata.get("config", {}),
                num_samples=metadata.get("num_samples", 0),
                class_distribution=metadata.get("class_distribution"),
                created_at=datetime.fromisoformat(metadata.get("created_at", datetime.now().isoformat())),
                updated_at=(
                    datetime.fromisoformat(metadata["updated_at"]) if metadata.get("updated_at") else None
                ),
                status=metadata.get("status", "unknown"),
                description=metadata.get("description"),
                feature_set_id=metadata.get("feature_set_id"),
            )
        except Exception as e:
            print(f"Failed to parse labeling set {labeling_set_id}: {e}")
            return None

    def get_labeling_data(
        self,
        labeling_set_id: str,
        limit: int = 1000,
    ) -> Dict[str, Any]:
        """Get labeling data"""
        data_file = self._data_file(labeling_set_id)
        if not data_file.exists():
            raise FileNotFoundError(f"Labeling data not found: {labeling_set_id}")

        try:
            df = pd.read_parquet(data_file)
            if limit > 0:
                df = df.head(limit)

            # Convert to dict
            return {
                "data": df.reset_index().to_dict(orient="records"),
                "num_rows": len(df),
            }
        except Exception as e:
            raise RuntimeError(f"Failed to read labeling data: {e}")

    def delete_labeling_set(self, labeling_set_id: str) -> bool:
        """Delete a labeling set"""
        set_dir = self._labeling_set_dir(labeling_set_id)
        if not set_dir.exists():
            return False

        try:
            import shutil

            shutil.rmtree(set_dir)
            return True
        except Exception as e:
            print(f"Failed to delete labeling set {labeling_set_id}: {e}")
            return False

    def save_labeling_result(
        self,
        labeling_set_id: str,
        name: str,
        dataset_id: str,
        method: str,
        config: Dict[str, Any],
        labels_df: pd.DataFrame,
        description: Optional[str] = None,
        feature_set_id: Optional[str] = None,
    ) -> LabelingSetInfo:
        """Save labeling result"""
        # Calculate statistics
        num_samples = len(labels_df)
        class_distribution = None

        # Calculate class distribution if label column exists
        if "label" in labels_df.columns:
            class_dist = labels_df["label"].value_counts().to_dict()
            class_distribution = {str(k): int(v) for k, v in class_dist.items()}

        # Save data (sanitize unsupported object types)
        safe_df = self._sanitize_dataframe(labels_df)
        data_file = self._data_file(labeling_set_id)
        data_file.parent.mkdir(parents=True, exist_ok=True)
        safe_df.to_parquet(data_file, index=True)

        # Save metadata
        metadata = {
            "name": name,
            "dataset_id": dataset_id,
            "method": method,
            "config": config,
            "num_samples": num_samples,
            "class_distribution": class_distribution,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "completed",
            "description": description,
            "feature_set_id": feature_set_id,
        }
        self._save_metadata(labeling_set_id, metadata)

        return LabelingSetInfo(
            id=labeling_set_id,
            name=name,
            dataset_id=dataset_id,
            method=LabelingMethod(method),
            config=config,
            num_samples=num_samples,
            class_distribution=class_distribution,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status="completed",
            description=description,
            feature_set_id=feature_set_id,
        )

    def update_metadata_status(
        self,
        labeling_set_id: str,
        status: str,
        num_samples: int = 0,
        class_distribution: Optional[Dict[str, int]] = None,
    ) -> None:
        """Update labeling set metadata status"""
        metadata = self._load_metadata(labeling_set_id)
        if not metadata:
            return

        metadata["status"] = status
        metadata["updated_at"] = datetime.now().isoformat()
        if num_samples > 0:
            metadata["num_samples"] = num_samples
        if class_distribution:
            metadata["class_distribution"] = class_distribution

        self._save_metadata(labeling_set_id, metadata)

    def _sanitize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure dataframe is parquet-compatible by converting unsupported object values.
        """
        sanitized = df.copy()

        def _convert_value(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (dict, list, tuple, set)):
                try:
                    return json.dumps(value, default=str)
                except Exception:
                    return str(value)
            return value

        for column in sanitized.columns:
            if sanitized[column].dtype == "object":
                sanitized[column] = sanitized[column].apply(_convert_value)

        return sanitized

