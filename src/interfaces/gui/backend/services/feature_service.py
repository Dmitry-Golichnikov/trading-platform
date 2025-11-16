"""
Feature Service

Business logic for feature generation operations.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.features.generator import FeatureGenerator
from src.interfaces.gui.backend.api.models import FeatureGenerationRequest, FeatureSetInfo
from src.interfaces.gui.backend.services.dataset_service import DatasetService


class FeatureService:
    """Service for feature operations"""

    def __init__(self, artifacts_dir: Path = Path("artifacts")):
        self.artifacts_dir = artifacts_dir
        self.features_dir = artifacts_dir / "features"
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = artifacts_dir / "data"
        self.dataset_service = DatasetService(artifacts_dir=artifacts_dir)
        self.default_chunk_size = 5000
        self.context_window = 512

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _slugify(self, value: str) -> str:
        """Simple slugify helper"""
        return value.replace(" ", "_").replace("-", "_").replace("/", "_").lower()

    def build_feature_set_id(self, dataset_id: str, name: str) -> str:
        """Build deterministic feature set identifier"""
        slug = self._slugify(name)
        return f"{dataset_id}_{slug}"

    def _metadata_path(self, feature_set_id: str) -> Path:
        return self.features_dir / f"{feature_set_id}_metadata.json"

    def _features_path(self, feature_set_id: str) -> Path:
        return self.features_dir / f"{feature_set_id}.parquet"

    def _calculate_config_hash(self, config: Dict[str, Any]) -> str:
        payload = json.dumps(config, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _load_metadata(self, feature_set_id: str) -> Optional[Dict[str, Any]]:
        metadata_file = self._metadata_path(feature_set_id)
        if not metadata_file.exists():
            return None
        with open(metadata_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_metadata(self, feature_set_id: str, metadata: Dict[str, Any]) -> None:
        metadata_file = self._metadata_path(feature_set_id)
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

    def _load_dataset_frame(self, dataset_id: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load dataset dataframe and catalog metadata"""
        dataset_info = self.dataset_service.get_dataset(dataset_id)
        if not dataset_info:
            raise ValueError(f"Dataset not found: {dataset_id}")

        df = self.dataset_service.get_dataset_data(dataset_id, limit=-1)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").set_index("timestamp")
        else:
            df = df.sort_index()

        dataset_meta = {
            "hash": dataset_info.metadata.get("hash") if dataset_info.metadata else None,
            "num_rows": dataset_info.num_rows,
            "start_date": dataset_info.start_date.isoformat() if dataset_info.start_date else None,
            "end_date": dataset_info.end_date.isoformat() if dataset_info.end_date else None,
        }
        return df, dataset_meta

    def _load_existing_features(self, feature_set_id: str) -> Optional[pd.DataFrame]:
        feature_file = self._features_path(feature_set_id)
        if not feature_file.exists():
            return None
        return pd.read_parquet(feature_file)

    def _write_features(self, feature_set_id: str, df: pd.DataFrame) -> None:
        feature_file = self._features_path(feature_set_id)
        feature_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(feature_file)

    def list_feature_sets(self, dataset_id: Optional[str] = None) -> List[FeatureSetInfo]:
        """List all feature sets"""
        feature_sets = []

        # Scan features directory
        for feature_file in self.features_dir.glob("*_metadata.json"):
            try:
                with open(feature_file, "r") as f:
                    metadata = json.load(f)

                # Filter by dataset_id if provided
                if dataset_id and metadata.get("dataset_id") != dataset_id:
                    continue

                created_at = datetime.fromisoformat(metadata.get("created_at", datetime.now().isoformat()))
                updated_at_raw = metadata.get("updated_at", metadata.get("created_at"))
                updated_at = datetime.fromisoformat(updated_at_raw) if updated_at_raw else created_at

                last_ts_raw = metadata.get("last_processed_timestamp")
                last_ts = datetime.fromisoformat(last_ts_raw) if last_ts_raw else None

                feature_sets.append(
                    FeatureSetInfo(
                        id=feature_file.stem.replace("_metadata", ""),
                        name=metadata.get("name", feature_file.stem),
                        dataset_id=metadata.get("dataset_id", ""),
                        config=metadata.get("config", {}),
                        num_features=metadata.get("num_features", 0),
                        num_rows=metadata.get("num_rows", 0),
                        created_at=created_at,
                        updated_at=updated_at,
                        status=metadata.get("status", "completed"),
                        config_hash=metadata.get("config_hash"),
                        dataset_hash=metadata.get("dataset_hash"),
                        last_processed_timestamp=last_ts,
                        description=metadata.get("description"),
                    )
                )
            except Exception as e:
                print(f"Error loading feature set {feature_file}: {e}")
                continue

        # Sort by creation date (newest first)
        feature_sets.sort(key=lambda f: f.created_at, reverse=True)

        return feature_sets

    def get_feature_set(self, feature_set_id: str) -> Optional[FeatureSetInfo]:
        """Get feature set by ID"""
        metadata = self._load_metadata(feature_set_id)
        if not metadata:
            return None

        created_at = datetime.fromisoformat(metadata.get("created_at", datetime.now().isoformat()))
        updated_at_raw = metadata.get("updated_at", metadata.get("created_at"))
        updated_at = datetime.fromisoformat(updated_at_raw) if updated_at_raw else created_at
        last_ts_raw = metadata.get("last_processed_timestamp")
        last_ts = datetime.fromisoformat(last_ts_raw) if last_ts_raw else None

        return FeatureSetInfo(
            id=feature_set_id,
            name=metadata.get("name", feature_set_id),
            dataset_id=metadata.get("dataset_id", ""),
            config=metadata.get("config", {}),
            num_features=metadata.get("num_features", 0),
            num_rows=metadata.get("num_rows", 0),
            created_at=created_at,
            updated_at=updated_at,
            status=metadata.get("status", "completed"),
            config_hash=metadata.get("config_hash"),
            dataset_hash=metadata.get("dataset_hash"),
            last_processed_timestamp=last_ts,
            description=metadata.get("description"),
        )

    def generate_features(self, request: FeatureGenerationRequest) -> str:
        """Generate features from dataset"""
        # Generate feature set ID
        feature_set_id = f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request.dataset_id}"

        # Parse dataset_id to get ticker and timeframe
        parts = request.dataset_id.split("_")
        if len(parts) < 2:
            raise ValueError(f"Invalid dataset ID: {request.dataset_id}")

        ticker = parts[0]
        timeframe = "_".join(parts[1:])

        # Load dataset
        dataset_path = self.data_dir / ticker / timeframe / "data.parquet"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {request.dataset_id}")

        import pandas as pd

        data = pd.read_parquet(dataset_path)

        # Determine config
        if request.config:
            config = request.config
        elif request.config_path:
            from src.common.config import load_yaml_config

            config = load_yaml_config(Path("configs") / request.config_path)
        else:
            # Use default config
            config = {
                "features": [
                    {"type": "indicator", "name": "SMA", "params": {"period": 20}},
                    {"type": "indicator", "name": "EMA", "params": {"period": 20}},
                    {"type": "indicator", "name": "RSI", "params": {"period": 14}},
                    {"type": "indicator", "name": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}},
                    {"type": "indicator", "name": "BollingerBands", "params": {"period": 20, "std_dev": 2}},
                    {"type": "lags", "lags": [1, 2, 3, 5, 10], "columns": ["close"]},
                    {
                        "type": "rolling",
                        "window": 20,
                        "functions": ["mean", "std"],
                        "columns": ["close"],
                    },
                ],
                "selection": {"enabled": False},
                "cache_enabled": False,
            }

        # Generate features
        try:
            generator = FeatureGenerator(config)
            features_df = generator.generate(data, dataset_id=request.dataset_id, use_cache=False)

            # Save features
            features_file = self.features_dir / f"{feature_set_id}.parquet"
            features_df.to_parquet(features_file)

            # Save metadata
            config_hash = self._calculate_config_hash(config)
            metadata = {
                "id": feature_set_id,
                "name": request.name,
                "dataset_id": request.dataset_id,
                "config": config,
                "num_features": len(features_df.columns),
                "num_rows": len(features_df),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "status": "completed",
                "description": request.description,
                "columns": list(features_df.columns),
                "config_hash": config_hash,
            }

            self._save_metadata(feature_set_id, metadata)

            return feature_set_id

        except Exception as e:
            # Save error status
            metadata = {
                "id": feature_set_id,
                "name": request.name,
                "dataset_id": request.dataset_id,
                "config": config,
                "num_features": 0,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e),
                "description": request.description,
            }

            self._save_metadata(feature_set_id, metadata)
            raise

    # ------------------------------------------------------------------ #
    # Advanced incremental generation
    # ------------------------------------------------------------------ #
    def generate_features_incremental(
        self,
        *,
        dataset_id: str,
        feature_set_name: str,
        config: Dict[str, Any],
        chunk_size: int = 5000,
        incremental: bool = True,
        progress_callback: Optional[Callable[[int, int, str, Optional[datetime]], None]] = None,
        wait_if_paused: Optional[Callable[[], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> FeatureSetInfo:
        """
        Generate (or update) feature set with incremental support.
        """

        if chunk_size <= 0:
            chunk_size = self.default_chunk_size

        base_df, dataset_meta = self._load_dataset_frame(dataset_id)
        total_rows = len(base_df)
        if total_rows == 0:
            raise ValueError(f"Dataset {dataset_id} is empty")

        feature_set_id = self.build_feature_set_id(dataset_id, feature_set_name)
        existing_metadata = self._load_metadata(feature_set_id)
        config_hash = self._calculate_config_hash(config)
        dataset_hash = dataset_meta.get("hash")

        generator = FeatureGenerator(config, cache_enabled=False)

        previous_df = None
        base_offset = 0
        target_df = base_df
        history_df = None
        incremental_allowed = False
        if incremental and existing_metadata is not None:
            incremental_allowed = (
                existing_metadata.get("config_hash") == config_hash
                and existing_metadata.get("dataset_hash") == dataset_hash
            )

        if incremental_allowed and existing_metadata is not None:
            previous_df = self._load_existing_features(feature_set_id)
            last_ts_raw = existing_metadata.get("last_processed_timestamp")
            if last_ts_raw:
                last_ts = pd.to_datetime(last_ts_raw)
                newer_mask = target_df.index > last_ts
                target_df = target_df.loc[newer_mask]
                base_offset = total_rows - len(target_df)
                history_df = base_df.loc[:last_ts].tail(self.context_window)

        if target_df.empty:
            # Nothing new to compute
            if progress_callback:
                progress_callback(total_rows, total_rows, "Dataset already up to date", None)
            if existing_metadata is not None:
                existing_metadata["updated_at"] = datetime.now().isoformat()
                existing_metadata["status"] = "completed"
                self._save_metadata(feature_set_id, existing_metadata)
                feature_info = self.get_feature_set(feature_set_id)
                if feature_info is None:
                    raise RuntimeError("Feature set metadata missing after update")
                return feature_info
            # Should not reach here because target empty implies metadata exists
            raise ValueError("No data to process and no existing feature set found")

        processed_rows = 0
        chunk_results: List[pd.DataFrame] = []

        def _check_control():
            if is_cancelled and is_cancelled():
                raise RuntimeError("cancelled")
            if wait_if_paused:
                wait_if_paused()

        chunk_iter = range(0, len(target_df), chunk_size)
        for start in chunk_iter:
            _check_control()
            chunk = target_df.iloc[start : start + chunk_size]
            if history_df is not None and not history_df.empty:
                chunk_input = pd.concat([history_df, chunk])
                chunk_input = chunk_input[~chunk_input.index.duplicated(keep="last")]
            else:
                chunk_input = chunk

            features_chunk = generator.generate(
                chunk_input,
                dataset_id=f"{feature_set_id}_{start}",
                use_cache=False,
            )
            if not features_chunk.empty:
                features_chunk = features_chunk.loc[chunk.index]
                chunk_results.append(features_chunk)

            history_df = chunk_input.tail(self.context_window)
            processed_rows += len(chunk)

            if progress_callback:
                last_point = chunk.index[-1] if len(chunk.index) else None
                if last_point is not None and hasattr(last_point, "to_pydatetime"):
                    last_point = last_point.to_pydatetime()
                progress_callback(
                    base_offset + processed_rows,
                    total_rows,
                    f"Processed {base_offset + processed_rows}/{total_rows} rows",
                    last_point,
                )

        if not chunk_results:
            raise RuntimeError("Feature generation produced no results")

        new_features_df = pd.concat(chunk_results).sort_index()
        if previous_df is not None and not previous_df.empty:
            combined_df = pd.concat([previous_df, new_features_df])
            combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
            combined_df = combined_df.sort_index()
        else:
            combined_df = new_features_df

        self._write_features(feature_set_id, combined_df)

        metadata = dict(existing_metadata) if existing_metadata is not None else {}
        now = datetime.now().isoformat()
        metadata.update(
            {
                "id": feature_set_id,
                "name": feature_set_name,
                "dataset_id": dataset_id,
                "config": config,
                "config_hash": config_hash,
                "dataset_hash": dataset_hash,
                "num_features": combined_df.shape[1],
                "num_rows": combined_df.shape[0],
                "created_at": metadata.get("created_at", now),
                "updated_at": now,
                "status": "completed",
                "description": metadata.get("description"),
                "columns": combined_df.columns.tolist(),
                "last_processed_timestamp": (
                    combined_df.index.max().isoformat()
                    if not combined_df.empty
                    else metadata.get("last_processed_timestamp")
                ),
            }
        )

        self._save_metadata(feature_set_id, metadata)

        if progress_callback:
            final_ts = combined_df.index.max() if not combined_df.empty else None
            if final_ts is not None and hasattr(final_ts, "to_pydatetime"):
                final_ts = final_ts.to_pydatetime()
            progress_callback(total_rows, total_rows, "Feature generation completed", final_ts)

        feature_info = self.get_feature_set(feature_set_id)
        if not feature_info:
            raise RuntimeError("Feature metadata missing after generation")
        return feature_info

    def delete_feature_set(self, feature_set_id: str) -> bool:
        """Delete feature set"""
        deleted = False

        # Delete feature file
        features_file = self.features_dir / f"{feature_set_id}.parquet"
        if features_file.exists():
            features_file.unlink()
            deleted = True

        # Delete metadata
        metadata_file = self.features_dir / f"{feature_set_id}_metadata.json"
        if metadata_file.exists():
            metadata_file.unlink()

        return deleted

    def get_feature_data(
        self, feature_set_id: str, columns: Optional[List[str]] = None, limit: int = 1000
    ) -> Dict[str, Any]:
        """Get feature data"""
        features_file = self.features_dir / f"{feature_set_id}.parquet"
        if not features_file.exists():
            raise FileNotFoundError(f"Feature set not found: {feature_set_id}")

        import pandas as pd

        df = pd.read_parquet(features_file)

        # Select columns if specified
        if columns:
            df = df[columns]

        # Apply limit
        if limit > 0:
            df = df.head(limit)

        return {
            "feature_set_id": feature_set_id,
            "num_rows": len(df),
            "num_features": len(df.columns),
            "columns": list(df.columns),
            "data": df.reset_index().to_dict(orient="records"),
        }
