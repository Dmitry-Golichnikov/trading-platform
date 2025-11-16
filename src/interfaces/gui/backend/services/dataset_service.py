"""
Dataset Service

Business logic for dataset operations.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - optional dependency
    pq = None

from src.data.quality.metrics import DataQualityMetrics
from src.data.storage.catalog import DatasetCatalog
from src.interfaces.gui.backend.api.models import DatasetInfo, DatasetQualityReport


class DatasetService:
    """Service for dataset operations"""

    def __init__(self, artifacts_dir: Path = Path("artifacts")):
        self.artifacts_dir = artifacts_dir
        self.data_dir = artifacts_dir / "data"
        self.catalog = DatasetCatalog()
        self.quality_checker = DataQualityMetrics()

    def _dataset_folder(self, ticker: str, timeframe: str) -> Path:
        return self.data_dir / ticker / timeframe

    def _collect_dataset_files(self, ticker: str, timeframe: str) -> List[Path]:
        """Return list of parquet files for dataset (single combined or multiple yearly)."""
        folder = self._dataset_folder(ticker, timeframe)
        if not folder.exists():
            return []

        combined = folder / "data.parquet"
        if combined.exists():
            return [combined]

        files = sorted(p for p in folder.glob("*.parquet") if p.is_file())
        return files

    def _parse_metadata_date(self, value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        value = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def _load_timeframe_metadata(self, timeframe_dir: Path) -> Dict[str, Any]:
        metadata_file = timeframe_dir / "metadata.json"
        if not metadata_file.exists():
            return {}
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            print(f"Failed to read metadata {metadata_file}: {exc}")
            return {}

    def _estimate_num_rows(self, files: List[Path]) -> int:
        if not files:
            return 0

        if pq is not None:
            total_rows = 0
            for file in files:
                try:
                    total_rows += pq.ParquetFile(file).metadata.num_rows
                except Exception:
                    continue
            if total_rows:
                return total_rows

        total_rows = 0
        for file in files:
            try:
                total_rows += len(pd.read_parquet(file, columns=["timestamp"]))
            except Exception:
                continue
        return total_rows

    def _list_datasets_from_files(self, ticker: Optional[str], timeframe: Optional[str]) -> Dict[str, DatasetInfo]:
        datasets: Dict[str, DatasetInfo] = {}
        if not self.data_dir.exists():
            return datasets

        for ticker_dir in sorted(p for p in self.data_dir.iterdir() if p.is_dir()):
            ticker_name = ticker_dir.name
            if ticker and ticker_name != ticker:
                continue

            for timeframe_dir in sorted(p for p in ticker_dir.iterdir() if p.is_dir()):
                timeframe_name = timeframe_dir.name
                if timeframe and timeframe_name != timeframe:
                    continue

                dataset_id = f"{ticker_name}_{timeframe_name}"
                files = sorted(timeframe_dir.glob("*.parquet"))
                size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024) if files else 0.0
                metadata = self._load_timeframe_metadata(timeframe_dir)

                num_rows = metadata.get("total_bars")
                if num_rows is None:
                    num_rows = self._estimate_num_rows(files)

                start_date = self._parse_metadata_date(metadata.get("start_date"))
                end_date = self._parse_metadata_date(metadata.get("end_date"))
                created_at = self._parse_metadata_date(metadata.get("created_at")) or datetime.utcnow()

                datasets[dataset_id] = DatasetInfo(
                    id=dataset_id,
                    ticker=ticker_name,
                    timeframe=timeframe_name,
                    source=metadata.get("source", "local"),
                    start_date=start_date,
                    end_date=end_date,
                    num_rows=num_rows,
                    size_mb=size_mb,
                    created_at=created_at,
                    updated_at=created_at,
                    quality_score=None,
                    metadata={"hash": metadata.get("hash"), "schema_version": metadata.get("schema_version")},
                )

        return datasets

    def list_datasets(self, ticker: Optional[str] = None, timeframe: Optional[str] = None) -> List[DatasetInfo]:
        """List all available datasets"""
        datasets_map: Dict[str, DatasetInfo] = self._list_datasets_from_files(ticker=ticker, timeframe=timeframe)

        # Supplement with catalog entries if missing on disk
        catalog_datasets = self.catalog.search(ticker=ticker, timeframe=timeframe)
        for ds_metadata in catalog_datasets:
            dataset_id = f"{ds_metadata.ticker}_{ds_metadata.timeframe}"
            if dataset_id in datasets_map:
                continue

            try:
                files = self._collect_dataset_files(ds_metadata.ticker, ds_metadata.timeframe)
                size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024) if files else 0.0
                datasets_map[dataset_id] = DatasetInfo(
                    id=dataset_id,
                    ticker=ds_metadata.ticker,
                    timeframe=ds_metadata.timeframe,
                    source=ds_metadata.source,
                    start_date=ds_metadata.start_date,
                    end_date=ds_metadata.end_date,
                    num_rows=ds_metadata.total_bars,
                    size_mb=size_mb,
                    created_at=ds_metadata.created_at,
                    updated_at=ds_metadata.created_at,
                    quality_score=None,
                    metadata={"hash": ds_metadata.hash, "schema_version": ds_metadata.schema_version},
                )
            except Exception as e:
                print(f"Error processing dataset {ds_metadata}: {e}")
                continue

        datasets = list(datasets_map.values())
        datasets.sort(key=lambda x: (x.ticker, x.timeframe))
        return datasets

    def get_dataset(self, dataset_id: str) -> Optional[DatasetInfo]:
        """Get dataset info by ID"""
        # Parse dataset_id: ticker_timeframe
        parts = dataset_id.split("_")
        if len(parts) < 2:
            return None

        ticker = parts[0]
        timeframe = "_".join(parts[1:])

        # Search for dataset by ticker and timeframe
        results = self.catalog.search(ticker=ticker, timeframe=timeframe)
        if not results:
            return None

        ds_metadata = results[0]  # Take first match

        # Calculate size from file if exists
        files = self._collect_dataset_files(ds_metadata.ticker, ds_metadata.timeframe)
        size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024) if files else 0.0

        return DatasetInfo(
            id=dataset_id,
            ticker=ds_metadata.ticker,
            timeframe=ds_metadata.timeframe,
            source=ds_metadata.source,
            start_date=ds_metadata.start_date,
            end_date=ds_metadata.end_date,
            num_rows=ds_metadata.total_bars,
            size_mb=size_mb,
            created_at=ds_metadata.created_at,
            updated_at=ds_metadata.created_at,
            quality_score=None,
            metadata={"hash": ds_metadata.hash, "schema_version": ds_metadata.schema_version},
        )

    def get_dataset_data(
        self, dataset_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None, limit: int = -1
    ) -> pd.DataFrame:
        """Get dataset data"""
        # Parse dataset_id
        parts = dataset_id.split("_")
        if len(parts) < 2:
            raise ValueError(f"Invalid dataset ID: {dataset_id}")

        ticker = parts[0]
        timeframe = "_".join(parts[1:])

        files = self._collect_dataset_files(ticker, timeframe)
        if not files:
            raise FileNotFoundError(f"Dataset not found: {dataset_id}")

        frames = []
        for file in files:
            try:
                frames.append(pd.read_parquet(file))
            except Exception as exc:
                print(f"Error reading dataset file {file}: {exc}")
                continue

        if not frames:
            raise ValueError(f"Dataset files unreadable for: {dataset_id}")

        df = pd.concat(frames, ignore_index=False)

        # Ensure timestamp sorted if column exists
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
            df = df.drop_duplicates(subset="timestamp", keep="last")

        # Filter by date range
        if start_date:
            df = df[df["timestamp"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["timestamp"] <= pd.to_datetime(end_date)]

        # Apply limit (take last N rows for most recent data)
        if limit and limit > 0:
            df = df.tail(limit)

        return df

    def generate_quality_report(self, dataset_id: str) -> DatasetQualityReport:
        """Generate quality report for dataset"""
        # Get dataset
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset not found: {dataset_id}")

        # Load data
        df = self.get_dataset_data(dataset_id, limit=-1)

        # Run quality checks
        metrics = self.quality_checker.get_all_metrics(df)

        # Calculate missing values
        num_missing = int(df.isna().sum().sum())

        # Calculate duplicates
        num_duplicates = len(df) - len(df.drop_duplicates())

        # Build issues list
        issues = []
        if metrics["completeness"] < 90:
            issues.append({"type": "low_completeness", "severity": "warning", "value": metrics["completeness"]})
        if metrics["validity"] < 95:
            issues.append({"type": "low_validity", "severity": "warning", "value": metrics["validity"]})
        if num_duplicates > 0:
            issues.append({"type": "duplicates", "severity": "info", "count": num_duplicates})

        # Build recommendations
        recommendations = []
        if metrics["completeness"] < 90:
            recommendations.append("Consider filling missing values or removing incomplete rows")
        if metrics["validity"] < 95:
            recommendations.append("Review data validity - check for negative prices or invalid OHLC relationships")
        if num_duplicates > 0:
            recommendations.append(f"Remove {num_duplicates} duplicate rows")

        return DatasetQualityReport(
            dataset_id=dataset_id,
            quality_score=metrics.get("quality_score", 0.0) / 100.0,  # Convert to 0-1 scale
            num_missing=num_missing,
            num_duplicates=num_duplicates,
            num_outliers=0,  # Not calculated by DataQualityMetrics
            issues=issues,
            recommendations=recommendations,
            generated_at=datetime.now(),
        )

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete dataset"""
        # Parse dataset_id
        parts = dataset_id.split("_")
        if len(parts) < 2:
            return False

        ticker = parts[0]
        timeframe = "_".join(parts[1:])

        # Find dataset in catalog
        results = self.catalog.search(ticker=ticker, timeframe=timeframe)
        if not results:
            return False

        # Delete from catalog (using UUID)
        for ds_metadata in results:
            try:
                self.catalog.delete(ds_metadata.dataset_id)
            except Exception as e:
                print(f"Error deleting dataset from catalog: {e}")

        # Delete files
        dataset_path = self.data_dir / ticker / timeframe
        if dataset_path.exists():
            import shutil

            shutil.rmtree(dataset_path)

        return True
