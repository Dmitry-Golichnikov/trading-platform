"""
Dataset Service

Business logic for dataset operations.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

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

    def list_datasets(self, ticker: Optional[str] = None, timeframe: Optional[str] = None) -> List[DatasetInfo]:
        """List all available datasets"""
        datasets = []

        # Get from catalog
        catalog_datasets = self.catalog.search(ticker=ticker, timeframe=timeframe)

        for ds_metadata in catalog_datasets:
            try:
                # Calculate size from file if exists
                files = self._collect_dataset_files(ds_metadata.ticker, ds_metadata.timeframe)
                size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024) if files else 0.0

                datasets.append(
                    DatasetInfo(
                        id=f"{ds_metadata.ticker}_{ds_metadata.timeframe}",
                        ticker=ds_metadata.ticker,
                        timeframe=ds_metadata.timeframe,
                        source=ds_metadata.source,
                        start_date=ds_metadata.start_date,
                        end_date=ds_metadata.end_date,
                        num_rows=ds_metadata.total_bars,
                        size_mb=size_mb,
                        created_at=ds_metadata.created_at,
                        updated_at=ds_metadata.created_at,  # Use created_at as updated_at
                        quality_score=None,  # Not stored in catalog
                        metadata={"hash": ds_metadata.hash, "schema_version": ds_metadata.schema_version},
                    )
                )
            except Exception as e:
                print(f"Error processing dataset {ds_metadata}: {e}")
                continue

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
        self, dataset_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None, limit: int = 1000
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

        # Filter by date range
        if start_date:
            df = df[df["timestamp"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["timestamp"] <= pd.to_datetime(end_date)]

        # Apply limit
        if limit > 0:
            df = df.head(limit)

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
