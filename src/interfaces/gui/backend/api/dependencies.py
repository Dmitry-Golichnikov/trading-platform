"""
API Dependencies

FastAPI dependencies for dependency injection.
"""

from pathlib import Path

from fastapi import Depends, HTTPException, status

from src.common.config import load_yaml_config
from src.orchestration.experiment_manager import ExperimentManager

# Paths
ARTIFACTS_DIR = Path("artifacts")
CONFIGS_DIR = Path("configs")
DATA_DIR = ARTIFACTS_DIR / "data"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"


def get_experiment_manager() -> ExperimentManager:
    """Get experiment manager instance"""
    try:
        return ExperimentManager()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize experiment manager: {str(e)}",
        )


def get_config(config_path: str) -> dict:
    """Load configuration from file"""
    try:
        config_file = CONFIGS_DIR / config_path
        if not config_file.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Configuration file not found: {config_path}"
            )
        return load_yaml_config(config_file)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to load configuration: {str(e)}"
        )


def validate_dataset_exists(dataset_id: str) -> Path:
    """Validate that dataset exists"""
    # Dataset format: {ticker}_{timeframe}
    parts = dataset_id.split("_")
    if len(parts) < 2:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid dataset ID format: {dataset_id}")

    ticker = parts[0]
    timeframe = "_".join(parts[1:])

    dataset_path = DATA_DIR / ticker / timeframe
    if not dataset_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset not found: {dataset_id}")

    return dataset_path


def validate_model_exists(model_id: str) -> Path:
    """Validate that model exists"""
    model_path = MODELS_DIR / f"{model_id}.pkl"
    if not model_path.exists():
        # Try with different extensions
        for ext in [".pt", ".pth", ".cbm", ".lgb"]:
            model_path = MODELS_DIR / f"{model_id}{ext}"
            if model_path.exists():
                return model_path

        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model not found: {model_id}")

    return model_path


def validate_experiment_exists(
    experiment_id: str, manager: ExperimentManager = Depends(get_experiment_manager)
) -> dict:
    """Validate that experiment exists"""
    try:
        experiment = manager.get_experiment(experiment_id)
        if experiment is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Experiment not found: {experiment_id}")
        return experiment
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get experiment: {str(e)}"
        )
