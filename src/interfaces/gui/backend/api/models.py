"""
API Models

Pydantic models for request/response validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Enums
class ExperimentStatus(str, Enum):
    """Experiment status enum"""

    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class TaskStatus(str, Enum):
    """Task status enum"""

    queued = "queued"
    running = "running"
    paused = "paused"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class TrainingPhase(str, Enum):
    """Training phase enum"""

    data_loading = "data_loading"
    feature_generation = "feature_generation"
    label_generation = "label_generation"
    model_training = "model_training"
    evaluation = "evaluation"
    completed = "completed"


# Request models
class DatasetUploadRequest(BaseModel):
    """Dataset upload request"""

    ticker: str
    timeframe: str
    source: str = "local"
    description: Optional[str] = None


class ExperimentCreateRequest(BaseModel):
    """Experiment creation request"""

    name: str = Field(..., description="Experiment name")
    description: Optional[str] = Field(None, description="Experiment description")
    config: Dict[str, Any] = Field(..., description="Experiment configuration")
    tags: Optional[Dict[str, str]] = Field(default_factory=dict)


class BacktestRunRequest(BaseModel):
    """Backtest run request"""

    model_id: str = Field(..., description="Model ID to backtest")
    config: Dict[str, Any] = Field(..., description="Backtest configuration")
    dataset: Optional[str] = Field(None, description="Dataset to use")


class PipelineRunRequest(BaseModel):
    """Pipeline run request"""

    pipeline_type: str = Field(..., description="Pipeline type")
    config: Dict[str, Any] = Field(..., description="Pipeline configuration")


class HyperoptRunRequest(BaseModel):
    """Hyperparameter optimization request"""

    model_type: str = Field(..., description="Model type")
    search_space: Dict[str, Any] = Field(..., description="Search space")
    n_trials: int = Field(100, description="Number of trials")
    config: Dict[str, Any] = Field(default_factory=dict)


# Response models
class DatasetInfo(BaseModel):
    """Dataset information"""

    id: str
    ticker: str
    timeframe: str
    source: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    num_rows: int = 0
    size_mb: float = 0.0
    created_at: datetime
    updated_at: datetime
    quality_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DatasetQualityReport(BaseModel):
    """Dataset quality report"""

    dataset_id: str
    quality_score: float
    num_missing: int
    num_duplicates: int
    num_outliers: int
    issues: List[Dict[str, Any]]
    recommendations: List[str]
    generated_at: datetime


class ExperimentInfo(BaseModel):
    """Experiment information"""

    id: str
    name: str
    description: Optional[str] = None
    status: ExperimentStatus
    config: Dict[str, Any]
    metrics: Dict[str, float] = Field(default_factory=dict)
    tags: Dict[str, str] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    artifacts: List[str] = Field(default_factory=list)


class ExperimentStatusResponse(BaseModel):
    """Experiment status response"""

    id: str
    status: str
    phase: Optional[TrainingPhase] = None
    progress: float = 0.0
    message: Optional[str] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    updated_at: datetime


class ModelInfo(BaseModel):
    """Model information"""

    id: str
    name: str
    type: str
    experiment_id: Optional[str] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    feature_importance: Optional[Dict[str, float]] = None
    created_at: datetime
    size_mb: float = 0.0
    is_deployed: bool = False


class ModelMetrics(BaseModel):
    """Model metrics"""

    model_id: str
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Optional[Dict[str, float]] = None
    feature_importance: Optional[Dict[str, float]] = None
    confusion_matrix: Optional[List[List[int]]] = None
    calibration_data: Optional[Dict[str, Any]] = None


class BacktestResult(BaseModel):
    """Backtest result"""

    id: str
    model_id: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    trades: List[Dict[str, Any]] = Field(default_factory=list)
    equity_curve: List[Dict[str, float]] = Field(default_factory=list)
    created_at: datetime
    duration_seconds: float


class PipelineInfo(BaseModel):
    """Pipeline information"""

    id: str
    type: str
    status: TaskStatus
    config: Dict[str, Any]
    progress: float = 0.0
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None


class TaskInfo(BaseModel):
    """Task information"""

    id: str
    type: str
    status: TaskStatus
    progress: float = 0.0
    message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class SystemHealth(BaseModel):
    """System health status"""

    status: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    gpu_available: bool
    gpu_usage: Optional[float] = None
    gpu_memory_usage: Optional[float] = None
    active_tasks: int
    timestamp: datetime


class ExperimentComparison(BaseModel):
    """Experiment comparison"""

    experiments: List[ExperimentInfo]
    metrics: Dict[str, List[float]]
    best_experiment_id: Optional[str] = None


# WebSocket messages
class WSMessage(BaseModel):
    """WebSocket message base"""

    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class TrainingUpdate(BaseModel):
    """Training update message"""

    experiment_id: str
    phase: TrainingPhase
    epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    progress: float
    metrics: Dict[str, float] = Field(default_factory=dict)
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class TaskUpdate(BaseModel):
    """Task update message"""

    task_id: str
    status: TaskStatus
    progress: float
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# Feature generation models
class FeatureSetInfo(BaseModel):
    """Feature set information"""

    id: str
    name: str
    dataset_id: str
    config: Dict[str, Any]
    num_features: int
    num_rows: int = 0
    created_at: datetime
    updated_at: Optional[datetime] = None
    status: str
    config_hash: Optional[str] = None
    dataset_hash: Optional[str] = None
    last_processed_timestamp: Optional[datetime] = None
    description: Optional[str] = None


class FeatureGenerationRequest(BaseModel):
    """Feature generation request"""

    name: str = Field(..., description="Feature set name")
    dataset_id: str = Field(..., description="Dataset ID to use")
    config_path: Optional[str] = Field(None, description="Path to feature config YAML")
    config: Optional[Dict[str, Any]] = Field(None, description="Inline feature configuration")
    description: Optional[str] = Field(None, description="Description")


class FeatureTaskCreateRequest(BaseModel):
    """Feature generation task creation request"""

    name: str = Field(..., description="Logical name of the feature set")
    dataset_ids: Optional[List[str]] = Field(None, description="Specific dataset IDs to process")
    apply_to_all: bool = Field(False, description="Generate for all available datasets")
    config: Dict[str, Any] = Field(..., description="Feature configuration payload")
    description: Optional[str] = Field(None, description="Notes or rationale")
    chunk_size: int = Field(5000, description="Processing chunk size")
    incremental: bool = Field(True, description="Only build missing portion when dataset updated")
    auto_start: bool = Field(True, description="Start tasks immediately after creation")


class FeatureTaskInfo(BaseModel):
    """Feature generation task information"""

    id: str
    name: str
    dataset_id: str
    feature_set_id: str
    status: TaskStatus
    progress: float = 0.0
    processed_rows: int = 0
    total_rows: int = 0
    config: Dict[str, Any]
    config_hash: str
    message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    last_processed_timestamp: Optional[datetime] = None
    dataset_hash: Optional[str] = None
    description: Optional[str] = None
    apply_to_all: bool = False
    batch_id: Optional[str] = None


class FeatureTaskActionResponse(BaseModel):
    """Response returned after task action"""

    task: FeatureTaskInfo
    message: str
