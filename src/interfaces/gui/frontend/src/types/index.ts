/**
 * TypeScript Type Definitions
 */

export interface DatasetInfo {
  id: string;
  ticker: string;
  timeframe: string;
  source: string;
  start_date?: string;
  end_date?: string;
  num_rows: number;
  size_mb: number;
  created_at: string;
  updated_at: string;
  quality_score?: number;
  metadata: Record<string, any>;
}

export interface DatasetDataResponse {
  dataset_id: string;
  num_rows: number;
  columns: string[];
  data: Array<Record<string, any>>;
}

export interface ExperimentInfo {
  id: string;
  name: string;
  description?: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  config: Record<string, any>;
  metrics: Record<string, number>;
  tags: Record<string, string>;
  created_at: string;
  updated_at: string;
  started_at?: string;
  finished_at?: string;
  duration_seconds?: number;
  artifacts: string[];
}

export interface ModelInfo {
  id: string;
  name: string;
  type: string;
  experiment_id?: string;
  metrics: Record<string, number>;
  hyperparameters: Record<string, any>;
  feature_importance?: Record<string, number>;
  created_at: string;
  size_mb: number;
  is_deployed: boolean;
}

export interface BacktestResult {
  id: string;
  model_id: string;
  config: Record<string, any>;
  metrics: Record<string, number>;
  trades: any[];
  equity_curve: Array<{ date: string; value: number }>;
  created_at: string;
  duration_seconds: number;
}

export interface SystemHealth {
  status: string;
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  gpu_available: boolean;
  gpu_usage?: number;
  gpu_memory_usage?: number;
  active_tasks: number;
  timestamp: string;
}

export interface TrainingUpdate {
  experiment_id: string;
  phase: string;
  epoch?: number;
  total_epochs?: number;
  progress: number;
  metrics: Record<string, number>;
  message?: string;
  timestamp: string;
}

export interface TaskUpdate {
  task_id: string;
  status: string;
  progress: number;
  message?: string;
  timestamp: string;
}

export interface FeatureSetInfo {
  id: string;
  name: string;
  dataset_id: string;
  config: Record<string, any>;
  num_features: number;
  num_rows: number;
  created_at: string;
  updated_at?: string;
  status: string;
  config_hash?: string;
  dataset_hash?: string;
  last_processed_timestamp?: string;
  description?: string;
}

export type FeatureTaskStatus = 'queued' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';

export interface FeatureTaskInfo {
  id: string;
  name: string;
  dataset_id: string;
  feature_set_id: string;
  status: FeatureTaskStatus;
  progress: number;
  processed_rows: number;
  total_rows: number;
  config: Record<string, any>;
  config_hash: string;
  message?: string;
  created_at: string;
  updated_at: string;
  started_at?: string;
  finished_at?: string;
  last_processed_timestamp?: string;
  dataset_hash?: string;
  description?: string;
  apply_to_all?: boolean;
  batch_id?: string;
}

export interface FeatureTaskCreatePayload {
  name: string;
  dataset_ids?: string[];
  apply_to_all?: boolean;
  config: Record<string, any>;
  description?: string;
  chunk_size?: number;
  incremental?: boolean;
  auto_start?: boolean;
}
