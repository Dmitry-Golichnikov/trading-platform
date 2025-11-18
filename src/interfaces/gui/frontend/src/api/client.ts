/**
 * API Client
 *
 * Axios-based client for backend API
 */
import axios from 'axios';
import type {
  DatasetInfo,
  ExperimentInfo,
  ModelInfo,
  BacktestResult,
  SystemHealth,
  FeatureSetInfo,
  FeatureTaskInfo,
  LabelingSetInfo,
  LabelingTaskInfo,
  LabelingTaskCreatePayload,
} from '@/types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle errors globally
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// Datasets API
export const datasetsAPI = {
  list: (params?: { ticker?: string; timeframe?: string }) =>
    apiClient.get<DatasetInfo[]>('/api/datasets', { params }),

  get: (datasetId: string) =>
    apiClient.get<DatasetInfo>(`/api/datasets/${datasetId}`),

  getData: (datasetId: string, params?: { start_date?: string; end_date?: string; limit?: number }) =>
    apiClient.get(`/api/datasets/${datasetId}/data`, { params }),

  getQuality: (datasetId: string) =>
    apiClient.get(`/api/datasets/${datasetId}/quality`),

  delete: (datasetId: string) =>
    apiClient.delete(`/api/datasets/${datasetId}`),
};

// Experiments API
export const experimentsAPI = {
  list: (params?: { status?: string; limit?: number }) =>
    apiClient.get<ExperimentInfo[]>('/api/experiments', { params }),

  get: (experimentId: string) =>
    apiClient.get<ExperimentInfo>(`/api/experiments/${experimentId}`),

  create: (data: { name: string; config: any; description?: string; tags?: Record<string, string> }) =>
    apiClient.post<ExperimentInfo>('/api/experiments', data),

  getStatus: (experimentId: string) =>
    apiClient.get(`/api/experiments/${experimentId}/status`),

  cancel: (experimentId: string) =>
    apiClient.post(`/api/experiments/${experimentId}/cancel`),

  delete: (experimentId: string) =>
    apiClient.delete(`/api/experiments/${experimentId}`),

  compare: (experimentIds: string[]) =>
    apiClient.post('/api/experiments/compare', experimentIds),
};

// Models API
export const modelsAPI = {
  list: (params?: { model_type?: string; experiment_id?: string; limit?: number }) =>
    apiClient.get<ModelInfo[]>('/api/models', { params }),

  get: (modelId: string) =>
    apiClient.get<ModelInfo>(`/api/models/${modelId}`),

  getMetrics: (modelId: string) =>
    apiClient.get(`/api/models/${modelId}/metrics`),

  deploy: (modelId: string) =>
    apiClient.post(`/api/models/${modelId}/deploy`),

  delete: (modelId: string) =>
    apiClient.delete(`/api/models/${modelId}`),
};

// Backtests API
export const backtestsAPI = {
  list: (params?: { model_id?: string; limit?: number }) =>
    apiClient.get<BacktestResult[]>('/api/backtests', { params }),

  get: (backtestId: string) =>
    apiClient.get<BacktestResult>(`/api/backtests/${backtestId}`),

  run: (data: { model_id: string; config: any; dataset?: string }) =>
    apiClient.post('/api/backtests/run', data),

  delete: (backtestId: string) =>
    apiClient.delete(`/api/backtests/${backtestId}`),
};

// Features API
export const featuresAPI = {
  list: (params?: { dataset_id?: string }) =>
    apiClient.get<FeatureSetInfo[]>('/api/features', { params }),

  get: (featureSetId: string) =>
    apiClient.get<FeatureSetInfo>(`/api/features/${featureSetId}`),

  getData: (featureSetId: string, params?: { columns?: string[]; limit?: number }) =>
    apiClient.get(`/api/features/${featureSetId}/data`, { params }),

  generate: (data: { name: string; dataset_id: string; config?: any; config_path?: string; description?: string }) =>
    apiClient.post('/api/features/generate', data),

  delete: (featureSetId: string) =>
    apiClient.delete(`/api/features/${featureSetId}`),

  listTasks: (params?: { status?: string; dataset_id?: string }) =>
    apiClient.get<FeatureTaskInfo[]>('/api/features/tasks', { params }),

  createTasks: (payload: any) =>
    apiClient.post<FeatureTaskInfo[]>('/api/features/tasks', payload),

  getTask: (taskId: string) =>
    apiClient.get<FeatureTaskInfo>(`/api/features/tasks/${taskId}`),

  pauseTask: (taskId: string) =>
    apiClient.post(`/api/features/tasks/${taskId}/pause`, {}),

  resumeTask: (taskId: string) =>
    apiClient.post(`/api/features/tasks/${taskId}/resume`, {}),

  cancelTask: (taskId: string) =>
    apiClient.post(`/api/features/tasks/${taskId}/cancel`, {}),

  restartTask: (taskId: string) =>
    apiClient.post(`/api/features/tasks/${taskId}/restart`, {}),
};

// Labeling API
export const labelingAPI = {
  list: (params?: { dataset_id?: string; method?: string }) =>
    apiClient.get<LabelingSetInfo[]>('/api/labeling', { params }),

  get: (labelingSetId: string) =>
    apiClient.get<LabelingSetInfo>(`/api/labeling/${labelingSetId}`),

  getData: (labelingSetId: string, params?: { limit?: number }) =>
    apiClient.get(`/api/labeling/${labelingSetId}/data`, { params }),

  delete: (labelingSetId: string) =>
    apiClient.delete(`/api/labeling/${labelingSetId}`),

  listTasks: (params?: { status?: string; dataset_id?: string }) =>
    apiClient.get<LabelingTaskInfo[]>('/api/labeling/tasks', { params }),

  createTasks: (payload: LabelingTaskCreatePayload) =>
    apiClient.post<LabelingTaskInfo[]>('/api/labeling/tasks', payload),

  getTask: (taskId: string) =>
    apiClient.get<LabelingTaskInfo>(`/api/labeling/tasks/${taskId}`),

  pauseTask: (taskId: string) =>
    apiClient.post(`/api/labeling/tasks/${taskId}/pause`, {}),

  resumeTask: (taskId: string) =>
    apiClient.post(`/api/labeling/tasks/${taskId}/resume`, {}),

  cancelTask: (taskId: string) =>
    apiClient.post(`/api/labeling/tasks/${taskId}/cancel`, {}),

  restartTask: (taskId: string) =>
    apiClient.post(`/api/labeling/tasks/${taskId}/restart`, {}),

  visualize: (labelingSetId: string, chartType: string) =>
    apiClient.get(`/api/labeling/${labelingSetId}/visualize`, { params: { chart_type: chartType } }),
};

// System API
export const systemAPI = {
  health: () =>
    apiClient.get<SystemHealth>('/api/system/health'),

  info: () =>
    apiClient.get('/api/system/info'),
};

export default apiClient;
