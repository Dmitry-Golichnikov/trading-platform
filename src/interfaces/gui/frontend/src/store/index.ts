/**
 * Global State Store (Zustand)
 */
import { create } from 'zustand';
import type {
  DatasetInfo,
  ExperimentInfo,
  ModelInfo,
  BacktestResult,
  SystemHealth,
} from '@/types';

interface AppState {
  // Datasets
  datasets: DatasetInfo[];
  selectedDataset: DatasetInfo | null;
  setDatasets: (datasets: DatasetInfo[]) => void;
  setSelectedDataset: (dataset: DatasetInfo | null) => void;

  // Experiments
  experiments: ExperimentInfo[];
  selectedExperiment: ExperimentInfo | null;
  setExperiments: (experiments: ExperimentInfo[]) => void;
  setSelectedExperiment: (experiment: ExperimentInfo | null) => void;

  // Models
  models: ModelInfo[];
  selectedModel: ModelInfo | null;
  setModels: (models: ModelInfo[]) => void;
  setSelectedModel: (model: ModelInfo | null) => void;

  // Backtests
  backtests: BacktestResult[];
  selectedBacktest: BacktestResult | null;
  setBacktests: (backtests: BacktestResult[]) => void;
  setSelectedBacktest: (backtest: BacktestResult | null) => void;

  // System
  systemHealth: SystemHealth | null;
  setSystemHealth: (health: SystemHealth | null) => void;

  // UI State
  darkMode: boolean;
  sidebarOpen: boolean;
  toggleDarkMode: () => void;
  toggleSidebar: () => void;

  // Notifications
  notifications: Array<{ id: string; type: 'info' | 'success' | 'warning' | 'error'; message: string }>;
  addNotification: (notification: { type: 'info' | 'success' | 'warning' | 'error'; message: string }) => void;
  removeNotification: (id: string) => void;
}

export const useStore = create<AppState>((set) => ({
  // Datasets
  datasets: [],
  selectedDataset: null,
  setDatasets: (datasets) => set({ datasets }),
  setSelectedDataset: (dataset) => set({ selectedDataset: dataset }),

  // Experiments
  experiments: [],
  selectedExperiment: null,
  setExperiments: (experiments) => set({ experiments }),
  setSelectedExperiment: (experiment) => set({ selectedExperiment: experiment }),

  // Models
  models: [],
  selectedModel: null,
  setModels: (models) => set({ models }),
  setSelectedModel: (model) => set({ selectedModel: model }),

  // Backtests
  backtests: [],
  selectedBacktest: null,
  setBacktests: (backtests) => set({ backtests }),
  setSelectedBacktest: (backtest) => set({ selectedBacktest: backtest }),

  // System
  systemHealth: null,
  setSystemHealth: (health) => set({ systemHealth: health }),

  // UI State
  darkMode: localStorage.getItem('darkMode') === 'true',
  sidebarOpen: true,
  toggleDarkMode: () =>
    set((state) => {
      const newDarkMode = !state.darkMode;
      localStorage.setItem('darkMode', String(newDarkMode));
      return { darkMode: newDarkMode };
    }),
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),

  // Notifications
  notifications: [],
  addNotification: (notification) =>
    set((state) => ({
      notifications: [
        ...state.notifications,
        { ...notification, id: Date.now().toString() },
      ],
    })),
  removeNotification: (id) =>
    set((state) => ({
      notifications: state.notifications.filter((n) => n.id !== id),
    })),
}));
