/**
 * Labeling Presets
 * 
 * Готовые конфигурации разметки для разных торговых стратегий
 */

export interface LabelingPreset {
  id: string;
  name: string;
  description: string;
  method: 'horizon' | 'triple_barrier' | 'regression';
  direction: 'long' | 'short' | 'long+short';
  config: Record<string, any>;
}

export const labelingPresets: LabelingPreset[] = [
  {
    id: 'long_conservative',
    name: 'Long консервативная 2%/1%',
    description: 'Консервативная long стратегия с take profit 2% и stop loss 1%',
    method: 'triple_barrier',
    direction: 'long',
    config: {
      upperBarrierType: 'percentage',
      upperBarrierValue: 2.0,
      lowerBarrierType: 'percentage',
      lowerBarrierValue: 1.0,
      timeBarrier: 20,
      minReturn: 0.0,
      asymmetricBarriers: true,
      enableSmoothing: true,
      smoothingWindow: 3,
      smoothingMethod: 'median',
      enableSequenceFilter: true,
      minSequenceLength: 2,
      enableDangerZones: true,
      volatilityThreshold: 3.0,
      balancingMethod: 'class_weights',
      considerCommissions: true,
      commissionRate: 0.05,
    },
  },
  {
    id: 'long_aggressive',
    name: 'Long агрессивная 3%/1.5%',
    description: 'Агрессивная long стратегия с большими профитами и рисками',
    method: 'triple_barrier',
    direction: 'long',
    config: {
      upperBarrierType: 'percentage',
      upperBarrierValue: 3.0,
      lowerBarrierType: 'percentage',
      lowerBarrierValue: 1.5,
      timeBarrier: 30,
      minReturn: 0.5,
      asymmetricBarriers: true,
      enableSmoothing: true,
      smoothingWindow: 5,
      smoothingMethod: 'exponential',
      enableSequenceFilter: true,
      minSequenceLength: 3,
      enableDangerZones: true,
      volatilityThreshold: 2.5,
      balancingMethod: 'class_weights',
      considerCommissions: true,
      commissionRate: 0.05,
    },
  },
  {
    id: 'long_short_balanced',
    name: 'Long+Short симметричная 1.5%',
    description: 'Симметричная long/short стратегия с одинаковыми барьерами',
    method: 'triple_barrier',
    direction: 'long+short',
    config: {
      upperBarrierType: 'percentage',
      upperBarrierValue: 1.5,
      lowerBarrierType: 'percentage',
      lowerBarrierValue: 1.5,
      timeBarrier: 15,
      minReturn: 0.0,
      asymmetricBarriers: false,
      enableSmoothing: true,
      smoothingWindow: 3,
      smoothingMethod: 'median',
      enableSequenceFilter: true,
      minSequenceLength: 2,
      enableMajorityVote: true,
      majorityWindow: 5,
      enableDangerZones: true,
      volatilityThreshold: 3.0,
      balancingMethod: 'class_weights',
      considerCommissions: true,
      commissionRate: 0.05,
    },
  },
  {
    id: 'atr_adaptive',
    name: 'ATR-адаптивные барьеры',
    description: 'Динамические барьеры на основе ATR для адаптации к волатильности',
    method: 'triple_barrier',
    direction: 'long',
    config: {
      upperBarrierType: 'atr',
      upperBarrierValue: 2.0,
      lowerBarrierType: 'atr',
      lowerBarrierValue: 1.0,
      timeBarrier: 20,
      minReturn: 0.0,
      asymmetricBarriers: true,
      enableSmoothing: true,
      smoothingWindow: 3,
      smoothingMethod: 'median',
      enableSequenceFilter: true,
      minSequenceLength: 2,
      enableDangerZones: true,
      volatilityThreshold: 3.5,
      balancingMethod: 'class_weights',
      considerCommissions: true,
      commissionRate: 0.05,
    },
  },
  {
    id: 'horizon_fixed',
    name: 'Horizon фиксированный 20 баров',
    description: 'Классификация на основе движения цены за 20 баров',
    method: 'horizon',
    direction: 'long+short',
    config: {
      horizonPeriod: 20,
      horizonAdaptive: false,
      horizonThreshold: 1.0,
      enableSmoothing: true,
      smoothingWindow: 3,
      smoothingMethod: 'median',
      enableSequenceFilter: true,
      minSequenceLength: 2,
      balancingMethod: 'class_weights',
      considerCommissions: true,
      commissionRate: 0.05,
    },
  },
  {
    id: 'horizon_adaptive',
    name: 'Horizon адаптивный (ATR)',
    description: 'Адаптивный горизонт на основе ATR, подстраивается под волатильность',
    method: 'horizon',
    direction: 'long',
    config: {
      horizonPeriod: 20,
      horizonAdaptive: true,
      horizonThreshold: 1.0,
      enableSmoothing: true,
      smoothingWindow: 5,
      smoothingMethod: 'exponential',
      enableSequenceFilter: true,
      minSequenceLength: 3,
      enableDangerZones: true,
      volatilityThreshold: 3.0,
      balancingMethod: 'class_weights',
      considerCommissions: true,
      commissionRate: 0.05,
    },
  },
  {
    id: 'regression_returns',
    name: 'Regression: Future Returns',
    description: 'Предсказание будущей доходности (регрессия)',
    method: 'regression',
    direction: 'long',
    config: {
      regressionTarget: 'future_return',
      regressionHorizon: 20,
      enableSmoothing: false,
      considerCommissions: true,
      commissionRate: 0.05,
    },
  },
  {
    id: 'regression_mfe',
    name: 'Regression: Max Favorable Excursion',
    description: 'Предсказание максимального благоприятного движения',
    method: 'regression',
    direction: 'long',
    config: {
      regressionTarget: 'mfe',
      regressionHorizon: 30,
      enableSmoothing: false,
      considerCommissions: false,
    },
  },
  {
    id: 'scalping',
    name: 'Скальпинг 0.5%/0.3%',
    description: 'Быстрая торговля с малыми барьерами',
    method: 'triple_barrier',
    direction: 'long+short',
    config: {
      upperBarrierType: 'percentage',
      upperBarrierValue: 0.5,
      lowerBarrierType: 'percentage',
      lowerBarrierValue: 0.3,
      timeBarrier: 5,
      minReturn: 0.0,
      asymmetricBarriers: true,
      enableSmoothing: false,
      enableSequenceFilter: false,
      enableDangerZones: true,
      volatilityThreshold: 2.0,
      balancingMethod: 'class_weights',
      considerCommissions: true,
      commissionRate: 0.05,
    },
  },
  {
    id: 'swing_trading',
    name: 'Swing Trading 5%/2.5%',
    description: 'Долгосрочная торговля с большими целями',
    method: 'triple_barrier',
    direction: 'long',
    config: {
      upperBarrierType: 'percentage',
      upperBarrierValue: 5.0,
      lowerBarrierType: 'percentage',
      lowerBarrierValue: 2.5,
      timeBarrier: 50,
      minReturn: 1.0,
      asymmetricBarriers: true,
      enableSmoothing: true,
      smoothingWindow: 7,
      smoothingMethod: 'exponential',
      enableSequenceFilter: true,
      minSequenceLength: 5,
      enableDangerZones: true,
      volatilityThreshold: 3.0,
      balancingMethod: 'undersampling',
      considerCommissions: true,
      commissionRate: 0.05,
    },
  },
];

/**
 * Получить пресет по ID
 */
export const getPresetById = (id: string): LabelingPreset | undefined => {
  return labelingPresets.find((preset) => preset.id === id);
};

/**
 * Получить все пресеты по методу
 */
export const getPresetsByMethod = (method: 'horizon' | 'triple_barrier' | 'regression'): LabelingPreset[] => {
  return labelingPresets.filter((preset) => preset.method === method);
};

