# GUI/UX детали

## 1. Архитектура GUI

### 1.1 Технологический стек

**Веб-приложение (рекомендуется):**
- **Frontend**: React + TypeScript
- **UI Library**: Material-UI или Ant Design
- **Charts**: Plotly.js или Lightweight Charts
- **State Management**: Redux Toolkit или Zustand
- **API Client**: Axios или React Query
- **Backend API**: FastAPI (Python)

**Альтернатива - Desktop:**
- **Electron** + React (кросс-платформенное)
- **Streamlit** (быстрое прототипирование)
- **Dash** (Plotly)

### 1.2 Компонентная структура

```
src/interfaces/gui/
├── components/
│   ├── common/
│   │   ├── Button.tsx
│   │   ├── Input.tsx
│   │   ├── Select.tsx
│   │   └── Table.tsx
│   ├── charts/
│   │   ├── CandlestickChart.tsx
│   │   ├── EquityCurve.tsx
│   │   ├── FeatureImportance.tsx
│   │   └── MetricsCard.tsx
│   ├── forms/
│   │   ├── ConfigForm.tsx
│   │   ├── BacktestForm.tsx
│   │   └── TrainingForm.tsx
│   └── layout/
│       ├── Header.tsx
│       ├── Sidebar.tsx
│       └── MainLayout.tsx
├── pages/
│   ├── Dashboard.tsx
│   ├── DataExplorer.tsx
│   ├── ModelTraining.tsx
│   ├── Backtesting.tsx
│   └── Results.tsx
├── hooks/
│   ├── useWebSocket.ts
│   ├── usePolling.ts
│   └── useLocalStorage.ts
├── services/
│   ├── api.ts
│   └── websocket.ts
└── utils/
    ├── formatters.ts
    └── validators.ts
```

## 2. Основные страницы и функции

### 2.1 Dashboard (Главная страница)

**Функционал:**
- Обзор активных пайплайнов
- Недавние эксперименты и результаты
- Метрики системы (CPU, GPU, Memory)
- Уведомления и алерты

**Компоненты:**
```tsx
// Dashboard.tsx
import React from 'react';
import { Grid, Card, CardContent } from '@mui/material';
import { ActivePipelines } from '../components/ActivePipelines';
import { SystemMetrics } from '../components/SystemMetrics';
import { RecentExperiments } from '../components/RecentExperiments';

export const Dashboard: React.FC = () => {
  return (
    <Grid container spacing={3}>
      <Grid item xs={12} md={8}>
        <ActivePipelines />
      </Grid>

      <Grid item xs={12} md={4}>
        <SystemMetrics />
      </Grid>

      <Grid item xs={12}>
        <RecentExperiments />
      </Grid>
    </Grid>
  );
};
```

### 2.2 Data Explorer

**Функционал:**
- Просмотр доступных датасетов
- Визуализация OHLCV данных с индикаторами
- Выбор тикеров и таймфреймов
- Детектирование аномалий и пропусков

**Ключевые компоненты:**

```tsx
// CandlestickChart.tsx
import React, { useMemo } from 'react';
import Plotly from 'plotly.js';
import createPlotlyComponent from 'react-plotly.js/factory';

const Plot = createPlotlyComponent(Plotly);

interface CandlestickChartProps {
  data: OHLCVData[];
  indicators?: IndicatorData[];
}

export const CandlestickChart: React.FC<CandlestickChartProps> = ({
  data,
  indicators = []
}) => {
  const chartData = useMemo(() => {
    const candlestick = {
      type: 'candlestick',
      x: data.map(d => d.timestamp),
      open: data.map(d => d.open),
      high: data.map(d => d.high),
      low: data.map(d => d.low),
      close: data.map(d => d.close),
      name: 'Price'
    };

    const indicatorTraces = indicators.map(ind => ({
      type: 'scatter',
      x: ind.timestamps,
      y: ind.values,
      name: ind.name,
      line: { color: ind.color }
    }));

    return [candlestick, ...indicatorTraces];
  }, [data, indicators]);

  return (
    <Plot
      data={chartData}
      layout={{
        title: 'Price Chart with Indicators',
        xaxis: { title: 'Time' },
        yaxis: { title: 'Price' },
        height: 600
      }}
      config={{
        responsive: true,
        displayModeBar: true
      }}
    />
  );
};
```

**Виртуализация для больших списков:**

```tsx
// TickerList.tsx (виртуализация с react-window)
import { FixedSizeList } from 'react-window';

interface TickerListProps {
  tickers: string[];
  onSelect: (ticker: string) => void;
}

export const TickerList: React.FC<TickerListProps> = ({ tickers, onSelect }) => {
  const Row = ({ index, style }) => (
    <div style={style} onClick={() => onSelect(tickers[index])}>
      {tickers[index]}
    </div>
  );

  return (
    <FixedSizeList
      height={600}
      itemCount={tickers.length}
      itemSize={35}
      width="100%"
    >
      {Row}
    </FixedSizeList>
  );
};
```

### 2.3 Model Training

**Функционал:**
- Настройка параметров обучения
- Выбор модели, признаков, таргетов
- Мониторинг прогресса обучения в реальном времени
- Визуализация loss curves

**Real-time Updates:**

```tsx
// TrainingProgress.tsx
import React, { useEffect, useState } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';

export const TrainingProgress: React.FC<{ trainingId: string }> = ({
  trainingId
}) => {
  const [progress, setProgress] = useState<TrainingProgress>({
    epoch: 0,
    totalEpochs: 100,
    loss: 0,
    valLoss: 0
  });

  // WebSocket для real-time updates
  const { data, connected } = useWebSocket(
    `ws://localhost:8000/training/${trainingId}`
  );

  useEffect(() => {
    if (data) {
      setProgress(data);
    }
  }, [data]);

  return (
    <div>
      <LinearProgress
        variant="determinate"
        value={(progress.epoch / progress.totalEpochs) * 100}
      />
      <Typography>
        Epoch {progress.epoch} / {progress.totalEpochs}
      </Typography>
      <Typography>
        Loss: {progress.loss.toFixed(4)} | Val Loss: {progress.valLoss.toFixed(4)}
      </Typography>
    </div>
  );
};
```

### 2.4 Backtesting

**Функционал:**
- Настройка параметров стратегии
- Запуск бэктестов
- Визуализация результатов (equity curve, drawdown)
- Сравнение стратегий

**Equity Curve:**

```tsx
// EquityCurve.tsx
export const EquityCurve: React.FC<{ results: BacktestResult[] }> = ({
  results
}) => {
  const data = [{
    type: 'scatter',
    mode: 'lines',
    x: results.map(r => r.timestamp),
    y: results.map(r => r.equity),
    name: 'Equity',
    line: { color: '#2196f3' }
  }];

  const layout = {
    title: 'Equity Curve',
    xaxis: { title: 'Time' },
    yaxis: { title: 'Equity' },
    hovermode: 'closest'
  };

  return <Plot data={data} layout={layout} />;
};
```

### 2.5 Results & Analysis

**Функционал:**
- Просмотр результатов экспериментов
- Сравнение метрик
- Feature importance
- Диагностика моделей

## 3. UX паттерны

### 3.1 Обработка Long-Running Operations

```tsx
// Компонент для долгих операций
export const LongRunningTask: React.FC<{
  taskId: string;
  onComplete: () => void;
}> = ({ taskId, onComplete }) => {
  const [status, setStatus] = useState<TaskStatus>('pending');
  const [progress, setProgress] = useState(0);
  const [canCancel, setCanCancel] = useState(true);

  // Polling для обновления статуса
  useEffect(() => {
    const interval = setInterval(async () => {
      const taskStatus = await api.getTaskStatus(taskId);
      setStatus(taskStatus.status);
      setProgress(taskStatus.progress);

      if (taskStatus.status === 'completed') {
        onComplete();
        clearInterval(interval);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [taskId]);

  const handleCancel = async () => {
    await api.cancelTask(taskId);
    setCanCancel(false);
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6">Task Running...</Typography>
        <LinearProgress variant="determinate" value={progress} />
        <Typography>{status}</Typography>

        {canCancel && (
          <Button onClick={handleCancel} color="error">
            Cancel
          </Button>
        )}
      </CardContent>
    </Card>
  );
};
```

### 3.2 Optimistic Updates

```tsx
// Optimistic UI updates
const useMutation = (mutationFn) => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const mutate = async (data, { onSuccess, onError, optimisticData }) => {
    setIsLoading(true);
    setError(null);

    // Немедленно обновить UI
    if (optimisticData) {
      updateCache(optimisticData);
    }

    try {
      const result = await mutationFn(data);
      onSuccess?.(result);
    } catch (err) {
      setError(err);
      // Откатить optimistic update
      if (optimisticData) {
        revertCache();
      }
      onError?.(err);
    } finally {
      setIsLoading(false);
    }
  };

  return { mutate, isLoading, error };
};
```

### 3.3 Error Boundaries

```tsx
// ErrorBoundary.tsx
class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error?: Error }
> {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
    // Отправить в систему мониторинга
    logErrorToService(error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <Card>
          <CardContent>
            <Typography variant="h5" color="error">
              Something went wrong
            </Typography>
            <Typography>
              {this.state.error?.message}
            </Typography>
            <Button onClick={() => this.setState({ hasError: false })}>
              Try Again
            </Button>
          </CardContent>
        </Card>
      );
    }

    return this.props.children;
  }
}
```

### 3.4 Keyboard Shortcuts

```tsx
// useKeyboardShortcuts hook
const useKeyboardShortcuts = (shortcuts: Record<string, () => void>) => {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const key = `${event.ctrlKey ? 'Ctrl+' : ''}${event.key}`;
      const handler = shortcuts[key];

      if (handler) {
        event.preventDefault();
        handler();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [shortcuts]);
};

// Использование
export const DataExplorer: React.FC = () => {
  useKeyboardShortcuts({
    'Ctrl+s': () => saveConfiguration(),
    'Ctrl+r': () => refreshData(),
    'Escape': () => closeModal()
  });

  return <div>...</div>;
};
```

## 4. Performance Optimization

### 4.1 Code Splitting

```tsx
// Lazy loading страниц
const Dashboard = React.lazy(() => import('./pages/Dashboard'));
const DataExplorer = React.lazy(() => import('./pages/DataExplorer'));
const ModelTraining = React.lazy(() => import('./pages/ModelTraining'));

function App() {
  return (
    <Suspense fallback={<Loading />}>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/data" element={<DataExplorer />} />
        <Route path="/training" element={<ModelTraining />} />
      </Routes>
    </Suspense>
  );
}
```

### 4.2 Memoization

```tsx
// Мемоизация дорогих вычислений
const ExpensiveChart = React.memo<ChartProps>(({ data }) => {
  const processedData = useMemo(() => {
    // Дорогая обработка данных
    return processChartData(data);
  }, [data]);

  return <Chart data={processedData} />;
});
```

### 4.3 Debouncing & Throttling

```tsx
// useDebounce hook
const useDebounce = <T,>(value: T, delay: number): T => {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => clearTimeout(handler);
  }, [value, delay]);

  return debouncedValue;
};

// Использование для поиска
const SearchInput: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const debouncedSearchTerm = useDebounce(searchTerm, 500);

  useEffect(() => {
    if (debouncedSearchTerm) {
      // Выполнить поиск
      performSearch(debouncedSearchTerm);
    }
  }, [debouncedSearchTerm]);

  return <input onChange={e => setSearchTerm(e.target.value)} />;
};
```

### 4.4 Virtualization

```tsx
// Виртуализация больших таблиц
import { useVirtual } from 'react-virtual';

export const VirtualizedTable: React.FC<{ rows: any[] }> = ({ rows }) => {
  const parentRef = useRef<HTMLDivElement>(null);

  const rowVirtualizer = useVirtual({
    size: rows.length,
    parentRef,
    estimateSize: useCallback(() => 35, []),
    overscan: 10
  });

  return (
    <div ref={parentRef} style={{ height: '600px', overflow: 'auto' }}>
      <div style={{ height: `${rowVirtualizer.totalSize}px` }}>
        {rowVirtualizer.virtualItems.map(virtualRow => (
          <div
            key={virtualRow.index}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: `${virtualRow.size}px`,
              transform: `translateY(${virtualRow.start}px)`
            }}
          >
            {rows[virtualRow.index]}
          </div>
        ))}
      </div>
    </div>
  );
};
```

## 5. Accessibility

### 5.1 ARIA Labels

```tsx
<button
  aria-label="Start training"
  aria-describedby="training-description"
  onClick={startTraining}
>
  <PlayIcon />
</button>

<p id="training-description">
  Start model training with current configuration
</p>
```

### 5.2 Keyboard Navigation

```tsx
// Все интерактивные элементы доступны с клавиатуры
<div
  role="button"
  tabIndex={0}
  onClick={handleClick}
  onKeyDown={e => {
    if (e.key === 'Enter' || e.key === ' ') {
      handleClick();
    }
  }}
>
  Interactive Element
</div>
```

### 5.3 Color Contrast

```tsx
// Использовать достаточный контраст (WCAG AA: 4.5:1)
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',  // Достаточный контраст с белым фоном
    },
    error: {
      main: '#d32f2f',
    }
  }
});
```

## 6. Responsive Design

### 6.1 Breakpoints

```tsx
import { useMediaQuery, useTheme } from '@mui/material';

export const ResponsiveLayout: React.FC = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const isTablet = useMediaQuery(theme.breakpoints.between('sm', 'md'));
  const isDesktop = useMediaQuery(theme.breakpoints.up('md'));

  return (
    <Grid container spacing={isMobile ? 1 : 3}>
      {/* Адаптивный layout */}
    </Grid>
  );
};
```

## 7. Offline Support

### 7.1 Service Worker

```tsx
// Регистрация service worker для offline mode
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js')
    .then(registration => {
      console.log('SW registered:', registration);
    });
}

// Показать offline banner
export const OfflineBanner: React.FC = () => {
  const [isOnline, setIsOnline] = useState(navigator.onLine);

  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  if (isOnline) return null;

  return (
    <Alert severity="warning">
      You are currently offline. Some features may be unavailable.
    </Alert>
  );
};
```

## 8. Best Practices

### 8.1 UX Guidelines
- ✅ Показывать прогресс для долгих операций
- ✅ Предоставлять возможность отмены операций
- ✅ Давать чёткую обратную связь на действия
- ✅ Использовать loading states
- ✅ Обрабатывать ошибки gracefully
- ✅ Сохранять состояние при navigation
- ✅ Keyboard shortcuts для power users
- ✅ Responsive design для разных экранов

### 8.2 Performance Guidelines
- ✅ Code splitting для больших приложений
- ✅ Lazy loading компонентов
- ✅ Virtualization для больших списков
- ✅ Debounce/throttle для частых updates
- ✅ Мемоизация дорогих вычислений
- ✅ Оптимизация ре-рендеров
- ✅ Prefetching критичных ресурсов

### 8.3 Антипаттерны
- ❌ Операции без индикации прогресса
- ❌ Блокирование UI при загрузке
- ❌ Отсутствие обработки ошибок
- ❌ Плохая доступность
- ❌ Игнорирование mobile users
- ❌ Ре-рендер всего дерева компонентов
- ❌ Отсутствие feedback на действия
