import { useMemo, useState } from 'react';
import { useQuery } from 'react-query';
import {
  Box,
  Typography,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  CircularProgress,
  IconButton,
  TextField,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  Tabs,
  Tab,
  LinearProgress,
  Chip,
  Stack,
  FormControlLabel,
  Checkbox,
  Card,
  CardContent,
  Divider,
  Tooltip,
  Grid,
  Switch,
  Slider,
  Radio,
  RadioGroup,
  FormLabel,
  Alert,
  Collapse,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  ToggleButton,
  ToggleButtonGroup,
  useTheme,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import VisibilityIcon from '@mui/icons-material/Visibility';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import StopIcon from '@mui/icons-material/Stop';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import RefreshIcon from '@mui/icons-material/Refresh';
import BarChartIcon from '@mui/icons-material/BarChart';
import TableChartIcon from '@mui/icons-material/TableChart';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import BookmarkIcon from '@mui/icons-material/Bookmark';
import Plot from 'react-plotly.js';
import { labelingAPI, datasetsAPI, featuresAPI } from '@/api/client';
import type { LabelingSetInfo, LabelingTaskInfo } from '@/types';
import { useStore } from '@/store';
import { labelingPresets, type LabelingPreset } from '@/configs/labelingPresets';

type TabValue = 'sets' | 'tasks';
type LabelingMethod = 'horizon' | 'triple_barrier' | 'regression';
type Direction = 'long' | 'short' | 'long+short';
type BarrierType = 'percentage' | 'atr' | 'volatility';

type BuilderState = {
  name: string;
  description: string;
  method: LabelingMethod;
  direction: Direction;

  // Horizon params
  horizonPeriod: number;
  horizonAdaptive: boolean;
  horizonThreshold: number;

  // Triple Barrier params
  upperBarrierType: BarrierType;
  upperBarrierValue: number;
  lowerBarrierType: BarrierType;
  lowerBarrierValue: number;
  timeBarrier: number;
  minReturn: number;
  asymmetricBarriers: boolean;

  // Regression params
  regressionTarget: 'future_return' | 'mfe' | 'mae' | 'sharpe';
  regressionHorizon: number;

  // Filters
  enableSmoothing: boolean;
  smoothingWindow: number;
  smoothingMethod: 'median' | 'mean' | 'exponential';

  enableSequenceFilter: boolean;
  minSequenceLength: number;

  enableMajorityVote: boolean;
  majorityWindow: number;

  enableDangerZones: boolean;
  volatilityThreshold: number;

  // Balancing
  balancingMethod: 'none' | 'class_weights' | 'oversampling' | 'undersampling';

  // Commission consideration
  considerCommissions: boolean;
  commissionRate: number;
};

const defaultBuilderState: BuilderState = {
  name: 'Long стратегия 2%-1%',
  description: '',
  method: 'triple_barrier',
  direction: 'long',

  horizonPeriod: 20,
  horizonAdaptive: false,
  horizonThreshold: 1.0,

  upperBarrierType: 'percentage',
  upperBarrierValue: 2.0,
  lowerBarrierType: 'percentage',
  lowerBarrierValue: 1.0,
  timeBarrier: 20,
  minReturn: 0.0,
  asymmetricBarriers: true,

  regressionTarget: 'future_return',
  regressionHorizon: 20,

  enableSmoothing: true,
  smoothingWindow: 3,
  smoothingMethod: 'median',

  enableSequenceFilter: true,
  minSequenceLength: 2,

  enableMajorityVote: false,
  majorityWindow: 5,

  enableDangerZones: true,
  volatilityThreshold: 3.0,

  balancingMethod: 'class_weights',

  considerCommissions: true,
  commissionRate: 0.05,
};

function TabPanel({
  value,
  current,
  children,
}: {
  value: TabValue;
  current: TabValue;
  children: React.ReactNode;
}) {
  if (value !== current) return null;
  return <Box mt={2}>{children}</Box>;
}

const statusColor: Record<string, 'default' | 'success' | 'warning' | 'info' | 'error'> = {
  queued: 'info',
  running: 'warning',
  paused: 'default',
  completed: 'success',
  failed: 'error',
  cancelled: 'default',
};

const buildLabelingConfig = (builder: BuilderState) => {
  const config: any = {
    method: builder.method,
    direction: builder.direction,
  };

  // Method-specific params
  if (builder.method === 'horizon') {
    config.horizon = builder.horizonPeriod;
    config.adaptive = builder.horizonAdaptive;
    config.threshold_pct = builder.horizonThreshold / 100;
  } else if (builder.method === 'triple_barrier') {
    config.upper_barrier = {
      type: builder.upperBarrierType,
      value: builder.upperBarrierValue / 100,
    };
    config.lower_barrier = {
      type: builder.lowerBarrierType,
      value: builder.lowerBarrierValue / 100,
    };
    config.time_barrier = builder.timeBarrier;
    config.min_return = builder.minReturn / 100;
  } else if (builder.method === 'regression') {
    config.target = builder.regressionTarget;
    config.horizon = builder.regressionHorizon;
  }

  // Filters
  const filters: any[] = [];

  if (builder.enableSmoothing) {
    filters.push({
      type: 'smoothing',
      params: {
        window: builder.smoothingWindow,
        method: builder.smoothingMethod,
      },
    });
  }

  if (builder.enableSequenceFilter) {
    filters.push({
      type: 'sequence',
      params: {
        min_length: builder.minSequenceLength,
      },
    });
  }

  if (builder.enableMajorityVote) {
    filters.push({
      type: 'majority_vote',
      params: {
        window: builder.majorityWindow,
      },
    });
  }

  if (builder.enableDangerZones) {
    filters.push({
      type: 'danger_zones',
      params: {
        high_volatility_threshold: builder.volatilityThreshold,
      },
    });
  }

  if (filters.length > 0) {
    config.filters = filters;
  }

  // Balancing
  if (builder.balancingMethod !== 'none') {
    config.balancing = {
      method: builder.balancingMethod,
      strategy: 'balanced',
    };
  }

  // Commissions
  if (builder.considerCommissions) {
    config.commission_rate = builder.commissionRate / 100;
  }

  return config;
};

export default function Labeling() {
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';
  const addNotification = useStore((state) => state.addNotification);
  const [tab, setTab] = useState<TabValue>('sets');
  const [datasetFilter, setDatasetFilter] = useState('');
  const [builder, setBuilder] = useState<BuilderState>(defaultBuilderState);
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const [selectedFeatureSet, setSelectedFeatureSet] = useState('');
  const [applyToAll, setApplyToAll] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [taskStatusFilter, setTaskStatusFilter] = useState('all');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showPresets, setShowPresets] = useState(false);

  // View Dialog State
  const [viewDialogOpen, setViewDialogOpen] = useState(false);
  const [viewMode, setViewMode] = useState<'table' | 'chart'>('table');
  const [chartType, setChartType] = useState<'candlestick' | 'line'>('candlestick');
  const [selectedLabelingSet, setSelectedLabelingSet] = useState<LabelingSetInfo | null>(null);

  const { data: datasets, isLoading: isDatasetsLoading } = useQuery('datasets', () =>
    datasetsAPI.list().then((res) => res.data)
  );

  const { data: featureSets } = useQuery('featureSets', () =>
    featuresAPI.list().then((res) => res.data)
  );

  const labelingSetsQuery = useQuery(['labelingSets', datasetFilter], () =>
    labelingAPI.list(datasetFilter ? { dataset_id: datasetFilter } : undefined).then((res) => res.data)
  );

  const tasksQuery = useQuery(
    ['labelingTasks', taskStatusFilter],
    () =>
      labelingAPI
        .listTasks(taskStatusFilter === 'all' ? undefined : { status: taskStatusFilter })
        .then((res) => res.data),
    { refetchInterval: 4000 }
  );

  const isLoading = labelingSetsQuery.isLoading || isDatasetsLoading;
  const filteredTasks = useMemo(() => tasksQuery.data ?? [], [tasksQuery.data]);

  const { data: viewData, isLoading: isViewDataLoading, error: viewError } = useQuery(
    ['labelingData', selectedLabelingSet?.id],
    () => {
      if (!selectedLabelingSet) return null;
      return labelingAPI.getData(selectedLabelingSet.id).then((res) => res.data);
    },
    {
      enabled: !!selectedLabelingSet && viewDialogOpen,
      retry: 1,
    }
  );

  const datasetsAvailable = datasets?.length ?? 0;
  const canSubmit = applyToAll ? datasetsAvailable > 0 : selectedDatasets.length > 0;

  const renderCellValue = (key: string, value: any, method?: string, config?: any) => {
    if (value === null || value === undefined) {
      return <Typography variant="caption" color="text.secondary">null</Typography>;
    }

    // Timestamp formatting
    if (['timestamp', 'date', 'time', 'created_at', 'updated_at'].includes(key)) {
      try {
        return new Date(value).toLocaleString();
      } catch (e) {
        return String(value);
      }
    }

    // Label handling
    if (key === 'label') {
      // Regression
      if (method === 'regression') {
        return (
          <Chip
            label={typeof value === 'number' ? value.toFixed(5) : String(value)}
            size="small"
            variant="outlined"
            color="info"
          />
        );
      }

      // Classification
      const numVal = Number(value);
      let color: 'default' | 'success' | 'error' | 'warning' = 'default';
      let text = String(value);
      const direction = config?.direction || 'long+short';

      if (numVal === 1) {
        // Label 1: Upper barrier hit
        if (direction === 'long') {
            color = 'success';
            text = 'LONG WIN (1)';
        } else if (direction === 'short') {
            color = 'error';
            text = 'SHORT LOSS (1)';
        } else {
            color = 'success';
            text = 'LONG (1)';
        }
      } else if (numVal === -1) {
        // Label -1: Lower barrier hit
        if (direction === 'long') {
            color = 'error';
            text = 'LONG LOSS (-1)';
        } else if (direction === 'short') {
            color = 'success';
            text = 'SHORT WIN (-1)';
        } else {
            color = 'error';
            text = 'SHORT (-1)';
        }
      } else if (numVal === 0) {
        color = 'default';
        text = 'HOLD (0)';
      }

      return <Chip label={text} color={color} size="small" />;
    }

    // Numeric formatting
    if (typeof value === 'number') {
      // Integer check
      if (Number.isInteger(value)) return value.toLocaleString();
      return value.toLocaleString(undefined, { maximumFractionDigits: 5 });
    }

    // Boolean
    if (typeof value === 'boolean') {
      return <Chip label={value ? 'TRUE' : 'FALSE'} size="small" variant="outlined" color={value ? 'success' : 'default'} />;
    }

    // Object/Array (sanitized as string from backend usually, but check just in case)
    if (typeof value === 'object') {
      return JSON.stringify(value);
    }

    return String(value);
  };

  const chartTraces = useMemo(() => {
    if (!viewData || !viewData.data || viewData.data.length === 0) return [];

    const data = viewData.data;
    // Ensure required columns exist
    const hasOHLC = data[0].open !== undefined && data[0].close !== undefined;

    if (!hasOHLC) return [];

    const timestamps = data.map((d: any) => d.timestamp);

    const traces: any[] = [];

    if (chartType === 'candlestick') {
      traces.push({
        x: timestamps,
        open: data.map((d: any) => d.open),
        high: data.map((d: any) => d.high),
        low: data.map((d: any) => d.low),
        close: data.map((d: any) => d.close),
        type: 'candlestick',
        name: 'Price',
        increasing: { line: { color: '#26a69a' } },
        decreasing: { line: { color: '#ef5350' } },
      });
    } else {
      traces.push({
        x: timestamps,
        y: data.map((d: any) => d.close),
        type: 'scatter',
        mode: 'lines',
        name: 'Close Price',
        line: { color: '#2196f3', width: 2 },
      });
    }

    // Add markers for labels
    if (selectedLabelingSet?.method !== 'regression') {
        const direction = selectedLabelingSet?.config?.direction || 'long+short';

        const longIndices = data.map((d: any, i: number) => Number(d.label) === 1 ? i : -1).filter((i: number) => i !== -1);
        const shortIndices = data.map((d: any, i: number) => Number(d.label) === -1 ? i : -1).filter((i: number) => i !== -1);

        // Helper to get y-coordinate based on chart type
        const getLongY = (i: number) => chartType === 'candlestick' ? data[i].low * 0.999 : data[i].close * 0.999;
        const getShortY = (i: number) => chartType === 'candlestick' ? data[i].high * 1.001 : data[i].close * 1.001;

        // Config based on direction
        let label1Name = 'Long';
        let label1Color = '#00c853'; // Green
        let labelMinus1Name = 'Short';
        let labelMinus1Color = '#d50000'; // Red

        if (direction === 'long') {
            label1Name = 'Long Win';
            labelMinus1Name = 'Long Loss';
        } else if (direction === 'short') {
            label1Name = 'Short Loss';
            label1Color = '#d50000'; // Red (price went up, bad for short)
            labelMinus1Name = 'Short Win';
            labelMinus1Color = '#00c853'; // Green (price went down, good for short)
        }

        if (longIndices.length > 0) {
            traces.push({
                x: longIndices.map((i: number) => timestamps[i]),
                y: longIndices.map((i: number) => getLongY(i)),
                mode: 'markers',
                type: 'scatter',
                name: label1Name,
                marker: { symbol: 'triangle-up', color: label1Color, size: 10 }
            });
        }

        if (shortIndices.length > 0) {
            traces.push({
                x: shortIndices.map((i: number) => timestamps[i]),
                y: shortIndices.map((i: number) => getShortY(i)),
                mode: 'markers',
                type: 'scatter',
                name: labelMinus1Name,
                marker: { symbol: 'triangle-down', color: labelMinus1Color, size: 10 }
            });
        }
    }

    return traces;
  }, [viewData, selectedLabelingSet, chartType]);

  const chartLayout = useMemo(() => ({
      autosize: true,
      height: 600,
      margin: { l: 50, r: 50, t: 30, b: 50 },
      paper_bgcolor: isDark ? '#1e1e1e' : '#fff',
      plot_bgcolor: isDark ? '#1e1e1e' : '#fff',
      font: { color: isDark ? '#fff' : '#000' },
      xaxis: {
          gridcolor: isDark ? '#333' : '#e0e0e0',
          rangeslider: { visible: false }
      },
      yaxis: { gridcolor: isDark ? '#333' : '#e0e0e0' },
      showlegend: true,
      legend: { orientation: 'h', y: 1.02, x: 0.5, xanchor: 'center' },
      uirevision: selectedLabelingSet?.id, // Critical for keeping zoom state on updates
  }), [isDark, selectedLabelingSet?.id]);

  const loadPreset = (preset: LabelingPreset) => {
    setBuilder({
      ...builder,
      name: preset.name,
      description: preset.description,
      method: preset.method,
      direction: preset.direction,
      ...preset.config,
    });
    setShowPresets(false);
    addNotification({ type: 'success', message: `Пресет "${preset.name}" загружен` });
  };

  const handleCreateTasks = async () => {
    try {
      if (!applyToAll && selectedDatasets.length === 0) {
        addNotification({ type: 'error', message: 'Выберите хотя бы один датасет' });
        return;
      }

      if (applyToAll && datasetsAvailable === 0) {
        addNotification({ type: 'error', message: 'Нет доступных датасетов для запуска' });
        return;
      }

      setIsCreating(true);

      const payload = {
        name: builder.name || 'Custom labeling',
        dataset_ids: applyToAll ? [] : selectedDatasets,
        feature_set_id: selectedFeatureSet || undefined,
        apply_to_all: applyToAll,
        method: builder.method,
        config: buildLabelingConfig(builder),
        description: builder.description,
        auto_start: true,
      };

      await labelingAPI.createTasks(payload);
      addNotification({ type: 'success', message: 'Задачи разметки поставлены в очередь' });
      tasksQuery.refetch();
      setTab('tasks');
    } catch (error: any) {
      addNotification({ type: 'error', message: error.response?.data?.detail || 'Не удалось создать задачи' });
    } finally {
      setIsCreating(false);
    }
  };

  const handleTaskAction = async (task: LabelingTaskInfo, action: 'pause' | 'resume' | 'cancel' | 'restart') => {
    try {
      if (action === 'pause') await labelingAPI.pauseTask(task.id);
      if (action === 'resume') await labelingAPI.resumeTask(task.id);
      if (action === 'cancel') await labelingAPI.cancelTask(task.id);
      if (action === 'restart') await labelingAPI.restartTask(task.id);
      tasksQuery.refetch();
    } catch (error: any) {
      addNotification({ type: 'error', message: error.response?.data?.detail || 'Операция не выполнена' });
    }
  };

  const deleteLabelingSet = async (labelingSet: LabelingSetInfo) => {
    if (!window.confirm(`Удалить набор разметки "${labelingSet.name}"?`)) {
      return;
    }

    try {
      await labelingAPI.delete(labelingSet.id);
      addNotification({ type: 'success', message: 'Набор разметки удалён' });
      labelingSetsQuery.refetch();
    } catch (error: any) {
      addNotification({ type: 'error', message: error.response?.data?.detail || 'Не удалось удалить набор' });
    }
  };

  const handleViewLabelingSet = (labelingSet: LabelingSetInfo) => {
    setSelectedLabelingSet(labelingSet);
    setViewDialogOpen(true);
  };

  const handleCloseViewDialog = () => {
    setViewDialogOpen(false);
    setSelectedLabelingSet(null);
  };

  const renderClassDistribution = (distribution?: Record<string, number>) => {
    if (!distribution || Object.keys(distribution).length === 0) return null;

    return (
      <Stack direction="row" spacing={1}>
        {Object.entries(distribution).map(([label, count]) => (
          <Chip
            key={label}
            label={`${label}: ${count}`}
            size="small"
            variant="outlined"
          />
        ))}
      </Stack>
    );
  };

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="80vh">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h4">Разметка таргетов (Labeling)</Typography>
        <Tabs value={tab} onChange={(_, value) => setTab(value)}>
          <Tab label="Наборы разметки" value="sets" />
          <Tab label="Задачи разметки" value="tasks" />
        </Tabs>
      </Box>

      <TabPanel value="sets" current={tab}>
        <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} mb={2} alignItems="flex-end">
          <FormControl sx={{ minWidth: 240 }} size="small">
            <InputLabel id="dataset-filter">Датасет</InputLabel>
            <Select
              labelId="dataset-filter"
              label="Датасет"
              value={datasetFilter}
              onChange={(e) => setDatasetFilter(e.target.value)}
            >
              <MenuItem value="">Все</MenuItem>
              {datasets?.map((ds) => (
                <MenuItem key={ds.id} value={ds.id}>
                  {ds.ticker} · {ds.timeframe}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Button startIcon={<RefreshIcon />} onClick={() => labelingSetsQuery.refetch()}>
            Обновить
          </Button>
        </Stack>

        <TableContainer component={Paper}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Название</TableCell>
                <TableCell>Датасет</TableCell>
                <TableCell>Метод</TableCell>
                <TableCell>Направление</TableCell>
                <TableCell align="right">Сэмплов</TableCell>
                <TableCell>Распределение классов</TableCell>
                <TableCell>Обновлено</TableCell>
                <TableCell>Статус</TableCell>
                <TableCell align="right">Действия</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {labelingSetsQuery.data && labelingSetsQuery.data.length > 0 ? (
                labelingSetsQuery.data.map((ls) => (
                  <TableRow key={ls.id} hover>
                    <TableCell>{ls.name}</TableCell>
                    <TableCell>{ls.dataset_id}</TableCell>
                    <TableCell>
                      <Chip size="small" label={ls.method} />
                    </TableCell>
                    <TableCell>
                      <Chip size="small" label={ls.config?.direction || 'N/A'} variant="outlined" />
                    </TableCell>
                    <TableCell align="right">{ls.num_samples?.toLocaleString() || 0}</TableCell>
                    <TableCell>{renderClassDistribution(ls.class_distribution)}</TableCell>
                    <TableCell>{ls.updated_at ? new Date(ls.updated_at).toLocaleString() : '—'}</TableCell>
                    <TableCell>
                      <Chip size="small" label={ls.status} color={statusColor[ls.status] || 'default'} />
                    </TableCell>
                    <TableCell align="right">
                      <Stack direction="row" spacing={1} justifyContent="flex-end">
                        <Tooltip title="Просмотреть данные">
                          <IconButton size="small" onClick={() => handleViewLabelingSet(ls)}>
                            <VisibilityIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Удалить набор">
                          <IconButton size="small" color="error" onClick={() => deleteLabelingSet(ls)}>
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </Stack>
                    </TableCell>
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell colSpan={9} align="center">
                    <Typography variant="body2" color="text.secondary">
                      Наборы разметки не найдены. Создайте их во вкладке «Задачи разметки».
                    </Typography>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </TabPanel>

      <TabPanel value="tasks" current={tab}>
        <Grid container spacing={3}>
          <Grid item xs={12} lg={7}>
            <Card variant="outlined">
              <CardContent>
                <Stack direction="row" justifyContent="space-between" alignItems="center" mb={2}>
                  <Typography variant="h6">Конфигуратор разметки</Typography>
                  <Stack direction="row" spacing={1}>
                    <Button
                      size="small"
                      startIcon={<BookmarkIcon />}
                      variant="outlined"
                      onClick={() => setShowPresets(!showPresets)}
                    >
                      Пресеты
                    </Button>
                    <Chip icon={<BarChartIcon />} label="Классификация & Регрессия" color="info" />
                  </Stack>
                </Stack>

                {/* Presets Section */}
                <Collapse in={showPresets}>
                  <Paper variant="outlined" sx={{ p: 2, mb: 2, bgcolor: 'action.hover' }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Выберите готовый пресет:
                    </Typography>
                    <Grid container spacing={1}>
                      {labelingPresets.map((preset) => (
                        <Grid item xs={12} md={6} key={preset.id}>
                          <Paper
                            variant="outlined"
                            sx={{
                              p: 1.5,
                              cursor: 'pointer',
                              '&:hover': { bgcolor: 'action.selected' },
                            }}
                            onClick={() => loadPreset(preset)}
                          >
                            <Stack direction="row" spacing={1} alignItems="center" mb={0.5}>
                              <Chip size="small" label={preset.method} color="primary" />
                              <Chip size="small" label={preset.direction} variant="outlined" />
                            </Stack>
                            <Typography variant="subtitle2">{preset.name}</Typography>
                            <Typography variant="caption" color="text.secondary">
                              {preset.description}
                            </Typography>
                          </Paper>
                        </Grid>
                      ))}
                    </Grid>
                  </Paper>
                </Collapse>

                <Stack spacing={3}>
                  <TextField
                    label="Название набора разметки"
                    value={builder.name}
                    onChange={(e) => setBuilder({ ...builder, name: e.target.value })}
                    fullWidth
                  />
                  <TextField
                    label="Описание (опционально)"
                    value={builder.description}
                    onChange={(e) => setBuilder({ ...builder, description: e.target.value })}
                    multiline
                    rows={2}
                  />

                  <Divider />

                  <FormControl component="fieldset">
                    <FormLabel component="legend">Метод разметки</FormLabel>
                    <RadioGroup
                      row
                      value={builder.method}
                      onChange={(e) => setBuilder({ ...builder, method: e.target.value as LabelingMethod })}
                    >
                      <FormControlLabel value="horizon" control={<Radio />} label="Horizon" />
                      <FormControlLabel value="triple_barrier" control={<Radio />} label="Triple Barrier" />
                      <FormControlLabel value="regression" control={<Radio />} label="Regression" />
                    </RadioGroup>
                  </FormControl>

                  {builder.method !== 'regression' && (
                    <FormControl component="fieldset">
                      <FormLabel component="legend">Направление торговли</FormLabel>
                      <RadioGroup
                        row
                        value={builder.direction}
                        onChange={(e) => setBuilder({ ...builder, direction: e.target.value as Direction })}
                      >
                        <FormControlLabel value="long" control={<Radio />} label="Long only" />
                        <FormControlLabel value="short" control={<Radio />} label="Short only" />
                        <FormControlLabel value="long+short" control={<Radio />} label="Long + Short" />
                      </RadioGroup>
                    </FormControl>
                  )}

                  <Divider />

                  {/* Horizon Method */}
                  {builder.method === 'horizon' && (
                    <Box>
                      <Alert severity="info" sx={{ mb: 2 }}>
                        <Typography variant="body2">
                          Метод Horizon: фиксированный или адаптивный горизонт прогнозирования
                        </Typography>
                      </Alert>

                      <Stack spacing={2}>
                        <TextField
                          label="Горизонт (баров)"
                          type="number"
                          value={builder.horizonPeriod}
                          onChange={(e) => setBuilder({ ...builder, horizonPeriod: parseInt(e.target.value, 10) })}
                          fullWidth
                        />
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={builder.horizonAdaptive}
                              onChange={(e) => setBuilder({ ...builder, horizonAdaptive: e.target.checked })}
                            />
                          }
                          label="Адаптивный горизонт (на основе ATR)"
                        />
                        <TextField
                          label="Порог для лонг/шорт (%)"
                          type="number"
                          value={builder.horizonThreshold}
                          onChange={(e) => setBuilder({ ...builder, horizonThreshold: parseFloat(e.target.value) })}
                          fullWidth
                          helperText="Минимальное изменение цены для присвоения класса"
                        />
                      </Stack>
                    </Box>
                  )}

                  {/* Triple Barrier Method */}
                  {builder.method === 'triple_barrier' && (
                    <Box>
                      <Alert severity="info" sx={{ mb: 2 }}>
                        <Typography variant="body2">
                          Triple Barrier: классический метод разметки с тремя барьерами (верхний, нижний, временной)
                        </Typography>
                      </Alert>

                      <Stack spacing={2}>
                        <Typography variant="subtitle2" color="primary">
                          Верхний барьер (Take Profit)
                        </Typography>
                        <Stack direction="row" spacing={2}>
                          <FormControl sx={{ minWidth: 150 }} size="small">
                            <InputLabel>Тип</InputLabel>
                            <Select
                              label="Тип"
                              value={builder.upperBarrierType}
                              onChange={(e) =>
                                setBuilder({ ...builder, upperBarrierType: e.target.value as BarrierType })
                              }
                            >
                              <MenuItem value="percentage">Процент</MenuItem>
                              <MenuItem value="atr">ATR</MenuItem>
                              <MenuItem value="volatility">Волатильность</MenuItem>
                            </Select>
                          </FormControl>
                          <TextField
                            label="Значение (%)"
                            type="number"
                            value={builder.upperBarrierValue}
                            onChange={(e) =>
                              setBuilder({ ...builder, upperBarrierValue: parseFloat(e.target.value) })
                            }
                            size="small"
                            fullWidth
                          />
                        </Stack>

                        <Typography variant="subtitle2" color="error">
                          Нижний барьер (Stop Loss)
                        </Typography>
                        <Stack direction="row" spacing={2}>
                          <FormControl sx={{ minWidth: 150 }} size="small">
                            <InputLabel>Тип</InputLabel>
                            <Select
                              label="Тип"
                              value={builder.lowerBarrierType}
                              onChange={(e) =>
                                setBuilder({ ...builder, lowerBarrierType: e.target.value as BarrierType })
                              }
                            >
                              <MenuItem value="percentage">Процент</MenuItem>
                              <MenuItem value="atr">ATR</MenuItem>
                              <MenuItem value="volatility">Волатильность</MenuItem>
                            </Select>
                          </FormControl>
                          <TextField
                            label="Значение (%)"
                            type="number"
                            value={builder.lowerBarrierValue}
                            onChange={(e) =>
                              setBuilder({ ...builder, lowerBarrierValue: parseFloat(e.target.value) })
                            }
                            size="small"
                            fullWidth
                          />
                        </Stack>

                        <TextField
                          label="Временной барьер (баров)"
                          type="number"
                          value={builder.timeBarrier}
                          onChange={(e) => setBuilder({ ...builder, timeBarrier: parseInt(e.target.value, 10) })}
                          fullWidth
                          helperText="Максимальное время удержания позиции"
                        />

                        <TextField
                          label="Минимальный return для учёта (%)"
                          type="number"
                          value={builder.minReturn}
                          onChange={(e) => setBuilder({ ...builder, minReturn: parseFloat(e.target.value) })}
                          fullWidth
                          helperText="Игнорировать сигналы с меньшим потенциальным профитом"
                        />

                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={builder.asymmetricBarriers}
                              onChange={(e) => setBuilder({ ...builder, asymmetricBarriers: e.target.checked })}
                            />
                          }
                          label="Асимметричные барьеры (разный TP и SL)"
                        />
                      </Stack>
                    </Box>
                  )}

                  {/* Regression Method */}
                  {builder.method === 'regression' && (
                    <Box>
                      <Alert severity="info" sx={{ mb: 2 }}>
                        <Typography variant="body2">
                          Regression: предсказание непрерывных значений (returns, MFE, MAE, Sharpe)
                        </Typography>
                      </Alert>

                      <Stack spacing={2}>
                        <FormControl fullWidth>
                          <InputLabel>Целевая переменная</InputLabel>
                          <Select
                            label="Целевая переменная"
                            value={builder.regressionTarget}
                            onChange={(e) =>
                              setBuilder({
                                ...builder,
                                regressionTarget: e.target.value as BuilderState['regressionTarget'],
                              })
                            }
                          >
                            <MenuItem value="future_return">Future Return</MenuItem>
                            <MenuItem value="mfe">Max Favorable Excursion (MFE)</MenuItem>
                            <MenuItem value="mae">Max Adverse Excursion (MAE)</MenuItem>
                            <MenuItem value="sharpe">Sharpe Ratio (rolling)</MenuItem>
                          </Select>
                        </FormControl>

                        <TextField
                          label="Горизонт (баров)"
                          type="number"
                          value={builder.regressionHorizon}
                          onChange={(e) =>
                            setBuilder({ ...builder, regressionHorizon: parseInt(e.target.value, 10) })
                          }
                          fullWidth
                        />
                      </Stack>
                    </Box>
                  )}

                  <Divider />

                  {/* Post-filters */}
                  <Box>
                    <Stack direction="row" justifyContent="space-between" alignItems="center" mb={1}>
                      <Typography variant="subtitle1">Постфильтры</Typography>
                      <Button
                        size="small"
                        startIcon={<InfoOutlinedIcon />}
                        onClick={() => setShowAdvanced(!showAdvanced)}
                      >
                        {showAdvanced ? 'Скрыть' : 'Показать'}
                      </Button>
                    </Stack>

                    <Collapse in={showAdvanced}>
                      <Stack spacing={2} mt={1}>
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={builder.enableSmoothing}
                              onChange={(e) => setBuilder({ ...builder, enableSmoothing: e.target.checked })}
                            />
                          }
                          label="Сглаживание сигналов"
                        />
                        {builder.enableSmoothing && (
                          <Stack direction="row" spacing={2} pl={4}>
                            <TextField
                              label="Окно"
                              type="number"
                              value={builder.smoothingWindow}
                              onChange={(e) =>
                                setBuilder({ ...builder, smoothingWindow: parseInt(e.target.value, 10) })
                              }
                              size="small"
                              sx={{ width: 120 }}
                            />
                            <FormControl size="small" sx={{ minWidth: 160 }}>
                              <InputLabel>Метод</InputLabel>
                              <Select
                                label="Метод"
                                value={builder.smoothingMethod}
                                onChange={(e) =>
                                  setBuilder({
                                    ...builder,
                                    smoothingMethod: e.target.value as BuilderState['smoothingMethod'],
                                  })
                                }
                              >
                                <MenuItem value="median">Median</MenuItem>
                                <MenuItem value="mean">Mean</MenuItem>
                                <MenuItem value="exponential">Exponential</MenuItem>
                              </Select>
                            </FormControl>
                          </Stack>
                        )}

                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={builder.enableSequenceFilter}
                              onChange={(e) => setBuilder({ ...builder, enableSequenceFilter: e.target.checked })}
                            />
                          }
                          label="Фильтр последовательностей (удаление одиночных сигналов)"
                        />
                        {builder.enableSequenceFilter && (
                          <TextField
                            label="Минимальная длина последовательности"
                            type="number"
                            value={builder.minSequenceLength}
                            onChange={(e) =>
                              setBuilder({ ...builder, minSequenceLength: parseInt(e.target.value, 10) })
                            }
                            size="small"
                            sx={{ width: 280, ml: 4 }}
                          />
                        )}

                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={builder.enableMajorityVote}
                              onChange={(e) => setBuilder({ ...builder, enableMajorityVote: e.target.checked })}
                            />
                          }
                          label="Majority Vote (голосование по соседним барам)"
                        />
                        {builder.enableMajorityVote && (
                          <TextField
                            label="Окно голосования"
                            type="number"
                            value={builder.majorityWindow}
                            onChange={(e) =>
                              setBuilder({ ...builder, majorityWindow: parseInt(e.target.value, 10) })
                            }
                            size="small"
                            sx={{ width: 200, ml: 4 }}
                          />
                        )}

                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={builder.enableDangerZones}
                              onChange={(e) => setBuilder({ ...builder, enableDangerZones: e.target.checked })}
                            />
                          }
                          label="Исключить опасные зоны (высокая волатильность)"
                        />
                        {builder.enableDangerZones && (
                          <Box pl={4}>
                            <Typography variant="caption" gutterBottom>
                              Порог волатильности (x σ)
                            </Typography>
                            <Slider
                              value={builder.volatilityThreshold}
                              onChange={(_, value) =>
                                setBuilder({ ...builder, volatilityThreshold: value as number })
                              }
                              min={1}
                              max={5}
                              step={0.1}
                              marks
                              valueLabelDisplay="auto"
                              sx={{ width: 280 }}
                            />
                          </Box>
                        )}
                      </Stack>
                    </Collapse>
                  </Box>

                  <Divider />

                  {/* Balancing */}
                  {builder.method !== 'regression' && (
                    <Box>
                      <Typography variant="subtitle1" gutterBottom>
                        Балансировка классов
                      </Typography>
                      <FormControl fullWidth size="small">
                        <InputLabel>Метод балансировки</InputLabel>
                        <Select
                          label="Метод балансировки"
                          value={builder.balancingMethod}
                          onChange={(e) =>
                            setBuilder({
                              ...builder,
                              balancingMethod: e.target.value as BuilderState['balancingMethod'],
                            })
                          }
                        >
                          <MenuItem value="none">Без балансировки</MenuItem>
                          <MenuItem value="class_weights">Class Weights</MenuItem>
                          <MenuItem value="oversampling">Oversampling (minority class)</MenuItem>
                          <MenuItem value="undersampling">Undersampling (majority class)</MenuItem>
                        </Select>
                      </FormControl>
                    </Box>
                  )}

                  <Divider />

                  {/* Commissions */}
                  <Box>
                    <FormControlLabel
                      control={
                        <Checkbox
                          checked={builder.considerCommissions}
                          onChange={(e) => setBuilder({ ...builder, considerCommissions: e.target.checked })}
                        />
                      }
                      label="Учитывать комиссии при разметке"
                    />
                    {builder.considerCommissions && (
                      <TextField
                        label="Комиссия (%)"
                        type="number"
                        value={builder.commissionRate}
                        onChange={(e) => setBuilder({ ...builder, commissionRate: parseFloat(e.target.value) })}
                        size="small"
                        sx={{ width: 200, ml: 4 }}
                        helperText="Для учёта в расчёте минимального профита"
                      />
                    )}
                  </Box>

                  <Divider />

                  {/* Dataset & Feature Set Selection */}
                  <Stack spacing={2}>
                    <FormControl size="small">
                      <InputLabel>Набор признаков (опционально)</InputLabel>
                      <Select
                        label="Набор признаков (опционально)"
                        value={selectedFeatureSet}
                        onChange={(e) => setSelectedFeatureSet(e.target.value)}
                      >
                        <MenuItem value="">Без привязки к признакам</MenuItem>
                        {featureSets?.map((fs) => (
                          <MenuItem key={fs.id} value={fs.id}>
                            {fs.name} · {fs.dataset_id}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>

                    <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} alignItems={{ xs: 'flex-start', md: 'center' }}>
                      <FormControl size="small" sx={{ minWidth: 240 }}>
                        <InputLabel>Датасеты для запуска</InputLabel>
                        <Select
                          multiple
                          disabled={applyToAll || datasetsAvailable === 0}
                          label="Датасеты для запуска"
                          value={selectedDatasets}
                          onChange={(e) => setSelectedDatasets(e.target.value as string[])}
                          renderValue={(selected) => (selected as string[]).length + ' выбрано'}
                        >
                          {datasets?.map((ds) => (
                            <MenuItem key={ds.id} value={ds.id}>
                              {ds.ticker} · {ds.timeframe}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                      <FormControlLabel
                        control={<Switch checked={applyToAll} onChange={(e) => setApplyToAll(e.target.checked)} />}
                        label="Для всех датасетов"
                      />
                    </Stack>
                  </Stack>

                  <Box display="flex" justifyContent="flex-end">
                    <Button variant="contained" onClick={handleCreateTasks} disabled={isCreating || !canSubmit}>
                      {isCreating ? 'Запуск...' : 'Запустить разметку'}
                    </Button>
                  </Box>
                  {applyToAll && datasetsAvailable === 0 && (
                    <Typography variant="caption" color="warning.main">
                      Нет доступных датасетов в каталоге — загрузите данные во вкладке Datasets.
                    </Typography>
                  )}
                </Stack>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} lg={5}>
            <Stack direction="row" justifyContent="space-between" alignItems="center" mb={2} spacing={2}>
              <Typography variant="h6">Активные задачи</Typography>
              <Stack direction="row" spacing={1}>
                <FormControl size="small" sx={{ minWidth: 160 }}>
                  <InputLabel>Статус</InputLabel>
                  <Select
                    label="Статус"
                    value={taskStatusFilter}
                    onChange={(e) => setTaskStatusFilter(e.target.value)}
                  >
                    <MenuItem value="all">Все</MenuItem>
                    <MenuItem value="queued">Queued</MenuItem>
                    <MenuItem value="running">Running</MenuItem>
                    <MenuItem value="paused">Paused</MenuItem>
                    <MenuItem value="completed">Completed</MenuItem>
                    <MenuItem value="failed">Failed</MenuItem>
                    <MenuItem value="cancelled">Cancelled</MenuItem>
                  </Select>
                </FormControl>
                <Button startIcon={<RefreshIcon />} onClick={() => tasksQuery.refetch()}>
                  Обновить
                </Button>
              </Stack>
            </Stack>

            <Stack spacing={2}>
              {filteredTasks.length === 0 && (
                <Paper variant="outlined">
                  <Box p={3} textAlign="center">
                    <Typography variant="body2" color="text.secondary">
                      Нет активных задач. Создайте задачу разметки слева.
                    </Typography>
                  </Box>
                </Paper>
              )}

              {filteredTasks.map((task) => (
                <Paper key={task.id} variant="outlined">
                  <Box p={2}>
                    <Stack direction="row" justifyContent="space-between" alignItems="center">
                      <Box>
                        <Typography variant="subtitle1">{task.name}</Typography>
                        <Stack direction="row" spacing={1} alignItems="center">
                          <Typography variant="body2" color="text.secondary">
                            {task.dataset_id}
                          </Typography>
                          <Chip size="small" label={task.method} />
                        </Stack>
                      </Box>
                      <Chip size="small" label={task.status} color={statusColor[task.status] || 'default'} />
                    </Stack>
                    <Box mt={2}>
                      <LinearProgress variant="determinate" value={task.progress * 100} />
                      <Stack direction="row" justifyContent="space-between" mt={0.5}>
                        <Typography variant="caption" color="text.secondary">
                          {Math.round(task.progress * 100)}%
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {task.processed_rows}/{task.total_rows || '—'} строк
                        </Typography>
                      </Stack>
                    </Box>
                    {task.class_distribution && (
                      <Box mt={1}>
                        <Typography variant="caption" color="text.secondary">
                          Распределение: {renderClassDistribution(task.class_distribution)}
                        </Typography>
                      </Box>
                    )}
                    {task.message && (
                      <Typography variant="caption" color="text.secondary" display="block" mt={1}>
                        {task.message}
                      </Typography>
                    )}
                    <Stack direction="row" spacing={1} mt={1}>
                      {task.status === 'running' && (
                        <Tooltip title="Пауза">
                          <IconButton size="small" onClick={() => handleTaskAction(task, 'pause')}>
                            <PauseIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      )}
                      {task.status === 'paused' && (
                        <Tooltip title="Возобновить">
                          <IconButton size="small" onClick={() => handleTaskAction(task, 'resume')}>
                            <PlayArrowIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      )}
                      {['queued', 'running', 'paused'].includes(task.status) && (
                        <Tooltip title="Отменить">
                          <IconButton size="small" color="error" onClick={() => handleTaskAction(task, 'cancel')}>
                            <StopIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      )}
                      {['failed', 'completed', 'cancelled'].includes(task.status) && (
                        <Tooltip title="Перезапустить">
                          <IconButton size="small" onClick={() => handleTaskAction(task, 'restart')}>
                            <RestartAltIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      )}
                    </Stack>
                  </Box>
                </Paper>
              ))}
            </Stack>
          </Grid>
        </Grid>
      </TabPanel>

      {/* View Data Dialog */}
      <Dialog open={viewDialogOpen} onClose={handleCloseViewDialog} maxWidth="xl" fullWidth>
        <DialogTitle>
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Box>
                <Typography variant="h6">Просмотр разметки: {selectedLabelingSet?.name}</Typography>
                {selectedLabelingSet && (
                    <Typography variant="caption" color="text.secondary">
                    Dataset: {selectedLabelingSet.dataset_id} | Samples: {selectedLabelingSet.num_samples}
                    </Typography>
                )}
            </Box>
            <Stack direction="row" spacing={2} alignItems="center">
                <ToggleButtonGroup
                    value={viewMode}
                    exclusive
                    onChange={(_, newMode) => newMode && setViewMode(newMode)}
                    size="small"
                >
                    <ToggleButton value="table">
                        <TableChartIcon sx={{ mr: 1 }} /> Таблица
                    </ToggleButton>
                    <ToggleButton value="chart">
                        <ShowChartIcon sx={{ mr: 1 }} /> График
                    </ToggleButton>
                </ToggleButtonGroup>

                {viewMode === 'chart' && (
                  <ToggleButtonGroup
                    value={chartType}
                    exclusive
                    onChange={(_, newType) => newType && setChartType(newType)}
                    size="small"
                    sx={{ ml: 2 }}
                  >
                    <Tooltip title="Свечи">
                      <ToggleButton value="candlestick">
                        <BarChartIcon />
                      </ToggleButton>
                    </Tooltip>
                    <Tooltip title="Линия">
                      <ToggleButton value="line">
                        <ShowChartIcon />
                      </ToggleButton>
                    </Tooltip>
                  </ToggleButtonGroup>
                )}

                <Chip
                size="small"
                label={selectedLabelingSet?.method}
                color="primary"
                variant="outlined"
                />
            </Stack>
          </Box>
        </DialogTitle>
        <DialogContent dividers sx={{ p: viewMode === 'chart' ? 1 : 2 }}>
          {isViewDataLoading ? (
            <Box display="flex" justifyContent="center" p={4}>
              <CircularProgress />
            </Box>
          ) : viewError ? (
             <Alert severity="error" sx={{ mt: 2 }}>
               Не удалось загрузить данные: {(viewError as any)?.response?.data?.detail || (viewError as any)?.message || 'Неизвестная ошибка'}
             </Alert>
          ) : viewData && viewData.data && viewData.data.length > 0 ? (
            viewMode === 'table' ? (
                <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 600 }}>
                <Table stickyHeader size="small">
                    <TableHead>
                    <TableRow>
                        {Object.keys(viewData.data[0]).map((key) => (
                        <TableCell
                            key={key}
                            sx={{
                            fontWeight: 'bold',
                            bgcolor: 'background.paper',
                            whiteSpace: 'nowrap'
                            }}
                        >
                            {key}
                        </TableCell>
                        ))}
                    </TableRow>
                    </TableHead>
                    <TableBody>
                    {viewData.data.map((row: any, idx: number) => (
                        <TableRow key={idx} hover>
                        {Object.entries(row).map(([key, val], cellIdx) => (
                            <TableCell key={cellIdx} sx={{ whiteSpace: 'nowrap' }}>
                            {renderCellValue(key, val, selectedLabelingSet?.method, selectedLabelingSet?.config)}
                            </TableCell>
                        ))}
                        </TableRow>
                    ))}
                    </TableBody>
                </Table>
                </TableContainer>
            ) : (
                <Box height={600}>
                    {chartTraces.length > 0 ? (
                        <Plot
                            data={chartTraces}
                            layout={chartLayout}
                            config={{ responsive: true, displayModeBar: true }}
                            style={{ width: '100%', height: '100%' }}
                        />
                    ) : (
                         <Box p={4} textAlign="center">
                            <Typography color="text.secondary">
                                Для отображения графика требуются колонки open, high, low, close.
                            </Typography>
                        </Box>
                    )}
                </Box>
            )
          ) : (
            <Box p={4} textAlign="center">
              <Typography color="text.secondary">Нет данных для отображения</Typography>
              {selectedLabelingSet && (
                  <Typography variant="caption" display="block" mt={1}>
                      ID: {selectedLabelingSet.id}
                  </Typography>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseViewDialog}>Закрыть</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
