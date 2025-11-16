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
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import StopIcon from '@mui/icons-material/Stop';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import RefreshIcon from '@mui/icons-material/Refresh';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import { featuresAPI, datasetsAPI } from '@/api/client';
import type { FeatureSetInfo, FeatureTaskInfo } from '@/types';
import { useStore } from '@/store';

type TabValue = 'sets' | 'tasks';

type IndicatorCatalogItem = {
  key: string;
  label: string;
  name: string;
  params: Record<string, any>;
};

type IndicatorCatalog = Record<'trend' | 'momentum' | 'volatility' | 'volume', IndicatorCatalogItem[]>;

type BuilderState = {
  presetName: string;
  description: string;
  trendIndicators: string[];
  momentumIndicators: string[];
  volatilityIndicators: string[];
  volumeIndicators: string[];
  includeCalendar: boolean;
  calendarFeatures: string[];
  includeTickerEncoding: boolean;
  tickerEncoding: 'onehot' | 'label' | 'target';
  includeHigherTimeframe: boolean;
  higherTimeframes: string[];
  includeLags: boolean;
  lags: number[];
  includeRolling: boolean;
  rollingWindows: number[];
  includeDifferences: boolean;
  chunkSize: number;
  incremental: boolean;
};

const indicatorCatalog: IndicatorCatalog = {
  trend: [
    { key: 'SMA_20', label: 'SMA (20)', name: 'SMA', params: { period: 20 } },
    { key: 'EMA_20', label: 'EMA (20)', name: 'EMA', params: { period: 20 } },
    { key: 'WMA_20', label: 'WMA (20)', name: 'WMA', params: { period: 20 } },
    { key: 'MACD_12_26', label: 'MACD 12/26', name: 'MACD', params: { fast: 12, slow: 26, signal: 9 } },
    { key: 'ADX_14', label: 'ADX (14)', name: 'ADX', params: { period: 14 } },
    { key: 'ICHIMOKU', label: 'Ichimoku', name: 'Ichimoku', params: {} },
    { key: 'SAR', label: 'Parabolic SAR', name: 'ParabolicSAR', params: {} },
  ],
  momentum: [
    { key: 'RSI_14', label: 'RSI (14)', name: 'RSI', params: { period: 14 } },
    { key: 'STOCH', label: 'Stochastic', name: 'Stochastic', params: { k_period: 14, d_period: 3 } },
    { key: 'STOCH_RSI', label: 'Stochastic RSI', name: 'StochasticRSI', params: {} },
    { key: 'CCI_20', label: 'CCI (20)', name: 'CCI', params: { period: 20 } },
    { key: 'WILLIAMS', label: "Williams %R", name: 'WilliamsR', params: { period: 14 } },
    { key: 'TRIX', label: 'TRIX', name: 'TRIX', params: { period: 15 } },
    { key: 'ELDER', label: 'Elder Force', name: 'ElderForce', params: { period: 13 } },
    { key: 'DPO', label: 'DPO', name: 'DPO', params: { period: 20 } },
  ],
  volatility: [
    { key: 'BBANDS', label: 'Bollinger Bands', name: 'BollingerBands', params: { period: 20, std_dev: 2 } },
    { key: 'ATR_14', label: 'ATR (14)', name: 'ATR', params: { period: 14 } },
    { key: 'KELTNER', label: 'Keltner', name: 'KeltnerChannels', params: { ema_period: 20, atr_period: 10 } },
    { key: 'DONCHIAN', label: 'Donchian', name: 'DonchianChannels', params: { period: 20 } },
    { key: 'HEIKIN', label: 'Heikin-Ashi', name: 'HeikinAshi', params: {} },
    { key: 'NADARAYA', label: 'Nadaraya-Watson', name: 'NadarayaWatson', params: { bandwidth: 20 } },
  ],
  volume: [
    { key: 'VWAP', label: 'VWAP', name: 'VWAP', params: {} },
    { key: 'OBV', label: 'OBV', name: 'OBV', params: {} },
    { key: 'CMF', label: 'Chaikin MF', name: 'ChaikinMoneyFlow', params: { period: 20 } },
    { key: 'MFI', label: 'MFI', name: 'MoneyFlowIndex', params: { period: 14 } },
    { key: 'ACCDIST', label: 'Accum/Dist', name: 'AccumulationDistribution', params: {} },
    { key: 'VOLUME_PROFILE', label: 'Volume Profile', name: 'VolumeProfile', params: { bins: 20 } },
  ],
};

const calendarOptions = [
  { key: 'hour', label: 'Hour' },
  { key: 'day_of_week', label: 'Day of Week' },
  { key: 'month', label: 'Month' },
  { key: 'is_month_start', label: 'Is Month Start' },
  { key: 'is_month_end', label: 'Is Month End' },
];

const higherTimeframes = ['4h', '1d'];

const defaultBuilderState: BuilderState = {
  presetName: 'Technical suite',
  description: '',
  trendIndicators: ['SMA_20', 'EMA_20', 'MACD_12_26', 'ADX_14'],
  momentumIndicators: ['RSI_14', 'STOCH'],
  volatilityIndicators: ['BBANDS', 'ATR_14'],
  volumeIndicators: ['OBV', 'VWAP'],
  includeCalendar: true,
  calendarFeatures: ['hour', 'day_of_week', 'is_month_end'],
  includeTickerEncoding: true,
  tickerEncoding: 'onehot',
  includeHigherTimeframe: false,
  higherTimeframes: [],
  includeLags: true,
  lags: [1, 2, 3, 5, 10],
  includeRolling: true,
  rollingWindows: [5, 20],
  includeDifferences: true,
  chunkSize: 4000,
  incremental: true,
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

const buildFeatureConfig = (builder: BuilderState) => {
  const features: any[] = [];

  const collect = (keys: string[], catalog: IndicatorCatalogItem[]) => {
    keys.forEach((key) => {
      const item = catalog.find((entry) => entry.key === key);
      if (item) {
        features.push({ type: 'indicator', name: item.name, params: item.params });
      }
    });
  };

  collect(builder.trendIndicators, indicatorCatalog.trend);
  collect(builder.momentumIndicators, indicatorCatalog.momentum);
  collect(builder.volatilityIndicators, indicatorCatalog.volatility);
  collect(builder.volumeIndicators, indicatorCatalog.volume);

  if (builder.includeCalendar && builder.calendarFeatures.length) {
    features.push({ type: 'calendar', features: builder.calendarFeatures });
  }
  if (builder.includeTickerEncoding) {
    features.push({ type: 'ticker', encoding: builder.tickerEncoding });
  }
  if (builder.includeLags && builder.lags.length) {
    features.push({ type: 'lags', lags: builder.lags, columns: ['close'] });
  }
  if (builder.includeRolling && builder.rollingWindows.length) {
    builder.rollingWindows.forEach((window) => {
      features.push({
        type: 'rolling',
        window,
        functions: ['mean', 'std'],
        columns: ['close', 'volume'],
      });
    });
  }
  if (builder.includeDifferences) {
    features.push({ type: 'differences', periods: [1, 5], columns: ['close'], method: 'diff' });
    features.push({ type: 'differences', periods: [1], columns: ['close'], method: 'pct_change' });
  }
  if (builder.includeHigherTimeframe && builder.higherTimeframes.length) {
    builder.higherTimeframes.forEach((tf) => {
      features.push({
        type: 'higher_timeframe',
        source_tf: tf,
        indicators: ['SMA', 'RSI', 'MACD'],
        alignment: 'forward_fill',
      });
    });
  }

  return {
    version: '1.0',
    features,
    selection: { enabled: false },
    cache_enabled: false,
  };
};

export default function Features() {
  const addNotification = useStore((state) => state.addNotification);
  const [tab, setTab] = useState<TabValue>('sets');
  const [datasetFilter, setDatasetFilter] = useState('');
  const [builder, setBuilder] = useState<BuilderState>(defaultBuilderState);
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
const [applyToAll, setApplyToAll] = useState(false);
const [isCreating, setIsCreating] = useState(false);
  const [taskStatusFilter, setTaskStatusFilter] = useState('all');

  const { data: datasets, isLoading: isDatasetsLoading } = useQuery('datasets', () =>
    datasetsAPI.list().then((res) => res.data)
  );

  const featureSetsQuery = useQuery(['featureSets', datasetFilter], () =>
    featuresAPI.list(datasetFilter ? { dataset_id: datasetFilter } : undefined).then((res) => res.data)
  );

  const tasksQuery = useQuery(
    ['featureTasks', taskStatusFilter],
    () =>
      featuresAPI
        .listTasks(taskStatusFilter === 'all' ? undefined : { status: taskStatusFilter })
        .then((res) => res.data),
    { refetchInterval: 4000 }
  );

  const isLoading = featureSetsQuery.isLoading || isDatasetsLoading;
  const filteredTasks = useMemo(() => tasksQuery.data ?? [], [tasksQuery.data]);

  const datasetsAvailable = datasets?.length ?? 0;
  const canSubmit = applyToAll ? datasetsAvailable > 0 : selectedDatasets.length > 0;

  const toggleIndicator = (group: keyof IndicatorCatalog, key: string) => {
    setBuilder((prev) => {
      const field = `${group}Indicators` as keyof BuilderState;
      const list = prev[field] as string[];
      const exists = list.includes(key);
      const updated = exists ? list.filter((item) => item !== key) : [...list, key];
      return { ...prev, [field]: updated };
    });
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
        name: builder.presetName || 'Custom features',
        dataset_ids: applyToAll ? [] : selectedDatasets,
        apply_to_all: applyToAll,
        config: buildFeatureConfig(builder),
        description: builder.description,
        chunk_size: builder.chunkSize,
        incremental: builder.incremental,
        auto_start: true,
      };

      await featuresAPI.createTasks(payload);
      addNotification({ type: 'success', message: 'Задачи генерации поставлены в очередь' });
      tasksQuery.refetch();
      setTab('tasks');
    } catch (error: any) {
      addNotification({ type: 'error', message: error.response?.data?.detail || 'Не удалось создать задачи' });
    } finally {
      setIsCreating(false);
    }
  };

  const handleTaskAction = async (task: FeatureTaskInfo, action: 'pause' | 'resume' | 'cancel' | 'restart') => {
    try {
      if (action === 'pause') await featuresAPI.pauseTask(task.id);
      if (action === 'resume') await featuresAPI.resumeTask(task.id);
      if (action === 'cancel') await featuresAPI.cancelTask(task.id);
      if (action === 'restart') await featuresAPI.restartTask(task.id);
      tasksQuery.refetch();
    } catch (error: any) {
      addNotification({ type: 'error', message: error.response?.data?.detail || 'Операция не выполнена' });
    }
  };

  const deleteFeatureSet = async (featureSet: FeatureSetInfo) => {
    if (!window.confirm(`Удалить набор признаков "${featureSet.name}"?`)) {
      return;
    }

    try {
      await featuresAPI.delete(featureSet.id);
      addNotification({ type: 'success', message: 'Набор признаков удалён' });
      featureSetsQuery.refetch();
    } catch (error: any) {
      addNotification({ type: 'error', message: error.response?.data?.detail || 'Не удалось удалить набор' });
    }
  };

  const renderIndicatorGroup = (
    title: string,
    group: keyof IndicatorCatalog,
    selected: string[],
  ) => (
    <Box>
      <Typography variant="subtitle2" gutterBottom>
        {title}
      </Typography>
      <Grid container spacing={1}>
        {indicatorCatalog[group].map((item) => (
          <Grid item xs={6} md={4} key={item.key}>
            <FormControlLabel
              control={
                <Checkbox checked={selected.includes(item.key)} onChange={() => toggleIndicator(group, item.key)} />
              }
              label={item.label}
            />
          </Grid>
        ))}
      </Grid>
    </Box>
  );

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
        <Typography variant="h4">Feature Engineering</Typography>
        <Tabs value={tab} onChange={(_, value) => setTab(value)}>
          <Tab label="Наборы признаков" value="sets" />
          <Tab label="Задачи генерации" value="tasks" />
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
          <Button startIcon={<RefreshIcon />} onClick={() => featureSetsQuery.refetch()}>
            Обновить
          </Button>
        </Stack>

        <TableContainer component={Paper}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Название</TableCell>
                <TableCell>Датасет</TableCell>
                <TableCell align="right">Признаков</TableCell>
                <TableCell align="right">Строк</TableCell>
                <TableCell>Обновлено</TableCell>
                <TableCell>Статус</TableCell>
                <TableCell align="right">Действия</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {featureSetsQuery.data && featureSetsQuery.data.length > 0 ? (
                featureSetsQuery.data.map((fs) => (
                  <TableRow key={fs.id} hover>
                    <TableCell>{fs.name}</TableCell>
                    <TableCell>{fs.dataset_id}</TableCell>
                    <TableCell align="right">{fs.num_features}</TableCell>
                    <TableCell align="right">{fs.num_rows}</TableCell>
                    <TableCell>{fs.updated_at ? new Date(fs.updated_at).toLocaleString() : '—'}</TableCell>
                    <TableCell>
                      <Chip size="small" label={fs.status} color={statusColor[fs.status] || 'default'} />
                    </TableCell>
                    <TableCell align="right">
                      <Tooltip title="Удалить набор">
                        <IconButton size="small" color="error" onClick={() => deleteFeatureSet(fs)}>
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell colSpan={7} align="center">
                    <Typography variant="body2" color="text.secondary">
                      Наборы признаков не найдены. Сформируйте их во вкладке «Задачи».
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
                  <Typography variant="h6">Конфигуратор признаков</Typography>
                  <Chip icon={<AutoAwesomeIcon />} label="Сохраняется и переиспользуется" color="info" />
                </Stack>

                <Stack spacing={2}>
                  <TextField
                    label="Название набора признаков"
                    value={builder.presetName}
                    onChange={(e) => setBuilder({ ...builder, presetName: e.target.value })}
                    fullWidth
                  />
                  <TextField
                    label="Описание (опционально)"
                    value={builder.description}
                    onChange={(e) => setBuilder({ ...builder, description: e.target.value })}
                    multiline
                    rows={2}
                  />

                  {renderIndicatorGroup('Трендовые индикаторы', 'trend', builder.trendIndicators)}
                  <Divider />
                  {renderIndicatorGroup('Моментум индикаторы', 'momentum', builder.momentumIndicators)}
                  <Divider />
                  {renderIndicatorGroup('Волатильность', 'volatility', builder.volatilityIndicators)}
                  <Divider />
                  {renderIndicatorGroup('Объёмные', 'volume', builder.volumeIndicators)}

                  <Divider />

                  <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
                    <FormControlLabel
                      control={
                        <Checkbox
                          checked={builder.includeCalendar}
                          onChange={(e) => setBuilder({ ...builder, includeCalendar: e.target.checked })}
                        />
                      }
                      label="Включить календарные признаки"
                    />
                    <FormControl sx={{ minWidth: 200 }} size="small" disabled={!builder.includeCalendar}>
                      <InputLabel>Календарь</InputLabel>
                      <Select
                        multiple
                        label="Календарь"
                        value={builder.calendarFeatures}
                        onChange={(e) =>
                          setBuilder({
                            ...builder,
                            calendarFeatures: e.target.value as string[],
                          })
                        }
                        renderValue={(selected) => (selected as string[]).join(', ')}
                      >
                        {calendarOptions.map((option) => (
                          <MenuItem key={option.key} value={option.key}>
                            {option.label}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  </Stack>

                  <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
                    <FormControlLabel
                      control={
                        <Checkbox
                          checked={builder.includeTickerEncoding}
                          onChange={(e) =>
                            setBuilder({ ...builder, includeTickerEncoding: e.target.checked })
                          }
                        />
                      }
                      label="One-hot кодирование тикера"
                    />
                    <FormControl size="small" sx={{ minWidth: 180 }} disabled={!builder.includeTickerEncoding}>
                      <InputLabel>Тип кодирования</InputLabel>
                      <Select
                        label="Тип кодирования"
                        value={builder.tickerEncoding}
                        onChange={(e) =>
                          setBuilder({
                            ...builder,
                            tickerEncoding: e.target.value as BuilderState['tickerEncoding'],
                          })
                        }
                      >
                        <MenuItem value="onehot">One-hot</MenuItem>
                        <MenuItem value="label">Label encoding</MenuItem>
                        <MenuItem value="target">Target encoding</MenuItem>
                      </Select>
                    </FormControl>
                  </Stack>

                  <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
                    <FormControlLabel
                      control={
                        <Checkbox
                          checked={builder.includeHigherTimeframe}
                          onChange={(e) =>
                            setBuilder({ ...builder, includeHigherTimeframe: e.target.checked })
                          }
                        />
                      }
                      label="Добавить признаки старших таймфреймов"
                    />
                    <FormControl size="small" sx={{ minWidth: 200 }} disabled={!builder.includeHigherTimeframe}>
                      <InputLabel>Таймфреймы</InputLabel>
                      <Select
                        multiple
                        label="Таймфреймы"
                        value={builder.higherTimeframes}
                        onChange={(e) =>
                          setBuilder({
                            ...builder,
                            higherTimeframes: e.target.value as string[],
                          })
                        }
                      >
                        {higherTimeframes.map((tf) => (
                          <MenuItem key={tf} value={tf}>
                            {tf}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  </Stack>

                  <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
                    <FormControlLabel
                      control={
                        <Checkbox checked={builder.includeLags} onChange={(e) => setBuilder({ ...builder, includeLags: e.target.checked })} />
                      }
                      label="Добавить лаги"
                    />
                    <TextField
                      label="Лаги (через запятую)"
                      value={builder.lags.join(', ')}
                      onChange={(e) =>
                        setBuilder({
                          ...builder,
                          lags: e.target.value
                            .split(',')
                            .map((v) => parseInt(v.trim(), 10))
                            .filter((v) => !Number.isNaN(v)),
                        })
                      }
                      size="small"
                      disabled={!builder.includeLags}
                    />
                  </Stack>

                  <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
                    <FormControlLabel
                      control={
                        <Checkbox
                          checked={builder.includeRolling}
                          onChange={(e) => setBuilder({ ...builder, includeRolling: e.target.checked })}
                        />
                      }
                      label="Rolling статистики"
                    />
                    <TextField
                      label="Окна"
                      value={builder.rollingWindows.join(', ')}
                      onChange={(e) =>
                        setBuilder({
                          ...builder,
                          rollingWindows: e.target.value
                            .split(',')
                            .map((v) => parseInt(v.trim(), 10))
                            .filter((v) => !Number.isNaN(v)),
                        })
                      }
                      size="small"
                      disabled={!builder.includeRolling}
                    />
                  </Stack>

                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={builder.includeDifferences}
                        onChange={(e) => setBuilder({ ...builder, includeDifferences: e.target.checked })}
                      />
                    }
                    label="Разности и процентные изменения"
                  />

                  <Divider />

                  <Stack
                    direction={{ xs: 'column', md: 'row' }}
                    spacing={2}
                    alignItems={{ xs: 'flex-start', md: 'center' }}
                  >
                    <FormControl size="small" sx={{ minWidth: 240 }}>
                      <InputLabel>Датасеты для запуска</InputLabel>
                      <Select
                        multiple
                        disabled={applyToAll || datasetsAvailable === 0}
                        label="Датасеты для запуска"
                        value={selectedDatasets}
                        onChange={(e) => setSelectedDatasets(e.target.value as string[])}
                        renderValue={(selected) => (selected as string[]).join(', ')}
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

                  <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
                    <TextField
                      label="Размер чанка"
                      type="number"
                      value={builder.chunkSize}
                      onChange={(e) =>
                        setBuilder({ ...builder, chunkSize: parseInt(e.target.value, 10) || 2000 })
                      }
                      size="small"
                    />
                    <FormControlLabel
                      control={
                        <Checkbox
                          checked={builder.incremental}
                          onChange={(e) => setBuilder({ ...builder, incremental: e.target.checked })}
                        />
                      }
                      label="Инкрементальное обновление"
                    />
                  </Stack>

                  <Box display="flex" justifyContent="flex-end">
                    <Button
                      variant="contained"
                      onClick={handleCreateTasks}
                      disabled={isCreating || !canSubmit}
                    >
                      {isCreating ? 'Запуск...' : 'Запустить генерацию'}
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
                      Нет активных задач. Создайте задачу генерации признаков слева.
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
                        <Typography variant="body2" color="text.secondary">
                          {task.dataset_id}
                        </Typography>
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
                    {task.message && (
                      <Typography variant="caption" color="text.secondary">
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
    </Box>
  );
}
