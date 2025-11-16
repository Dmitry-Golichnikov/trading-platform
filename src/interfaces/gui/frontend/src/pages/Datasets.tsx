import { useEffect, useMemo, useState, type MouseEvent } from 'react';
import { useQuery } from 'react-query';
import {
  Box,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  CircularProgress,
  Chip,
  IconButton,
  Stack,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  ToggleButtonGroup,
  ToggleButton,
  Divider,
  type SelectChangeEvent,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import VisibilityIcon from '@mui/icons-material/Visibility';
import RefreshIcon from '@mui/icons-material/Refresh';
import { datasetsAPI } from '@/api/client';
import { useStore } from '@/store';
import { PriceChart } from '@/components/charts';
import type { DatasetDataResponse, DatasetInfo } from '@/types';

export default function Datasets() {
  const setDatasets = useStore((state) => state.setDatasets);
  const setSelectedDataset = useStore((state) => state.setSelectedDataset);
  const addNotification = useStore((state) => state.addNotification);
  const [activeDataset, setActiveDataset] = useState<DatasetInfo | null>(null);
  const [chartMode, setChartMode] = useState<'candles' | 'line'>('candles');
  const [viewerOpen, setViewerOpen] = useState(false);

  const { data: datasets, isLoading, refetch } = useQuery(
    'datasets',
    () => datasetsAPI.list().then((res) => {
      setDatasets(res.data);
      return res.data;
    })
  );

  useEffect(() => {
    if (datasets && datasets.length > 0) {
      if (!activeDataset) {
        handleSelectDataset(datasets[0]);
      } else {
        const stillExists = datasets.some((ds) => ds.id === activeDataset.id);
        if (!stillExists) {
          handleSelectDataset(datasets[0]);
        }
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasets]);

  const {
    data: datasetSeries,
    isFetching: isChartLoading,
    refetch: refetchDatasetSeries,
  } = useQuery<DatasetDataResponse>(
    ['datasetData', activeDataset?.id],
    () => datasetsAPI.getData(activeDataset!.id, { limit: 1500 }).then((res) => res.data),
    {
      enabled: Boolean(activeDataset),
      onError: (error: any) => {
        addNotification({
          type: 'error',
          message: error?.response?.data?.detail || 'Не удалось загрузить данные датасета',
        });
      },
      refetchOnWindowFocus: false,
    }
  );

  const columnLookup = useMemo(() => {
    const columns = datasetSeries?.columns ?? [];
    return columns.reduce<Record<string, string>>((acc, col) => {
      acc[col.toLowerCase()] = col;
      return acc;
    }, {});
  }, [datasetSeries]);

  const findColumn = (aliases: string[]): string | undefined => {
    for (const alias of aliases) {
      const key = columnLookup[alias];
      if (key) {
        return key;
      }
    }
    return undefined;
  };

  const timestampKey = findColumn(['timestamp', 'date', 'datetime', 'time', 'index']);
  const openKey = findColumn(['open', 'o']);
  const highKey = findColumn(['high', 'h']);
  const lowKey = findColumn(['low', 'l']);
  const closeKey = findColumn(['close', 'c', 'price']);
  const volumeKey = findColumn(['volume', 'vol', 'v']);

  const parseNumber = (value: unknown): number | undefined => {
    if (value === undefined || value === null || value === '') return undefined;
    const num = Number(value);
    return Number.isFinite(num) ? num : undefined;
  };

  const formatTimestamp = (value: unknown, index: number): string | undefined => {
    if (value === undefined || value === null) {
      return index.toString();
    }

    if (value instanceof Date) {
      return value.toISOString();
    }

    if (typeof value === 'number') {
      return new Date(value).toISOString();
    }

    const parsed = Date.parse(String(value));
    if (!Number.isNaN(parsed)) {
      return new Date(parsed).toISOString();
    }

    return String(value);
  };

  const parsedChartData = useMemo(() => {
    if (!datasetSeries) return [];

    return datasetSeries.data
      .map((row, index) => ({
        timestamp: formatTimestamp(timestampKey ? row[timestampKey] : undefined, index),
        open: parseNumber(openKey ? row[openKey] : undefined),
        high: parseNumber(highKey ? row[highKey] : undefined),
        low: parseNumber(lowKey ? row[lowKey] : undefined),
        close: parseNumber(closeKey ? row[closeKey] : undefined),
        volume: parseNumber(volumeKey ? row[volumeKey] : undefined),
      }))
      .filter((row) => row.timestamp);
  }, [datasetSeries, timestampKey, openKey, highKey, lowKey, closeKey, volumeKey]);

  const candleData = useMemo(
    () => parsedChartData.filter((row) => row.open !== undefined && row.high !== undefined && row.low !== undefined && row.close !== undefined),
    [parsedChartData]
  );

  const lineData = useMemo(
    () => parsedChartData.filter((row) => row.close !== undefined),
    [parsedChartData]
  );

  const chartData = chartMode === 'line' ? lineData : candleData;

  const handleSelectDataset = (dataset: DatasetInfo | null, openViewer = false) => {
    setSelectedDataset(dataset);
    setActiveDataset(dataset);
    if (openViewer && dataset) {
      setViewerOpen(true);
    }
  };

  const handleChartModeChange = (_: MouseEvent<HTMLElement>, value: 'candles' | 'line' | null) => {
    if (value) {
      setChartMode(value);
    }
  };

  const handleDatasetChange = (event: SelectChangeEvent<string>) => {
    const datasetId = event.target.value as string;
    const dataset = datasets?.find((ds) => ds.id === datasetId) ?? null;
    handleSelectDataset(dataset);
  };

  const datasetName = activeDataset ? `${activeDataset.ticker} · ${activeDataset.timeframe}` : 'Выберите датасет';

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="80vh">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Datasets
      </Typography>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Ticker</TableCell>
              <TableCell>Timeframe</TableCell>
              <TableCell>Source</TableCell>
              <TableCell>Rows</TableCell>
              <TableCell>Size (MB)</TableCell>
              <TableCell>Quality</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {datasets && datasets.length > 0 ? (
              datasets.map((dataset) => (
                <TableRow key={dataset.id} selected={activeDataset?.id === dataset.id}>
                  <TableCell>{dataset.ticker}</TableCell>
                  <TableCell>{dataset.timeframe}</TableCell>
                  <TableCell>{dataset.source}</TableCell>
                  <TableCell>{dataset.num_rows.toLocaleString()}</TableCell>
                  <TableCell>{dataset.size_mb.toFixed(2)}</TableCell>
                  <TableCell>
                    {dataset.quality_score !== undefined && dataset.quality_score !== null ? (
                      <Chip
                        label={`${(dataset.quality_score * 100).toFixed(0)}%`}
                        color={dataset.quality_score > 0.8 ? 'success' : 'warning'}
                        size="small"
                      />
                    ) : (
                      <Typography variant="body2" color="text.secondary">
                        N/A
                      </Typography>
                    )}
                  </TableCell>
                  <TableCell>
                    <IconButton
                      size="small"
                      onClick={() => handleSelectDataset(dataset, true)}
                      color={activeDataset?.id === dataset.id && viewerOpen ? 'primary' : 'default'}
                    >
                      <VisibilityIcon />
                    </IconButton>
                    <IconButton
                      size="small"
                      color="error"
                      onClick={() => {
                        if (window.confirm('Are you sure you want to delete this dataset?')) {
                          datasetsAPI.delete(dataset.id).then(() => refetch());
                        }
                      }}
                    >
                      <DeleteIcon />
                    </IconButton>
                  </TableCell>
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={7} align="center">
                  <Typography variant="body2" color="text.secondary">
                    No datasets found
                  </Typography>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>

      <Dialog
        open={viewerOpen}
        onClose={() => setViewerOpen(false)}
        fullWidth
        maxWidth="lg"
      >
        <DialogTitle>Просмотр датасета</DialogTitle>
        <DialogContent dividers>
          <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} justifyContent="space-between" alignItems={{ xs: 'flex-start', md: 'center' }}>
            <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} alignItems={{ xs: 'stretch', sm: 'center' }} width="100%">
              <FormControl size="small" sx={{ minWidth: 220 }}>
                <InputLabel id="dataset-chart-select">Датасет</InputLabel>
                <Select
                  labelId="dataset-chart-select"
                  value={activeDataset?.id ?? ''}
                  label="Датасет"
                  onChange={handleDatasetChange}
                >
                  {datasets?.map((dataset) => (
                    <MenuItem key={dataset.id} value={dataset.id}>
                      {dataset.ticker} · {dataset.timeframe}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <IconButton
                aria-label="refresh dataset chart"
                onClick={() => refetchDatasetSeries()}
                disabled={!activeDataset || isChartLoading}
              >
                <RefreshIcon />
              </IconButton>
            </Stack>
            <ToggleButtonGroup
              exclusive
              size="small"
              value={chartMode}
              onChange={handleChartModeChange}
            >
              <ToggleButton value="candles">Бары</ToggleButton>
              <ToggleButton value="line">Линия</ToggleButton>
            </ToggleButtonGroup>
          </Stack>

          <Divider sx={{ my: 2 }} />

          {!activeDataset ? (
            <Box display="flex" justifyContent="center" alignItems="center" height={400}>
              <Typography color="text.secondary">Выберите датасет, чтобы увидеть график</Typography>
            </Box>
          ) : isChartLoading ? (
            <Box display="flex" justifyContent="center" alignItems="center" height={400}>
              <CircularProgress />
            </Box>
          ) : chartData.length === 0 ? (
            <Box display="flex" justifyContent="center" alignItems="center" height={400}>
              <Typography color="text.secondary">
                Для выбранного режима нет достаточных данных (нужны столбцы {chartMode === 'candles' ? 'Open, High, Low и Close' : 'Close'}).
              </Typography>
            </Box>
          ) : (
            <PriceChart
              data={chartData}
              title={datasetName}
              variant={chartMode}
              showVolume={chartMode === 'candles'}
            />
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setViewerOpen(false)}>Закрыть</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
