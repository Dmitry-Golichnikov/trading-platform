import React, { useState, useMemo } from 'react';
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
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  useTheme,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import VisibilityIcon from '@mui/icons-material/Visibility';
import CloseIcon from '@mui/icons-material/Close';
import { datasetsAPI } from '@/api/client';
import { useStore } from '@/store';
import { DatasetChart } from '@/components/charts';
import type { DatasetInfo } from '@/types';

interface DataPoint {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export default function Datasets() {
  const theme = useTheme();
  const isDarkMode = theme.palette.mode === 'dark';
  const addNotification = useStore((state) => state.addNotification);
  const setDatasets = useStore((state) => state.setDatasets);

  const [viewDialogOpen, setViewDialogOpen] = useState(false);
  const [selectedDatasetForView, setSelectedDatasetForView] = useState<DatasetInfo | null>(null);
  const [chartType, setChartType] = useState<'candlestick' | 'line'>('candlestick');

  const { data: datasets, isLoading, refetch } = useQuery(
    'datasets',
    () => datasetsAPI.list().then((res) => {
      setDatasets(res.data);
      return res.data;
    })
  );

  const { data: datasetData, isLoading: isDataLoading } = useQuery(
    ['datasetData', selectedDatasetForView?.id],
    () => {
      if (!selectedDatasetForView) return null;
      return datasetsAPI.getData(selectedDatasetForView.id).then((res) => res.data);
    },
    {
      enabled: !!selectedDatasetForView && viewDialogOpen,
    }
  );

  const handleViewDataset = (dataset: DatasetInfo) => {
    setSelectedDatasetForView(dataset);
    setViewDialogOpen(true);
  };

  const handleCloseViewDialog = () => {
    setViewDialogOpen(false);
    setSelectedDatasetForView(null);
  };

  const handleDeleteDataset = async (dataset: DatasetInfo) => {
    if (window.confirm(`Вы уверены, что хотите удалить датасет ${dataset.ticker} ${dataset.timeframe}?`)) {
      try {
        await datasetsAPI.delete(dataset.id);
        addNotification({ type: 'success', message: 'Датасет успешно удалён' });
        refetch();
      } catch (error: any) {
        addNotification({ type: 'error', message: error.response?.data?.detail || 'Не удалось удалить датасет' });
      }
    }
  };

  // Group datasets by ticker
  const groupedDatasets = useMemo(() => {
    if (!datasets) return {};

    const grouped: Record<string, DatasetInfo[]> = {};
    datasets.forEach((dataset) => {
      if (!grouped[dataset.ticker]) {
        grouped[dataset.ticker] = [];
      }
      grouped[dataset.ticker].push(dataset);
    });

    // Sort timeframes within each ticker
    Object.keys(grouped).forEach((ticker) => {
      grouped[ticker].sort((a, b) => {
        const timeframeOrder: Record<string, number> = {
          '1m': 1, '5m': 2, '15m': 3, '1h': 4, '4h': 5, '1d': 6
        };
        return (timeframeOrder[a.timeframe] || 999) - (timeframeOrder[b.timeframe] || 999);
      });
    });

    return grouped;
  }, [datasets]);

  const tickers = useMemo(() => Object.keys(groupedDatasets).sort(), [groupedDatasets]);

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
            {tickers.length > 0 ? (
              tickers.map((ticker) => (
                <React.Fragment key={ticker}>
                  {groupedDatasets[ticker].map((dataset, index) => (
                    <TableRow key={dataset.id}>
                      {index === 0 && (
                        <TableCell
                          rowSpan={groupedDatasets[ticker].length}
                          sx={{
                            fontWeight: 'bold',
                            verticalAlign: 'top',
                            borderRight: '1px solid',
                            borderColor: 'divider',
                          }}
                        >
                          {dataset.ticker}
                        </TableCell>
                      )}
                      <TableCell>
                        <Chip label={dataset.timeframe} size="small" variant="outlined" />
                      </TableCell>
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
                          onClick={() => handleViewDataset(dataset)}
                          title="Просмотр графика"
                        >
                          <VisibilityIcon />
                        </IconButton>
                        <IconButton
                          size="small"
                          color="error"
                          onClick={() => handleDeleteDataset(dataset)}
                          title="Удалить датасет"
                        >
                          <DeleteIcon />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </React.Fragment>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={7} align="center">
                  <Typography variant="body2" color="text.secondary">
                    Нет доступных датасетов
                  </Typography>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>

      {/* View Dataset Dialog */}
      <Dialog
        open={viewDialogOpen}
        onClose={handleCloseViewDialog}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Typography variant="h6">
              {selectedDatasetForView?.ticker} · {selectedDatasetForView?.timeframe}
            </Typography>
            <IconButton onClick={handleCloseViewDialog} size="small">
              <CloseIcon />
            </IconButton>
          </Box>
        </DialogTitle>
        <DialogContent>
          {isDataLoading ? (
            <Box display="flex" justifyContent="center" alignItems="center" height={400}>
              <CircularProgress />
            </Box>
          ) : datasetData?.data ? (
            <DatasetChart
              data={datasetData.data as DataPoint[]}
              chartType={chartType}
              onChartTypeChange={setChartType}
              isDarkMode={isDarkMode}
            />
          ) : (
            <Box display="flex" justifyContent="center" alignItems="center" height={400}>
              <Typography variant="body2" color="text.secondary">
                Нет данных для отображения
              </Typography>
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
