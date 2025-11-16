import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { Box, ToggleButtonGroup, ToggleButton, CircularProgress, Typography } from '@mui/material';
import { ShowChart, CandlestickChart } from '@mui/icons-material';

interface DataPoint {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface DatasetChartProps {
  data: DataPoint[];
  chartType: 'candlestick' | 'line';
  onChartTypeChange: (type: 'candlestick' | 'line') => void;
  isDarkMode: boolean;
}

export default function DatasetChart({ data, chartType, onChartTypeChange, isDarkMode }: DatasetChartProps) {
  const plotData = useMemo(() => {
    if (!data || data.length === 0) return [];

    const timestamps = data.map((d) => d.timestamp);
    const opens = data.map((d) => d.open);
    const highs = data.map((d) => d.high);
    const lows = data.map((d) => d.low);
    const closes = data.map((d) => d.close);
    const volumes = data.map((d) => d.volume);

    if (chartType === 'candlestick') {
      return [
        {
          type: 'candlestick' as const,
          x: timestamps,
          open: opens,
          high: highs,
          low: lows,
          close: closes,
          name: 'Price',
          increasing: { line: { color: isDarkMode ? '#26a69a' : '#089981' } },
          decreasing: { line: { color: isDarkMode ? '#ef5350' : '#f23645' } },
          xaxis: 'x',
          yaxis: 'y',
        },
        {
          type: 'bar' as const,
          x: timestamps,
          y: volumes,
          name: 'Volume',
          marker: {
            color: volumes.map((_, i) => (closes[i] >= opens[i] ? (isDarkMode ? '#26a69a' : '#089981') : (isDarkMode ? '#ef5350' : '#f23645'))),
            opacity: 0.5,
          },
          xaxis: 'x',
          yaxis: 'y2',
        },
      ];
    } else {
      return [
        {
          type: 'scatter' as const,
          mode: 'lines' as const,
          x: timestamps,
          y: closes,
          name: 'Close Price',
          line: { color: isDarkMode ? '#2196f3' : '#1976d2', width: 2 },
          xaxis: 'x',
          yaxis: 'y',
        },
        {
          type: 'bar' as const,
          x: timestamps,
          y: volumes,
          name: 'Volume',
          marker: {
            color: volumes.map((_, i) => (closes[i] >= opens[i] ? (isDarkMode ? '#26a69a' : '#089981') : (isDarkMode ? '#ef5350' : '#f23645'))),
            opacity: 0.5,
          },
          xaxis: 'x',
          yaxis: 'y2',
        },
      ];
    }
  }, [data, chartType, isDarkMode]);

  const layout = useMemo(
    () => ({
      autosize: true,
      height: 600,
      margin: { l: 60, r: 60, t: 40, b: 60 },
      paper_bgcolor: isDarkMode ? '#1e1e1e' : '#ffffff',
      plot_bgcolor: isDarkMode ? '#2d2d2d' : '#fafafa',
      font: {
        color: isDarkMode ? '#e0e0e0' : '#333333',
        family: 'Roboto, sans-serif',
      },
      xaxis: {
        type: 'date' as const,
        rangeslider: { visible: false },
        gridcolor: isDarkMode ? '#3d3d3d' : '#e0e0e0',
        showgrid: true,
        zeroline: false,
      },
      yaxis: {
        title: 'Price',
        domain: [0.25, 1],
        gridcolor: isDarkMode ? '#3d3d3d' : '#e0e0e0',
        showgrid: true,
        zeroline: false,
      },
      yaxis2: {
        title: 'Volume',
        domain: [0, 0.2],
        gridcolor: isDarkMode ? '#3d3d3d' : '#e0e0e0',
        showgrid: false,
        zeroline: false,
      },
      hovermode: 'x unified' as const,
      showlegend: true,
      legend: {
        orientation: 'h' as const,
        yanchor: 'bottom' as const,
        y: 1.02,
        xanchor: 'right' as const,
        x: 1,
        bgcolor: isDarkMode ? 'rgba(30, 30, 30, 0.8)' : 'rgba(255, 255, 255, 0.8)',
      },
    }),
    [isDarkMode]
  );

  const config = {
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    responsive: true,
  };

  if (!data || data.length === 0) {
    return (
      <Box display="flex" flexDirection="column" alignItems="center" justifyContent="center" height={400}>
        <CircularProgress />
        <Typography variant="body2" color="text.secondary" mt={2}>
          Загрузка данных...
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Box display="flex" justifyContent="flex-end" mb={2}>
        <ToggleButtonGroup
          value={chartType}
          exclusive
          onChange={(_, newType) => {
            if (newType !== null) {
              onChartTypeChange(newType);
            }
          }}
          size="small"
        >
          <ToggleButton value="candlestick" aria-label="candlestick chart">
            <CandlestickChart fontSize="small" sx={{ mr: 0.5 }} />
            Свечи
          </ToggleButton>
          <ToggleButton value="line" aria-label="line chart">
            <ShowChart fontSize="small" sx={{ mr: 0.5 }} />
            Линия
          </ToggleButton>
        </ToggleButtonGroup>
      </Box>
      <Plot data={plotData} layout={layout} config={config} style={{ width: '100%' }} />
    </Box>
  );
}
