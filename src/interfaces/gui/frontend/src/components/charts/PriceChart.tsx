import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { Box, Typography, useTheme } from '@mui/material';

interface CandlestickData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface PriceChartProps {
  data: CandlestickData[];
  title?: string;
  height?: number;
  showVolume?: boolean;
}

export default function PriceChart({
  data,
  title = 'Price Chart',
  height = 500,
  showVolume = true
}: PriceChartProps) {
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';

  const plotData = useMemo(() => {
    if (!data || data.length === 0) return [];

    const traces: any[] = [
      {
        x: data.map(d => d.timestamp),
        open: data.map(d => d.open),
        high: data.map(d => d.high),
        low: data.map(d => d.low),
        close: data.map(d => d.close),
        type: 'candlestick' as const,
        name: 'Price',
        yaxis: 'y',
        increasing: { line: { color: '#26a69a' } },
        decreasing: { line: { color: '#ef5350' } },
      },
    ];

    if (showVolume && data[0]?.volume !== undefined) {
      traces.push({
        x: data.map(d => d.timestamp),
        y: data.map(d => d.volume),
        type: 'bar' as const,
        name: 'Volume',
        yaxis: 'y2',
        marker: {
          color: data.map(d => d.close >= d.open ? '#26a69a' : '#ef5350'),
        },
      });
    }

    return traces;
  }, [data, showVolume]);

  const layout = useMemo(() => ({
    title: title,
    autosize: true,
    height: height,
    margin: { l: 50, r: 50, t: 50, b: 50 },
    paper_bgcolor: isDark ? '#1e1e1e' : '#fff',
    plot_bgcolor: isDark ? '#1e1e1e' : '#fff',
    font: {
      color: isDark ? '#fff' : '#000',
    },
    xaxis: {
      rangeslider: { visible: false },
      gridcolor: isDark ? '#333' : '#e0e0e0',
    },
    yaxis: {
      title: 'Price',
      domain: showVolume ? [0.3, 1] : [0, 1],
      gridcolor: isDark ? '#333' : '#e0e0e0',
    },
    ...(showVolume && {
      yaxis2: {
        title: 'Volume',
        domain: [0, 0.2],
        gridcolor: isDark ? '#333' : '#e0e0e0',
      },
    }),
  }), [title, height, isDark, showVolume]);

  if (!data || data.length === 0) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height={height}>
        <Typography color="text.secondary">No data available</Typography>
      </Box>
    );
  }

  return (
    <Plot
      data={plotData}
      layout={layout}
      config={{ responsive: true, displayModeBar: true }}
      style={{ width: '100%', height: '100%' }}
    />
  );
}
