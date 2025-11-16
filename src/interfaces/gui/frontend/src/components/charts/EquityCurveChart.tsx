import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { Box, Typography, useTheme } from '@mui/material';

interface EquityCurveData {
  date: string;
  value: number;
}

interface EquityCurveChartProps {
  data: EquityCurveData[];
  title?: string;
  height?: number;
}

export default function EquityCurveChart({
  data,
  title = 'Equity Curve',
  height = 400
}: EquityCurveChartProps) {
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';

  const plotData = useMemo(() => {
    if (!data || data.length === 0) return [];

    return [
      {
        x: data.map(d => d.date),
        y: data.map(d => d.value),
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: 'Equity',
        line: {
          color: theme.palette.primary.main,
          width: 2,
        },
      },
    ];
  }, [data, theme.palette.primary.main]);

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
      title: 'Date',
      gridcolor: isDark ? '#333' : '#e0e0e0',
    },
    yaxis: {
      title: 'Equity',
      gridcolor: isDark ? '#333' : '#e0e0e0',
    },
  }), [title, height, isDark]);

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
