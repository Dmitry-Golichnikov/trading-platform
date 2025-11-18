import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { Box, Typography, useTheme } from '@mui/material';

interface MetricsChartProps {
  metrics: Record<string, number>;
  title?: string;
  height?: number;
  type?: 'bar' | 'horizontal-bar';
}

export default function MetricsChart({
  metrics,
  title = 'Metrics',
  height = 400,
  type = 'bar'
}: MetricsChartProps) {
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';

  const plotData = useMemo(() => {
    if (!metrics || Object.keys(metrics).length === 0) return [];

    const labels = Object.keys(metrics);
    const values = Object.values(metrics);

    if (type === 'horizontal-bar') {
      return [
        {
          x: values,
          y: labels,
          type: 'bar' as const,
          orientation: 'h' as const,
          marker: {
            color: theme.palette.primary.main,
          },
        },
      ];
    }

    return [
      {
        x: labels,
        y: values,
        type: 'bar' as const,
        marker: {
          color: theme.palette.primary.main,
        },
      },
    ];
  }, [metrics, type, theme.palette.primary.main]);

  const layout = useMemo(() => ({
    title: title,
    autosize: true,
    height: height,
    margin: { l: type === 'horizontal-bar' ? 120 : 50, r: 50, t: 50, b: 100 },
    paper_bgcolor: isDark ? '#1e1e1e' : '#fff',
    plot_bgcolor: isDark ? '#1e1e1e' : '#fff',
    font: {
      color: isDark ? '#fff' : '#000',
    },
    xaxis: {
      title: type === 'horizontal-bar' ? 'Value' : 'Metric',
      gridcolor: isDark ? '#333' : '#e0e0e0',
      ...(type === 'bar' && { tickangle: -45 }),
    },
    yaxis: {
      title: type === 'horizontal-bar' ? 'Metric' : 'Value',
      gridcolor: isDark ? '#333' : '#e0e0e0',
    },
  }), [title, height, type, isDark]);

  if (!metrics || Object.keys(metrics).length === 0) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height={height}>
        <Typography color="text.secondary">No metrics available</Typography>
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
