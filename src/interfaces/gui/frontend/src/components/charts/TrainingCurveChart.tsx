import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { Box, Typography, useTheme } from '@mui/material';

interface TrainingData {
  epoch: number;
  train_loss?: number;
  val_loss?: number;
  train_metric?: number;
  val_metric?: number;
}

interface TrainingCurveChartProps {
  data: TrainingData[];
  title?: string;
  height?: number;
  metricName?: string;
}

export default function TrainingCurveChart({
  data,
  title = 'Training Curves',
  height = 400,
  metricName = 'Metric'
}: TrainingCurveChartProps) {
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';

  const plotData = useMemo(() => {
    if (!data || data.length === 0) return [];

    const traces: any[] = [];
    const epochs = data.map(d => d.epoch);

    // Loss curves
    if (data[0]?.train_loss !== undefined) {
      traces.push({
        x: epochs,
        y: data.map(d => d.train_loss),
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: 'Train Loss',
        line: { color: '#ef5350', width: 2 },
      });
    }

    if (data[0]?.val_loss !== undefined) {
      traces.push({
        x: epochs,
        y: data.map(d => d.val_loss),
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: 'Val Loss',
        line: { color: '#ff7043', width: 2, dash: 'dash' },
      });
    }

    // Metric curves
    if (data[0]?.train_metric !== undefined) {
      traces.push({
        x: epochs,
        y: data.map(d => d.train_metric),
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: `Train ${metricName}`,
        line: { color: '#26a69a', width: 2 },
        yaxis: 'y2',
      });
    }

    if (data[0]?.val_metric !== undefined) {
      traces.push({
        x: epochs,
        y: data.map(d => d.val_metric),
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: `Val ${metricName}`,
        line: { color: '#66bb6a', width: 2, dash: 'dash' },
        yaxis: 'y2',
      });
    }

    return traces;
  }, [data, metricName]);

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
      title: 'Epoch',
      gridcolor: isDark ? '#333' : '#e0e0e0',
    },
    yaxis: {
      title: 'Loss',
      gridcolor: isDark ? '#333' : '#e0e0e0',
    },
    yaxis2: {
      title: metricName,
      overlaying: 'y',
      side: 'right',
      gridcolor: isDark ? '#333' : '#e0e0e0',
    },
    legend: {
      x: 1.1,
      y: 1,
    },
  }), [title, height, metricName, isDark]);

  if (!data || data.length === 0) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height={height}>
        <Typography color="text.secondary">No training data available</Typography>
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
