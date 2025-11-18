import { useMemo } from 'react';
import { Box, Typography, Paper, Stack, Chip } from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';

interface LabelDistributionChartProps {
  distribution: Record<string, number>;
  chartType?: 'bar' | 'pie';
  isDarkMode?: boolean;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

export default function LabelDistributionChart({
  distribution,
  chartType = 'bar',
  isDarkMode = false,
}: LabelDistributionChartProps) {
  const chartData = useMemo(() => {
    return Object.entries(distribution).map(([label, count]) => ({
      label,
      count,
      percentage: 0, // Will be calculated
    }));
  }, [distribution]);

  const totalSamples = useMemo(() => {
    return chartData.reduce((sum, item) => sum + item.count, 0);
  }, [chartData]);

  const chartDataWithPercentages = useMemo(() => {
    return chartData.map((item) => ({
      ...item,
      percentage: ((item.count / totalSamples) * 100).toFixed(1),
    }));
  }, [chartData, totalSamples]);

  const labelMapping: Record<string, string> = {
    '-1': 'Short',
    '0': 'Neutral / Hold',
    '1': 'Long',
  };

  const gridColor = isDarkMode ? '#444' : '#ddd';
  const textColor = isDarkMode ? '#ccc' : '#333';

  if (chartType === 'pie') {
    return (
      <Box>
        <Stack direction="row" spacing={2} mb={2} alignItems="center">
          <Typography variant="h6">Распределение классов</Typography>
          <Chip label={`Всего: ${totalSamples.toLocaleString()}`} color="primary" />
        </Stack>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={chartDataWithPercentages}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={(entry) => `${labelMapping[entry.label] || entry.label}: ${entry.percentage}%`}
              outerRadius={80}
              fill="#8884d8"
              dataKey="count"
            >
              {chartDataWithPercentages.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>

        <Stack direction="row" spacing={2} mt={2} justifyContent="center">
          {chartDataWithPercentages.map((item, index) => (
            <Chip
              key={item.label}
              label={`${labelMapping[item.label] || item.label}: ${item.count} (${item.percentage}%)`}
              sx={{ backgroundColor: COLORS[index % COLORS.length], color: '#fff' }}
            />
          ))}
        </Stack>
      </Box>
    );
  }

  return (
    <Box>
      <Stack direction="row" spacing={2} mb={2} alignItems="center">
        <Typography variant="h6">Распределение классов</Typography>
        <Chip label={`Всего: ${totalSamples.toLocaleString()}`} color="primary" />
      </Stack>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartDataWithPercentages}>
          <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
          <XAxis
            dataKey="label"
            stroke={textColor}
            tickFormatter={(label) => labelMapping[label] || label}
          />
          <YAxis stroke={textColor} />
          <Tooltip
            contentStyle={{
              backgroundColor: isDarkMode ? '#333' : '#fff',
              border: `1px solid ${gridColor}`,
              color: textColor,
            }}
            labelFormatter={(label) => labelMapping[label] || label}
            formatter={(value: number, name: string) => {
              if (name === 'percentage') return [`${value}%`, 'Percentage'];
              return [value.toLocaleString(), 'Count'];
            }}
          />
          <Legend />
          <Bar dataKey="count" fill="#8884d8" name="Количество сэмплов" />
        </BarChart>
      </ResponsiveContainer>

      <Stack direction="row" spacing={2} mt={2} justifyContent="center">
        {chartDataWithPercentages.map((item) => (
          <Paper key={item.label} variant="outlined" sx={{ p: 1, minWidth: 120 }}>
            <Typography variant="caption" color="text.secondary">
              {labelMapping[item.label] || item.label}
            </Typography>
            <Typography variant="h6">{item.count.toLocaleString()}</Typography>
            <Typography variant="caption" color="text.secondary">
              {item.percentage}%
            </Typography>
          </Paper>
        ))}
      </Stack>
    </Box>
  );
}

