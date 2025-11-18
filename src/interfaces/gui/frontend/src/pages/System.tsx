import { useQuery } from 'react-query';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  CircularProgress,
  LinearProgress,
  Chip,
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import WarningIcon from '@mui/icons-material/Warning';
import ErrorIcon from '@mui/icons-material/Error';
import { systemAPI } from '@/api/client';

const getStatusIcon = (status: string) => {
  switch (status) {
    case 'healthy':
      return <CheckCircleIcon color="success" />;
    case 'warning':
      return <WarningIcon color="warning" />;
    case 'critical':
      return <ErrorIcon color="error" />;
    default:
      return null;
  }
};

const getStatusColor = (status: string) => {
  switch (status) {
    case 'healthy':
      return 'success';
    case 'warning':
      return 'warning';
    case 'critical':
      return 'error';
    default:
      return 'default';
  }
};

export default function System() {
  const { data: systemHealth, isLoading: healthLoading } = useQuery(
    'systemHealth',
    () => systemAPI.health().then((res) => res.data),
    { refetchInterval: 5000 }
  );

  const { data: systemInfo, isLoading: infoLoading } = useQuery(
    'systemInfo',
    () => systemAPI.info().then((res) => res.data)
  );

  if (healthLoading || infoLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="80vh">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        System Status
      </Typography>

      <Grid container spacing={3}>
        {/* Overall Status */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" gap={2}>
                {systemHealth && getStatusIcon(systemHealth.status)}
                <Box>
                  <Typography variant="h6">Overall Status</Typography>
                  {systemHealth && (
                    <Chip
                      label={systemHealth.status.toUpperCase()}
                      color={getStatusColor(systemHealth.status) as any}
                      size="small"
                    />
                  )}
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* CPU Usage */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                CPU Usage
              </Typography>
              {systemHealth && (
                <>
                  <Typography variant="h4">
                    {systemHealth.cpu_usage.toFixed(1)}%
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={systemHealth.cpu_usage}
                    color={systemHealth.cpu_usage > 80 ? 'error' : 'primary'}
                  />
                  {systemInfo && (
                    <Typography variant="body2" color="text.secondary" mt={1}>
                      {systemInfo.cpu.count} cores
                    </Typography>
                  )}
                </>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Memory Usage */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Memory Usage
              </Typography>
              {systemHealth && (
                <>
                  <Typography variant="h4">
                    {systemHealth.memory_usage.toFixed(1)}%
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={systemHealth.memory_usage}
                    color={systemHealth.memory_usage > 80 ? 'error' : 'primary'}
                  />
                  {systemInfo && (
                    <Typography variant="body2" color="text.secondary" mt={1}>
                      {systemInfo.memory.total_gb.toFixed(1)} GB total
                    </Typography>
                  )}
                </>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Disk Usage */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Disk Usage
              </Typography>
              {systemHealth && (
                <>
                  <Typography variant="h4">
                    {systemHealth.disk_usage.toFixed(1)}%
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={systemHealth.disk_usage}
                    color={systemHealth.disk_usage > 80 ? 'error' : 'primary'}
                  />
                  {systemInfo && (
                    <Typography variant="body2" color="text.secondary" mt={1}>
                      {systemInfo.disk.free_gb.toFixed(1)} GB free
                    </Typography>
                  )}
                </>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* GPU Status */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                GPU Status
              </Typography>
              {systemHealth && (
                <>
                  <Typography variant="h4">
                    {systemHealth.gpu_available ? 'Available' : 'Not Available'}
                  </Typography>
                  {systemHealth.gpu_available && systemHealth.gpu_usage !== undefined && (
                    <>
                      <LinearProgress
                        variant="determinate"
                        value={systemHealth.gpu_usage}
                        color="primary"
                      />
                      <Typography variant="body2" color="text.secondary" mt={1}>
                        {systemHealth.gpu_usage.toFixed(1)}% utilization
                      </Typography>
                    </>
                  )}
                  {systemInfo && systemInfo.gpu.devices && systemInfo.gpu.devices.length > 0 && (
                    <Typography variant="body2" color="text.secondary" mt={1}>
                      {systemInfo.gpu.devices[0].name}
                    </Typography>
                  )}
                </>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Active Tasks */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Active Tasks
              </Typography>
              {systemHealth && (
                <Typography variant="h4">
                  {systemHealth.active_tasks}
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
