import { useEffect } from 'react';
import { useQuery } from 'react-query';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  CircularProgress,
  Alert,
} from '@mui/material';
import { experimentsAPI, modelsAPI, systemAPI } from '@/api/client';
import { useStore } from '@/store';

export default function Dashboard() {
  const setSystemHealth = useStore((state) => state.setSystemHealth);

  // Fetch experiments
  const { data: experiments, isLoading: experimentsLoading } = useQuery(
    'experiments',
    () => experimentsAPI.list({ limit: 5 }).then((res) => res.data)
  );

  // Fetch models
  const { data: models, isLoading: modelsLoading } = useQuery(
    'models',
    () => modelsAPI.list({ limit: 5 }).then((res) => res.data)
  );

  // Fetch system health
  const { data: systemHealth, isLoading: systemLoading } = useQuery(
    'systemHealth',
    () => systemAPI.health().then((res) => res.data),
    { refetchInterval: 5000 } // Refresh every 5 seconds
  );

  useEffect(() => {
    if (systemHealth) {
      setSystemHealth(systemHealth);
    }
  }, [systemHealth, setSystemHealth]);

  if (experimentsLoading || modelsLoading || systemLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="80vh">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>

      <Grid container spacing={3}>
        {/* System Health */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Health
              </Typography>
              {systemHealth && (
                <Grid container spacing={2}>
                  <Grid item xs={3}>
                    <Typography variant="body2" color="text.secondary">
                      Status
                    </Typography>
                    <Typography variant="h6">
                      {systemHealth.status.toUpperCase()}
                    </Typography>
                  </Grid>
                  <Grid item xs={3}>
                    <Typography variant="body2" color="text.secondary">
                      CPU Usage
                    </Typography>
                    <Typography variant="h6">
                      {systemHealth.cpu_usage.toFixed(1)}%
                    </Typography>
                  </Grid>
                  <Grid item xs={3}>
                    <Typography variant="body2" color="text.secondary">
                      Memory Usage
                    </Typography>
                    <Typography variant="h6">
                      {systemHealth.memory_usage.toFixed(1)}%
                    </Typography>
                  </Grid>
                  <Grid item xs={3}>
                    <Typography variant="body2" color="text.secondary">
                      GPU Available
                    </Typography>
                    <Typography variant="h6">
                      {systemHealth.gpu_available ? 'Yes' : 'No'}
                    </Typography>
                  </Grid>
                </Grid>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Experiments */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Experiments
              </Typography>
              {experiments && experiments.length > 0 ? (
                <Box>
                  {experiments.map((exp) => (
                    <Box key={exp.id} mb={1}>
                      <Typography variant="body1">{exp.name}</Typography>
                      <Typography variant="body2" color="text.secondary">
                        Status: {exp.status}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No experiments found
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Models */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Models
              </Typography>
              {models && models.length > 0 ? (
                <Box>
                  {models.map((model) => (
                    <Box key={model.id} mb={1}>
                      <Typography variant="body1">{model.name}</Typography>
                      <Typography variant="body2" color="text.secondary">
                        Type: {model.type} | Size: {model.size_mb.toFixed(2)} MB
                      </Typography>
                    </Box>
                  ))}
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No models found
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Quick Stats */}
        <Grid item xs={12}>
          <Alert severity="info">
            Welcome to Trading Platform GUI! Use the sidebar to navigate to different sections.
          </Alert>
        </Grid>
      </Grid>
    </Box>
  );
}
