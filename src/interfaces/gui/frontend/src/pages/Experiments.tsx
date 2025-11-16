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
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import CancelIcon from '@mui/icons-material/Cancel';
import VisibilityIcon from '@mui/icons-material/Visibility';
import { experimentsAPI } from '@/api/client';
import { useStore } from '@/store';

const getStatusColor = (status: string) => {
  switch (status) {
    case 'completed':
      return 'success';
    case 'running':
      return 'primary';
    case 'failed':
      return 'error';
    case 'cancelled':
      return 'default';
    default:
      return 'default';
  }
};

export default function Experiments() {
  const setExperiments = useStore((state) => state.setExperiments);
  const setSelectedExperiment = useStore((state) => state.setSelectedExperiment);

  const { data: experiments, isLoading, refetch } = useQuery(
    'experiments',
    () => experimentsAPI.list().then((res) => {
      setExperiments(res.data);
      return res.data;
    })
  );

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
        Experiments
      </Typography>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Name</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Created</TableCell>
              <TableCell>Duration</TableCell>
              <TableCell>Metrics</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {experiments && experiments.length > 0 ? (
              experiments.map((experiment) => (
                <TableRow key={experiment.id}>
                  <TableCell>{experiment.name}</TableCell>
                  <TableCell>
                    <Chip
                      label={experiment.status}
                      color={getStatusColor(experiment.status)}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    {new Date(experiment.created_at).toLocaleString()}
                  </TableCell>
                  <TableCell>
                    {experiment.duration_seconds
                      ? `${experiment.duration_seconds.toFixed(1)}s`
                      : 'N/A'}
                  </TableCell>
                  <TableCell>
                    {Object.keys(experiment.metrics).length > 0 ? (
                      <Box>
                        {Object.entries(experiment.metrics)
                          .slice(0, 2)
                          .map(([key, value]) => (
                            <Typography key={key} variant="body2">
                              {key}: {typeof value === 'number' ? value.toFixed(4) : value}
                            </Typography>
                          ))}
                      </Box>
                    ) : (
                      <Typography variant="body2" color="text.secondary">
                        No metrics
                      </Typography>
                    )}
                  </TableCell>
                  <TableCell>
                    <IconButton
                      size="small"
                      onClick={() => setSelectedExperiment(experiment)}
                    >
                      <VisibilityIcon />
                    </IconButton>
                    {experiment.status === 'running' && (
                      <IconButton
                        size="small"
                        color="warning"
                        onClick={() => {
                          experimentsAPI.cancel(experiment.id).then(() => refetch());
                        }}
                      >
                        <CancelIcon />
                      </IconButton>
                    )}
                    <IconButton
                      size="small"
                      color="error"
                      onClick={() => {
                        if (window.confirm('Are you sure you want to delete this experiment?')) {
                          experimentsAPI.delete(experiment.id).then(() => refetch());
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
                <TableCell colSpan={6} align="center">
                  <Typography variant="body2" color="text.secondary">
                    No experiments found
                  </Typography>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}
