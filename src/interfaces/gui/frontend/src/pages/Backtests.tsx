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
  IconButton,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import VisibilityIcon from '@mui/icons-material/Visibility';
import { backtestsAPI } from '@/api/client';
import { useStore } from '@/store';

export default function Backtests() {
  const setBacktests = useStore((state) => state.setBacktests);
  const setSelectedBacktest = useStore((state) => state.setSelectedBacktest);

  const { data: backtests, isLoading, refetch } = useQuery(
    'backtests',
    () => backtestsAPI.list().then((res) => {
      setBacktests(res.data);
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
        Backtests
      </Typography>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>ID</TableCell>
              <TableCell>Model ID</TableCell>
              <TableCell>Created</TableCell>
              <TableCell>Duration</TableCell>
              <TableCell>Metrics</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {backtests && backtests.length > 0 ? (
              backtests.map((backtest) => (
                <TableRow key={backtest.id}>
                  <TableCell>{backtest.id}</TableCell>
                  <TableCell>{backtest.model_id}</TableCell>
                  <TableCell>
                    {new Date(backtest.created_at).toLocaleString()}
                  </TableCell>
                  <TableCell>{backtest.duration_seconds.toFixed(1)}s</TableCell>
                  <TableCell>
                    {Object.keys(backtest.metrics).length > 0 ? (
                      <Box>
                        {Object.entries(backtest.metrics)
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
                      onClick={() => setSelectedBacktest(backtest)}
                    >
                      <VisibilityIcon />
                    </IconButton>
                    <IconButton
                      size="small"
                      color="error"
                      onClick={() => {
                        if (window.confirm('Are you sure you want to delete this backtest?')) {
                          backtestsAPI.delete(backtest.id).then(() => refetch());
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
                    No backtests found
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
