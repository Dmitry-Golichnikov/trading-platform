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
import VisibilityIcon from '@mui/icons-material/Visibility';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import { modelsAPI } from '@/api/client';
import { useStore } from '@/store';

export default function Models() {
  const setModels = useStore((state) => state.setModels);
  const setSelectedModel = useStore((state) => state.setSelectedModel);

  const { data: models, isLoading, refetch } = useQuery(
    'models',
    () => modelsAPI.list().then((res) => {
      setModels(res.data);
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
        Models
      </Typography>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Name</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>Size (MB)</TableCell>
              <TableCell>Created</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {models && models.length > 0 ? (
              models.map((model) => (
                <TableRow key={model.id}>
                  <TableCell>{model.name}</TableCell>
                  <TableCell>{model.type}</TableCell>
                  <TableCell>{model.size_mb.toFixed(2)}</TableCell>
                  <TableCell>
                    {new Date(model.created_at).toLocaleString()}
                  </TableCell>
                  <TableCell>
                    {model.is_deployed ? (
                      <Chip label="Deployed" color="success" size="small" />
                    ) : (
                      <Chip label="Not Deployed" color="default" size="small" />
                    )}
                  </TableCell>
                  <TableCell>
                    <IconButton
                      size="small"
                      onClick={() => setSelectedModel(model)}
                    >
                      <VisibilityIcon />
                    </IconButton>
                    {!model.is_deployed && (
                      <IconButton
                        size="small"
                        color="primary"
                        onClick={() => {
                          modelsAPI.deploy(model.id).then(() => refetch());
                        }}
                      >
                        <RocketLaunchIcon />
                      </IconButton>
                    )}
                    <IconButton
                      size="small"
                      color="error"
                      onClick={() => {
                        if (window.confirm('Are you sure you want to delete this model?')) {
                          modelsAPI.delete(model.id).then(() => refetch());
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
                    No models found
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
