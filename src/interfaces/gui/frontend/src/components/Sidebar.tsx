import { List, ListItem, ListItemButton, ListItemIcon, ListItemText, Divider } from '@mui/material';
import { useNavigate, useLocation } from 'react-router-dom';
import DashboardIcon from '@mui/icons-material/Dashboard';
import DatasetIcon from '@mui/icons-material/Dataset';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import LabelIcon from '@mui/icons-material/Label';
import ScienceIcon from '@mui/icons-material/Science';
import ModelTrainingIcon from '@mui/icons-material/ModelTraining';
import AssessmentIcon from '@mui/icons-material/Assessment';
import SettingsIcon from '@mui/icons-material/Settings';

const menuItems = [
  { text: 'Dashboard', icon: <DashboardIcon />, path: '/dashboard' },
  { text: 'Datasets', icon: <DatasetIcon />, path: '/datasets' },
  { text: 'Features', icon: <AutoAwesomeIcon />, path: '/features' },
  { text: 'Labeling', icon: <LabelIcon />, path: '/labeling' },
  { text: 'Experiments', icon: <ScienceIcon />, path: '/experiments' },
  { text: 'Models', icon: <ModelTrainingIcon />, path: '/models' },
  { text: 'Backtests', icon: <AssessmentIcon />, path: '/backtests' },
];

const systemItems = [
  { text: 'System', icon: <SettingsIcon />, path: '/system' },
];

export default function Sidebar() {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <>
      <List>
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton
              selected={location.pathname === item.path}
              onClick={() => navigate(item.path)}
            >
              <ListItemIcon>{item.icon}</ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>

      <Divider />

      <List>
        {systemItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton
              selected={location.pathname === item.path}
              onClick={() => navigate(item.path)}
            >
              <ListItemIcon>{item.icon}</ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </>
  );
}
