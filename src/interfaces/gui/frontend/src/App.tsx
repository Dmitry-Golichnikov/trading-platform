import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import { QueryClient, QueryClientProvider } from 'react-query';
import { useStore } from '@/store';
import Layout from '@/components/Layout';
import Dashboard from '@/pages/Dashboard';
import Datasets from '@/pages/Datasets';
import Features from '@/pages/Features';
import Labeling from '@/pages/Labeling';
import Experiments from '@/pages/Experiments';
import Models from '@/pages/Models';
import Backtests from '@/pages/Backtests';
import System from '@/pages/System';

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

function App() {
  const darkMode = useStore((state) => state.darkMode);

  const theme = createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: {
        main: '#1976d2',
      },
      secondary: {
        main: '#dc004e',
      },
    },
  });

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Router>
          <Layout>
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/datasets" element={<Datasets />} />
              <Route path="/features" element={<Features />} />
              <Route path="/labeling" element={<Labeling />} />
              <Route path="/experiments" element={<Experiments />} />
              <Route path="/models" element={<Models />} />
              <Route path="/backtests" element={<Backtests />} />
              <Route path="/system" element={<System />} />
            </Routes>
          </Layout>
        </Router>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
