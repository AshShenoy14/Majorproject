import { Container, CssBaseline, ThemeProvider, createTheme, Box } from '@mui/material';
import PredictionForm from './components/PredictionForm';
import RealDataVerifier from './components/RealDataVerifier';

const theme = createTheme({
    palette: {
        mode: 'light',
        primary: {
            main: '#2196f3',
        },
        background: {
            default: '#f0f2f5',
        },
    },
    typography: {
        fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    },
});

function App() {
    return (
        <ThemeProvider theme={theme}>
            <CssBaseline />
            <Container maxWidth="md">
                <Box sx={{ py: 8 }}>
                    <PredictionForm />
                    <RealDataVerifier />
                </Box>
            </Container>
        </ThemeProvider>
    );
}

export default App;
