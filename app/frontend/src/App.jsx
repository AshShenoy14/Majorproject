import { ThemeProvider, CssBaseline, Box, Container, Typography } from '@mui/material';
import PredictionForm from './components/PredictionForm';
import RealDataVerifier from './components/RealDataVerifier';
import Layout from './components/Layout';
import theme from './theme';
import { motion } from 'framer-motion';

function App() {
    return (
        <ThemeProvider theme={theme}>
            <CssBaseline />
            <Layout>
                {/* Hero Section */}
                <Box sx={{
                    pt: 8,
                    pb: 6,
                    background: 'linear-gradient(135deg, #e0f2f1 0%, #e3f2fd 100%)',
                    borderRadius: '0 0 50% 50% / 40px',
                    mb: 6
                }}>
                    <Container maxWidth="md">
                        <Typography
                            component={motion.h1}
                            initial={{ y: -20, opacity: 0 }}
                            animate={{ y: 0, opacity: 1 }}
                            variant="h2"
                            align="center"
                            color="text.primary"
                            gutterBottom
                        >
                            Predict Protein Interactions
                        </Typography>
                        <Typography variant="h5" align="center" color="text.secondary" paragraph>
                            Advanced PPI prediction using Hybrid Ensemble Learning (ESM-2 + GAT)
                        </Typography>
                    </Container>
                </Box>

                <Container maxWidth="lg" sx={{ mb: 8 }}>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                        <PredictionForm />
                        <RealDataVerifier />
                    </Box>
                </Container>
            </Layout>
        </ThemeProvider>
    );
}

export default App;
