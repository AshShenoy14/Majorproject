import React, { useContext } from 'react';
import { Container, Typography, Box, Chip } from '@mui/material';
import PredictionForm from '../components/PredictionForm';
import { motion } from 'framer-motion';
import { ThemeContext } from '../ThemeContext';

const stagger = {
    animate: { transition: { staggerChildren: 0.1 } },
};

const fadeUp = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0, transition: { duration: 0.6, ease: [0.4, 0, 0.2, 1] } },
};

const Predict = () => {
    const { mode } = useContext(ThemeContext);
    const isDark = mode === 'dark';

    return (
        <Container maxWidth="lg" sx={{ py: { xs: 4, md: 6 }, minHeight: '80vh' }}>
            <motion.div variants={stagger} initial="initial" animate="animate">
                <motion.div variants={fadeUp}>
                    <Box textAlign="center" mb={6}>
                        <Chip
                            label="🔬 AI-Powered Analysis"
                            variant="outlined"
                            size="small"
                            sx={{
                                mb: 2,
                                borderColor: isDark ? 'rgba(0, 229, 255, 0.25)' : 'rgba(0, 105, 92, 0.25)',
                                color: isDark ? '#00e5ff' : '#00695c',
                                fontFamily: '"Outfit", sans-serif',
                                fontWeight: 600,
                                background: isDark ? 'rgba(0, 229, 255, 0.05)' : 'rgba(0, 105, 92, 0.03)',
                            }}
                        />
                        <Typography
                            variant="h2"
                            gutterBottom
                            sx={{
                                fontWeight: 800,
                                background: isDark
                                    ? 'linear-gradient(135deg, #00e5ff 0%, #7c4dff 100%)'
                                    : 'linear-gradient(135deg, #00695c 0%, #1565c0 100%)',
                                WebkitBackgroundClip: 'text',
                                WebkitTextFillColor: 'transparent',
                            }}
                        >
                            Interaction Predictor
                        </Typography>
                        <Typography
                            variant="body1"
                            color="text.secondary"
                            sx={{ maxWidth: 600, mx: 'auto', lineHeight: 1.7 }}
                        >
                            Enter two protein sequences or UniProt IDs to predict their likelihood of interacting
                            using our hybrid ensemble model.
                        </Typography>
                    </Box>
                </motion.div>

                <motion.div variants={fadeUp}>
                    <PredictionForm />
                </motion.div>
            </motion.div>
        </Container>
    );
};

export default Predict;
