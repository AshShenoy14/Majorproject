import React from 'react';
import { Container, Typography, Box } from '@mui/material';
import PredictionForm from '../components/PredictionForm';
import { motion } from 'framer-motion';

const Predict = () => {
    return (
        <Container maxWidth="lg" sx={{ py: 6, minHeight: '80vh' }}>
            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
            >
                <Box textAlign="center" mb={6}>
                    <Typography variant="h2" gutterBottom color="primary">
                        Interaction Predictor
                    </Typography>
                    <Typography variant="body1" color="text.secondary">
                        Enter two protein sequences or IDs to predict their likelihood of interacting.
                    </Typography>
                </Box>
            </motion.div>

            <PredictionForm />
        </Container>
    );
};

export default Predict;
