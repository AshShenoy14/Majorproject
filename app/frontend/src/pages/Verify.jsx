import React from 'react';
import { Container, Typography, Box } from '@mui/material';
import RealDataVerifier from '../components/RealDataVerifier';
import { motion } from 'framer-motion';

const Verify = () => {
    return (
        <Container maxWidth="lg" sx={{ py: 6, minHeight: '80vh' }}>
            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
            >
                <Box textAlign="center" mb={6}>
                    <Typography variant="h2" gutterBottom color="primary">
                        Network Verification
                    </Typography>
                    <Typography variant="body1" color="text.secondary">
                        Verify interactions against real-world biological datasets and ChEMBL drug targets.
                    </Typography>
                </Box>
            </motion.div>

            <RealDataVerifier />
        </Container>
    );
};

export default Verify;
