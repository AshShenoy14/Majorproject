import React, { useContext } from 'react';
import { Container, Typography, Box, Chip } from '@mui/material';
import RealDataVerifier from '../components/RealDataVerifier';
import { motion } from 'framer-motion';
import { ThemeContext } from '../ThemeContext';

const stagger = {
    animate: { transition: { staggerChildren: 0.1 } },
};

const fadeUp = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0, transition: { duration: 0.6, ease: [0.4, 0, 0.2, 1] } },
};

const Verify = () => {
    const { mode } = useContext(ThemeContext);
    const isDark = mode === 'dark';

    return (
        <Container maxWidth="lg" sx={{ py: { xs: 4, md: 6 }, minHeight: '80vh' }}>
            <motion.div variants={stagger} initial="initial" animate="animate">
                <motion.div variants={fadeUp}>
                    <Box textAlign="center" mb={6}>
                        <Chip
                            label="🧬 Biological Validation"
                            variant="outlined"
                            size="small"
                            sx={{
                                mb: 2,
                                borderColor: isDark ? 'rgba(213, 0, 249, 0.25)' : 'rgba(21, 101, 192, 0.25)',
                                color: isDark ? '#d500f9' : '#1565c0',
                                fontFamily: '"Outfit", sans-serif',
                                fontWeight: 600,
                                background: isDark ? 'rgba(213, 0, 249, 0.05)' : 'rgba(21, 101, 192, 0.03)',
                            }}
                        />
                        <Typography
                            variant="h2"
                            gutterBottom
                            sx={{
                                fontWeight: 800,
                                background: isDark
                                    ? 'linear-gradient(135deg, #d500f9 0%, #00e5ff 100%)'
                                    : 'linear-gradient(135deg, #1565c0 0%, #00695c 100%)',
                                WebkitBackgroundClip: 'text',
                                WebkitTextFillColor: 'transparent',
                            }}
                        >
                            Network Verification
                        </Typography>
                        <Typography
                            variant="body1"
                            color="text.secondary"
                            sx={{ maxWidth: 600, mx: 'auto', lineHeight: 1.7 }}
                        >
                            Verify interactions against real-world biological datasets and ChEMBL drug targets
                            to validate predictions.
                        </Typography>
                    </Box>
                </motion.div>

                <motion.div variants={fadeUp}>
                    <RealDataVerifier />
                </motion.div>
            </motion.div>
        </Container>
    );
};

export default Verify;
