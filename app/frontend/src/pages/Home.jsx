import React, { useContext } from 'react';
import { Box, Container, Typography } from '@mui/material';
import { motion } from 'framer-motion';
import { ThemeContext } from '../ThemeContext';

const Home = () => {
    const { mode } = useContext(ThemeContext);

    return (
        <Box sx={{
            pt: { xs: 8, md: 12 },
            pb: { xs: 6, md: 10 },
            position: 'relative',
            overflow: 'hidden',
            mb: 6,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '60vh'
        }}>
            {/* Background glow effects */}
            {mode === 'dark' && (
                <Box sx={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    width: '100vw',
                    height: '100%',
                    background: 'radial-gradient(ellipse at top, rgba(0, 229, 255, 0.15) 0%, transparent 60%)',
                    zIndex: -1,
                    pointerEvents: 'none'
                }} />
            )}

            <Container maxWidth="md" sx={{ position: 'relative', zIndex: 1 }}>
                <motion.div
                    initial={{ y: 30, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                >
                    <Typography
                        variant="h1"
                        align="center"
                        gutterBottom
                        sx={{
                            fontSize: { xs: '2.5rem', sm: '3.5rem', md: '4.5rem' },
                            background: mode === 'dark' ? 'linear-gradient(135deg, #ffffff 0%, #ccd6f6 100%)' : 'linear-gradient(135deg, #1a202c 0%, #4a5568 100%)',
                            WebkitBackgroundClip: 'text',
                            WebkitTextFillColor: 'transparent',
                            mb: 2
                        }}
                    >
                        Predict Protein <Box component="span" sx={{
                            background: mode === 'dark'
                                ? `linear-gradient(135deg, #00e5ff 0%, #d500f9 100%)`
                                : `linear-gradient(135deg, #00695c 0%, #1565c0 100%)`,
                            WebkitBackgroundClip: 'text',
                            WebkitTextFillColor: 'transparent',
                        }}>Interactions</Box>
                    </Typography>
                </motion.div>

                <motion.div
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ duration: 0.8, delay: 0.2, ease: "easeOut" }}
                >
                    <Typography variant="h5" align="center" color="text.secondary" sx={{ maxWidth: '80%', mx: 'auto', lineHeight: 1.6 }}>
                        Advanced PPI prediction using Hybrid Ensemble Learning (ESM-2 + GAT)
                    </Typography>
                </motion.div>
            </Container>
        </Box>
    );
};

export default Home;
