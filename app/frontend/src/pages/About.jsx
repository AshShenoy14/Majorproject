import React, { useContext } from 'react';
import { Container, Typography, Box, Paper, Grid, Chip } from '@mui/material';
import { motion } from 'framer-motion';
import { ThemeContext } from '../ThemeContext';

const stagger = {
    animate: { transition: { staggerChildren: 0.12 } },
};

const fadeUp = {
    initial: { opacity: 0, y: 25 },
    animate: { opacity: 1, y: 0, transition: { duration: 0.65, ease: [0.4, 0, 0.2, 1] } },
};

const About = () => {
    const { mode } = useContext(ThemeContext);
    const isDark = mode === 'dark';

    const cardStyles = {
        p: 4,
        borderRadius: 4,
        background: isDark ? 'rgba(16, 33, 65, 0.5)' : 'rgba(255, 255, 255, 0.6)',
        backdropFilter: 'blur(20px) saturate(180%)',
        border: `1px solid ${isDark ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.06)'}`,
        transition: 'all 0.35s cubic-bezier(0.4, 0, 0.2, 1)',
        '&:hover': {
            transform: 'translateY(-4px)',
            boxShadow: isDark
                ? '0 20px 60px rgba(0, 0, 0, 0.3)'
                : '0 20px 60px rgba(0, 0, 0, 0.08)',
            borderColor: isDark ? 'rgba(0, 229, 255, 0.15)' : 'rgba(0, 105, 92, 0.15)',
        },
    };

    return (
        <Container maxWidth="md" sx={{ py: { xs: 6, md: 8 }, minHeight: '80vh' }}>
            <motion.div variants={stagger} initial="initial" animate="animate">
                {/* Header */}
                <motion.div variants={fadeUp}>
                    <Box textAlign="center" mb={6}>
                        <Chip
                            label="📖 Research Project"
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
                                    ? 'linear-gradient(135deg, #00e5ff 0%, #7c4dff 50%, #d500f9 100%)'
                                    : 'linear-gradient(135deg, #00695c 0%, #1565c0 100%)',
                                WebkitBackgroundClip: 'text',
                                WebkitTextFillColor: 'transparent',
                            }}
                        >
                            About TransGraph-PPI
                        </Typography>
                        <Typography
                            variant="h6"
                            color="text.secondary"
                            sx={{ maxWidth: 600, mx: 'auto', lineHeight: 1.7, fontWeight: 400 }}
                        >
                            A Hybrid Artificial Intelligence model for Protein-Protein Interaction Prediction.
                        </Typography>
                    </Box>
                </motion.div>

                {/* The Science */}
                <motion.div variants={fadeUp}>
                    <Paper elevation={0} sx={{ ...cardStyles, mb: 4 }}>
                        <Typography
                            variant="h4"
                            gutterBottom
                            sx={{
                                fontWeight: 700,
                                background: isDark
                                    ? 'linear-gradient(90deg, #d500f9, #ff5bff)'
                                    : 'linear-gradient(90deg, #1565c0, #42a5f5)',
                                WebkitBackgroundClip: 'text',
                                WebkitTextFillColor: 'transparent',
                            }}
                        >
                            The Science
                        </Typography>
                        <Typography variant="body1" paragraph sx={{ lineHeight: 1.8 }}>
                            Protein-Protein Interactions (PPIs) are central to almost every cellular process.
                            Predicting how proteins interact helps scientists understand devastating diseases,
                            formulate new drugs, and decode the complex machinery of life exactly.
                        </Typography>
                        <Typography variant="body1" paragraph sx={{ lineHeight: 1.8 }}>
                            Unfortunately, wet-lab experiments to find interactions are incredibly expensive and
                            unbelievably slow. Computational prediction models offer a massively
                            scalable alternative to scan vast networks for promising interaction targets.
                        </Typography>
                    </Paper>
                </motion.div>

                {/* Model Cards */}
                <Grid container spacing={3} sx={{ mb: 4 }}>
                    <Grid item xs={12} sm={6}>
                        <motion.div variants={fadeUp} style={{ height: '100%' }}>
                            <Paper elevation={0} sx={{ ...cardStyles, height: '100%' }}>
                                <Box sx={{
                                    display: 'inline-flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    width: 48,
                                    height: 48,
                                    borderRadius: 3,
                                    mb: 2,
                                    background: isDark
                                        ? 'linear-gradient(135deg, rgba(0, 229, 255, 0.15), rgba(0, 229, 255, 0.05))'
                                        : 'linear-gradient(135deg, rgba(0, 105, 92, 0.12), rgba(0, 105, 92, 0.04))',
                                    border: `1px solid ${isDark ? 'rgba(0, 229, 255, 0.2)' : 'rgba(0, 105, 92, 0.15)'}`,
                                }}>
                                    <Typography sx={{ fontSize: '1.5rem' }}>🧠</Typography>
                                </Box>
                                <Typography
                                    variant="h6"
                                    gutterBottom
                                    sx={{
                                        fontWeight: 700,
                                        color: isDark ? '#00e5ff' : '#00695c',
                                    }}
                                >
                                    ESM-2
                                </Typography>
                                <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.7 }}>
                                    An advanced Evolutionary Scale Modeling transformer. ESM-2 looks at millions
                                    of protein sequences to understand the fundamental grammar and evolution of proteins,
                                    generating rich numerical embeddings from simple amino-acid strings.
                                </Typography>
                            </Paper>
                        </motion.div>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                        <motion.div variants={fadeUp} style={{ height: '100%' }}>
                            <Paper elevation={0} sx={{ ...cardStyles, height: '100%' }}>
                                <Box sx={{
                                    display: 'inline-flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    width: 48,
                                    height: 48,
                                    borderRadius: 3,
                                    mb: 2,
                                    background: isDark
                                        ? 'linear-gradient(135deg, rgba(213, 0, 249, 0.15), rgba(213, 0, 249, 0.05))'
                                        : 'linear-gradient(135deg, rgba(21, 101, 192, 0.12), rgba(21, 101, 192, 0.04))',
                                    border: `1px solid ${isDark ? 'rgba(213, 0, 249, 0.2)' : 'rgba(21, 101, 192, 0.15)'}`,
                                }}>
                                    <Typography sx={{ fontSize: '1.5rem' }}>🕸️</Typography>
                                </Box>
                                <Typography
                                    variant="h6"
                                    gutterBottom
                                    sx={{
                                        fontWeight: 700,
                                        color: isDark ? '#d500f9' : '#1565c0',
                                    }}
                                >
                                    GAT
                                </Typography>
                                <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.7 }}>
                                    Graph Attention Networks process proteins as a connected 3D network.
                                    Unlike standard neural networks, Graph Attention allows the model to learn
                                    the structural 'shape' of interactions by giving specific attention to
                                    highly-relevant neighboring nodes in the biological graph.
                                </Typography>
                            </Paper>
                        </motion.div>
                    </Grid>
                </Grid>
            </motion.div>
        </Container>
    );
};

export default About;
