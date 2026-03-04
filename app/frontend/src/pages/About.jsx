import React, { useContext } from 'react';
import { Container, Typography, Box, Paper, Grid } from '@mui/material';
import { motion } from 'framer-motion';
import { ThemeContext } from '../ThemeContext';

const About = () => {
    const { mode } = useContext(ThemeContext);

    const cardBgColor = mode === 'dark' ? 'rgba(16, 33, 65, 0.6)' : 'rgba(255, 255, 255, 0.6)';
    const borderColor = mode === 'dark' ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.08)';

    return (
        <Container maxWidth="md" sx={{ py: 8, minHeight: '80vh' }}>
            <motion.div
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
            >
                <Typography variant="h2" align="center" color="primary" gutterBottom>
                    About TransGraph-PPI
                </Typography>
                <Typography variant="h5" align="center" color="text.secondary" paragraph sx={{ mb: 6 }}>
                    A Hybrid Artificial Intelligence model for Protein-Protein Interaction Prediction.
                </Typography>

                <Paper
                    elevation={0}
                    sx={{
                        p: 4,
                        borderRadius: 4,
                        background: cardBgColor,
                        backdropFilter: 'blur(20px)',
                        border: `1px solid ${borderColor}`,
                        mb: 4
                    }}
                >
                    <Typography variant="h4" gutterBottom color="secondary">
                        The Science
                    </Typography>
                    <Typography variant="body1" paragraph>
                        Protein-Protein Interactions (PPIs) are central to almost every cellular process.
                        Predicting how proteins interact helps scientists understand devastating diseases,
                        formulate new drugs, and decode the complex machinery of life exactly.
                    </Typography>
                    <Typography variant="body1" paragraph>
                        Unfortunately, wet-lab experiments to find interactions are incredibly expensive and
                        unbelievably slow. Computational prediction models offer a massively
                        scalable alternative to scan vast networks for promising interaction targets.
                    </Typography>
                </Paper>

                <Grid container spacing={4} sx={{ mb: 4 }}>
                    <Grid item xs={12} sm={6}>
                        <Paper
                            elevation={0}
                            sx={{
                                p: 4,
                                height: '100%',
                                borderRadius: 4,
                                background: cardBgColor,
                                backdropFilter: 'blur(20px)',
                                border: `1px solid ${borderColor}`,
                            }}
                        >
                            <Typography variant="h6" color="primary" gutterBottom>
                                ESM-2
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                                An advanced Evolutionary Scale Modeling transformer. ESM-2 looks at millions
                                of protein sequences to understand the fundamental grammar and evolution of proteins,
                                generating rich numerical embeddings from simple amino-acid strings.
                            </Typography>
                        </Paper>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                        <Paper
                            elevation={0}
                            sx={{
                                p: 4,
                                height: '100%',
                                borderRadius: 4,
                                background: cardBgColor,
                                backdropFilter: 'blur(20px)',
                                border: `1px solid ${borderColor}`,
                            }}
                        >
                            <Typography variant="h6" color="secondary" gutterBottom>
                                GAT
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                                Graph Attention Networks process proteins as a connected 3D network.
                                Unlike standard neural networks, Graph Attention allows the model to learn
                                the structural 'shape' of interactions by giving specific attention to
                                highly-relevant neighboring nodes in the biological graph.
                            </Typography>
                        </Paper>
                    </Grid>
                </Grid>
            </motion.div>
        </Container>
    );
};

export default About;
