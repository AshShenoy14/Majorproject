import React from 'react';
import { Box, Typography, Stack, Paper, Chip } from '@mui/material';
import { motion } from 'framer-motion';

const InteractionVisualizer = ({ result, id1, id2 }) => {
    if (!result) return null;

    const { interaction_probability, explanation } = result;
    const isInteracting = interaction_probability > 0.5;

    // Decide colors and texts based on probability
    const glowColor = isInteracting ? 'rgba(76, 175, 80, 0.6)' : 'rgba(244, 67, 54, 0.6)';
    const mainColor = isInteracting ? '#4caf50' : '#f44336';
    const statusText = isInteracting ? 'High Probability of Interaction' : 'Low Probability of Interaction';

    // Create a plain English explanation
    const getExplanation = () => {
        if (interaction_probability > 0.8) {
            return "These proteins fit together almost perfectly, like two pieces of a puzzle. They are highly likely to bind and work together in the cell.";
        } else if (interaction_probability > 0.5) {
            return "These proteins show signs that they might interact. Their structures have some complementary areas that could allow them to connect.";
        } else if (interaction_probability > 0.2) {
            return "It's unlikely these proteins interact directly. Their shapes don't quite match up well enough to form a stable connection.";
        } else {
            return "These proteins repel each other or have completely mismatched shapes. They are very unlikely to interact in nature.";
        }
    };

    const getModelInsight = () => {
        const seq = explanation.Sequence_Model_Contribution;
        const graph = explanation.Graph_Model_Contribution;

        if (seq > 0.5 && graph > 0.5) {
            return "Both their amino acid sequences AND their position in the cellular network strongly suggest they work together.";
        } else if (seq > 0.5) {
            return "Their chemical sequences look like a match, even though they aren't known to be close in the protein network.";
        } else if (graph > 0.5) {
            return "They operate in the same neighborhood of the cell, making an interaction more likely despite their sequences.";
        } else {
            return "Neither their sequences nor their network positions suggest they would interact.";
        }
    };

    return (
        <Paper elevation={0} sx={{ p: 4, bgcolor: '#f8f9fa', borderRadius: 3, mt: 4 }}>
            <Typography variant="h6" color="primary" gutterBottom sx={{ fontWeight: 600, textAlign: 'center', mb: 4 }}>
                What Does This Mean?
            </Typography>

            <Stack direction={{ xs: 'column', md: 'row' }} spacing={4} alignItems="center">

                {/* Animation Container */}
                <Box
                    sx={{
                        width: { xs: '100%', md: '50%' },
                        height: 250,
                        position: 'relative',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        bgcolor: '#ffffff',
                        borderRadius: 4,
                        boxShadow: `0 8px 32px ${glowColor}`,
                        overflow: 'hidden'
                    }}
                >
                    {/* Protein 1 (Left shape) */}
                    <motion.div
                        initial={{ x: -100, opacity: 0 }}
                        animate={{
                            x: isInteracting ? -15 : -60,
                            opacity: 1,
                            rotate: isInteracting ? 0 : -10
                        }}
                        transition={{ duration: 1.5, type: 'spring', bounce: 0.4 }}
                        style={{
                            position: 'absolute',
                            width: 120,
                            height: 120,
                            backgroundColor: '#1976d2',
                            borderRadius: isInteracting ? '20% 0 0 20%' : '20%',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            color: 'white',
                            fontWeight: 'bold',
                            clipPath: isInteracting ? 'polygon(0% 0%, 75% 0%, 100% 50%, 75% 100%, 0% 100%)' : 'none'
                        }}
                    >
                        <Typography variant="caption" sx={{ px: 2, textAlign: 'center' }}>
                            {id1 || "Protein A"}
                        </Typography>
                    </motion.div>

                    {/* Protein 2 (Right shape) */}
                    <motion.div
                        initial={{ x: 100, opacity: 0 }}
                        animate={{
                            x: isInteracting ? 15 : 60,
                            opacity: 1,
                            rotate: isInteracting ? 0 : 10
                        }}
                        transition={{ duration: 1.5, type: 'spring', bounce: 0.4 }}
                        style={{
                            position: 'absolute',
                            width: 120,
                            height: 120,
                            backgroundColor: '#9c27b0',
                            borderRadius: isInteracting ? '0 20% 20% 0' : '20%',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            color: 'white',
                            fontWeight: 'bold',
                            clipPath: isInteracting ? 'polygon(25% 0%, 100% 0%, 100% 100%, 25% 100%, 0% 50%)' : 'none'
                        }}
                    >
                        <Typography variant="caption" sx={{ px: 2, textAlign: 'center' }}>
                            {id2 || "Protein B"}
                        </Typography>
                    </motion.div>

                    {/* Connection Sparks (only if interacting) */}
                    {isInteracting && (
                        <motion.div
                            initial={{ scale: 0, opacity: 0 }}
                            animate={{ scale: [1, 1.5, 1], opacity: [0, 0.8, 0] }}
                            transition={{ delay: 1, duration: 2, repeat: Infinity }}
                            style={{
                                position: 'absolute',
                                width: 60,
                                height: 60,
                                backgroundColor: '#ffeb3b',
                                filter: 'blur(15px)',
                                borderRadius: '50%',
                                zIndex: 0
                            }}
                        />
                    )}
                </Box>

                {/* Text Explanation */}
                <Box sx={{ width: { xs: '100%', md: '50%' } }}>
                    <Chip
                        label={statusText}
                        sx={{
                            bgcolor: mainColor,
                            color: 'white',
                            fontWeight: 'bold',
                            mb: 2
                        }}
                    />
                    <Typography variant="body1" paragraph sx={{ fontSize: '1.1rem', color: '#333' }}>
                        {getExplanation()}
                    </Typography>

                    <Box sx={{ mt: 3, p: 2, bgcolor: '#e3f2fd', borderRadius: 2, borderLeft: '4px solid #1976d2' }}>
                        <Typography variant="subtitle2" color="primary" gutterBottom>
                            🧠 How the AI knows this:
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                            {getModelInsight()}
                        </Typography>
                    </Box>
                </Box>

            </Stack>
        </Paper>
    );
};

export default InteractionVisualizer;
