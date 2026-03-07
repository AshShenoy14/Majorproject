import { Box, Typography, Stack, Paper, Chip, useTheme } from '@mui/material';
import { motion } from 'framer-motion';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip } from 'recharts';

const InteractionVisualizer = ({ result, id1, id2 }) => {
    const theme = useTheme();
    const isDark = theme.palette.mode === 'dark';
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

    const radarData = [
        { subject: 'Sequence Match', A: explanation.Sequence_Model_Contribution * 100, fullMark: 100 },
        { subject: 'Network Proximity', A: explanation.Graph_Model_Contribution * 100, fullMark: 100 }
    ];
    if (explanation.SHAP_Sequence !== undefined) {
        // Normalize SHAP to 0-100 range roughly for visual purposes (absolute magnitude of influence)
        const seqShap = Math.min(Math.abs(explanation.SHAP_Sequence) * 200, 100);
        const graphShap = Math.min(Math.abs(explanation.SHAP_Graph) * 200, 100);
        radarData.push({ subject: 'Seq SHAP Impact', A: seqShap, fullMark: 100 });
        radarData.push({ subject: 'Graph SHAP Impact', A: graphShap, fullMark: 100 });
    }

    return (
        <Paper elevation={0} sx={{
            p: 4,
            background: isDark ? 'rgba(255, 255, 255, 0.02)' : 'rgba(0, 0, 0, 0.02)',
            border: `1px solid ${isDark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)'}`,
            borderRadius: 3,
            mt: 4
        }}>
            <Typography variant="h5" color="secondary" gutterBottom sx={{ fontWeight: 700, textAlign: 'center', mb: 4, textShadow: isDark ? '0 0 10px rgba(213, 0, 249, 0.3)' : 'none' }}>
                Explainability Insights
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
                        background: isDark ? 'rgba(0, 0, 0, 0.2)' : 'rgba(0, 0, 0, 0.02)',
                        border: `1px solid ${isDark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)'}`,
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
                            bgcolor: isInteracting ? (isDark ? 'rgba(0, 255, 136, 0.2)' : 'rgba(46, 125, 50, 0.1)') : (isDark ? 'rgba(255, 75, 75, 0.2)' : 'rgba(211, 47, 47, 0.1)'),
                            color: isInteracting ? (isDark ? '#00ff88' : '#2e7d32') : (isDark ? '#ff4b4b' : '#d32f2f'),
                            border: `1px solid ${isInteracting ? (isDark ? 'rgba(0, 255, 136, 0.5)' : 'rgba(46, 125, 50, 0.5)') : (isDark ? 'rgba(255, 75, 75, 0.5)' : 'rgba(211, 47, 47, 0.5)')}`,
                            fontWeight: 'bold',
                            mb: 2,
                            boxShadow: `0 0 10px ${isInteracting ? (isDark ? 'rgba(0, 255, 136, 0.2)' : 'rgba(46, 125, 50, 0.1)') : (isDark ? 'rgba(255, 75, 75, 0.2)' : 'rgba(211, 47, 47, 0.1)')}`
                        }}
                    />
                    <Typography variant="body1" paragraph sx={{ fontSize: '1.1rem', color: isDark ? '#e6f1ff' : 'text.primary', lineHeight: 1.7 }}>
                        {getExplanation()}
                    </Typography>

                    <Box sx={{ mt: 3, p: 3, bgcolor: isDark ? 'rgba(0, 229, 255, 0.05)' : 'rgba(0, 105, 92, 0.05)', borderRadius: 3, borderLeft: `4px solid ${isDark ? '#00e5ff' : '#00695c'}` }}>
                        <Typography variant="subtitle2" color="primary" gutterBottom sx={{ fontWeight: 600 }}>
                            🧠 How the AI knows this:
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.6 }}>
                            {getModelInsight()}
                        </Typography>
                    </Box>

                    {radarData.length > 2 && (
                        <Box sx={{ mt: 3, height: 250, width: '100%', bgcolor: isDark ? 'rgba(0,0,0,0.2)' : 'rgba(0,0,0,0.02)', borderRadius: 3, p: 2, border: `1px solid ${isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)'}` }}>
                            <Typography variant="caption" color="text.secondary" align="center" display="block">Model Weighting (Radar)</Typography>
                            <ResponsiveContainer width="100%" height="100%">
                                <RadarChart cx="50%" cy="50%" outerRadius="70%" data={radarData}>
                                    <PolarGrid stroke={isDark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.2)'} />
                                    <PolarAngleAxis dataKey="subject" tick={{ fill: isDark ? '#aaa' : '#555', fontSize: 10 }} />
                                    <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                                    <Radar name="Weights" dataKey="A" stroke={mainColor} fill={glowColor} fillOpacity={0.5} />
                                    <Tooltip contentStyle={{ backgroundColor: isDark ? '#102141' : '#fff', color: isDark ? '#fff' : '#000', borderRadius: '8px' }} />
                                </RadarChart>
                            </ResponsiveContainer>
                        </Box>
                    )}
                </Box>

            </Stack>
        </Paper>
    );
};

export default InteractionVisualizer;
