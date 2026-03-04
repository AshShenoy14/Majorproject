import { useState } from 'react';
import axios from 'axios';
import {
  Box,
  Button,
  TextField,
  Typography,
  Paper,
  CircularProgress,
  Grid,
  Divider,
  Fade,
  Stack,
  Alert
} from '@mui/material';
import { motion, AnimatePresence } from 'framer-motion';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import InteractionVisualizer from './InteractionVisualizer';

const PredictionForm = () => {
  const [id1, setId1] = useState('');
  const [id2, setId2] = useState('');
  const [seq1, setSeq1] = useState('');
  const [seq2, setSeq2] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Replace with actual API URL
      const response = await axios.post('http://localhost:8000/predict', {
        protein1_id: id1 || "Protein A",
        protein2_id: id2 || "Protein B",
        protein1_seq: seq1,
        protein2_seq: seq2,
      });
      setResult(response.data);
    } catch (err) {
      setError('Failed to fetch prediction. Ensure backend is running.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const probData = result ? [
    { name: 'Interaction', value: result.interaction_probability },
    { name: 'No Interaction', value: 1 - result.interaction_probability }
  ] : [];

  const COLORS = ['#00e5ff', 'rgba(255, 255, 255, 0.1)'];

  return (
    <Paper
      elevation={0}
      component={motion.div}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      sx={{
        p: 4,
        borderRadius: 4,
        overflow: 'hidden',
        background: 'rgba(16, 33, 65, 0.6)',
        backdropFilter: 'blur(20px)',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        boxShadow: '0 20px 40px rgba(0,0,0,0.2)'
      }}
    >
      <Grid container spacing={4}>
        {/* Input Section */}
        <Grid item xs={12} md={6}>
          <Typography variant="h4" color="primary" gutterBottom sx={{ fontWeight: 700, textShadow: '0 0 15px rgba(0, 229, 255, 0.3)' }}>
            Analyze Interaction
          </Typography>
          <Typography variant="body1" color="text.secondary" paragraph sx={{ mb: 4 }}>
            Enter protein IDs (UniProt/Ensembl) or sequences to predict interaction probability.
          </Typography>

          <form onSubmit={handleSubmit}>
            <Stack spacing={3}>
              <Box p={2} sx={{ background: 'rgba(0,0,0,0.2)', borderRadius: 2, border: '1px solid rgba(255,255,255,0.05)' }}>
                <Typography variant="caption" fontWeight="bold" color="secondary" sx={{ letterSpacing: '0.1em' }}>PROTEIN 1</Typography>
                <TextField
                  label="ID (e.g., P12345)"
                  fullWidth
                  value={id1}
                  onChange={(e) => setId1(e.target.value)}
                  margin="dense"
                  size="small"
                  variant="outlined"
                  sx={{ mt: 1 }}
                />
                <TextField
                  label="Sequence (Optional)"
                  fullWidth
                  multiline
                  rows={2}
                  value={seq1}
                  onChange={(e) => setSeq1(e.target.value)}
                  margin="dense"
                  size="small"
                  placeholder="MVLSPADKTN..."
                />
              </Box>

              <Box p={2} sx={{ background: 'rgba(0,0,0,0.2)', borderRadius: 2, border: '1px solid rgba(255,255,255,0.05)' }}>
                <Typography variant="caption" fontWeight="bold" color="secondary" sx={{ letterSpacing: '0.1em' }}>PROTEIN 2</Typography>
                <TextField
                  label="ID (e.g., Q98765)"
                  fullWidth
                  value={id2}
                  onChange={(e) => setId2(e.target.value)}
                  margin="dense"
                  size="small"
                  sx={{ mt: 1 }}
                />
                <TextField
                  label="Sequence (Optional)"
                  fullWidth
                  multiline
                  rows={2}
                  value={seq2}
                  onChange={(e) => setSeq2(e.target.value)}
                  margin="dense"
                  size="small"
                />
              </Box>

              <Button
                type="submit"
                variant="contained"
                color="primary"
                size="large"
                fullWidth
                disabled={loading}
                sx={{ mt: 2, height: 52, fontSize: '1.1rem' }}
              >
                {loading ? <CircularProgress size={24} color="inherit" /> : 'Run Prediction'}
              </Button>
            </Stack>
          </form>

          {error && (
            <Fade in>
              <Alert severity="error" variant="filled" sx={{ mt: 3 }}>{error}</Alert>
            </Fade>
          )}
        </Grid>

        {/* Results Section */}
        <Grid item xs={12} md={6} sx={{ borderLeft: { md: '1px solid rgba(255,255,255,0.1)' }, pl: { md: 5 } }}>
          <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <AnimatePresence mode="wait">
              {!result && !loading && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <Box sx={{ textAlign: 'center', opacity: 0.5, py: 8 }}>
                    <Typography variant="h6">Ready to Analyze</Typography>
                    <Typography variant="body2">Enter protein details to view prediction results.</Typography>
                  </Box>
                </motion.div>
              )}

              {loading && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <Box sx={{ textAlign: 'center', py: 8 }}>
                    <Typography variant="h6" color="primary">Analyzing Sequences...</Typography>
                    <Typography variant="caption">Calculating embeddings and graph attention</Typography>
                  </Box>
                </motion.div>
              )}

              {result && (
                <Stack
                  spacing={3}
                  component={motion.div}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Box>
                      <Typography variant="h6">Probability</Typography>
                      <Typography variant="h3" color="primary" fontWeight="bold">
                        {(result.interaction_probability * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                    <Box sx={{ width: 100, height: 100 }}>
                      <ResponsiveContainer>
                        <PieChart>
                          <Pie
                            data={probData}
                            innerRadius={35}
                            outerRadius={45}
                            paddingAngle={5}
                            dataKey="value"
                          >
                            {probData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip />
                        </PieChart>
                      </ResponsiveContainer>
                    </Box>
                  </Box>

                  <Divider />

                  {result.esm_probability !== undefined && result.gat_probability !== undefined && (
                    <Box>
                      <Typography variant="subtitle2" color="text.secondary" gutterBottom>Model Predictions</Typography>
                      <Stack spacing={1.5}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', bgcolor: 'rgba(0, 229, 255, 0.05)', border: '1px solid rgba(0, 229, 255, 0.1)', p: 1.5, borderRadius: 2 }}>
                          <Typography variant="body2">ESM-MLP</Typography>
                          <Typography variant="body2" fontWeight="bold" color="primary">
                            {(result.esm_probability * 100).toFixed(1)}%
                          </Typography>
                        </Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', bgcolor: 'rgba(213, 0, 249, 0.05)', border: '1px solid rgba(213, 0, 249, 0.1)', p: 1.5, borderRadius: 2 }}>
                          <Typography variant="body2">Graph Attention (GAT)</Typography>
                          <Typography variant="body2" fontWeight="bold" color="secondary">
                            {(result.gat_probability * 100).toFixed(1)}%
                          </Typography>
                        </Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', bgcolor: 'rgba(0, 255, 136, 0.05)', border: '1px solid rgba(0, 255, 136, 0.2)', p: 1.5, borderRadius: 2 }}>
                          <Typography variant="body2">Ensemble (Meta-learner)</Typography>
                          <Typography variant="body2" fontWeight="bold" sx={{ color: '#00ff88' }}>
                            {(result.interaction_probability * 100).toFixed(1)}%
                          </Typography>
                        </Box>
                      </Stack>
                    </Box>
                  )}

                  <Box>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>Explainability (SHAP)</Typography>
                    <Stack spacing={1}>
                      {Object.entries(result.explanation).map(([key, value]) => (
                        <Box key={key} sx={{ display: 'flex', justifyContent: 'space-between', bgcolor: 'rgba(255, 255, 255, 0.03)', p: 1, px: 1.5, borderRadius: 1 }}>
                          <Typography variant="caption" sx={{ textTransform: 'capitalize' }}>
                            {key.replace('_', ' ')}
                          </Typography>
                          <Typography variant="caption" fontWeight="bold" color="text.primary">
                            {typeof value === 'number' ? value.toFixed(4) : value}
                          </Typography>
                        </Box>
                      ))}
                    </Stack>
                  </Box>

                  <Box sx={{ textAlign: 'right' }}>
                    <Typography variant="caption" color="text.secondary">
                      Confidence Score: {(result.confidence_score * 100).toFixed(1)}%
                    </Typography>
                  </Box>

                  {/* Accessible Visualizer */}
                  <InteractionVisualizer result={result} id1={id1} id2={id2} />
                </Stack>
              )}
            </AnimatePresence>
          </Box>
        </Grid>
      </Grid>
    </Paper>
  );
};

export default PredictionForm;
