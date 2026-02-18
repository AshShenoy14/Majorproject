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

  const COLORS = ['#00695c', '#e0e0e0'];

  return (
    <Paper
      elevation={3}
      component={motion.div}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      sx={{ p: 4, borderRadius: 3, overflow: 'hidden' }}
    >
      <Grid container spacing={4}>
        {/* Input Section */}
        <Grid item xs={12} md={6}>
          <Typography variant="h5" color="primary" gutterBottom sx={{ fontWeight: 600 }}>
            Analyze Interaction
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            Enter protein IDs (UniProt/Ensembl) or sequences to predict interaction probability.
          </Typography>

          <form onSubmit={handleSubmit}>
            <Stack spacing={2}>
              <Box>
                <Typography variant="caption" fontWeight="bold" color="secondary">PROTEIN 1</Typography>
                <TextField
                  label="ID (e.g., P12345)"
                  fullWidth
                  value={id1}
                  onChange={(e) => setId1(e.target.value)}
                  margin="dense"
                  size="small"
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

              <Box>
                <Typography variant="caption" fontWeight="bold" color="secondary">PROTEIN 2</Typography>
                <TextField
                  label="ID (e.g., Q98765)"
                  fullWidth
                  value={id2}
                  onChange={(e) => setId2(e.target.value)}
                  margin="dense"
                  size="small"
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
                size="large"
                fullWidth
                disabled={loading}
                sx={{ mt: 2, height: 48 }}
              >
                {loading ? <CircularProgress size={24} color="inherit" /> : 'Run Prediction'}
              </Button>
            </Stack>
          </form>

          {error && (
            <Fade in>
              <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>
            </Fade>
          )}
        </Grid>

        {/* Results Section */}
        <Grid item xs={12} md={6} sx={{ borderLeft: { md: '1px solid #eee' }, pl: { md: 4 } }}>
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

                  <Box>
                    <Typography variant="subtitle2" gutterBottom>Explainability (SHAP)</Typography>
                    <Stack spacing={1}>
                      {Object.entries(result.explanation).map(([key, value]) => (
                        <Box key={key} sx={{ display: 'flex', justifyContent: 'space-between', bgcolor: '#f5f5f5', p: 1, borderRadius: 1 }}>
                          <Typography variant="caption" sx={{ textTransform: 'capitalize' }}>
                            {key.replace('_', ' ')}
                          </Typography>
                          <Typography variant="caption" fontWeight="bold" color="secondary">
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
