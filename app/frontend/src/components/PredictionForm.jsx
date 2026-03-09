import { useState, useEffect } from 'react';
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
  Alert,
  useTheme,
  List,
  ListItem,
  ListItemText,
  ListItemButton,
  Chip
} from '@mui/material';
import { motion, AnimatePresence } from 'framer-motion';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import InteractionVisualizer from './InteractionVisualizer';
import ProteinViewer from './ProteinViewer';
import html2pdf from 'html2pdf.js';

const inputBoxStyles = (isDark) => ({
  p: 2.5,
  borderRadius: 3,
  background: isDark ? 'rgba(0, 0, 0, 0.25)' : 'rgba(0, 0, 0, 0.02)',
  border: `1px solid ${isDark ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.06)'}`,
  transition: 'all 0.35s cubic-bezier(0.4, 0, 0.2, 1)',
  '&:hover': {
    borderColor: isDark ? 'rgba(0, 229, 255, 0.2)' : 'rgba(0, 105, 92, 0.2)',
    transform: 'translateY(-2px)',
    boxShadow: isDark
      ? '0 8px 30px rgba(0, 229, 255, 0.08)'
      : '0 8px 30px rgba(0, 0, 0, 0.04)',
  },
  '&:focus-within': {
    borderColor: isDark ? 'rgba(0, 229, 255, 0.35)' : 'rgba(0, 105, 92, 0.35)',
    boxShadow: isDark
      ? '0 0 20px rgba(0, 229, 255, 0.1)'
      : '0 0 20px rgba(0, 105, 92, 0.06)',
  },
});

const PredictionForm = () => {
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';
  const [id1, setId1] = useState('');
  const [id2, setId2] = useState('');
  const [seq1, setSeq1] = useState('');
  const [seq2, setSeq2] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [batchResults, setBatchResults] = useState([]);

  useEffect(() => {
    const saved = localStorage.getItem('predictionHistory');
    if (saved) {
      setHistory(JSON.parse(saved));
    }
  }, []);

  const saveToHistory = (predictionData) => {
    const newHistory = [predictionData, ...history].slice(0, 10); // Keep last 10
    setHistory(newHistory);
    localStorage.setItem('predictionHistory', JSON.stringify(newHistory));
  };

  const loadFromHistory = (item) => {
    setId1(item.id1);
    setId2(item.id2);
    setResult(item.result);
    setBatchResults([]);
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);
    setBatchResults([]);

    const reader = new FileReader();
    reader.onload = async (event) => {
      try {
        const text = event.target.result;
        const lines = text.split('\n').map(l => l.trim()).filter(l => l);

        const pairs = [];
        for (let line of lines) {
          // Basic CSV parse: id1,id2
          const parts = line.split(',');
          if (parts.length >= 2) {
            pairs.push({
              protein1_id: parts[0].trim(),
              protein2_id: parts[1].trim(),
              protein1_seq: "",
              protein2_seq: ""
            });
          }
        }

        if (pairs.length === 0) {
          throw new Error("No valid pairs found in CSV.");
        }

        const response = await axios.post('http://localhost:8000/predict_batch', { pairs });
        setBatchResults(response.data);
        // Set the first one as active result
        if (response.data.length > 0) {
          setResult(response.data[0]);
          setId1(pairs[0].protein1_id);
          setId2(pairs[0].protein2_id);
          saveToHistory({ id1: pairs[0].protein1_id, id2: pairs[0].protein2_id, result: response.data[0] });
        }
      } catch (err) {
        setError("Failed to process batch file.");
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    reader.readAsText(file);
  };

  const handleExportPDF = () => {
    const element = document.getElementById('prediction-report');
    if (!element) return;

    const opt = {
      margin: 0.5,
      filename: `PPI_Report_${id1}_${id2}.pdf`,
      image: { type: 'jpeg', quality: 0.98 },
      html2canvas: { scale: 2, useCORS: true, backgroundColor: isDark ? '#0a192f' : '#f4f6f8' },
      jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' }
    };
    html2pdf().set(opt).from(element).save();
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    setBatchResults([]);

    try {
      // Replace with actual API URL
      const payload = {
        protein1_id: id1 || "Protein A",
        protein2_id: id2 || "Protein B",
        protein1_seq: seq1,
        protein2_seq: seq2,
      };
      const response = await axios.post('http://localhost:8000/predict', payload);
      setResult(response.data);
      saveToHistory({ id1: id1 || "Protein A", id2: id2 || "Protein B", result: response.data });
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
      transition={{ duration: 0.6, ease: [0.4, 0, 0.2, 1] }}
      sx={{
        p: { xs: 3, md: 4 },
        borderRadius: 4,
        overflow: 'hidden',
        background: isDark ? 'rgba(16, 33, 65, 0.5)' : 'rgba(255, 255, 255, 0.6)',
        backdropFilter: 'blur(24px) saturate(180%)',
        border: `1px solid ${isDark ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.06)'}`,
        boxShadow: isDark ? '0 24px 48px rgba(0,0,0,0.25)' : '0 24px 48px rgba(0,0,0,0.06)',
      }}
    >
      <Grid container spacing={4}>
        {/* Input Section */}
        <Grid item xs={12} md={6}>
          <Typography variant="h4" gutterBottom sx={{
            fontWeight: 800,
            background: isDark
              ? 'linear-gradient(90deg, #00e5ff, #7c4dff)'
              : 'linear-gradient(90deg, #00695c, #1565c0)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}>
            Analyze Interaction
          </Typography>
          <Typography variant="body1" color="text.secondary" paragraph sx={{ mb: 4 }}>
            Enter protein IDs (UniProt/Ensembl) or sequences to predict interaction probability.
          </Typography>

          <form onSubmit={handleSubmit}>
            <Stack spacing={3}>
              <Box sx={inputBoxStyles(isDark)}>
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

              <Box sx={inputBoxStyles(isDark)}>
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
                sx={{
                  mt: 2,
                  height: 54,
                  fontSize: '1.1rem',
                  fontWeight: 700,
                  borderRadius: '50px',
                  boxShadow: isDark
                    ? '0 0 25px rgba(0, 229, 255, 0.2)'
                    : '0 8px 25px rgba(0, 105, 92, 0.15)',
                  '&:hover': {
                    boxShadow: isDark
                      ? '0 0 40px rgba(0, 229, 255, 0.35)'
                      : '0 12px 35px rgba(0, 105, 92, 0.25)',
                    transform: 'translateY(-2px)',
                  },
                  transition: 'all 0.35s cubic-bezier(0.4, 0, 0.2, 1)',
                }}
              >
                {loading ? <CircularProgress size={24} color="inherit" /> : '⚡ Run Prediction'}
              </Button>

              <Button
                variant="outlined"
                component="label"
                color="secondary"
                disabled={loading}
                sx={{ mt: 1 }}
              >
                Upload Batch CSV
                <input
                  type="file"
                  hidden
                  accept=".csv,.txt"
                  onChange={handleFileUpload}
                />
              </Button>
            </Stack>
          </form>

          {history.length > 0 && (
            <Box sx={{ mt: 4, p: 2, background: isDark ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.02)', borderRadius: 2 }}>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>Recent Predictions</Typography>
              <List dense>
                {history.map((h, i) => (
                  <ListItem key={i} disablePadding>
                    <ListItemButton onClick={() => loadFromHistory(h)} sx={{ borderRadius: 1, mb: 0.5 }}>
                      <ListItemText
                        primary={`${h.id1} ↔ ${h.id2}`}
                        secondary={`Confidence: ${(h.result.confidence_score * 100).toFixed(1)}%`}
                      />
                      <Chip
                        size="small"
                        label={(h.result.interaction_probability * 100).toFixed(1) + '%'}
                        color={h.result.interaction_probability > 0.5 ? 'success' : 'error'}
                        variant="outlined"
                      />
                    </ListItemButton>
                  </ListItem>
                ))}
              </List>
            </Box>
          )}

          {error && (
            <Fade in>
              <Alert severity="error" variant="filled" sx={{ mt: 3 }}>{error}</Alert>
            </Fade>
          )}
        </Grid>

        {/* Results Section */}
        <Grid item xs={12} md={6} sx={{ borderLeft: { md: `1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}` }, pl: { md: 5 } }}>
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
                    <Typography variant="body2">Enter protein details or upload a batch CSV to view prediction results.</Typography>
                  </Box>
                </motion.div>
              )}

              {loading && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <Box sx={{
                    textAlign: 'center',
                    py: 8,
                    animation: 'pulseGlow 2s ease-in-out infinite',
                  }}>
                    <CircularProgress
                      size={48}
                      sx={{
                        mb: 3,
                        color: isDark ? '#00e5ff' : '#00695c',
                      }}
                    />
                    <Typography variant="h6" color="primary" sx={{ fontWeight: 600 }}>Analyzing Sequences...</Typography>
                    <Typography variant="caption">Calculating embeddings and graph attention</Typography>
                  </Box>
                </motion.div>
              )}

              {result && (
                <Stack
                  id="prediction-report"
                  spacing={3}
                  component={motion.div}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  sx={{ p: 2, bgcolor: isDark ? 'rgba(0,0,0,0.2)' : '#fff', borderRadius: 2 }}
                >
                  {batchResults.length > 1 && (
                    <Alert severity="success" sx={{ mb: 2 }}>
                      Batch processed successfully! Showing first pair. Check history or implement a viewer to see the other {batchResults.length - 1} results.
                    </Alert>
                  )}
                  <Button
                    variant="contained"
                    color="secondary"
                    onClick={handleExportPDF}
                    sx={{ mt: 2 }}
                  >
                    Export as PDF
                  </Button>
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
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', bgcolor: isDark ? 'rgba(0, 229, 255, 0.05)' : 'rgba(0, 105, 92, 0.05)', border: `1px solid ${isDark ? 'rgba(0, 229, 255, 0.1)' : 'rgba(0, 105, 92, 0.1)'}`, p: 1.5, borderRadius: 2 }}>
                          <Typography variant="body2">ESM-MLP</Typography>
                          <Typography variant="body2" fontWeight="bold" color="primary">
                            {(result.esm_probability * 100).toFixed(1)}%
                          </Typography>
                        </Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', bgcolor: isDark ? 'rgba(213, 0, 249, 0.05)' : 'rgba(21, 101, 192, 0.05)', border: `1px solid ${isDark ? 'rgba(213, 0, 249, 0.1)' : 'rgba(21, 101, 192, 0.1)'}`, p: 1.5, borderRadius: 2 }}>
                          <Typography variant="body2">Graph Attention (GAT)</Typography>
                          <Typography variant="body2" fontWeight="bold" color="secondary">
                            {(result.gat_probability * 100).toFixed(1)}%
                          </Typography>
                        </Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', bgcolor: isDark ? 'rgba(0, 255, 136, 0.05)' : 'rgba(46, 125, 50, 0.05)', border: `1px solid ${isDark ? 'rgba(0, 255, 136, 0.2)' : 'rgba(46, 125, 50, 0.2)'}`, p: 1.5, borderRadius: 2 }}>
                          <Typography variant="body2">Ensemble (Meta-learner)</Typography>
                          <Typography variant="body2" fontWeight="bold" sx={{ color: isDark ? '#00ff88' : '#2e7d32' }}>
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
                        <Box key={key} sx={{ display: 'flex', justifyContent: 'space-between', bgcolor: isDark ? 'rgba(255, 255, 255, 0.03)' : 'rgba(0, 0, 0, 0.03)', p: 1, px: 1.5, borderRadius: 1 }}>
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

                  {/* 3D Protein Viewers */}
                  <Box>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>3D Structure Analysis</Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={12} sm={6}>
                        <Typography variant="caption" color="primary" gutterBottom display="block" align="center">
                          {id1 || "Protein A"}
                        </Typography>
                        <ProteinViewer proteinId={result.protein1_uniprot_id || id1} />
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <Typography variant="caption" color="secondary" gutterBottom display="block" align="center">
                          {id2 || "Protein B"}
                        </Typography>
                        <ProteinViewer proteinId={result.protein2_uniprot_id || id2} />
                      </Grid>
                    </Grid>
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
