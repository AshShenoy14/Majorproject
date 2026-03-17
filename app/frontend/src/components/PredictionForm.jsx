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
  Chip,
  IconButton,
  Tooltip
} from '@mui/material';
import { motion, AnimatePresence } from 'framer-motion';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip as RechartsTooltip } from 'recharts';
import {
  Delete as DeleteIcon,
  DeleteSweep as DeleteSweepIcon,
  InfoOutlined as InfoIcon
} from '@mui/icons-material';
import InteractionVisualizer from './InteractionVisualizer';
import ProteinViewer from './ProteinViewer';
import MutationScanner from './MutationScanner';
import ResidueInteractionGraph from './ResidueInteractionGraph';
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
  const [bioInfo, setBioInfo] = useState(null);
  const [feasibility, setFeasibility] = useState(null);

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

  const removeFromHistory = (index, e) => {
    e.stopPropagation();
    const newHistory = history.filter((_, i) => i !== index);
    setHistory(newHistory);
    localStorage.setItem('predictionHistory', JSON.stringify(newHistory));
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem('predictionHistory');
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

    // Use specific options for professional PDF output
    const opt = {
      margin: [0.5, 0.5],
      filename: `TransGraph_PPI_Report_${id1 || 'Analysis'}.pdf`,
      image: { type: 'jpeg', quality: 1.0 },
      html2canvas: { 
        scale: 2, 
        useCORS: true, 
        backgroundColor: '#ffffff', // Force white for professional PDF
        logging: false,
        letterRendering: true
      },
      jsPDF: { unit: 'in', format: 'a4', orientation: 'portrait' }
    };

    // Temporarily add a "Printing" class to handle special alignment if needed
    element.classList.add('is-printing');
    
    html2pdf().set(opt).from(element).save().then(() => {
      element.classList.remove('is-printing');
    });
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
      
      // Fetch Bio Metadata and Feasibility
      fetchBioData(id1, id2);
    } catch (err) {
      setError('Failed to fetch prediction. Ensure backend is running.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const fetchBioData = async (p1, p2) => {
    try {
      if (p1 && p2) {
        const [bioRes, feasRes] = await Promise.all([
          axios.get(`http://localhost:8000/bio/metadata?proteins=${p1},${p2}`),
          axios.get(`http://localhost:8000/bio/feasibility?p1=${p1}&p2=${p2}`)
        ]);
        setBioInfo(bioRes.data);
        setFeasibility(feasRes.data);
      }
    } catch (err) {
      console.error("Failed to fetch biological context", err);
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
      <Stack spacing={4}>
        {/* Input Section */}
        <Box>
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
                <Stack direction="row" alignItems="center" spacing={1}>
                  <Typography variant="caption" fontWeight="bold" color="secondary" sx={{ letterSpacing: '0.1em' }}>PROTEIN 1</Typography>
                  <Tooltip title="Ensembl Protein ID (e.g., ENSP...) or UniProt ID. These identifiers allow the system to fetch known sequences and structures." arrow>
                    <InfoIcon sx={{ fontSize: '1rem', color: 'text.secondary', cursor: 'pointer', opacity: 0.7 }} />
                  </Tooltip>
                </Stack>
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
                <Stack direction="row" alignItems="center" spacing={1} sx={{ mt: 1.5 }}>
                  <Typography variant="caption" fontWeight="bold" color="secondary" sx={{ letterSpacing: '0.1em' }}>SEQUENCE (OPTIONAL)</Typography>
                  <Tooltip title="Raw amino acid sequence in FASTA-like format. If provided, this overrides the ID lookup for embedding generation." arrow>
                    <InfoIcon sx={{ fontSize: '1rem', color: 'text.secondary', cursor: 'pointer', opacity: 0.7 }} />
                  </Tooltip>
                </Stack>
                <TextField
                  label="Sequence"
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
                <Stack direction="row" alignItems="center" spacing={1}>
                  <Typography variant="caption" fontWeight="bold" color="secondary" sx={{ letterSpacing: '0.1em' }}>PROTEIN 2</Typography>
                  <Tooltip title="Ensembl Protein ID (e.g., ENSP...) or UniProt ID for the second interactive partner." arrow>
                    <InfoIcon sx={{ fontSize: '1rem', color: 'text.secondary', cursor: 'pointer', opacity: 0.7 }} />
                  </Tooltip>
                </Stack>
                <TextField
                  label="ID (e.g., Q98765)"
                  fullWidth
                  value={id2}
                  onChange={(e) => setId2(e.target.value)}
                  margin="dense"
                  size="small"
                  sx={{ mt: 1 }}
                />
                <Stack direction="row" alignItems="center" spacing={1} sx={{ mt: 1.5 }}>
                  <Typography variant="caption" fontWeight="bold" color="secondary" sx={{ letterSpacing: '0.1em' }}>SEQUENCE (OPTIONAL)</Typography>
                  <Tooltip title="Amino acid sequence for the second protein. Use this if the protein ID is not in our database." arrow>
                    <InfoIcon sx={{ fontSize: '1rem', color: 'text.secondary', cursor: 'pointer', opacity: 0.7 }} />
                  </Tooltip>
                </Stack>
                <TextField
                  label="Sequence"
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
                {loading ? <CircularProgress size={24} color="inherit" /> : ' Run Prediction'}
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
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                <Typography variant="subtitle2" color="text.secondary">Recent Predictions</Typography>
                <Button
                  size="small"
                  color="error"
                  startIcon={<DeleteSweepIcon />}
                  onClick={clearHistory}
                  sx={{ fontSize: '0.7rem', opacity: 0.7, '&:hover': { opacity: 1 } }}
                >
                  Clear All
                </Button>
              </Box>
              <List dense>
                {history.map((h, i) => (
                  <ListItem
                    key={i}
                    disablePadding
                    secondaryAction={
                      <IconButton edge="end" aria-label="delete" size="small" onClick={(e) => removeFromHistory(i, e)} sx={{ color: 'error.main', opacity: 0.5, '&:hover': { opacity: 1 } }}>
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    }
                  >
                    <ListItemButton onClick={() => loadFromHistory(h)} sx={{ borderRadius: 1, mb: 0.5, pr: 5 }}>
                      <ListItemText
                        primary={`${h.id1} ↔ ${h.id2}`}
                        secondary={`Confidence: ${(h.result.confidence_score * 100).toFixed(1)}%`}
                        primaryTypographyProps={{ sx: { fontSize: '0.85rem', fontWeight: 600 } }}
                        secondaryTypographyProps={{ sx: { fontSize: '0.75rem' } }}
                      />
                      <Chip
                        size="small"
                        label={(h.result.interaction_probability * 100).toFixed(1) + '%'}
                        color={h.result.interaction_probability > 0.5 ? 'success' : 'error'}
                        variant="outlined"
                        sx={{ ml: 1, height: 20, fontSize: '0.7rem' }}
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
        </Box>

        <Divider sx={{ opacity: 0.1 }} />

        {/* Results Section */}
        <Box>
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
                  sx={{ 
                    p: 4, 
                    bgcolor: isDark ? 'rgba(0,0,0,0.3)' : '#fff', 
                    borderRadius: 3,
                    position: 'relative',
                    // Print-specific styles
                    '@media print': {
                      bgcolor: '#fff !important',
                      color: '#000 !important',
                      p: '2cm !important',
                      boxShadow: 'none !important',
                      border: 'none !important'
                    }
                  }}
                >
                  <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: -2, '@media print': { display: 'none' } }}>
                    <Button
                      variant="outlined"
                      color="secondary"
                      size="small"
                      startIcon={<InfoIcon />}
                      onClick={handleExportPDF}
                      sx={{ 
                        borderRadius: '20px',
                        fontSize: '0.7rem',
                        px: 2,
                        textTransform: 'none',
                        borderWidth: '1px !important'
                      }}
                    >
                      Export PDF
                    </Button>
                  </Box>

                  {/* PDF Only Header */}
                  <Box className="pdf-header" sx={{ 
                    display: 'none', 
                    flexDirection: 'column', 
                    alignItems: 'center', 
                    mb: 4,
                    borderBottom: '2px solid #00e5ff',
                    pb: 2,
                    '.is-printing &': { display: 'flex' }
                  }}>
                    <Typography variant="h4" sx={{ color: '#00695c', fontWeight: 800 }}>TransGraph-PPI</Typography>
                    <Typography variant="subtitle2" color="text.secondary">Protein-Protein Interaction Analysis Report</Typography>
                    <Typography variant="caption" sx={{ mt: 1 }}>Generated on: {new Date().toLocaleDateString()}</Typography>
                  </Box>

                  {batchResults.length > 1 && (
                    <Alert severity="success" sx={{ mb: 2 }}>
                      Batch processed successfully! Check history or implement a viewer to see the other {batchResults.length - 1} results.
                    </Alert>
                  )}
                  
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
                          <RechartsTooltip />
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
                          <Typography variant="body2">Hybrid Ensemble (v2.0)</Typography>
                          <Typography variant="body2" fontWeight="bold" sx={{ color: isDark ? '#00ff88' : '#2e7d32' }}>
                            {(result.final_prob * 100).toFixed(1)}%
                          </Typography>
                        </Box>
                      </Stack>
                    </Box>
                  )}

                  {/* Explainability (SHAP) Restored */}
                  {result.shap_explanations && (
                    <Box sx={{ p: 2, bgcolor: isDark ? 'rgba(255, 255, 255, 0.02)' : 'rgba(0, 0, 0, 0.02)', borderRadius: 2 }}>
                      <Typography variant="subtitle2" color="text.secondary" gutterBottom>Feature Attribution (SHAP)</Typography>
                      <Stack spacing={1}>
                        {[
                          { label: 'Seq Evidence', val: result.shap_explanations[0] },
                          { label: 'Graph Evidence', val: result.shap_explanations[1] },
                          { label: 'Seq Confidence', val: result.shap_explanations[2] },
                          { label: 'Graph Confidence', val: result.shap_explanations[3] }
                        ].map((item, idx) => (
                          <Box key={idx} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Typography variant="caption" sx={{ minWidth: 100 }}>{item.label}</Typography>
                            <Box sx={{ flex: 1, height: 4, bgcolor: 'rgba(0,0,0,0.1)', borderRadius: 2, overflow: 'hidden' }}>
                              <Box sx={{ 
                                height: '100%', 
                                width: `${Math.min(100, Math.abs(item.val) * 500)}%`, 
                                bgcolor: item.val > 0 ? 'success.main' : 'error.main',
                                ml: item.val < 0 ? 'auto' : 0
                              }} />
                            </Box>
                            <Typography variant="caption" sx={{ minWidth: 30, textAlign: 'right', fontWeight: 'bold' }}>
                              {item.val > 0 ? '+' : ''}{(item.val * 100).toFixed(1)}%
                            </Typography>
                          </Box>
                        ))}
                      </Stack>
                    </Box>
                  )}

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
                  
                  {/* Residue Interaction Graphs for Topology Visualization */}
                  <Box>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>Sequence Topology Networks</Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={6}>
                        <ResidueInteractionGraph 
                          proteinId={id1 || "Protein A"} 
                          sequence={seq1 || result.protein1_seq} 
                          isDark={isDark} 
                        />
                      </Grid>
                      <Grid item xs={12} md={6}>
                        <ResidueInteractionGraph 
                          proteinId={id2 || "Protein B"} 
                          sequence={seq2 || result.protein2_seq} 
                          isDark={isDark} 
                        />
                      </Grid>
                    </Grid>
                  </Box>
                  
                  {/* Novelty 1: Localization Feasibility */}
                  {feasibility && (
                    <Alert 
                      severity={feasibility.compatible ? "success" : "warning"} 
                      variant="outlined"
                      sx={{ borderRadius: 2 }}
                    >
                      <Typography variant="subtitle2" fontWeight="bold">Biological Feasibility</Typography>
                      <Typography variant="caption">{feasibility.reason}</Typography>
                      {feasibility.intersection?.length > 0 && (
                        <Box sx={{ mt: 1 }}>
                          <Typography variant="caption" sx={{ opacity: 0.8 }}>Shared Locations: </Typography>
                          {feasibility.intersection.map(l => <Chip key={l} label={l} size="small" sx={{ height: 18, fontSize: '0.6rem', ml: 0.5 }} />)}
                        </Box>
                      )}
                    </Alert>
                  )}

                  {/* Novelty 2: Biological Context (Pathways) */}
                  {bioInfo && bioInfo.length > 0 && (
                    <Box sx={{ p: 2, bgcolor: isDark ? 'rgba(0, 229, 255, 0.05)' : 'rgba(0, 105, 92, 0.05)', borderRadius: 2 }}>
                      <Typography variant="subtitle2" color="secondary" gutterBottom fontWeight="bold">Biological Context</Typography>
                      {bioInfo.map(info => (
                        <Box key={info.protein_id} sx={{ mb: 1.5 }}>
                          <Typography variant="caption" fontWeight="bold">{info.protein_id}</Typography>
                          <Typography variant="caption" display="block" color="text.secondary">
                            Pathways: {info.pathways || "Unknown"}
                          </Typography>
                        </Box>
                      ))}
                    </Box>
                  )}

                  {/* Novelty 3: Mutation Impact Scanner */}
                  <MutationScanner 
                    isDark={isDark}
                    protein1={{ id: id1, seq: seq1 || result.protein1_seq }}
                    protein2={{ id: id2, seq: seq2 || result.protein2_seq }}
                  />

                  {/* Accessible Visualizer */}
                  <InteractionVisualizer result={result} id1={id1} id2={id2} />
                </Stack>
              )}
            </AnimatePresence>
          </Box>
        </Box>
      </Stack>
    </Paper>
  );
};

export default PredictionForm;
