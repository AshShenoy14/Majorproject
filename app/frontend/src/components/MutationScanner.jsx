import { useState } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  Grid,
  Card,
  CardContent,
  Chip,
  LinearProgress,
  Stack,
  Alert,
  IconButton,
  CircularProgress,
  Paper
} from '@mui/material';
import {
  Science as ScienceIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  KeyboardDoubleArrowUp as EnhancingIcon,
  KeyboardDoubleArrowDown as DisruptingIcon,
  Minimize as NeutralIcon
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

const MutationScanner = ({ protein1, protein2, isDark }) => {
  const [mutations, setMutations] = useState([
    { protein: 1, pos: '', orig: '', mut: '' }
  ]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [vulnerability, setVulnerability] = useState(null);
  const [designing, setDesigning] = useState(false);

  const addMutation = () => {
    setMutations([...mutations, { protein: 1, pos: '', orig: '', mut: '' }]);
  };

  const updateMutation = (index, field, value) => {
    const newMutations = [...mutations];
    newMutations[index][field] = value;
    setMutations(newMutations);
  };

  const removeMutation = (index) => {
    setMutations(mutations.filter((_, i) => i !== index));
  };

  const handleOptimize = async (mode = 'disrupt') => {
    if (!protein1.seq || !protein2.seq) {
      setError("Sequence data is required for optimization.");
      return;
    }
    setDesigning(true);
    setError(null);
    try {
      const resp = await axios.post('http://localhost:8000/analysis/optimize', {
        protein1_id: protein1.id,
        protein1_seq: protein1.seq,
        protein2_id: protein2.id,
        protein2_seq: protein2.seq
      }, { params: { mode } });
      
      const suggestions = resp.data.suggestions;
      setMutations(suggestions.map(s => ({
        protein: 1, pos: s.pos, orig: s.orig, mut: s.mut
      })));
      setResults(null); 
    } catch (err) {
      setError("AI Optimizer unavailable.");
    } finally {
      setDesigning(false);
    }
  };

  const handleScan = async () => {
    if (!protein1.seq || !protein2.seq) {
      setError("Sequence data is required for mutation scanning.");
      return;
    }

    setLoading(true);
    setError(null);
    setVulnerability(null);
    try {
      const payload = {
        protein1_id: protein1.id,
        protein1_seq: protein1.seq,
        protein2_id: protein2.id,
        protein2_seq: protein2.seq,
        mutations: mutations.map(m => ({
          ...m,
          protein: parseInt(m.protein),
          pos: parseInt(m.pos)
        })).filter(m => m.pos && m.orig && m.mut)
      };

      const resp = await axios.post('http://localhost:8000/analysis/mutate', payload);
      setResults(resp.data.mutation_results);

      // Check Pathway Vulnerability for the most disruptive mutation
      const maxDisrupt = resp.data.mutation_results.sort((a, b) => a.impact_delta - b.impact_delta)[0];
      if (maxDisrupt && maxDisrupt.impact_delta < -0.1) {
          const vResp = await axios.get('http://localhost:8000/analysis/vulnerability', {
              params: { p1: protein1.id, p2: protein2.id, delta: maxDisrupt.impact_delta }
          });
          setVulnerability(vResp.data);
      }
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to scan mutations.");
    } finally {
      setLoading(false);
    }
  };

  const getImpactColor = (delta) => {
    if (delta < -0.05) return '#f44336';
    if (delta > 0.05) return '#4caf50';
    return '#757575';
  };

  const getImpactIcon = (delta) => {
    if (delta < -0.05) return <DisruptingIcon sx={{ color: '#f44336' }} />;
    if (delta > 0.05) return <EnhancingIcon sx={{ color: '#4caf50' }} />;
    return <NeutralIcon sx={{ color: '#757575' }} />;
  };

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h6" gutterBottom display="flex" alignItems="center" sx={{ fontWeight: 700 }}>
        <ScienceIcon sx={{ mr: 1, color: isDark ? '#00e5ff' : '#00695c' }} />
        Genetic Mutation Impact Scanner
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Predict how individual residues affects the interaction probability.
      </Typography>

      <Card sx={{ 
        bgcolor: isDark ? 'rgba(0,0,0,0.2)' : 'rgba(255,255,255,0.4)', 
        borderRadius: 2,
        border: `1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}`
      }}>
        <CardContent>
          <Stack spacing={2}>
            {mutations.map((m, i) => (
              <Grid container spacing={2} key={i} alignItems="center">
                <Grid item xs={12} sm={3}>
                  <TextField
                    select
                    fullWidth
                    label="Target"
                    size="small"
                    value={m.protein}
                    onChange={(e) => updateMutation(i, 'protein', e.target.value)}
                    SelectProps={{ native: true }}
                  >
                    <option value={1}>{protein1.id || 'Protein 1'}</option>
                    <option value={2}>{protein2.id || 'Protein 2'}</option>
                  </TextField>
                </Grid>
                <Grid item xs={4} sm={2}>
                  <TextField
                    fullWidth
                    label="Pos"
                    size="small"
                    type="number"
                    value={m.pos}
                    onChange={(e) => updateMutation(i, 'pos', e.target.value)}
                  />
                </Grid>
                <Grid item xs={3} sm={2}>
                  <TextField
                    fullWidth
                    label="From"
                    size="small"
                    inputProps={{ maxLength: 1 }}
                    value={m.orig}
                    onChange={(e) => updateMutation(i, 'orig', e.target.value.toUpperCase())}
                  />
                </Grid>
                <Grid item xs={3} sm={2}>
                  <TextField
                    fullWidth
                    label="To"
                    size="small"
                    inputProps={{ maxLength: 1 }}
                    value={m.mut}
                    onChange={(e) => updateMutation(i, 'mut', e.target.value.toUpperCase())}
                  />
                </Grid>
                <Grid item xs={2} sm={1}>
                  <IconButton color="error" size="small" onClick={() => removeMutation(i)} disabled={mutations.length <= 1}>
                    <DeleteIcon />
                  </IconButton>
                </Grid>
              </Grid>
            ))}
            
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
              <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                <Button 
                   size="small" 
                   startIcon={<ScienceIcon />} 
                   onClick={() => handleOptimize('disrupt')}
                   disabled={designing}
                   color="secondary"
                >
                  {designing ? <CircularProgress size={16} /> : 'AI Suggest Inhibition'}
                </Button>
                <Button size="small" startIcon={<AddIcon />} onClick={addMutation}>Add Row</Button>
              </Box>
              <Button 
                variant="contained" 
                size="medium" 
                onClick={handleScan}
                disabled={loading}
                sx={{ borderRadius: '20px', px: 4, fontWeight: 'bold' }}
              >
                {loading ? <CircularProgress size={20} /> : 'Biological Scan'}
              </Button>
            </Box>
          </Stack>
        </CardContent>
      </Card>

      {vulnerability && (
          <Alert 
            severity={vulnerability.risk_level === 'High' ? 'error' : 'warning'} 
            sx={{ mt: 2, borderRadius: 2 }}
            variant="filled"
          >
            <Typography variant="subtitle2" fontWeight="bold">Pathway Vulnerability Alert: {vulnerability.risk_level} Risk</Typography>
            <Typography variant="body2">{vulnerability.description}</Typography>
            {vulnerability.affected_pathways.length > 0 && (
                <Box sx={{ mt: 1 }}>
                    <Typography variant="caption" sx={{ opacity: 0.9 }}>Potentially affected: {vulnerability.affected_pathways.slice(0, 3).join(", ")}...</Typography>
                </Box>
            )}
          </Alert>
      )}

      {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}

      <AnimatePresence>
        {results && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}>
            <Box sx={{ mt: 3 }}>
              <Typography variant="subtitle1" fontWeight="bold" gutterBottom>Scan Results</Typography>
              <Stack spacing={1.5}>
                {results.map((r, i) => (
                  <Paper 
                    key={i} 
                    sx={{ 
                      p: 2, 
                      borderRadius: 2, 
                      bgcolor: isDark ? 'rgba(255,255,255,0.03)' : '#fff',
                      borderLeft: `4px solid ${getImpactColor(r.impact_delta)}`
                    }}
                  >
                    <Grid container alignItems="center">
                      <Grid item xs={12} sm={4}>
                        <Typography variant="body2" fontWeight="bold">
                          {r.protein === 1 ? protein1.id : protein2.id}: {r.orig}{r.pos}{r.mut}
                        </Typography>
                      </Grid>
                      <Grid item xs={8} sm={5}>
                        <Box sx={{ width: '100%', mr: 1 }}>
                          <Typography variant="caption" color="text.secondary">
                            Impact: {r.impact_delta > 0 ? '+' : ''}{(r.impact_delta * 100).toFixed(2)}% (Score: {(r.mutated_score * 100).toFixed(1)}%)
                          </Typography>
                          <LinearProgress 
                            variant="determinate" 
                            value={Math.min(100, Math.max(0, r.mutated_score * 100))} 
                            sx={{ height: 6, borderRadius: 3, bgcolor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)', '& .MuiLinearProgress-bar': { bgcolor: getImpactColor(r.impact_delta) }}}
                          />
                        </Box>
                      </Grid>
                      <Grid item xs={4} sm={3} sx={{ textAlign: 'right' }}>
                        <Chip 
                          size="small" 
                          icon={getImpactIcon(r.impact_delta)}
                          label={r.interpretation} 
                          sx={{ fontWeight: 'bold' }}
                        />
                      </Grid>
                    </Grid>
                  </Paper>
                ))}
              </Stack>
            </Box>
          </motion.div>
        )}
      </AnimatePresence>
    </Box>
  );
};

export default MutationScanner;
