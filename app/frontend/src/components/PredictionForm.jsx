import { useState } from 'react';
import axios from 'axios';
import {
  Box,
  Button,
  TextField,
  Typography,
  Paper,
  CircularProgress,
  Card,
  CardContent
} from '@mui/material';
import { motion } from 'framer-motion';

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

  return (
    <Box component={motion.div} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
      <Typography variant="h4" gutterBottom sx={{ fontWeight: 'bold', color: '#1976d2' }}>
        TransGraph-PPI Predictor
      </Typography>

      <Paper elevation={3} sx={{ p: 4, mb: 4, borderRadius: 2 }}>
        <form onSubmit={handleSubmit}>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <TextField
              label="Protein ID 1 (e.g., ENSP...)"
              fullWidth
              value={id1}
              onChange={(e) => setId1(e.target.value)}
              margin="normal"
              variant="outlined"
            />
            <TextField
              label="Protein ID 2 (e.g., ENSP...)"
              fullWidth
              value={id2}
              onChange={(e) => setId2(e.target.value)}
              margin="normal"
              variant="outlined"
            />
          </Box>
          <Typography variant="caption" color="text.secondary">
            Enter valid IDs (UniProt/Ensembl) to use real graph data. Leave sequences empty to auto-fetch.
          </Typography>

          <TextField
            label="Protein Sequence 1 (Optional)"
            multiline
            rows={2}
            fullWidth
            value={seq1}
            onChange={(e) => setSeq1(e.target.value)}
            margin="normal"
            variant="outlined"
          />
          <TextField
            label="Protein Sequence 2 (Optional)"
            multiline
            rows={2}
            fullWidth
            value={seq2}
            onChange={(e) => setSeq2(e.target.value)}
            margin="normal"
            variant="outlined"
          />

          <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
            <Button
              type="submit"
              variant="contained"
              size="large"
              disabled={loading}
              sx={{ px: 5, py: 1.5, fontSize: '1.1rem' }}
            >
              {loading ? <CircularProgress size={24} /> : 'Predict Interaction'}
            </Button>
          </Box>
        </form>
      </Paper>

      {error && (
        <Typography color="error" align="center" sx={{ mb: 2 }}>
          {error}
        </Typography>
      )}

      {result && (
        <Card component={motion.div} initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }}>
          <CardContent>
            <Typography variant="h5" gutterBottom>
              Prediction Result
            </Typography>

            <Box sx={{ display: 'flex', justifyContent: 'space-around', alignItems: 'center', my: 2 }}>
              <Box textAlign="center">
                <Typography variant="subtitle1" color="text.secondary">Probability</Typography>
                <Typography variant="h3" color="primary">
                  {(result.interaction_probability * 100).toFixed(1)}%
                </Typography>
              </Box>

              <Box textAlign="center">
                <Typography variant="subtitle1" color="text.secondary">Confidence</Typography>
                <Typography variant="h4" color={result.confidence_score > 0.8 ? "success.main" : "warning.main"}>
                  {(result.confidence_score * 100).toFixed(1)}%
                </Typography>
              </Box>
            </Box>

            <Typography variant="subtitle1" sx={{ mt: 2 }}>Explainability (SHAP):</Typography>
            <Box sx={{ bgcolor: '#f5f5f5', p: 2, borderRadius: 1 }}>
              {Object.entries(result.explanation).map(([key, value]) => (
                <Box key={key} sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body1" sx={{ fontWeight: 'medium' }}>
                    {key.replace('_', ' ')}:
                  </Typography>
                  <Typography variant="body1" color="primary">
                    {typeof value === 'number' ? value.toFixed(4) : value}
                  </Typography>
                </Box>
              ))}
            </Box>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default PredictionForm;
