import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const ppiService = {
  predict: (protein1_id, protein2_id, p1_seq = null, p2_seq = null) => 
    api.post('/predict', { 
      protein1_id, 
      protein2_id, 
      protein1_seq: p1_seq, 
      protein2_seq: p2_seq 
    }),

  getNetwork: (limit = 100) => api.get(`/network?limit=${limit}`),

  getDrugTargets: (proteins = null) => 
    api.get(`/drug_targets${proteins ? `?proteins=${proteins}` : ''}`),

  getCentrality: (topK = 10) => api.get(`/analysis/centrality?top_k=${topK}`),

  getNetworkStats: () => api.get('/analysis/stats'),

  getBioMetadata: (proteins) => api.get(`/bio/metadata?proteins=${proteins}`),

  mutate: (p1_id, p1_seq, p2_id, p2_seq, mutations) => 
    api.post('/analysis/mutate', {
      protein1_id: p1_id,
      protein1_seq: p1_seq,
      protein2_id: p2_id,
      protein2_seq: p2_seq,
      mutations
    }),

  getHotspots: (p1_id, p2_id, p1_seq = null, p2_seq = null) => 
    api.post('/analysis/hotspots', { 
      protein1_id: p1_id, 
      protein2_id: p2_id,
      protein1_seq: p1_seq,
      protein2_seq: p2_seq
    }),

  getResidueGraph: (protein_id, sequence = null) => 
    api.post('/analysis/residue_graph', { protein_id, sequence }),
};

export default api;
