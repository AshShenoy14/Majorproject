import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json',
  },
});

const GAT_API_BASE_URL = import.meta.env.VITE_GAT_API_BASE_URL || 'http://127.0.0.1:8001';

const gatApi = axios.create({
  baseURL: GAT_API_BASE_URL,
  timeout: 120000,
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

  predictGAT: (protein1_id, protein2_id, p1_seq = null, p2_seq = null) => 
    gatApi.post('/predict', { 
      protein1_id, 
      protein2_id, 
      protein1_seq: p1_seq, 
      protein2_seq: p2_seq 
    }),

  checkGATHealth: () => gatApi.get('/health'),

  explainGNN: (protein1_id, protein2_id) =>
    api.post('/analysis/explain-gnn', { protein1_id, protein2_id }),

  getNetwork: (limit = 100) => api.get(`/network?limit=${limit}`),

  getDrugTargets: (proteins = null) => 
    api.get(`/drug_targets${proteins ? `?proteins=${proteins}` : ''}`),

  // network: 'predicted' (known + ensemble-predicted interactions) or 'known' (training interactions only)
  getTherapeuticTargets: (limit = 50, wDegree = 0.40, wBetweenness = 0.35, wChembl = 0.25, network = 'predicted') =>
    api.get(`/analysis/therapeutic-targets?limit=${limit}&w_degree=${wDegree}&w_betweenness=${wBetweenness}&w_chembl=${wChembl}&network=${network}`),


  getCentrality: (topK = 10, network = 'predicted') => api.get(`/analysis/centrality?top_k=${topK}&network=${network}`),

  getNetworkStats: (network = 'predicted') => api.get(`/analysis/stats?network=${network}`),

  getFinalEvaluation: () => api.get('/evaluation/final'),
  getAllBenchmarks: () => api.get('/evaluation/benchmarks'),

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

  predictBatch: (pairs) => api.post('/predict_batch', { pairs }),

  getFeasibility: (p1, p2) => api.get(`/bio/feasibility?p1=${p1}&p2=${p2}`),

  optimize: (p1_id, p1_seq, p2_id, p2_seq, mode = 'disrupt') => 
    api.post(`/analysis/optimize?mode=${mode}`, {
      protein1_id: p1_id,
      protein1_seq: p1_seq,
      protein2_id: p2_id,
      protein2_seq: p2_seq
    }),

  // Long analyses (1-2 min on CPU) run as background jobs: start, then poll getJob(job_id) until status is done/error.
  startHotspotJob: (p1_id, p2_id) =>
    api.post('/analysis/jobs/hotspots', { protein1_id: p1_id, protein2_id: p2_id }),
  startOptimizeJob: (p1_id, p2_id, mode = 'disrupt') =>
    api.post(`/analysis/jobs/optimize?mode=${mode}`, { protein1_id: p1_id, protein2_id: p2_id }),
  getJob: (jobId) => api.get(`/analysis/jobs/${jobId}`),

  getVulnerability: (p1, p2, delta) =>
    api.get(`/analysis/vulnerability?p1=${p1}&p2=${p2}&delta=${delta}`),

  // AI Assistant
  getChatGreeting: () => api.get('/chat/greeting'),
  sendChatMessage: (message) => api.post('/chat', { message }),
};

export default api;
