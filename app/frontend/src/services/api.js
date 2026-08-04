import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

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

  predictBatch: (pairs) => api.post('/predict_batch', { pairs }),

  getFeasibility: (p1, p2) => api.get(`/bio/feasibility?p1=${p1}&p2=${p2}`),

  optimize: (p1_id, p1_seq, p2_id, p2_seq, mode = 'disrupt') => 
    api.post(`/analysis/optimize?mode=${mode}`, {
      protein1_id: p1_id,
      protein1_seq: p1_seq,
      protein2_id: p2_id,
      protein2_seq: p2_seq
    }),

  getVulnerability: (p1, p2, delta) => 
    api.get(`/analysis/vulnerability?p1=${p1}&p2=${p2}&delta=${delta}`),

  localizeInteractionRegions: (p1_id, p2_id, p1_seq = null, p2_seq = null, base_prob = 0.5) =>
    api.post('/analysis/localize', {
      protein1_id: p1_id,
      protein2_id: p2_id,
      protein1_seq: p1_seq,
      protein2_seq: p2_seq,
      base_probability: base_prob
    }),

  // AI Assistant
  getChatGreeting: () => api.get('/chat/greeting'),
  sendChatMessage: (message) => api.post('/chat', { message }),

  // Telemetry/Logging
  logTelemetry: (location, message, data = {}) => {
    const TELEMETRY_URL = import.meta.env.VITE_TELEMETRY_URL || 'http://127.0.0.1:7656/ingest/a2c6930f-0198-499d-9920-7d735f885f13';
    return axios.post(TELEMETRY_URL, {
      sessionId: 'e579db',
      runId: 'pre-fix',
      hypothesisId: 'H1',
      location,
      message,
      data: {
        baseURL: API_BASE_URL,
        ...data
      },
      timestamp: Date.now()
    }, {
      headers: {
        'Content-Type': 'application/json',
        'X-Debug-Session-Id': 'e579db'
      }
    }).catch(() => {});
  }
};

export default api;
