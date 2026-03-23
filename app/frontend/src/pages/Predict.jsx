import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Search,
  Activity,
  BarChart3,
  Info,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Loader2,
  ChevronRight,
  ShieldCheck
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie
} from 'recharts';
import { ppiService } from '../services/api';

const TooltipIcon = ({ text }) => (
  <div className="group relative inline-block ml-1 align-middle">
    <Info size={14} className="text-slate-400 cursor-help" />
    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 p-2 bg-slate-800 text-white text-[10px] rounded shadow-xl opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
      {text}
    </div>
  </div>
);

const Predict = () => {
  const [protein1, setProtein1] = useState('ENSP00000327694');
  const [protein2, setProtein2] = useState('ENSP00000373627');
  const [p1Seq, setP1Seq] = useState('');
  const [p2Seq, setP2Seq] = useState('');
  const [showSeq, setShowSeq] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Derived data for charts
  const predictions = result ? [
    { name: 'ESM-MLP', value: (result.esm_probability * 100).toFixed(1), color: '#6366F1' },
    { name: 'GAT', value: (result.gat_probability * 100).toFixed(1), color: '#22D3EE' },
    { name: 'Ensemble', value: (result.interaction_probability * 100).toFixed(1), color: '#22C55E' },
  ] : [];

  const shapValues = result?.explanation?.SHAP_Values;
  const shapData = shapValues ? [
    { name: 'Sequence Signal', value: shapValues[0] },
    { name: 'Graph Signal', value: shapValues[1] },
    { name: 'Seq. Confidence', value: shapValues[2] },
    { name: 'Graph Confidence', value: shapValues[3] },
    { name: 'Bio Localization', value: shapValues[4] || 0 },
  ] : [];

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const response = await ppiService.predict(protein1, protein2, p1Seq || null, p2Seq || null);
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || "Prediction failed. Please check IDs.");
    } finally {
      setLoading(false);
    }
  };

  // The getModelData function is no longer needed as predictions are set directly in handlePredict
  // and consumed from the 'predictions' state.

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* Input Section */}
      <div className="glass-card p-8">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 bg-scientific-primary/10 rounded-lg text-scientific-primary">
            <Search size={24} />
          </div>
          <h2 className="text-2xl font-bold text-slate-800">New PPI Prediction</h2>
        </div>

        <form onSubmit={handlePredict} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label className="text-xs font-bold text-slate-500 uppercase tracking-wider">Protein A ID <TooltipIcon text="Ensembl Protein ID (e.g. ENSP...)" /></label>
              <input
                type="text"
                value={protein1}
                onChange={(e) => setProtein1(e.target.value)}
                placeholder="Enter ENSP ID..."
                className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-scientific-primary outline-none transition-all"
              />
            </div>
            <div className="space-y-2">
              <label className="text-xs font-bold text-slate-500 uppercase tracking-wider">Protein B ID <TooltipIcon text="The partner protein to test interaction against." /></label>
              <input
                type="text"
                value={protein2}
                onChange={(e) => setProtein2(e.target.value)}
                placeholder="Enter ENSP ID..."
                className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-scientific-primary outline-none transition-all"
              />
            </div>
          </div>

          <div className="flex items-center gap-2 mb-2">
            <button
              type="button"
              onClick={() => setShowSeq(!showSeq)}
              className="text-xs font-bold text-scientific-secondary flex items-center gap-1 hover:underline"
            >
              <Activity size={12} /> {showSeq ? 'Hide Sequence Inputs' : 'Provide Manual Sequences (Optional)'}
            </button>
          </div>

          <AnimatePresence>
            {showSeq && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="grid grid-cols-1 md:grid-cols-2 gap-6 overflow-hidden"
              >
                <div className="space-y-2">
                  <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Protein A Sequence</label>
                  <textarea
                    value={p1Seq}
                    onChange={(e) => setP1Seq(e.target.value)}
                    placeholder="PAAA..."
                    className="w-full h-24 px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-scientific-secondary outline-none transition-all font-mono text-xs"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Protein B Sequence</label>
                  <textarea
                    value={p2Seq}
                    onChange={(e) => setP2Seq(e.target.value)}
                    placeholder="PAAA..."
                    className="w-full h-24 px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-scientific-secondary outline-none transition-all font-mono text-xs"
                  />
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <button
            type="submit"
            disabled={loading}
            className="w-full btn-primary py-3.5 flex items-center justify-center gap-2 group"
          >
            {loading ? <Loader2 className="animate-spin" size={20} /> : <Activity size={20} />}
            Predict Interaction
            <ChevronRight size={18} className="group-hover:translate-x-1 transition-transform" />
          </button>
        </form>
        {error && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-4 p-4 bg-red-50 text-red-600 rounded-xl flex items-center gap-2 text-sm border border-red-100">
            <AlertCircle size={18} /> {error}
          </motion.div>
        )}
      </div>

      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-8 pb-12"
          >
            {/* Summary Result Card */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              <div className="lg:col-span-1 glass-card p-8 flex flex-col items-center text-center justify-center relative overflow-hidden">
                <div className="absolute top-0 left-0 w-full h-1 bg-scientific-gradient" />
                <h3 className="text-sm font-bold text-slate-500 uppercase tracking-widest mb-6">Interaction Result</h3>

                <div className="relative w-48 h-48 mb-6">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={[
                          { value: result.interaction_probability * 100 },
                          { value: 100 - (result.interaction_probability * 100) }
                        ]}
                        innerRadius={60}
                        outerRadius={80}
                        startAngle={180}
                        endAngle={-180}
                        paddingAngle={5}
                        dataKey="value"
                      >
                        <Cell fill={result.interaction_probability > 0.5 ? "#22C55E" : "#F59E0B"} />
                        <Cell fill="#F1F5F9" />
                      </Pie>
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <span className="text-4xl font-extrabold text-slate-800">{(Number(result.interaction_probability || 0) * 100).toFixed(1)}%</span>
                    <span className="text-[10px] text-slate-400 font-bold uppercase tracking-tighter">Probability</span>
                  </div>
                </div>

                <div className={`flex items-center gap-2 px-6 py-2 rounded-full mb-4 ${
                  result.interaction_probability > 0.5 
                    ? (Number(result.confidence_score) > 0.3 ? 'bg-green-50 text-green-600 border border-green-100' : 'bg-amber-50 text-amber-600 border border-amber-100')
                    : 'bg-orange-50 text-orange-600 border border-orange-100'
                }`}>
                  {result.interaction_probability > 0.5 
                    ? (Number(result.confidence_score) > 0.3 ? <CheckCircle2 size={18} /> : <AlertCircle size={18} />)
                    : <XCircle size={18} />
                  }
                  <span className="font-bold uppercase tracking-wider">
                    {result.interaction_probability > 0.5 
                      ? (Number(result.confidence_score) > 0.3 ? 'Strong Interaction' : 'Interaction (Unverified)')
                      : 'No Interaction'
                    }
                  </span>
                </div>

                <div className={`flex items-center gap-2 px-6 py-2 rounded-full mb-4 ${result.explanation?.Biological_Match ? 'bg-indigo-50 text-indigo-600 border border-indigo-100' : 'bg-slate-50 text-slate-400 border border-slate-100'}`}>
                  <ShieldCheck size={18} />
                  <span className="font-bold uppercase tracking-wider text-[10px]">
                    {result.explanation?.Biological_Match ? 'Bio-Match: Compatible' : 'Bio-Match: Unverified'}
                  </span>
                </div>

                <div className={`p-4 rounded-xl w-full mb-2 ${Number(result.confidence_score) > 0.6 ? 'bg-green-50/50' : Number(result.confidence_score) > 0.3 ? 'bg-amber-50/50' : 'bg-rose-50/50'}`}>
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-bold text-slate-500 uppercase">Confidence Score</span>
                    <span className={`font-extrabold ${(Number(result.confidence_score || 0) * 100) > 60 ? 'text-green-600' : (Number(result.confidence_score || 0) * 100) > 30 ? 'text-amber-600' : 'text-rose-600'}`}>
                      {(Number(result.confidence_score || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                  {Number(result.confidence_score) < 0.3 && (
                    <p className="text-[10px] text-rose-500 font-medium mt-1 leading-tight">
                      ⚠️ Caution: Low consensus between sequence and graph signals. High risk of false positive.
                    </p>
                  )}
                </div>
                <div className="text-[10px] text-slate-400 flex items-center justify-center gap-1">
                  <Info size={10} /> Statistical reliability metric
                </div>
              </div>

              {/* Model Breakdown */}
              <div className="lg:col-span-2 glass-card p-8">
                <div className="flex items-center gap-3 mb-8">
                  <BarChart3 className="text-scientific-secondary" />
                  <h4 className="text-lg font-bold text-slate-800">Model Component Predictions</h4>
                </div>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={predictions} layout="vertical" barSize={32}>
                      <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#E2E8F0" />
                      <XAxis type="number" domain={[0, 100]} hide />
                      <YAxis
                        dataKey="name"
                        type="category"
                        axisLine={false}
                        tickLine={false}
                        width={100}
                        tick={{ fontSize: 12, fontWeight: 700, fill: '#64748B' }}
                      />
                      <Tooltip
                        cursor={{ fill: '#F8FAFC' }}
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            return (
                              <div className="bg-slate-800 text-white p-3 rounded-xl shadow-xl border border-slate-700">
                                <p className="text-xs font-bold mb-1 uppercase tracking-wider">{payload[0].payload.name}</p>
                                <p className="text-xl font-extrabold">{payload[0].value}%</p>
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Bar dataKey="value" radius={[0, 8, 8, 0]}>
                        {predictions.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-6 flex justify-between gap-4">
                  <div className="flex-1 p-3 bg-slate-50 rounded-xl border border-slate-100">
                    <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">ESM-2 Latent Space</p>
                    <p className="text-sm font-medium text-slate-600">Sequence embeddings capture semantic bio-logic.</p>
                  </div>
                  <div className="flex-1 p-3 bg-slate-50 rounded-xl border border-slate-100">
                    <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">GAT Topography</p>
                    <p className="text-sm font-medium text-slate-600">Graph nodes analyze functional network proximity.</p>
                  </div>
                </div>
              </div>
            </div>

            {/* SHAP Explanation Section (Restored) */}
            {shapData.length > 0 && (
              <div className="pt-6 border-t border-slate-100">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-bold text-slate-700 uppercase tracking-tight">AI Interpretation (SHAP)</h3>
                  <div className="flex items-center gap-1.5 px-2 py-0.5 bg-amber-50 rounded-full">
                    <ShieldCheck size={12} className="text-amber-500" />
                    <span className="text-[10px] font-bold text-amber-600 uppercase">Ensemble Validated</span>
                  </div>
                </div>

                <div className="space-y-3">
                  {shapData.map((item, i) => (
                    <div key={i} className="space-y-1">
                      <div className="flex justify-between text-[10px]">
                        <span className="font-bold text-slate-500 uppercase">{item.name}</span>
                        <span className={`font-bold ${item.value > 0 ? 'text-green-500' : 'text-rose-500'}`}>
                          {item.value > 0 ? '+' : ''}{item.value.toFixed(1)}%
                        </span>
                      </div>
                      <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${Math.abs(item.value) * 2}%` }}
                          className={`h-full ${item.value > 0 ? 'bg-green-400' : 'bg-rose-400'}`}
                        />
                      </div>
                    </div>
                  ))}
                </div>
                <p className="mt-3 text-[10px] text-slate-400 font-medium italic">
                  Shapley values indicate the relative contribution of each feature to the final ensemble decision.
                </p>
              </div>
            )}
            <div className="glass-card p-8 bg-slate-50/50">
              <div className="flex items-center gap-3 mb-8">
                <Activity className="text-scientific-accent" />
                <h4 className="text-lg font-bold text-slate-800">Model Interpretation</h4>
              </div>

              <p className="text-slate-500 mb-6 leading-relaxed">
                Our advanced <span className="text-scientific-primary font-bold">Hybrid Stacking Ensemble</span> aggregates multiple
                biological evidence streams. By combining protein sequence motifs with localized graph topology,
                we achieve state-of-the-art predictive performance.
              </p>
              <div className="max-w-3xl">
                <div className="p-6 bg-white rounded-2xl border border-slate-200 shadow-sm relative italic text-slate-600 leading-relaxed">
                  <div className="absolute -top-3 left-6 p-1 px-2 bg-scientific-accent text-white text-[10px] font-bold rounded uppercase tracking-widest">AI Interpreter</div>
                  "The {result.esm_probability > 0.5 ? 'sequence' : 'sequence'} model suggests {result.esm_probability > 0.5 ? 'strong' : 'weak'} biological alignment.
                  The graph network {result.gat_probability > 0.5 ? 'reinforces' : 'contradicts'} this with its neighbor analysis.
                  The final consensus interaction is {result.interaction_probability > 0.5 ? 'likely' : 'unlikely'} based on the combined evidence."
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Predict;
