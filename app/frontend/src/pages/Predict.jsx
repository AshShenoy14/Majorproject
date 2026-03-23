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
  ChevronRight
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

  const getModelData = () => {
    if (!result) return [];
    return [
      { name: 'ESM-MLP', value: (Number(result.esm_probability || 0) * 100).toFixed(1), color: '#0D9488' },
      { name: 'GAT Graph', value: (Number(result.gat_probability || 0) * 100).toFixed(1), color: '#3B82F6' },
      { name: 'Ensemble', value: (Number(result.interaction_probability || 0) * 100).toFixed(1), color: '#7C3AED' },
    ];
  };

  const getShapData = () => {
    if (!result || !result.explanation) return [];
    const exp = result.explanation;
    return [
      { name: 'Seq Evidence', value: exp.SHAP_Sequence || 0 },
      { name: 'Graph Evidence', value: exp.SHAP_Graph || 0 },
      { name: 'Seq Conf', value: exp.SHAP_Seq_Conf || 0 },
      { name: 'Graph Conf', value: exp.SHAP_Graph_Conf || 0 },
    ].sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  };

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

                <div className={`flex items-center gap-2 px-6 py-2 rounded-full mb-4 ${result.interaction_probability > 0.5 ? 'bg-green-50 text-green-600 border border-green-100' : 'bg-orange-50 text-orange-600 border border-orange-100'}`}>
                  {result.interaction_probability > 0.5 ? <CheckCircle2 size={18} /> : <XCircle size={18} />}
                  <span className="font-bold uppercase tracking-wider">
                    {result.interaction_probability > 0.5 ? 'Strong Interaction' : 'No Interaction'}
                  </span>
                </div>
                
                <div className="text-xs text-slate-400 font-medium">
                  Confidence Score: <span className="text-slate-700 font-bold">{(Number(result.confidence_score || 0) * 100).toFixed(1)}%</span>
                  <TooltipIcon text="Score based on the agreement between Sequence and graph models." />
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
                    <BarChart data={getModelData()} layout="vertical" barSize={32}>
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
                        {getModelData().map((entry, index) => (
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

            {/* Explainability Section */}
            <div className="glass-card p-8 bg-slate-50/50">
              <div className="flex items-center gap-3 mb-8">
                <Activity className="text-scientific-accent" />
                <h4 className="text-lg font-bold text-slate-800">SHAP Explainability Insights <TooltipIcon text="Values show how much each feature contributed to the final probability." /></h4>
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
                <div className="h-64">
                   <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={getShapData()} layout="vertical">
                       <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#E2E8F0" />
                       <XAxis type="number" hide />
                       <YAxis 
                        dataKey="name" 
                        type="category" 
                        axisLine={false} 
                        tickLine={false} 
                        width={120}
                        tick={{ fontSize: 11, fontWeight: 600, fill: '#475569' }}
                      />
                      <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                        {getShapData().map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#22C55E' : '#EF4444'} />
                        ))}
                      </Bar>
                    </BarChart>
                   </ResponsiveContainer>
                </div>
                <div className="flex flex-col justify-center">
                   <div className="p-6 bg-white rounded-2xl border border-slate-200 shadow-sm relative italic text-slate-600 leading-relaxed">
                      <div className="absolute -top-3 left-6 p-1 px-2 bg-scientific-accent text-white text-[10px] font-bold rounded uppercase tracking-widest">AI Interpreter</div>
                      "The {result.esm_probability > 0.5 ? 'sequence' : 'sequence'} model suggests {result.esm_probability > 0.5 ? 'strong' : 'weak'} biological alignment. 
                      The graph network {result.gat_probability > 0.5 ? 'reinforces' : 'contradicts'} this with its neighbor analysis. 
                      Overall, the ensemble meta-learner concludes that this interaction is {result.interaction_probability > 0.5 ? 'likely' : 'unlikely'} based on the combined evidence."
                   </div>
                   <div className="mt-6 flex gap-3 flex-wrap">
                      <span className="px-3 py-1 bg-green-50 text-green-600 text-[10px] font-bold rounded-full border border-green-100">PRO-INTERACTION FACTORS</span>
                      <span className="px-3 py-1 bg-red-50 text-red-600 text-[10px] font-bold rounded-full border border-red-100">ANTI-INTERACTION FACTORS</span>
                   </div>
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
