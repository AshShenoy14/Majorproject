import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Dna, 
  ArrowRight, 
  Plus, 
  Trash2, 
  Activity, 
  AlertTriangle,
  Loader2,
  TrendingDown,
  TrendingUp,
  Minus
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  ReferenceLine,
  AreaChart,
  Area
} from 'recharts';
import { ppiService } from '../services/api';
import IRLMVisualizer from '../components/IRLMVisualizer';

const MutationAnalysis = () => {
  const [protein1, setProtein1] = useState('ENSP00000327694');
  const [protein2, setProtein2] = useState('ENSP00000373627');
  const [mutations, setMutations] = useState([{ protein: 1, pos: 45, orig: 'A', mut: 'T' }]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [irlmData, setIrlmData] = useState(null);
  const [error, setError] = useState(null);

  const addMutation = () => {
    setMutations([...mutations, { protein: 1, pos: 0, orig: '', mut: '' }]);
  };

  const removeMutation = (index) => {
    setMutations(mutations.filter((_, i) => i !== index));
  };

  const updateMutation = (index, field, value) => {
    const newMutations = [...mutations];
    if (field === 'pos' || field === 'protein') {
      newMutations[index][field] = parseInt(value) || 0;
    } else {
      newMutations[index][field] = value.toUpperCase();
    }
    setMutations(newMutations);
  };

  const handleAnalysis = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      // Map mutations to the expected format
      const formattedMutations = mutations.map(m => ({
        protein: m.protein,
        pos: m.pos,
        orig: m.orig,
        mut: m.mut
      }));

      const response = await ppiService.mutate(protein1, null, protein2, null, formattedMutations);
      setResult(response.data);

      try {
        const locRes = await ppiService.localizeInteractionRegions(
          protein1, 
          protein2, 
          null, 
          null, 
          response.data.mutation_results[0]?.base_score || 0.5
        );
        setIrlmData(locRes.data);
      } catch (locErr) {
        console.warn("IRLM region localization in mutation scan failed:", locErr);
      }
    } catch (err) {
      setError(err.response?.data?.detail || "Mutation scan failed. Ensure correct IDs and residue positions.");
    } finally {
      setLoading(false);
    }
  };

  const getImpactData = () => {
    if (!result || !result.mutation_results) return [];
    return result.mutation_results.map(res => ({
      pos: res.pos,
      impact: res.impact_delta,
      orig: res.orig,
      mut: res.mut
    }));
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8 pb-12">
      <div className="glass-card p-8">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 bg-scientific-accent/10 rounded-lg text-scientific-accent">
            <Dna size={24} />
          </div>
          <h2 className="text-2xl font-bold text-slate-800">Mutation Impact Scan</h2>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-100 rounded-xl flex items-center gap-3 text-red-600 text-sm">
            <AlertTriangle size={18} />
            {error}
          </div>
        )}

        <form onSubmit={handleAnalysis} className="space-y-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label className="text-xs font-bold text-slate-500 uppercase tracking-wider">Protein A ID</label>
              <input 
                type="text" 
                value={protein1}
                onChange={(e) => setProtein1(e.target.value)}
                className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-scientific-accent outline-none"
              />
            </div>
            <div className="space-y-2">
              <label className="text-xs font-bold text-slate-500 uppercase tracking-wider">Protein B ID</label>
              <input 
                type="text" 
                value={protein2}
                onChange={(e) => setProtein2(e.target.value)}
                className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-scientific-accent outline-none"
              />
            </div>
          </div>

          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="text-sm font-bold text-slate-600 uppercase tracking-widest">Mutation Batch</h3>
              <button 
                type="button" 
                onClick={addMutation}
                className="text-xs font-bold text-scientific-accent flex items-center gap-1 hover:underline"
              >
                <Plus size={14} /> ADD RESIDUE
              </button>
            </div>

            <div className="grid gap-3">
              {mutations.map((m, i) => (
                <div key={i} className="flex gap-4 items-center animate-in fade-in slide-in-from-left-2 transition-all">
                  <div className="flex-1 grid grid-cols-4 gap-3">
                    <select
                      value={m.protein}
                      onChange={(e) => updateMutation(i, 'protein', e.target.value)}
                      className="px-3 py-2 bg-white border border-slate-200 rounded-lg outline-none text-xs font-bold"
                    >
                      <option value={1}>Protein A</option>
                      <option value={2}>Protein B</option>
                    </select>
                    <input 
                      type="number" 
                      placeholder="Pos" 
                      value={m.pos} 
                      onChange={(e) => updateMutation(i, 'pos', e.target.value)}
                      className="px-3 py-2 bg-white border border-slate-200 rounded-lg outline-none" 
                    />
                    <input 
                      type="text" 
                      placeholder="Orig" 
                      value={m.orig} 
                      maxLength={1}
                      onChange={(e) => updateMutation(i, 'orig', e.target.value)}
                      className="px-3 py-2 bg-white border border-slate-200 rounded-lg outline-none" 
                    />
                    <input 
                      type="text" 
                      placeholder="Mut" 
                      value={m.mut} 
                      maxLength={1}
                      onChange={(e) => updateMutation(i, 'mut', e.target.value)}
                      className="px-3 py-2 bg-white border border-slate-200 rounded-lg outline-none" 
                    />
                  </div>
                  <button 
                    type="button" 
                    onClick={() => removeMutation(i)}
                    className="p-2 text-red-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                  >
                    <Trash2 size={18} />
                  </button>
                </div>
              ))}
            </div>
          </div>

          <button 
            type="submit" 
            disabled={loading}
            className="w-full py-4 bg-scientific-accent text-white rounded-xl font-bold flex items-center justify-center gap-2 hover:bg-purple-700 transition-all shadow-lg shadow-purple-200 disabled:opacity-50"
          >
            {loading ? <Loader2 className="animate-spin" size={20} /> : <Activity size={20} />}
            RUN IN-SILICO SCAN
          </button>
        </form>
      </div>

      {result && result.mutation_results && result.mutation_results.length > 0 && (
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="grid grid-cols-1 lg:grid-cols-3 gap-8"
        >
          {/* Result Card (showing first mutation summary) */}
          <div className="lg:col-span-1 glass-card p-8 flex flex-col items-center">
            <h3 className="text-sm font-bold text-slate-500 uppercase tracking-widest mb-8 text-center">Relative Impact Summary</h3>
            
            <div className="p-6 bg-slate-50 rounded-3xl w-full text-center space-y-6 mb-8 border border-slate-100">
               <div>
                  <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Wild-type Interaction Probability</p>
                  <p className="text-2xl font-bold text-slate-700">{(Number(result.mutation_results[0].base_score || 0) * 100).toFixed(1)}%</p>
               </div>
               <div className="flex items-center justify-center gap-2">
                  <div className="h-[1px] flex-1 bg-slate-200" />
                  <ArrowRight size={16} className="text-slate-300" />
                  <div className="h-[1px] flex-1 bg-slate-200" />
               </div>
               <div>
                  <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Mutant Interaction Probability</p>
                  <p className="text-4xl font-extrabold text-scientific-accent">{(Number(result.mutation_results[0].mutated_score || 0) * 100).toFixed(1)}%</p>
               </div>
            </div>

            <div className={`p-4 rounded-2xl w-full flex items-center justify-between border ${
              result.mutation_results[0].impact_delta > 0 
                ? 'bg-emerald-50 text-emerald-700 border-emerald-200' 
                : result.mutation_results[0].impact_delta < 0
                  ? 'bg-rose-50 text-rose-700 border-rose-200'
                  : 'bg-slate-50 text-slate-700 border-slate-200'
            }`}>
              <div className="flex items-center gap-2">
                 {result.mutation_results[0].impact_delta > 0 ? (
                   <TrendingUp size={20} className="text-emerald-600" />
                 ) : result.mutation_results[0].impact_delta < 0 ? (
                   <TrendingDown size={20} className="text-rose-600" />
                 ) : (
                   <Minus size={20} className="text-slate-500" />
                 )}
                  <span className="text-sm font-extrabold">
                    {result.mutation_results[0].impact_delta > 0 ? '+' : ''}
                    {(Number(result.mutation_results[0].impact_delta || 0) * 100).toFixed(1)}% Δ
                  </span>
              </div>
              <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-white/80 shadow-xs">
                {result.mutation_results[0].impact_delta > 0 ? 'Increase (Stabilizing)' : result.mutation_results[0].impact_delta < 0 ? 'Decrease (Disruptive)' : 'Neutral'}
              </span>
            </div>

            <div className="mt-6 p-4 bg-slate-50 rounded-xl w-full">
               <p className="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em] mb-2 text-center">AI Interpretation</p>
               <p className="text-xs text-slate-600 italic leading-relaxed text-center">
                 "{result.mutation_results[0].interpretation}"
               </p>
            </div>
          </div>

          {/* Charts */}
          <div className="lg:col-span-2 glass-card p-8">
            <div className="flex items-center gap-3 mb-8">
              <Activity className="text-scientific-accent" size={20} />
              <h4 className="text-lg font-bold text-slate-800">ΔProbability (Mutant − Wild-type)</h4>
            </div>
            
            <div className="h-64 mb-6">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={getImpactData()}>
                  <defs>
                    <linearGradient id="colorImpact" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#7C3AED" stopOpacity={0.15}/>
                      <stop offset="95%" stopColor="#7C3AED" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                  <XAxis dataKey="pos" axisLine={false} tickLine={false} tick={{ fontSize: 10 }} />
                  <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 10 }} />
                  <Tooltip 
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const val = Number(payload[0].value || 0);
                        const isIncrease = val > 0;
                        const isDecrease = val < 0;
                        return (
                          <div className={`p-3 rounded-xl shadow-xl border text-white ${
                            isIncrease ? 'bg-emerald-900 border-emerald-500' : isDecrease ? 'bg-rose-900 border-rose-500' : 'bg-slate-800 border-slate-600'
                          }`}>
                            <p className="text-xs font-bold mb-1 tracking-tight">Residue {payload[0].payload.pos} ({payload[0].payload.orig} → {payload[0].payload.mut})</p>
                            <p className={`text-lg font-extrabold flex items-center gap-1 ${
                              isIncrease ? 'text-emerald-300' : isDecrease ? 'text-rose-300' : 'text-slate-200'
                            }`}>
                              {isIncrease ? '+' : ''}{(val * 100).toFixed(2)}% Δ
                            </p>
                            <p className="text-[10px] uppercase font-bold tracking-wider opacity-80">
                              {isIncrease ? 'Green = Increase' : isDecrease ? 'Red = Decrease' : 'Neutral'}
                            </p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <ReferenceLine y={0} stroke="#94A3B8" strokeDasharray="3 3" />
                  <Area 
                    type="monotone" 
                    dataKey="impact" 
                    stroke="#7C3AED" 
                    fillOpacity={1} 
                    fill="url(#colorImpact)" 
                    strokeWidth={3}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Color Legend */}
            <div className="flex items-center justify-center gap-6 mb-6 text-xs font-bold">
               <div className="flex items-center gap-2 text-emerald-700 bg-emerald-50 px-3 py-1.5 rounded-lg border border-emerald-200">
                  <div className="w-3 h-3 rounded-full bg-emerald-500" />
                  <span>Green = Increase (Mutant &gt; Wild-type)</span>
               </div>
               <div className="flex items-center gap-2 text-rose-700 bg-rose-50 px-3 py-1.5 rounded-lg border border-rose-200">
                  <div className="w-3 h-3 rounded-full bg-rose-500" />
                  <span>Red = Decrease (Mutant &lt; Wild-type)</span>
               </div>
            </div>

            <div className="p-4 bg-orange-50 rounded-xl flex gap-3 border border-orange-100">
               <AlertTriangle className="text-orange-500 shrink-0" size={20} />
               <div>
                  <p className="text-xs font-bold text-orange-700 uppercase tracking-widest mb-1">Biological Context</p>
                  <p className="text-xs text-orange-600 leading-relaxed">
                    Mutations in high-affinity regions (hotspots) often lead to significant disruption of binding interfaces. 
                    Monitor the GAT Topography metrics if the delta is significant.
                  </p>
               </div>
            </div>
          </div>

          {/* IRLM VISUALIZER WITH MUTATION OVERLAYS */}
          {irlmData && (
            <div className="lg:col-span-3">
              <IRLMVisualizer 
                irlmData={irlmData} 
                id1={protein1} 
                id2={protein2} 
                mutations={mutations} 
                isDark={false} 
              />
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
};

export default MutationAnalysis;
