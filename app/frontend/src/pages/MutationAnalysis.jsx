import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { motion } from 'framer-motion';
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
import HotspotDesignPanel from '../components/HotspotDesignPanel';

const MutationAnalysis = () => {
  const [searchParams] = useSearchParams();
  const paramP1 = searchParams.get('p1') || searchParams.get('protein1') || '';
  const paramP2 = searchParams.get('p2') || searchParams.get('protein2') || '';
  const [protein1, setProtein1] = useState(paramP1 || 'ENSP00000327694');
  const [protein2, setProtein2] = useState(paramP2 || 'ENSP00000373627');

  useEffect(() => {
    if (paramP1) setProtein1(paramP1);
    if (paramP2) setProtein2(paramP2);
  }, [paramP1, paramP2]);
  const [mutations, setMutations] = useState([{ protein: 1, pos: 45, orig: 'K', mut: 'A' }]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
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
    } catch (err) {
      setError(err.response?.data?.detail || "Mutation scan failed. Ensure correct IDs and residue positions.");
    } finally {
      setLoading(false);
    }
  };

  const getImpactData = () => {
    if (!result || !result.mutation_results) return [];
    return result.mutation_results.filter(res => !res.error).map(res => ({
      pos: res.pos,
      impact: res.impact_delta,
      orig: res.orig,
      mut: res.mut
    }));
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8 pb-12">
      <div className="glass-card p-8">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 bg-scientific-accent/10 rounded-lg text-scientific-accent">
            <Dna size={24} aria-hidden="true" />
          </div>
          <h1 className="text-2xl font-bold text-slate-900">Mutations</h1>
        </div>
        <p className="text-slate-700 mb-6 max-w-3xl">
          A mutation changes one amino acid (one letter) in a protein. Enter the change below to see whether the
          predicted interaction between the two proteins gets stronger or weaker.
        </p>

        {error && (
          <div role="alert" className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl flex items-center gap-3 text-red-700 text-sm">
            <AlertTriangle size={18} aria-hidden="true" />
            {error}
          </div>
        )}

        <form onSubmit={handleAnalysis} className="space-y-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label htmlFor="mut-protein1" className="text-sm font-bold text-slate-800">Protein 1</label>
              <input
                id="mut-protein1"
                type="text"
                value={protein1}
                onChange={(e) => setProtein1(e.target.value)}
                className="w-full px-4 py-3 bg-slate-50 border border-slate-300 rounded-xl focus:ring-2 focus:ring-scientific-accent outline-none font-mono"
              />
            </div>
            <div className="space-y-2">
              <label htmlFor="mut-protein2" className="text-sm font-bold text-slate-800">Protein 2</label>
              <input
                id="mut-protein2"
                type="text"
                value={protein2}
                onChange={(e) => setProtein2(e.target.value)}
                className="w-full px-4 py-3 bg-slate-50 border border-slate-300 rounded-xl focus:ring-2 focus:ring-scientific-accent outline-none font-mono"
              />
            </div>
          </div>

          <fieldset className="space-y-3">
            <div className="flex justify-between items-center">
              <legend className="text-base font-bold text-slate-900">Changes to test</legend>
              <button
                type="button"
                onClick={addMutation}
                className="text-sm font-bold text-scientific-accent flex items-center gap-1 hover:underline"
              >
                <Plus size={16} aria-hidden="true" /> Add another change
              </button>
            </div>
            <p className="text-sm text-slate-600">Example: protein 1, position 45, original letter K, new letter A means "replace the K at position 45 with an A".</p>

            <div className="hidden sm:grid grid-cols-4 gap-3 pr-12 text-xs font-bold text-slate-700" aria-hidden="true">
              <span>Which protein</span><span>Position</span><span>Original letter</span><span>New letter</span>
            </div>
            <div className="grid gap-3">
              {mutations.map((m, i) => (
                <div key={i} className="flex gap-4 items-center">
                  <div className="flex-1 grid grid-cols-2 sm:grid-cols-4 gap-3">
                    <select
                      aria-label={`Change ${i + 1}: which protein`}
                      value={m.protein}
                      onChange={(e) => updateMutation(i, 'protein', e.target.value)}
                      className="px-3 py-2 bg-white border border-slate-300 rounded-lg outline-none text-sm font-bold"
                    >
                      <option value={1}>Protein 1</option>
                      <option value={2}>Protein 2</option>
                    </select>
                    <input
                      type="number"
                      aria-label={`Change ${i + 1}: position`}
                      placeholder="Position"
                      value={m.pos}
                      onChange={(e) => updateMutation(i, 'pos', e.target.value)}
                      className="px-3 py-2 bg-white border border-slate-300 rounded-lg outline-none"
                    />
                    <input
                      type="text"
                      aria-label={`Change ${i + 1}: original letter`}
                      placeholder="e.g. K"
                      value={m.orig}
                      maxLength={1}
                      onChange={(e) => updateMutation(i, 'orig', e.target.value)}
                      className="px-3 py-2 bg-white border border-slate-300 rounded-lg outline-none"
                    />
                    <input
                      type="text"
                      aria-label={`Change ${i + 1}: new letter`}
                      placeholder="e.g. A"
                      value={m.mut}
                      maxLength={1}
                      onChange={(e) => updateMutation(i, 'mut', e.target.value)}
                      className="px-3 py-2 bg-white border border-slate-300 rounded-lg outline-none"
                    />
                  </div>
                  <button
                    type="button"
                    onClick={() => removeMutation(i)}
                    aria-label={`Remove change ${i + 1}`}
                    title="Remove this change"
                    className="p-2 text-red-700 hover:bg-red-50 rounded-lg transition-colors"
                  >
                    <Trash2 size={18} aria-hidden="true" />
                  </button>
                </div>
              ))}
            </div>
          </fieldset>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-4 bg-scientific-accent text-white rounded-xl font-bold flex items-center justify-center gap-2 hover:brightness-95 transition-all shadow-lg disabled:opacity-50 text-base"
          >
            {loading ? <Loader2 className="animate-spin" size={20} aria-hidden="true" /> : <Activity size={20} aria-hidden="true" />}
            {loading ? 'Testing…' : 'Test these changes'}
          </button>
        </form>
      </div>

      <HotspotDesignPanel
        protein1={protein1}
        protein2={protein2}
        onLoadMutations={(muts) => { setMutations(muts); setResult(null); window.scrollTo({ top: 0, behavior: 'smooth' }); }}
      />

      {result && result.mutation_results && result.mutation_results.length > 0 && (
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="grid grid-cols-1 lg:grid-cols-3 gap-8"
        >
          {/* Result Cards for all mutations */}
          <div className="lg:col-span-1 space-y-6">
            <h2 className="text-base font-bold text-slate-900 text-center">Results</h2>
            {result.mutation_results.map((res, idx) => {
              if (res.error) {
                return (
                  <div key={idx} className="glass-card p-6 space-y-2 border border-amber-200">
                    <span className="text-xs font-bold text-slate-600 uppercase tracking-wider">
                      Protein {res.protein === 2 ? '2' : '1'}: position {res.pos}, {res.orig} → {res.mut}
                    </span>
                    <p className="text-xs text-amber-700 flex items-start gap-2">
                      <AlertTriangle size={14} className="shrink-0 mt-0.5" /> Not evaluated: {res.error}
                    </p>
                  </div>
                );
              }
              // same thresholds as the backend's interpretation (|Δ| > 0.05 = Enhancing / Disruptive)
              const isIncrease = res.interpretation === 'Enhancing';
              const isDecrease = res.interpretation === 'Disruptive';
              return (
                <div key={idx} className="glass-card p-6 flex flex-col items-center space-y-4">
                  <div className="flex items-center justify-between w-full gap-2">
                    <span className="text-xs font-bold text-slate-600 uppercase tracking-wider">
                      Protein {res.protein === 2 ? '2' : '1'}: position {res.pos}, {res.orig} → {res.mut}
                    </span>
                  </div>

                  <div className="p-4 bg-slate-50 rounded-2xl w-full text-center space-y-3 border border-slate-100">
                     <div className="flex justify-between items-center text-xs text-slate-500">
                        <span>Original: <b>{(Number(res.base_score || 0) * 100).toFixed(1)}%</b></span>
                        <ArrowRight size={14} className="text-slate-500" />
                        <span>After change: <b className="text-scientific-accent">{(Number(res.mutated_score || 0) * 100).toFixed(1)}%</b></span>
                     </div>
                  </div>

                  <div className={`p-3 rounded-xl w-full flex items-center justify-between border ${
                    isIncrease 
                      ? 'bg-emerald-50 text-emerald-700 border-emerald-200' 
                      : isDecrease
                        ? 'bg-rose-50 text-rose-700 border-rose-200'
                        : 'bg-slate-50 text-slate-700 border-slate-200'
                  }`}>
                    <div className="flex items-center gap-2">
                       {isIncrease ? (
                         <TrendingUp size={18} className="text-emerald-600" />
                       ) : isDecrease ? (
                         <TrendingDown size={18} className="text-rose-600" />
                       ) : (
                         <Minus size={18} className="text-slate-500" />
                       )}
                        <span className="text-xs font-extrabold">
                          {res.impact_delta > 0 ? '+' : ''}
                          {(Number(res.impact_delta || 0) * 100).toFixed(1)}% Δ
                        </span>
                    </div>
                    <span className="text-xs font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-white/80">
                      {isIncrease ? 'Stronger' : isDecrease ? 'Weaker' : 'Little effect'}
                    </span>
                  </div>

                  <div className="p-3 bg-slate-50 rounded-xl w-full">
                     <p className="text-[11px] font-bold text-slate-600 mb-1 text-center">Model's reading</p>
                     <p className="text-xs text-slate-600 italic leading-relaxed text-center">
                       "{res.interpretation}"
                     </p>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Charts */}
          <div className="lg:col-span-2 glass-card p-8">
            <div className="flex items-center gap-3 mb-8">
              <Activity className="text-scientific-accent" size={20} />
              <h2 className="text-lg font-bold text-slate-900">Change in predicted chance (after change minus original)</h2>
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
                            <p className="text-xs uppercase font-bold tracking-wider opacity-80">
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
                  <span>Green = stronger after the change</span>
               </div>
               <div className="flex items-center gap-2 text-rose-700 bg-rose-50 px-3 py-1.5 rounded-lg border border-rose-200">
                  <div className="w-3 h-3 rounded-full bg-rose-500" />
                  <span>Red = weaker after the change</span>
               </div>
            </div>

            <div className="p-4 bg-orange-50 rounded-xl flex gap-3 border border-orange-100">
               <AlertTriangle className="text-orange-500 shrink-0" size={20} />
               <div>
                  <p className="text-xs font-bold text-orange-700 mb-1">Please note</p>
                  <p className="text-xs text-orange-600 leading-relaxed">
                    These numbers come from the sequence model only: they show how a change in the amino-acid sequence moves its
                    estimate. They are predictions, not measured binding strengths.
                  </p>
               </div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default MutationAnalysis;
