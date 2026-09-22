import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  GitCompare, Dna, ChevronRight, Loader2, AlertCircle,
  TrendingUp, TrendingDown, Minus, Zap, Map, CheckCircle
} from 'lucide-react';
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Cell, PieChart, Pie
} from 'recharts';
import { ppiService } from '../services/api';

// ────── Comparison Mode Page ──────────────────────────────────────────────────
// Shows wild-type vs single-point mutant side-by-side with:
//   - Interaction probability diff
//   - Probability delta visualization
//   - Mutation impact interpretation

const ComparisonMode = () => {
  const [searchParams] = useSearchParams();
  const paramP1 = searchParams.get('p1') || searchParams.get('protein1') || '';
  const paramP2 = searchParams.get('p2') || searchParams.get('protein2') || '';

  const [protein1, setProtein1] = useState(paramP1);
  const [protein2, setProtein2] = useState(paramP2);

  useEffect(() => {
    if (paramP1) setProtein1(paramP1);
    if (paramP2) setProtein2(paramP2);
  }, [paramP1, paramP2]);
  const [mutPos, setMutPos] = useState('');
  const [mutOrig, setMutOrig] = useState('');
  const [mutAlt, setMutAlt] = useState('');
  const [mutProtein, setMutProtein] = useState('1');

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Results for WT and mutant
  const [wtResult, setWtResult] = useState(null);
  const [mutResult, setMutResult] = useState(null);

  const handleCompare = async () => {
    if (!protein1 || !protein2 || !mutPos || !mutOrig || !mutAlt) {
      setError('Please fill in all fields before comparing.');
      return;
    }
    setError(null);
    setLoading(true);
    setWtResult(null);
    setMutResult(null);

    try {
      // 1. Wild-type prediction
      const wtPred = await ppiService.predict(protein1, protein2);
      setWtResult(wtPred.data);

      // 2. Mutant prediction via mutation endpoint
      const mutRes = await ppiService.mutate(protein1, null, protein2, null, [{
        protein: parseInt(mutProtein),
        pos: parseInt(mutPos),
        orig: mutOrig.toUpperCase(),
        mut: mutAlt.toUpperCase(),
      }]);
      const mutEntry = mutRes.data.mutation_results[0];
      setMutResult(mutEntry);
    } catch (err) {
      setError(err?.response?.data?.detail || 'Comparison failed. Check backend.');
    } finally {
      setLoading(false);
    }
  };

  const delta = mutResult ? (mutResult.mutated_score - mutResult.base_score) : null;
  const wtProb = wtResult?.interaction_probability;
  const mutProb = mutResult?.mutated_score;

  const chartData = wtProb != null && mutProb != null ? [
    { label: 'Wild Type', prob: +(wtProb * 100).toFixed(1), fill: '#0d9488' },
    { label: 'Mutant',    prob: +(mutProb * 100).toFixed(1), fill: delta < 0 ? '#f43f5e' : '#f59e0b' },
  ] : [];

  return (
    <div className="max-w-7xl mx-auto space-y-8 py-2">
      {/* Header */}
      <div className="flex items-center gap-4">
        <div className="p-3 bg-gradient-to-br from-amber-400 to-orange-500 rounded-2xl shadow-lg shadow-amber-100">
          <GitCompare size={28} className="text-white" />
        </div>
        <div>
          <h1 className="text-3xl font-black text-slate-800 tracking-tight">Comparison Mode</h1>
          <p className="text-sm text-slate-400 font-medium">Wild Type vs Single-Point Mutant — side-by-side interaction analysis</p>
        </div>
      </div>

      {/* Input Form */}
      <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-8">
        <h2 className="text-sm font-black text-slate-600 uppercase tracking-widest mb-6">Configure Comparison</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div>
            <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest block mb-2">Protein A (ENSP ID)</label>
            <input value={protein1} onChange={e => setProtein1(e.target.value)}
              placeholder="e.g. ENSP00000327694"
              className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl text-sm font-medium focus:ring-2 focus:ring-teal-300 outline-none"
            />
          </div>
          <div>
            <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest block mb-2">Protein B (ENSP ID)</label>
            <input value={protein2} onChange={e => setProtein2(e.target.value)}
              placeholder="e.g. ENSP00000373627"
              className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl text-sm font-medium focus:ring-2 focus:ring-teal-300 outline-none"
            />
          </div>
        </div>

        <div className="bg-amber-50 border border-amber-100 rounded-xl p-5 mb-6">
          <h3 className="text-xs font-black text-amber-700 uppercase tracking-widest mb-4 flex items-center gap-2">
            <Dna size={14} /> Mutation Configuration
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest block mb-1.5">Which Protein</label>
              <select value={mutProtein} onChange={e => setMutProtein(e.target.value)}
                className="w-full px-3 py-2.5 bg-white border border-slate-200 rounded-xl text-sm font-medium focus:ring-2 focus:ring-amber-300 outline-none">
                <option value="1">Protein A</option>
                <option value="2">Protein B</option>
              </select>
            </div>
            <div>
              <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest block mb-1.5">Position</label>
              <input type="number" min="1" value={mutPos} onChange={e => setMutPos(e.target.value)}
                placeholder="e.g. 152"
                className="w-full px-3 py-2.5 bg-white border border-slate-200 rounded-xl text-sm font-medium focus:ring-2 focus:ring-amber-300 outline-none"
              />
            </div>
            <div>
              <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest block mb-1.5">Original AA</label>
              <input maxLength={1} value={mutOrig} onChange={e => setMutOrig(e.target.value)}
                placeholder="e.g. A"
                className="w-full px-3 py-2.5 bg-white border border-slate-200 rounded-xl text-sm font-bold uppercase focus:ring-2 focus:ring-amber-300 outline-none"
              />
            </div>
            <div>
              <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest block mb-1.5">Mutant AA</label>
              <input maxLength={1} value={mutAlt} onChange={e => setMutAlt(e.target.value)}
                placeholder="e.g. V"
                className="w-full px-3 py-2.5 bg-white border border-slate-200 rounded-xl text-sm font-bold uppercase focus:ring-2 focus:ring-amber-300 outline-none"
              />
            </div>
          </div>
        </div>

        {error && (
          <div className="flex items-center gap-2 text-rose-600 bg-rose-50 border border-rose-100 px-4 py-3 rounded-xl mb-4 text-sm">
            <AlertCircle size={14} /> {error}
          </div>
        )}

        <button onClick={handleCompare} disabled={loading}
          className="w-full py-3.5 bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-600 hover:to-orange-600 text-white font-black rounded-xl shadow-lg shadow-amber-100 transition-all hover:scale-[1.01] active:scale-95 disabled:opacity-50 flex items-center justify-center gap-2">
          {loading ? <><Loader2 size={18} className="animate-spin" /> Analyzing...</> : <><GitCompare size={18} /> Run Comparison</>}
        </button>
      </div>

      {/* Results */}
      <AnimatePresence>
        {(wtResult || mutResult) && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
            {/* Summary Bar */}
            <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6 flex flex-wrap items-center gap-8">
              <div className="flex-1 min-w-[200px]">
                <ResponsiveContainer width="100%" height={100}>
                  <BarChart data={chartData} margin={{ top: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                    <XAxis dataKey="label" tick={{ fontSize: 11, fontWeight: 700 }} />
                    <YAxis domain={[0, 100]} unit="%" tick={{ fontSize: 10 }} />
                    <Tooltip formatter={v => `${v}%`} />
                    <Bar dataKey="prob" radius={[6, 6, 0, 0]}>
                      {chartData.map((d, i) => <Cell key={i} fill={d.fill} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="space-y-3 flex-1 min-w-[200px]">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-teal-50 flex items-center justify-center">
                    <CheckCircle size={18} className="text-teal-500" />
                  </div>
                  <div>
                    <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Wild Type</p>
                    <p className="text-xl font-black text-teal-600">{wtProb != null ? `${(wtProb*100).toFixed(1)}%` : '—'}</p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${delta < 0 ? 'bg-rose-50' : 'bg-amber-50'}`}>
                    {delta < 0 ? <TrendingDown size={18} className="text-rose-500" /> : <TrendingUp size={18} className="text-amber-500" />}
                  </div>
                  <div>
                    <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Mutant ({mutOrig}{mutPos}{mutAlt})</p>
                    <p className={`text-xl font-black ${delta < 0 ? 'text-rose-600' : 'text-amber-600'}`}>
                      {mutProb != null ? `${(mutProb*100).toFixed(1)}%` : '—'}
                    </p>
                  </div>
                </div>
              </div>

              <div className="flex-1 min-w-[180px]">
                {delta != null && (
                  <div className={`rounded-2xl p-5 text-center ${delta < -0.05 ? 'bg-rose-50 border border-rose-100' : delta > 0.05 ? 'bg-emerald-50 border border-emerald-100' : 'bg-slate-50 border border-slate-200'}`}>
                    <p className="text-[10px] font-black uppercase tracking-widest text-slate-400 mb-1">Δ Impact</p>
                    <p className={`text-3xl font-black ${delta < 0 ? 'text-rose-600' : delta > 0 ? 'text-emerald-600' : 'text-slate-600'}`}>
                      {delta > 0 ? '+' : ''}{(delta * 100).toFixed(1)}%
                    </p>
                    <p className="text-[10px] font-bold text-slate-500 mt-1">
                      {Math.abs(delta) < 0.05 ? '🟡 Neutral' : delta < 0 ? '🔴 Disruptive' : '🟢 Stabilizing'}
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Mutation impact summary */}
            {mutResult && (
              <div className={`rounded-2xl p-6 border ${Math.abs(delta) >= 0.05 ? 'bg-amber-50 border-amber-200' : 'bg-slate-50 border-slate-200'}`}>
                <h3 className="text-sm font-black uppercase tracking-widest mb-2 flex items-center gap-2 text-slate-700">
                  <Zap size={14} /> Mutation Impact Analysis
                </h3>
                <p className="text-sm text-slate-600">
                  {Math.abs(delta) >= 0.05
                    ? `⚠️ Position ${mutPos} mutation (${mutOrig}→${mutAlt}) shows a significant impact on interaction probability (${delta > 0 ? '+' : ''}${(delta * 100).toFixed(1)}% Δ).`
                    : `ℹ️ Position ${mutPos} mutation (${mutOrig}→${mutAlt}) shows minimal effect on overall binding affinity (${delta > 0 ? '+' : ''}${(delta * 100).toFixed(1)}% Δ).`}
                </p>
                {mutResult.interpretation && (
                  <p className="text-xs text-slate-500 mt-2 italic">Model Interpretation: {mutResult.interpretation}</p>
                )}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ComparisonMode;
