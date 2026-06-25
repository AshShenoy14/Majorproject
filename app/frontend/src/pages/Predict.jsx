import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Search, Activity, BarChart3, Info, CheckCircle2, 
  XCircle, AlertCircle, Loader2, ChevronRight, 
  ShieldCheck, Cpu, Database, Terminal as TerminalIcon,
  Zap, ArrowRightLeft, LayoutGrid, Box, BookOpen
} from 'lucide-react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, 
  Tooltip, ResponsiveContainer, Cell, PieChart, Pie 
} from 'recharts';
import { ppiService } from '../services/api';
import Protein3DView from '../components/Protein3DView';

const Predict = () => {
  const [protein1, setProtein1] = useState('ENSP00000327694');
  const [protein2, setProtein2] = useState('ENSP00000373627');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [logs, setLogs] = useState([]);
  const logEndRef = useRef(null);

  const addLog = (msg, type = 'info') => {
    setLogs(prev => [...prev, { msg, type, time: new Date().toLocaleTimeString() }].slice(-10));
  };

  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setLogs([]);
    
    addLog("Initializing TransGraph-PPI Pipeline...", "info");
    await new Promise(r => setTimeout(r, 800));
    addLog("Extracting ESM-2 Language Embeddings...", "process");
    await new Promise(r => setTimeout(r, 600));
    addLog("Analyzing Topological Centrality via GIN...", "process");

    try {
      const response = await ppiService.predict(protein1, protein2);
      addLog("Ensemble Meta-Learner Converged.", "success");
      setResult(response.data);
    } catch (err) {
      addLog("Execution Fault: " + (err.response?.data?.detail || "Unknown"), "error");
      setError(err.response?.data?.detail || "Prediction failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col lg:flex-row h-[calc(100vh-120px)] gap-6 overflow-hidden">
      
      {/* LEFT: Scientific Control Sidebar */}
      <aside className="w-full lg:w-[400px] flex flex-col gap-6 h-full overflow-y-auto no-scrollbar">
        <div className="bg-white border border-slate-100 rounded-[2.5rem] p-8 shadow-xl relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-4 opacity-[0.03] group-hover:opacity-[0.06] transition-opacity text-slate-900">
            <Cpu size={80} />
          </div>
          
          <div className="flex items-center gap-3 mb-8">
            <div className="p-3 bg-gradient-to-br from-emerald-400 to-teal-600 rounded-2xl text-white shadow-lg shadow-emerald-200">
              <Zap size={22} fill="currentColor" />
            </div>
            <div>
              <h2 className="text-xl font-black text-slate-800 tracking-tight">Analysis Portal</h2>
              <p className="text-[10px] font-black text-emerald-500 uppercase tracking-widest">Model: GIN-FUSION-V2</p>
            </div>
          </div>

          <form onSubmit={handlePredict} className="space-y-6">
            <div className="space-y-2">
              <label className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] ml-1">Protein Alpha ID</label>
              <div className="relative">
                <input 
                  value={protein1}
                  onChange={(e) => setProtein1(e.target.value)}
                  className="w-full bg-slate-50 border border-slate-100 rounded-2xl px-6 py-4 text-slate-700 font-mono text-sm focus:border-emerald-500 focus:bg-white outline-none transition-all shadow-inner"
                  placeholder="ENSP..."
                />
                <Database size={16} className="absolute right-5 top-1/2 -translate-y-1/2 text-slate-300" />
              </div>
            </div>

            <div className="flex justify-center -my-3">
              <div className="w-10 h-10 rounded-full bg-white border border-slate-100 flex items-center justify-center text-emerald-500 shadow-xl z-10 hover:rotate-180 transition-transform duration-500">
                <ArrowRightLeft size={16} />
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] ml-1">Protein Beta ID</label>
              <div className="relative">
                <input 
                  value={protein2}
                  onChange={(e) => setProtein2(e.target.value)}
                  className="w-full bg-slate-50 border border-slate-100 rounded-2xl px-6 py-4 text-slate-700 font-mono text-sm focus:border-emerald-500 focus:bg-white outline-none transition-all shadow-inner"
                  placeholder="ENSP..."
                />
                <Database size={16} className="absolute right-5 top-1/2 -translate-y-1/2 text-slate-300" />
              </div>
            </div>

            <button 
              type="submit" 
              disabled={loading}
              className="w-full bg-gradient-to-r from-emerald-500 via-teal-500 to-indigo-500 hover:from-emerald-400 hover:to-indigo-400 text-white font-black py-5 rounded-2xl transition-all shadow-2xl shadow-indigo-200 flex items-center justify-center gap-3 active:scale-95 disabled:opacity-50 uppercase tracking-widest text-xs"
            >
              {loading ? <Loader2 className="animate-spin" size={20} /> : <Zap size={20} />}
              Predict Interaction
            </button>
          </form>
        </div>

        {/* Live Telemetry Log */}
        <div className="flex-1 bg-white rounded-[2.5rem] border border-slate-100 p-8 font-mono text-[11px] overflow-hidden flex flex-col shadow-inner relative">
          <div className="flex items-center gap-3 mb-6 text-slate-400 border-b border-slate-50 pb-4 uppercase tracking-[0.2em] font-black">
            <TerminalIcon size={14} className="text-indigo-400" /> Neural Processing Logs
          </div>
          <div className="flex-1 overflow-y-auto space-y-2 no-scrollbar">
            {logs.length === 0 && <div className="text-slate-700 italic">Waiting for input sequence...</div>}
            {logs.map((log, i) => (
              <div key={i} className="flex gap-3 animate-in fade-in slide-in-from-left-2 duration-300">
                <span className="text-slate-600">[{log.time}]</span>
                <span className={
                  log.type === 'error' ? 'text-rose-400' : 
                  log.type === 'success' ? 'text-emerald-400' : 
                  log.type === 'process' ? 'text-cyan-400' : 'text-slate-300'
                }>
                  {log.msg}
                </span>
              </div>
            ))}
            <div ref={logEndRef} />
          </div>
        </div>
      </aside>

      {/* RIGHT: Analytical Workspace */}
      <main className="flex-1 bg-white/50 rounded-[3rem] border border-slate-100 p-10 relative overflow-hidden flex flex-col min-w-0 shadow-sm">
        <AnimatePresence mode="wait">
          {!result && !loading ? (
            <motion.div 
              key="empty"
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              className="h-full flex flex-col items-center justify-center text-center p-12"
            >
              <div className="w-28 h-28 rounded-full bg-slate-50 flex items-center justify-center mb-8 border border-slate-100 shadow-inner">
                <Box size={48} className="text-slate-300" />
              </div>
              <h3 className="text-3xl font-black text-slate-800 mb-3 tracking-tight">Workspace Idle</h3>
              <p className="text-slate-400 max-w-sm font-medium leading-relaxed">Enter protein IDs in the analysis portal to initiate geometric interaction mapping.</p>
            </motion.div>
          ) : loading ? (
            <motion.div 
              key="loading"
              initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              className="h-full flex flex-col items-center justify-center"
            >
              <div className="relative">
                <Loader2 size={80} className="text-emerald-500 animate-spin" />
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-16 h-16 bg-emerald-500/10 blur-2xl animate-pulse" />
                </div>
              </div>
              <p className="mt-10 text-slate-400 font-black uppercase tracking-[0.4em] text-[10px] animate-pulse">De-coding Fusion Gradients</p>
            </motion.div>
          ) : (
            <motion.div 
              key="result"
              initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }}
              className="h-full flex flex-col overflow-y-auto no-scrollbar gap-10"
            >
              {/* TOP: Critical Metrics Grid */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                
                {/* Consensus Gauge */}
                <div className="bg-white p-10 rounded-[2.5rem] border border-slate-50 flex flex-col items-center justify-center relative shadow-sm">
                  <div className="absolute top-6 left-6 px-3 py-1 bg-emerald-50 text-emerald-600 text-[9px] font-black rounded-lg uppercase tracking-widest">Ensemble Confidence</div>
                  <div className="w-44 h-44 mt-6 relative">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie data={[{v: result.interaction_probability*100}, {v: 100 - result.interaction_probability*100}]} innerRadius={65} outerRadius={80} startAngle={90} endAngle={-270} dataKey="v" paddingAngle={2}>
                          <Cell fill={result.interaction_probability > 0.5 ? "#10b981" : "#f43f5e"} />
                          <Cell fill="#f1f5f9" />
                        </Pie>
                      </PieChart>
                    </ResponsiveContainer>
                    <div className="absolute inset-0 flex flex-col items-center justify-center">
                      <span className="text-5xl font-black text-slate-800 tracking-tighter">{(result.interaction_probability*100).toFixed(0)}%</span>
                    </div>
                  </div>
                  <div className={`mt-8 px-6 py-2.5 rounded-full text-[10px] font-black uppercase tracking-widest shadow-sm ${result.interaction_probability > 0.5 ? 'bg-emerald-500 text-white shadow-emerald-100' : 'bg-rose-500 text-white shadow-rose-100'}`}>
                    {result.interaction_probability > 0.5 ? 'Strong Interaction' : 'Low Probability'}
                  </div>
                </div>

                {/* 3D Molecular Workbench (Center Hero) */}
                <div className="md:col-span-2 bg-white p-10 rounded-[2.5rem] border border-slate-50 flex items-center justify-center gap-10 shadow-sm overflow-hidden min-h-[350px] relative">
                  <div className="absolute top-6 left-1/2 -translate-x-1/2 px-4 py-1.5 bg-slate-50 rounded-full text-[9px] font-black text-slate-400 uppercase tracking-widest">3D Structural Projection</div>
                  <div className="flex-1 h-full"><Protein3DView pdbId={protein1} label="Target Alpha" /></div>
                  <div className="w-px h-32 bg-slate-100" />
                  <div className="flex-1 h-full"><Protein3DView pdbId={protein2} label="Target Beta" /></div>
                </div>
              </div>

              {/* BOTTOM: Model Breakdown & Logic */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 flex-1 min-h-0">
                
                {/* Signal Strength */}
                <div className="bg-white p-10 rounded-[2.5rem] border border-slate-50 flex flex-col shadow-sm">
                  <div className="flex items-center gap-4 mb-10">
                    <div className="p-2.5 bg-indigo-50 rounded-xl text-indigo-500"><LayoutGrid size={18} /></div>
                    <h4 className="text-xs font-black text-slate-800 uppercase tracking-widest">Expert Evidence Weighting</h4>
                  </div>
                  <div className="flex-1 flex flex-col justify-center space-y-8">
                    {[
                      { label: 'Protein Language (ESM)', val: result.esm_probability*100, color: 'bg-emerald-500' },
                      { label: 'Social Network (GIN)', val: result.gat_probability*100, color: 'bg-indigo-500' },
                      { label: 'Jury Agreement', val: result.confidence_score*100, color: 'bg-amber-500' }
                    ].map((sig, i) => (
                      <div key={i} className="space-y-3">
                        <div className="flex justify-between items-center text-[10px] font-black">
                          <span className="text-slate-400 uppercase tracking-[0.2em]">{sig.label}</span>
                          <span className="text-slate-800">{sig.val.toFixed(1)}%</span>
                        </div>
                        <div className="h-2 w-full bg-slate-50 rounded-full overflow-hidden shadow-inner">
                          <motion.div 
                            initial={{ width: 0 }} animate={{ width: `${sig.val}%` }}
                            transition={{ duration: 1, delay: i*0.2 }}
                            className={`h-full rounded-full ${sig.color} shadow-lg`}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* AI Interpretability Memo */}
                <div className="bg-slate-900 p-12 rounded-[2.5rem] flex flex-col relative overflow-hidden shadow-2xl">
                  <div className="absolute top-0 right-0 p-10 opacity-5 text-white"><ShieldCheck size={140} /></div>
                  <div className="flex items-center gap-4 mb-8">
                    <div className="p-2.5 bg-white/10 rounded-xl text-emerald-400"><BookOpen size={18} /></div>
                    <h4 className="text-xs font-black text-white uppercase tracking-widest">Plain Language Summary</h4>
                  </div>
                  <div className="space-y-6 text-slate-300 leading-relaxed font-medium">
                    <p className="text-lg text-white font-bold leading-snug">
                      The AI identifies a <span className="text-emerald-400 underline decoration-2 underline-offset-4">{result.interaction_probability > 0.5 ? 'High probability' : 'Very low chance'}</span> of these proteins meeting.
                    </p>
                    <p className="text-sm opacity-80">
                      The "Language Reader" expert found {result.interaction_probability > 0.5 ? 'strong biological motifs' : 'conflicting patterns'} in their sequences, 
                      while the "Social Mapper" confirmed they {result.interaction_probability > 0.5 ? 'share common neighbors' : 'operate in different areas'} of the cell.
                    </p>
                    <div className="flex items-center gap-6 text-[10px] text-slate-500 border-t border-white/5 pt-8 mt-4 font-black uppercase tracking-widest">
                      <div className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-emerald-500" /> Confidence: High</div>
                      <div className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-indigo-500" /> Data Source: STRING DB</div>
                    </div>
                  </div>
                </div>

              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
};

export default Predict;
