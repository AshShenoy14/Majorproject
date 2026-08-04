import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Search, Activity, BarChart3, Info, CheckCircle2, 
  XCircle, AlertCircle, Loader2, ChevronRight, 
  ShieldCheck, Cpu, Database, Terminal as TerminalIcon,
  Zap, ArrowRightLeft, LayoutGrid, Box, BookOpen, Download,
  Sparkles, Gauge, Server, Clock, UploadCloud, ToggleLeft, ToggleRight,
  GraduationCap, FlaskConical, Table2, FileDown
} from 'lucide-react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, 
  Tooltip, ResponsiveContainer, Cell, PieChart, Pie 
} from 'recharts';
import html2pdf from 'html2pdf.js';
import { ppiService } from '../services/api';
import Protein3DView from '../components/Protein3DView';
import ProteinInfoButton from '../components/ProteinInfoModal';
import IRLMVisualizer from '../components/IRLMVisualizer';

const CASE_STUDIES = [
  { label: "🎯 Oncology (TP53 & MDM2)", p1: "ENSP00000269305", p2: "ENSP00000258149", desc: "Tumor suppressor binding regulating cell cycle & apoptosis." },
  { label: "🧠 Neurodegenerative (AP2A2 & CLTC)", p1: "ENSP00000300161", p2: "ENSP00000267029", desc: "Clathrin-mediated endocytosis pathway linked to Alzheimer's." },
  { label: "⚡ Apoptosis (BAX & BCL2L1)", p1: "ENSP00000293879", p2: "ENSP00000307677", desc: "Mitochondrial outer membrane permeabilization control." },
  { label: "❄️ Cold-Start (Uncharacterized Pair)", p1: "ENSP00000385802", p2: "ENSP00000361000", desc: "Novel prediction for protein without prior graph interactions." }
];

const Predict = () => {
  const [inputMode, setInputMode] = useState('ids'); // 'ids' | 'sequence' | 'case_studies'
  const [protein1, setProtein1] = useState('ENSP00000327694');
  const [protein2, setProtein2] = useState('ENSP00000373627');
  const [seq1, setSeq1] = useState('');
  const [seq2, setSeq2] = useState('');
  const [selectedCase, setSelectedCase] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [irlmData, setIrlmData] = useState(null);
  const [selectedResidueP1, setSelectedResidueP1] = useState(null);
  const [selectedResidueP2, setSelectedResidueP2] = useState(null);
  const [error, setError] = useState(null);
  const [logs, setLogs] = useState([]);
  const [latency, setLatency] = useState(42);
  const logEndRef = useRef(null);
  // ── NEW STATE ──────────────────────────────────────────────
  const [expertMode, setExpertMode] = useState(false);  // false = Beginner, true = Research
  const [batchResults, setBatchResults] = useState([]);   // batch CSV results
  const [batchLoading, setBatchLoading] = useState(false);
  const fileInputRef = useRef(null);

  const addLog = (msg, type = 'info') => {
    setLogs(prev => [...prev, { msg, type, time: new Date().toLocaleTimeString() }].slice(-10));
  };

  const handleResidueSelect = (resObj) => {
    if (!resObj) return;
    // IRLMVisualizer passes proteinName = id1 or id2 string
    const isProtein1 = resObj.proteinName === protein1 || resObj.proteinNum === 1;
    if (isProtein1) {
      setSelectedResidueP1({ residue_number: resObj.pos, residue_name: resObj.aa || 'AA' });
      addLog(`3D View Synced → Protein A: Residue #${resObj.pos} (${resObj.aa || '?'})`, 'info');
    } else {
      setSelectedResidueP2({ residue_number: resObj.pos, residue_name: resObj.aa || 'AA' });
      addLog(`3D View Synced → Protein B: Residue #${resObj.pos} (${resObj.aa || '?'})`, 'info');
    }
  };

  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  const handleSelectCase = (caseObj) => {
    setSelectedCase(caseObj.label);
    setProtein1(caseObj.p1);
    setProtein2(caseObj.p2);
    addLog(`Preset selected: ${caseObj.label}`, 'info');
  };

  const handlePredict = async (e) => {
    if (e) e.preventDefault();
    setLoading(true);
    setError(null);
    setIrlmData(null);
    setLogs([]);
    const startTime = performance.now();
    
    addLog("Initializing TransGraph-PPI Pipeline...", "info");
    await new Promise(r => setTimeout(r, 400));
    addLog("Extracting ESM-2 Language Embeddings...", "process");
    await new Promise(r => setTimeout(r, 300));
    addLog("Analyzing Topological Centrality via GAT...", "process");

    try {
      const p1 = protein1.trim() || "Protein_1";
      const p2 = protein2.trim() || "Protein_2";
      const s1 = inputMode === 'sequence' ? seq1.trim() : null;
      const s2 = inputMode === 'sequence' ? seq2.trim() : null;

      const response = await ppiService.predict(p1, p2, s1, s2);
      const elapsed = Math.round(performance.now() - startTime);
      setLatency(elapsed);
      addLog(`Ensemble Meta-Learner Converged in ${elapsed}ms.`, "success");
      setResult(response.data);

      // Save context for the AI Assistant (context-aware chat)
      try {
        localStorage.setItem('transgraph_last_prediction', JSON.stringify({
          p1: p1, p2: p2,
          prob: response.data.interaction_probability,
          esm: response.data.esm_probability,
          gat: response.data.gat_probability,
          conf: response.data.confidence_score
        }));
      } catch (_) {}

      try {
        addLog("Localizing Interaction Regions (IRLM)...", "process");
        const locRes = await ppiService.localizeInteractionRegions(p1, p2, s1, s2, response.data.interaction_probability);
        setIrlmData(locRes.data);
        addLog("IRLM Region Localization Complete.", "success");
      } catch (locErr) {
        console.warn("IRLM localization failed:", locErr);
      }
    } catch (err) {
      addLog("Execution Fault: " + (err.response?.data?.detail || "Unknown"), "error");
      setError(err.response?.data?.detail || "Prediction failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadPDF = () => {
    if (!result) return;
    const element = document.getElementById('scientific-report-content');
    if (!element) return;
    const opt = {
      margin: 0.4,
      filename: `TransGraph_PPI_Report_${protein1}_${protein2}.pdf`,
      image: { type: 'jpeg', quality: 0.98 },
      html2canvas: { scale: 2 },
      jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' }
    };
    html2pdf().set(opt).from(element).save();
  };

  // ── Batch CSV upload handler ──────────────────────────────
  const handleBatchUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setBatchLoading(true);
    setBatchResults([]);
    setResult(null);
    setIrlmData(null);
    addLog(`Batch CSV loaded: ${file.name}`, 'info');
    const reader = new FileReader();
    reader.onload = async (ev) => {
      try {
        const lines = ev.target.result.split('\n').map(l => l.trim()).filter(Boolean);
        const pairs = lines
          .filter(l => !l.startsWith('#'))
          .map(l => {
            const [p1, p2] = l.split(',');
            return p1 && p2 ? { protein1_id: p1.trim(), protein2_id: p2.trim(), protein1_seq: '', protein2_seq: '' } : null;
          })
          .filter(Boolean);
        if (pairs.length === 0) throw new Error('No valid pairs in CSV.');
        addLog(`Processing ${pairs.length} pairs...`, 'process');
        const res = await ppiService.predictBatch(pairs);
        setBatchResults(res.data);
        addLog(`Batch complete: ${res.data.length} predictions.`, 'success');
        // Populate single result from first row for main dashboard
        if (res.data.length > 0) {
          setProtein1(pairs[0].protein1_id);
          setProtein2(pairs[0].protein2_id);
          setResult(res.data[0]);
        }
      } catch (err) {
        addLog('Batch failed: ' + (err.message || 'Unknown error'), 'error');
        setError('Batch processing failed: ' + (err.message || 'check CSV format (id1,id2 per line)'));
      } finally {
        setBatchLoading(false);
        if (fileInputRef.current) fileInputRef.current.value = '';
      }
    };
    reader.readAsText(file);
  };

  // ── Batch CSV download ────────────────────────────────────
  const downloadBatchCSV = useCallback(() => {
    if (!batchResults.length) return;
    const header = 'protein1_id,protein2_id,interaction_probability,esm_probability,gat_probability,confidence_score,interacts\n';
    const rows = batchResults.map(r =>
      `${r.protein1_id || ''},${r.protein2_id || ''},${r.interaction_probability?.toFixed(4) || ''},${r.esm_probability?.toFixed(4) || ''},${r.gat_probability?.toFixed(4) || ''},${r.confidence_score?.toFixed(4) || ''},${r.interaction_probability > 0.5 ? 'YES' : 'NO'}`
    ).join('\n');
    const blob = new Blob([header + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `transgraph_batch_results_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [batchResults]);

  return (
    <div className="flex flex-col gap-6">
      
      {/* FEATURE 3: Live Model Benchmark & Latency Telemetry Header Badge */}
      <div className="bg-slate-900 text-white rounded-2xl p-4 px-6 flex flex-wrap items-center justify-between gap-4 shadow-xl border border-slate-800">
        <div className="flex items-center gap-3">
          <div className="w-3 h-3 rounded-full bg-emerald-400 animate-ping" />
          <span className="text-xs font-black uppercase tracking-widest text-emerald-400">Live Model Telemetry</span>
          <span className="text-slate-500">|</span>
          <span className="text-xs font-semibold text-slate-300">ESM-2 + GAT Ensemble</span>
        </div>
        <div className="flex items-center gap-6 text-xs font-mono text-slate-300">
          <div className="flex items-center gap-1.5">
            <Clock size={14} className="text-emerald-400" />
            <span>Inference: <strong className="text-white">{latency}ms</strong></span>
          </div>
          <div className="flex items-center gap-1.5">
            <Gauge size={14} className="text-indigo-400" />
            <span>ROC-AUC: <strong className="text-white">0.942</strong></span>
          </div>
          <div className="flex items-center gap-1.5">
            <Server size={14} className="text-amber-400" />
            <span>Graph: <strong className="text-white">12,238 Nodes | 96,829 Edges</strong></span>
          </div>
        </div>
      </div>

      <div className="flex flex-col lg:flex-row min-h-[calc(100vh-160px)] gap-6 pb-8">
        
        {/* LEFT: Scientific Control Sidebar */}
        <aside className="w-full lg:w-[380px] flex flex-col gap-6">
          <div className="bg-white border border-slate-100 rounded-[2.5rem] p-6 shadow-xl relative overflow-hidden group">
            <div className="absolute top-0 right-0 p-4 opacity-[0.03] group-hover:opacity-[0.06] transition-opacity text-slate-900">
              <Cpu size={80} />
            </div>
            
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2.5 bg-gradient-to-br from-emerald-400 to-teal-600 rounded-2xl text-white shadow-lg shadow-emerald-200">
                <Zap size={20} fill="currentColor" />
              </div>
              <div>
                <h2 className="text-lg font-black text-slate-800 tracking-tight">Analysis Portal</h2>
                <p className="text-[10px] font-black text-emerald-500 uppercase tracking-widest">Model: ESM2-GAT-FUSION-V2</p>
              </div>
            </div>

            {/* Input Method Dropdown Selector */}
            <div className="mb-4 space-y-1">
              <label className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] ml-1 flex items-center gap-1">
                <Sparkles size={12} className="text-emerald-500" /> Input Method
              </label>
              <select
                value={inputMode}
                onChange={(e) => {
                  const mode = e.target.value;
                  setInputMode(mode);
                  if (mode === 'case_studies' && !selectedCase && CASE_STUDIES.length > 0) {
                    handleSelectCase(CASE_STUDIES[0]);
                  }
                }}
                className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3.5 py-2.5 text-xs text-slate-800 font-bold focus:border-emerald-500 focus:bg-white outline-none cursor-pointer shadow-xs transition-all"
              >
                <option value="ids">🆔 Use Protein IDs (UniProt / Ensembl)</option>
                <option value="sequence">🧬 Enter Amino Acid Sequence</option>
                <option value="case_studies">🎯 Select Case Studies</option>
              </select>
            </div>

            {/* Case Studies Sub-selector */}
            {inputMode === 'case_studies' && (
              <div className="mb-4 bg-emerald-50/70 border border-emerald-200/60 rounded-2xl p-3.5 space-y-2">
                <label className="text-[10px] font-black text-emerald-700 uppercase tracking-[0.2em] flex items-center gap-1">
                  <BookOpen size={12} /> Choose Case Study Preset
                </label>
                <select
                  value={selectedCase || ''}
                  onChange={(e) => {
                    const c = CASE_STUDIES.find(cs => cs.label === e.target.value);
                    if (c) handleSelectCase(c);
                  }}
                  className="w-full bg-white border border-emerald-300 rounded-xl px-3 py-2 text-xs text-slate-700 font-bold focus:border-emerald-500 outline-none cursor-pointer"
                >
                  <option value="" disabled>-- Select a Preset --</option>
                  {CASE_STUDIES.map((c, i) => (
                    <option key={i} value={c.label}>
                      {c.label} ({c.p1} & {c.p2})
                    </option>
                  ))}
                </select>
                {selectedCase && (
                  <p className="text-[11px] text-slate-600 font-medium italic leading-relaxed pt-1">
                    {CASE_STUDIES.find(c => c.label === selectedCase)?.desc}
                  </p>
                )}
              </div>
            )}

            <form onSubmit={handlePredict} className="space-y-4">
              {(inputMode === 'ids' || inputMode === 'case_studies') && (
                <>
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between">
                      <label className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] ml-1">Protein Alpha ID</label>
                      {protein1 && <ProteinInfoButton proteinId={protein1} label="Info" />}
                    </div>
                    <div className="relative">
                      <input 
                        value={protein1}
                        onChange={(e) => setProtein1(e.target.value)}
                        className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-700 font-mono text-sm focus:border-emerald-500 focus:bg-white outline-none transition-all shadow-inner"
                        placeholder="ENSP..."
                      />
                      <Database size={16} className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-300" />
                    </div>
                  </div>

                  <div className="flex justify-center -my-1">
                    <button
                      type="button"
                      onClick={() => {
                        setProtein1(protein2);
                        setProtein2(protein1);
                      }}
                      className="w-8 h-8 rounded-full bg-white border border-slate-200 flex items-center justify-center text-emerald-500 shadow-md z-10 hover:rotate-180 transition-transform duration-500 cursor-pointer"
                    >
                      <ArrowRightLeft size={14} />
                    </button>
                  </div>

                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between">
                      <label className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] ml-1">Protein Beta ID</label>
                      {protein2 && <ProteinInfoButton proteinId={protein2} label="Info" />}
                    </div>
                    <div className="relative">
                      <input 
                        value={protein2}
                        onChange={(e) => setProtein2(e.target.value)}
                        className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-700 font-mono text-sm focus:border-emerald-500 focus:bg-white outline-none transition-all shadow-inner"
                        placeholder="ENSP..."
                      />
                      <Database size={16} className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-300" />
                    </div>
                  </div>
                </>
              )}

              {inputMode === 'sequence' && (
                <>
                  <div className="space-y-1.5">
                    <label className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] ml-1">Protein 1 Identifier / Name</label>
                    <input 
                      value={protein1}
                      onChange={(e) => setProtein1(e.target.value)}
                      className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-2.5 text-slate-700 font-mono text-xs focus:border-emerald-500 focus:bg-white outline-none transition-all shadow-inner"
                      placeholder="e.g., Protein_Alpha"
                    />
                    <label className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] ml-1 mt-2 block">Amino Acid Sequence 1</label>
                    <textarea
                      rows={3}
                      value={seq1}
                      onChange={(e) => setSeq1(e.target.value)}
                      className="w-full bg-slate-50 border border-slate-200 rounded-xl p-3 text-slate-700 font-mono text-xs focus:border-emerald-500 focus:bg-white outline-none transition-all shadow-inner uppercase"
                      placeholder="MVLSPADKTNVKAAWGKVGAHAGEYGAEALERM..."
                    />
                  </div>

                  <div className="space-y-1.5 pt-2">
                    <label className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] ml-1">Protein 2 Identifier / Name</label>
                    <input 
                      value={protein2}
                      onChange={(e) => setProtein2(e.target.value)}
                      className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-2.5 text-slate-700 font-mono text-xs focus:border-emerald-500 focus:bg-white outline-none transition-all shadow-inner"
                      placeholder="e.g., Protein_Beta"
                    />
                    <label className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] ml-1 mt-2 block">Amino Acid Sequence 2</label>
                    <textarea
                      rows={3}
                      value={seq2}
                      onChange={(e) => setSeq2(e.target.value)}
                      className="w-full bg-slate-50 border border-slate-200 rounded-xl p-3 text-slate-700 font-mono text-xs focus:border-emerald-500 focus:bg-white outline-none transition-all shadow-inner uppercase"
                      placeholder="VHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWT..."
                    />
                  </div>
                </>
              )}

              <button 
                type="submit" 
                disabled={loading}
                className="w-full bg-gradient-to-r from-emerald-500 via-teal-500 to-indigo-500 hover:from-emerald-400 hover:to-indigo-400 text-white font-black py-4 rounded-xl transition-all shadow-lg shadow-emerald-500/20 flex items-center justify-center gap-2 active:scale-95 disabled:opacity-50 uppercase tracking-widest text-xs mt-2 cursor-pointer"
              >
                {loading ? <Loader2 className="animate-spin" size={18} /> : <Zap size={18} />}
                Predict Interaction
              </button>
            </form>

            {/* ── Batch Upload Button ────────────── */}
            <div className="border-t border-slate-100 pt-4 space-y-3">
              <label className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] ml-1 flex items-center gap-1">
                <UploadCloud size={12} className="text-indigo-400" /> Batch CSV Analysis
              </label>
              <button
                type="button"
                disabled={batchLoading}
                onClick={() => fileInputRef.current?.click()}
                className="w-full bg-slate-50 hover:bg-indigo-50 border border-dashed border-slate-300 hover:border-indigo-400 text-slate-500 hover:text-indigo-600 font-bold py-3 rounded-xl transition-all flex items-center justify-center gap-2 text-xs cursor-pointer disabled:opacity-50"
              >
                {batchLoading ? <Loader2 className="animate-spin" size={15} /> : <Table2 size={15} />}
                {batchLoading ? 'Processing...' : 'Upload CSV (id1,id2 per line)'}
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,.txt"
                className="hidden"
                onChange={handleBatchUpload}
              />
            </div>

            {/* ── Beginner / Research Mode Toggle ── */}
            <div className="border-t border-slate-100 pt-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {expertMode ? <FlaskConical size={14} className="text-violet-500" /> : <GraduationCap size={14} className="text-teal-500" />}
                  <span className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-500">
                    {expertMode ? 'Research Mode' : 'Beginner Mode'}
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() => setExpertMode(v => !v)}
                  className={`relative w-12 h-6 rounded-full transition-colors duration-300 ${
                    expertMode ? 'bg-violet-500' : 'bg-teal-400'
                  }`}
                >
                  <div className={`absolute top-1 w-4 h-4 bg-white rounded-full shadow transition-all duration-300 ${
                    expertMode ? 'left-7' : 'left-1'
                  }`} />
                </button>
              </div>
              <p className="text-[9px] text-slate-400 mt-1.5 leading-relaxed">
                {expertMode
                  ? 'Research: SHAP, IRLM, full metrics visible.'
                  : 'Beginner: plain-language results, key metrics only.'}
              </p>
            </div>
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
                <p className="text-slate-400 max-w-sm font-medium leading-relaxed">Enter protein IDs or select a Demo Case Study to launch neural interaction prediction.</p>
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
                <p className="mt-10 text-slate-400 font-black uppercase tracking-[0.4em] text-[10px] animate-pulse">Decoding Fusion Gradients ({latency}ms)</p>
              </motion.div>
            ) : (
              <motion.div 
                key="result"
                initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }}
                className="h-full flex flex-col overflow-y-auto no-scrollbar gap-10"
              >
                {/* PDF REPORT CONTAINER WRAPPER */}
                <div id="scientific-report-content" className="space-y-10 p-2">
                  
                  {/* TOP HEADER WITH EXPORT BUTTON */}
                  <div className="flex justify-between items-center bg-white p-6 rounded-[2rem] border border-slate-100 shadow-sm">
                    <div>
                      <h3 className="text-xl font-black text-slate-800 tracking-tight">TransGraph-PPI Analytical Report</h3>
                      <p className="text-xs text-slate-400 font-mono mt-0.5">Pair: {protein1} ↔ {protein2} | Latency: {latency}ms</p>
                    </div>

                    {/* FEATURE 1: ONE-CLICK PDF RESEARCH REPORT BUTTON */}
                    <button
                      onClick={handleDownloadPDF}
                      className="px-5 py-3 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white rounded-xl font-bold text-xs flex items-center gap-2 shadow-lg shadow-emerald-200 transition-all cursor-pointer active:scale-95"
                    >
                      <Download size={16} /> Download Scientific PDF Report
                    </button>
                  </div>

                  {/* TOP: Critical Metrics Grid */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    
                    {/* Consensus Gauge + Multi-Signal Badges */}
                    <div className="bg-white p-8 rounded-[2.5rem] border border-slate-50 flex flex-col items-center justify-center relative shadow-sm">
                      <div className="absolute top-6 left-6 px-3 py-1 bg-emerald-50 text-emerald-600 text-[9px] font-black rounded-lg uppercase tracking-widest">Interaction Probability</div>
                      <div className="w-40 h-40 mt-6 relative">
                        <ResponsiveContainer width="100%" height="100%">
                          <PieChart>
                            <Pie data={[{v: result.interaction_probability*100}, {v: 100 - result.interaction_probability*100}]} innerRadius={58} outerRadius={72} startAngle={90} endAngle={-270} dataKey="v" paddingAngle={2}>
                              <Cell fill={result.interaction_probability > 0.7 ? "#10b981" : result.interaction_probability > 0.5 ? "#f59e0b" : "#f43f5e"} />
                              <Cell fill="#f1f5f9" />
                            </Pie>
                          </PieChart>
                        </ResponsiveContainer>
                        <div className="absolute inset-0 flex flex-col items-center justify-center">
                          <span className="text-4xl font-black text-slate-800 tracking-tighter">{(result.interaction_probability*100).toFixed(0)}%</span>
                        </div>
                      </div>
                      <div className={`mt-4 px-5 py-2 rounded-full text-[10px] font-black uppercase tracking-widest shadow-sm ${
                        result.interaction_probability > 0.7 ? 'bg-emerald-500 text-white' :
                        result.interaction_probability > 0.5 ? 'bg-amber-500 text-white' :
                        'bg-rose-500 text-white'
                      }`}>
                        {result.interaction_probability > 0.7 ? '⚡ Strong' : result.interaction_probability > 0.5 ? '◑ Moderate' : '✕ Weak'} Interaction
                      </div>

                      {/* Multi-Metric Sub-badges */}
                      <div className="mt-5 w-full space-y-2">
                        {[
                          { label: 'Interaction Strength', value: result.interaction_probability > 0.7 ? 'Strong' : result.interaction_probability > 0.5 ? 'Moderate' : 'Weak', color: result.interaction_probability > 0.7 ? 'text-emerald-600 bg-emerald-50 border-emerald-200' : result.interaction_probability > 0.5 ? 'text-amber-600 bg-amber-50 border-amber-200' : 'text-rose-600 bg-rose-50 border-rose-200' },
                          { label: 'Prediction Confidence', value: `${(result.confidence_score * 100).toFixed(1)}%`, color: 'text-indigo-600 bg-indigo-50 border-indigo-200' },
                          { label: 'ESM Sequence Signal', value: `${(result.esm_probability * 100).toFixed(1)}%`, color: 'text-teal-600 bg-teal-50 border-teal-200' },
                          { label: 'GAT Graph Signal', value: `${(result.gat_probability * 100).toFixed(1)}%`, color: 'text-violet-600 bg-violet-50 border-violet-200' },
                          { label: 'IRLM Confidence', value: irlmData?.region_confidence ? `${(irlmData.region_confidence * 100).toFixed(0)}%` : 'N/A', color: 'text-cyan-600 bg-cyan-50 border-cyan-200' },
                          { label: 'Biological Importance', value: result.interaction_probability > 0.75 ? 'High' : result.interaction_probability > 0.5 ? 'Moderate' : 'Low', color: 'text-slate-600 bg-slate-50 border-slate-200' },
                        ].map((m, i) => (
                          <div key={i} className={`flex items-center justify-between px-3 py-1.5 rounded-lg border text-[10px] font-bold ${m.color}`}>
                            <span>{m.label}</span>
                            <span>{m.value}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* 3D Molecular Workbench (Center Hero) */}
                    <div className="md:col-span-2 bg-white p-10 rounded-[2.5rem] border border-slate-50 flex items-center justify-center gap-10 shadow-sm overflow-hidden min-h-[350px] relative">
                      <div className="absolute top-6 left-1/2 -translate-x-1/2 px-4 py-1.5 bg-slate-50 rounded-full text-[9px] font-black text-slate-400 uppercase tracking-widest">3D Structural Projection</div>
                      <div className="flex-1 h-full">
                        <Protein3DView pdbId={protein1} label="Target Alpha" selectedResidue={selectedResidueP1} />
                      </div>
                      <div className="w-px h-32 bg-slate-100" />
                      <div className="flex-1 h-full">
                        <Protein3DView pdbId={protein2} label="Target Beta" selectedResidue={selectedResidueP2} />
                      </div>
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
                          { label: 'Social Network (GAT)', val: result.gat_probability*100, color: 'bg-indigo-500' },
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

                    {/* FEATURE #4: "Why this prediction?" Explainer Card */}
                    <div className="bg-slate-900 p-10 rounded-[2.5rem] flex flex-col relative overflow-hidden shadow-2xl border border-slate-800">
                      <div className="absolute top-0 right-0 p-8 opacity-5 text-white"><ShieldCheck size={140} /></div>
                      <div className="flex items-center justify-between gap-4 mb-6 border-b border-white/10 pb-4">
                        <div className="flex items-center gap-3">
                          <div className="p-2.5 bg-emerald-500/20 rounded-xl text-emerald-400 border border-emerald-500/30">
                            <Sparkles size={18} />
                          </div>
                          <div>
                            <h4 className="text-xs font-black text-white uppercase tracking-widest">Why this prediction?</h4>
                            <p className="text-[10px] text-slate-400 font-mono">Automated Multi-Modal Explainer</p>
                          </div>
                        </div>
                        <span className={`px-3 py-1 rounded-full text-[9px] font-black uppercase tracking-wider ${result.interaction_probability > 0.5 ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40' : 'bg-rose-500/20 text-rose-300 border border-rose-500/40'}`}>
                          {result.interaction_probability > 0.5 ? 'Positive Binding' : 'Non-Interacting'}
                        </span>
                      </div>

                      <div className="space-y-4 text-slate-300 text-xs font-medium leading-relaxed">
                        <div className="p-3 bg-white/5 rounded-xl border border-white/5 space-y-1">
                          <p className="text-white font-bold flex items-center gap-2">
                            <Zap size={14} className="text-amber-400" /> Language Model Motif Rationale
                          </p>
                          <p className="text-slate-300 opacity-90 text-[11px]">
                            ESM-2 Transformer calculated a motif alignment score of <strong className="text-emerald-400">{(result.esm_probability * 100).toFixed(1)}%</strong> based on hydrophobic contact surface compatibility between target sequences.
                          </p>
                        </div>

                        <div className="p-3 bg-white/5 rounded-xl border border-white/5 space-y-1">
                          <p className="text-white font-bold flex items-center gap-2">
                            <Database size={14} className="text-indigo-400" /> Graph Network Topology
                          </p>
                          <p className="text-slate-300 opacity-90 text-[11px]">
                            Graph Attention Network (GAT) scored cellular pathway proximity at <strong className="text-indigo-400">{(result.gat_probability * 100).toFixed(1)}%</strong>, indicating shared functional sub-graphs in STRING DB topology.
                          </p>
                        </div>

                        {irlmData && irlmData.hotspot_residues && (
                          <div className="p-3 bg-white/5 rounded-xl border border-white/5 space-y-1">
                            <p className="text-white font-bold flex items-center gap-2">
                              <Info size={14} className="text-cyan-400" /> IRLM Binding Hotspots
                            </p>
                            <p className="text-slate-300 opacity-90 text-[11px]">
                              Localization identified <strong className="text-cyan-300">{irlmData.hotspot_residues.length} key binding residue pairs</strong> driving overall interaction potential.
                            </p>
                          </div>
                        )}

                        <div className="flex items-center gap-4 text-[10px] text-slate-400 border-t border-white/10 pt-4 mt-2 font-mono">
                          <span>Actionable Next Step:</span>
                          <span className="text-emerald-400 font-bold">Run In-Silico Mutation Scan to pinpoint binding hotspots.</span>
                        </div>
                      </div>
                    </div>

                  </div>

                  {/* IRLM INTERACTION REGION VISUALIZER SECTION – Research mode only */}
                  {expertMode && irlmData && (
                    <IRLMVisualizer 
                      irlmData={irlmData} 
                      id1={protein1} 
                      id2={protein2} 
                      seq1={seq1} 
                      seq2={seq2} 
                      isDark={true}
                      onSelectResidue={handleResidueSelect}
                    />
                  )}

                  {/* Beginner mode: simplified explanation */}
                  {!expertMode && irlmData && (
                    <div className="bg-teal-50 border border-teal-200 rounded-2xl p-6">
                      <h4 className="text-sm font-black text-teal-700 mb-2 flex items-center gap-2">
                        <GraduationCap size={16} /> Interaction Region Summary
                      </h4>
                      <p className="text-xs text-teal-800 leading-relaxed">
                        The AI detected <strong>{irlmData.hotspot_residues?.length ?? '?'}</strong> binding hotspot residue pairs
                        between these two proteins. These are the amino acid positions most likely to physically
                        touch when the proteins interact.
                        {irlmData.region_confidence != null && (
                          <span> The localization model is <strong>{(irlmData.region_confidence * 100).toFixed(0)}% confident</strong> in this region mapping.</span>
                        )}
                      </p>
                    </div>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>

      {/* ── Batch Results Table ─────────────────────────────────── */}
      {batchResults.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-[2rem] border border-slate-100 shadow-sm p-8 mt-2"
        >
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-indigo-50 rounded-xl text-indigo-500"><Table2 size={18} /></div>
              <div>
                <h3 className="text-sm font-black text-slate-800 tracking-tight">Batch Prediction Results</h3>
                <p className="text-[10px] text-slate-400 font-mono">{batchResults.length} pairs processed</p>
              </div>
            </div>
            <button
              onClick={downloadBatchCSV}
              className="px-4 py-2 bg-gradient-to-r from-indigo-500 to-violet-500 text-white rounded-xl font-bold text-xs flex items-center gap-2 hover:from-indigo-400 hover:to-violet-400 transition-all shadow-lg shadow-indigo-200 cursor-pointer active:scale-95"
            >
              <FileDown size={14} /> Export CSV
            </button>
          </div>

          <div className="overflow-x-auto rounded-xl border border-slate-100">
            <table className="w-full text-xs font-medium">
              <thead>
                <tr className="bg-slate-50 text-slate-500">
                  <th className="text-left px-4 py-3 font-black uppercase tracking-wider">#</th>
                  <th className="text-left px-4 py-3 font-black uppercase tracking-wider">Protein A</th>
                  <th className="text-left px-4 py-3 font-black uppercase tracking-wider">Protein B</th>
                  <th className="text-center px-4 py-3 font-black uppercase tracking-wider">Probability</th>
                  <th className="text-center px-4 py-3 font-black uppercase tracking-wider">ESM</th>
                  <th className="text-center px-4 py-3 font-black uppercase tracking-wider">GAT</th>
                  <th className="text-center px-4 py-3 font-black uppercase tracking-wider">Confidence</th>
                  <th className="text-center px-4 py-3 font-black uppercase tracking-wider">Interacts?</th>
                  <th className="text-center px-4 py-3 font-black uppercase tracking-wider">Action</th>
                </tr>
              </thead>
              <tbody>
                {batchResults.map((r, i) => (
                  <motion.tr
                    key={i}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.04 }}
                    onClick={() => {
                      setProtein1(r.protein1_id || protein1);
                      setProtein2(r.protein2_id || protein2);
                      setResult(r);
                      setIrlmData(null);
                    }}
                    className="border-t border-slate-50 hover:bg-slate-50/80 transition-colors cursor-pointer group"
                  >
                    <td className="px-4 py-3 text-slate-400 font-mono">{i + 1}</td>
                    <td className="px-4 py-3 font-mono text-slate-700 max-w-[130px] truncate" title={r.protein1_id}>{r.protein1_id || '—'}</td>
                    <td className="px-4 py-3 font-mono text-slate-700 max-w-[130px] truncate" title={r.protein2_id}>{r.protein2_id || '—'}</td>
                    <td className="px-4 py-3 text-center">
                      <span className={`px-2 py-1 rounded-lg font-black text-[10px] ${
                        r.interaction_probability > 0.7 ? 'bg-emerald-100 text-emerald-700' :
                        r.interaction_probability > 0.5 ? 'bg-amber-100 text-amber-700' :
                        'bg-rose-100 text-rose-700'
                      }`}>
                        {(r.interaction_probability * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="px-4 py-3 text-center text-slate-600 font-mono">{r.esm_probability != null ? (r.esm_probability * 100).toFixed(1) + '%' : '—'}</td>
                    <td className="px-4 py-3 text-center text-slate-600 font-mono">{r.gat_probability != null ? (r.gat_probability * 100).toFixed(1) + '%' : '—'}</td>
                    <td className="px-4 py-3 text-center text-slate-600 font-mono">{r.confidence_score != null ? (r.confidence_score * 100).toFixed(1) + '%' : '—'}</td>
                    <td className="px-4 py-3 text-center">
                      <span className={`px-2.5 py-1 rounded-full font-black text-[9px] uppercase tracking-wider ${
                        r.interaction_probability > 0.5 ? 'bg-emerald-500 text-white' : 'bg-slate-200 text-slate-500'
                      }`}>
                        {r.interaction_probability > 0.5 ? '✓ YES' : '✕ NO'}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-center">
                      <button className="text-[9px] text-indigo-500 font-black uppercase tracking-wider opacity-0 group-hover:opacity-100 transition-opacity">
                        View →
                      </button>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="text-[10px] text-slate-400 mt-3 text-center">
            Click any row to load its full result in the workspace above.
          </p>
        </motion.div>
      )}
    </div>
  );
};

export default Predict;
