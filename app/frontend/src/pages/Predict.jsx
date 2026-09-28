import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Search, Activity, BarChart3, Info, CheckCircle2, 
  XCircle, AlertCircle, Loader2, ChevronRight, 
  ShieldCheck, Cpu, Database, Terminal as TerminalIcon,
  Zap, ArrowRightLeft, LayoutGrid, Box, BookOpen, Download,
  Sparkles, Gauge, Server, Clock, UploadCloud, ToggleLeft, ToggleRight,
  GraduationCap, FlaskConical, Table2, FileDown, Camera, FileText, Layers, Image,
  Dna, Pill, Share2, GitCompare, Bot, Boxes, ArrowRight, ExternalLink, Compass
} from 'lucide-react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, 
  Tooltip, ResponsiveContainer, Cell, PieChart, Pie 
} from 'recharts';
import html2pdf from 'html2pdf.js';
import { ppiService } from '../services/api';
import Protein3DView from '../components/Protein3DView';
import ProteinInfoButton from '../components/ProteinInfoModal';
import GATAttentionPanel from '../components/GATAttentionPanel';

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
  const [selectedModel, setSelectedModel] = useState('graphsage'); // 'graphsage' | 'gat'
  const [selectedCase, setSelectedCase] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [selectedResidueP1, setSelectedResidueP1] = useState(null);
  const [selectedResidueP2, setSelectedResidueP2] = useState(null);
  const [error, setError] = useState(null);
  const [logs, setLogs] = useState([]);
  const [latency, setLatency] = useState(42);
  const logEndRef = useRef(null);
  
  // ── UI PAGE & EXPORT STATE ──────────────────────────────────
  const [activeResultPage, setActiveResultPage] = useState('probability'); // 'probability' | 'evidence' | 'discovery'
  const [exportingPdf, setExportingPdf] = useState(false);
  const [expertMode, setExpertMode] = useState(false);  // false = Beginner, true = Research
  const [batchResults, setBatchResults] = useState([]);   // batch CSV results
  const [batchLoading, setBatchLoading] = useState(false);
  const fileInputRef = useRef(null);

  const addLog = (msg, type = 'info') => {
    setLogs(prev => [...prev, { msg, type, time: new Date().toLocaleTimeString() }].slice(-10));
  };

  const handleResidueSelect = (resObj) => {
    if (!resObj) return;
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
    if (inputMode === 'sequence') {
      setInputMode('ids');
    }
    addLog(`Preset selected: ${caseObj.label}`, 'info');
  };

  const handlePredict = async (e) => {
    if (e) e.preventDefault();
    setLoading(true);
    setError(null);
    setLogs([]);

    const p1 = protein1.trim();
    const p2 = protein2.trim();
    if (!p1 || !p2) {
      setError("Both Protein 1 and Protein 2 identifiers are required.");
      setLoading(false);
      return;
    }
    const s1 = inputMode === 'sequence' ? seq1.trim() : null;
    const s2 = inputMode === 'sequence' ? seq2.trim() : null;

    const modelLabel = selectedModel === 'graphsage' 
      ? 'GraphSAGE Ensemble (Primary Final Model)' 
      : 'GAT Ensemble (Controlled Comparison)';
    addLog(`Dispatching prediction query to ${modelLabel}...`, "process");

    const startTime = performance.now();
    try {
      let response;
      if (selectedModel === 'gat') {
        try {
          response = await ppiService.predictGAT(p1, p2, s1, s2);
        } catch (gatErr) {
          if (!gatErr.response || gatErr.code === 'ERR_NETWORK') {
            throw new Error("Controlled Comparison GAT backend (Port 8001) is offline. Start it with 'python ESMGAT/backend/main.py' or switch to the Primary Final Model (GraphSAGE on Port 8000).");
          }
          throw gatErr;
        }
      } else {
        response = await ppiService.predict(p1, p2, s1, s2);
      }

      const elapsed = Math.round(performance.now() - startTime);
      setLatency(elapsed);
      addLog(`${modelLabel} completed in ${elapsed}ms.`, "success");

      const resData = {
        ...response.data,
        active_model: selectedModel,
        model_title: selectedModel === 'graphsage' ? 'ESM-2 + GraphSAGE + XGBoost' : 'ESM-2 + Standard GAT + XGBoost',
        model_role: selectedModel === 'graphsage' ? 'Primary Final Model' : 'Controlled Experimental Comparison',
        threshold: response.data.threshold ?? (selectedModel === 'graphsage' ? 0.45 : 0.50),
      };
      setResult(resData);
      setActiveResultPage('probability');

      try {
        localStorage.setItem('transgraph_last_prediction', JSON.stringify({
          p1: p1, p2: p2,
          model: selectedModel,
          prob: resData.interaction_probability,
          esm: resData.esm_probability,
          gat: resData.gat_probability,
          conf: resData.confidence_score
        }));
      } catch (_) {}
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || "Prediction failed.";
      addLog("Execution Fault: " + msg, "error");
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  // ── FULL SCIENTIFIC PDF EXPORT (ALL PAGES) ─────────────────
  const handleDownloadPDF = async () => {
    if (!result) return;
    setExportingPdf(true);
    addLog("Packaging Multi-Page Scientific PDF Report...", "process");

    try {
      const element = document.getElementById('scientific-report-pdf-content');
      if (!element) throw new Error("Report element missing.");

      // Temporarily reveal offscreen PDF element for rendering
      element.style.display = 'block';

      const opt = {
        margin: 0.3,
        filename: `TransGraph_PPI_Scientific_Report_${protein1}_${protein2}.pdf`,
        image: { type: 'jpeg', quality: 0.98 },
        html2canvas: { scale: 2, useCORS: true, logging: false },
        jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' },
        pagebreak: { mode: ['avoid-all', 'css', 'legacy'] }
      };

      await html2pdf().set(opt).from(element).save();
      addLog("Scientific PDF Report downloaded successfully.", "success");
    } catch (err) {
      console.error("PDF Export Error:", err);
      addLog("PDF Export failed: " + (err.message || 'Unknown error'), "error");
    } finally {
      const element = document.getElementById('scientific-report-pdf-content');
      if (element) element.style.display = 'none';
      setExportingPdf(false);
    }
  };

  // ── INDIVIDUAL CARD / FIGURE EXPORT FUNCTION ───────────────
  const handleExportCardFigure = async (cardId, figureTitle) => {
    const element = document.getElementById(cardId);
    if (!element) {
      addLog(`Figure element #${cardId} not found`, "error");
      return;
    }
    addLog(`Exporting ${figureTitle} figure...`, "process");
    try {
      const opt = {
        margin: 0.3,
        filename: `TransGraph_PPI_Figure_${figureTitle}_${protein1}_${protein2}.pdf`,
        image: { type: 'jpeg', quality: 0.98 },
        html2canvas: { scale: 2, useCORS: true, logging: false },
        jsPDF: { unit: 'in', format: 'letter', orientation: 'landscape' }
      };
      await html2pdf().set(opt).from(element).save();
      addLog(`Figure ${figureTitle} exported successfully.`, "success");
    } catch (err) {
      console.error("Export Figure Error:", err);
      addLog("Export figure error: " + err.message, "error");
    }
  };

  // ── BATCH CSV UPLOAD HANDLER ──────────────────────────────
  const handleBatchUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setBatchLoading(true);
    setBatchResults([]);
    setResult(null);
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
        if (res.data.length > 0) {
          setProtein1(pairs[0].protein1_id);
          setProtein2(pairs[0].protein2_id);
          setResult(res.data[0]);
          setActiveResultPage('probability');
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

  // ── BATCH CSV DOWNLOAD ────────────────────────────────────
  const downloadBatchCSV = useCallback(() => {
    if (!batchResults.length) return;
    const header = 'protein1_id,protein2_id,status,interaction_probability,esm_probability,gat_probability,confidence_score,interacts,error\n';
    const rows = batchResults.map(r => {
      const p1 = r.protein1_id || '';
      const p2 = r.protein2_id || '';
      const status = r.status || (r.error ? 'error' : 'success');
      const prob = r.interaction_probability != null ? r.interaction_probability.toFixed(4) : '';
      const esm = r.esm_probability != null ? r.esm_probability.toFixed(4) : '';
      const gat = r.gat_probability != null ? r.gat_probability.toFixed(4) : '';
      const conf = r.confidence_score != null ? r.confidence_score.toFixed(4) : '';
      const interacts = status === 'error' ? 'ERROR' : (r.interaction_probability > 0.5 ? 'YES' : 'NO');
      const err = r.error ? `"${r.error.replace(/"/g, '""')}"` : '';
      return `${p1},${p2},${status},${prob},${esm},${gat},${conf},${interacts},${err}`;
    }).join('\n');
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
      
      {/* Telemetry Header */}
      <div className="bg-slate-900 text-white rounded-2xl p-4 px-6 flex flex-wrap items-center justify-between gap-4 shadow-xl border border-slate-800">
        <div className="flex items-center gap-3">
          <div className="w-3 h-3 rounded-full bg-emerald-400 animate-ping" />
          <span className="text-xs font-black uppercase tracking-widest text-emerald-400">Live Model Telemetry</span>
          <span className="text-slate-500">|</span>
          <span className="text-xs font-semibold text-slate-300">
            {selectedModel === 'graphsage' 
              ? 'ESM-2 + GraphSAGE Ensemble (Primary Final Model)' 
              : 'ESM-2 + Standard GAT Ensemble (Controlled Comparison)'}
          </span>
        </div>
        <div className="flex items-center gap-6 text-xs font-mono text-slate-300">
          <div className="flex items-center gap-1.5">
            <Clock size={14} className="text-emerald-400" />
            <span>Inference: <strong className="text-white">{latency}ms</strong></span>
          </div>
          <div className="flex items-center gap-1.5">
            <Gauge size={14} className="text-indigo-400" />
            <span>ROC-AUC: <strong className="text-white">{selectedModel === 'graphsage' ? '0.9753' : '0.9582'}</strong></span>
          </div>
          <div className="flex items-center gap-1.5">
            <Server size={14} className="text-amber-400" />
            <span>Graph: <strong className="text-white">12,323 Nodes | 80,685 Edges</strong></span>
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
                <p className="text-[10px] font-black uppercase tracking-widest text-emerald-600">
                  {selectedModel === 'graphsage' ? 'Primary: GraphSAGE' : 'Comparison: Standard GAT'}
                </p>
              </div>
            </div>

            {/* Quick 1-Click Viva Demo Presets */}
            <div className="mb-4 bg-slate-50/80 border border-slate-200/80 rounded-2xl p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-black uppercase tracking-[0.15em] text-slate-500 flex items-center gap-1.5">
                  <Sparkles size={12} className="text-amber-500" /> Quick Demo Presets
                </span>
                <span className="text-[10px] text-slate-400 font-medium">1-click populate</span>
              </div>
              <div className="grid grid-cols-2 gap-2">
                {CASE_STUDIES.map((c, i) => (
                  <button
                    key={i}
                    type="button"
                    onClick={() => handleSelectCase(c)}
                    className={`p-2 text-left rounded-xl border transition-all text-xs flex flex-col justify-between ${
                      selectedCase === c.label
                        ? 'bg-emerald-50 border-emerald-400 text-emerald-900 shadow-xs ring-1 ring-emerald-400/30'
                        : 'bg-white hover:bg-slate-100/80 border-slate-200 text-slate-700 hover:border-slate-300'
                    }`}
                  >
                    <span className="font-bold text-[11px] truncate">{c.label}</span>
                    <span className="text-[9px] text-slate-400 font-mono mt-0.5 truncate">{c.p1.slice(0, 11)}... & {c.p2.slice(0, 11)}...</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Model Architecture Selector (PART 10) */}
            <div className="mb-4 space-y-1.5 bg-slate-50/80 border border-slate-200/80 rounded-2xl p-3">
              <div className="flex items-center justify-between">
                <label className="text-[10px] font-black text-slate-600 uppercase tracking-[0.15em] flex items-center gap-1.5">
                  <Cpu size={12} className="text-emerald-500" /> Model Architecture
                </label>
                <span className={`text-[9px] font-black px-2 py-0.5 rounded-full uppercase tracking-wider ${
                  selectedModel === 'graphsage' 
                    ? 'bg-emerald-100 text-emerald-800 border border-emerald-300' 
                    : 'bg-amber-100 text-amber-800 border border-amber-300'
                }`}>
                  {selectedModel === 'graphsage' ? 'Primary' : 'Comparison'}
                </span>
              </div>
              <select
                value={selectedModel}
                onChange={(e) => {
                  const m = e.target.value;
                  setSelectedModel(m);
                  addLog(`Architecture selected: ${m === 'graphsage' ? 'GraphSAGE Ensemble (Primary Final Model)' : 'GAT Ensemble (Controlled Comparison)'}`, 'info');
                }}
                className="w-full bg-white border border-slate-200 rounded-xl px-3 py-2 text-xs text-slate-800 font-bold focus:border-emerald-500 outline-none cursor-pointer shadow-xs transition-all"
              >
                <option value="graphsage">GraphSAGE Ensemble (Primary Final Model)</option>
                <option value="gat">GAT Ensemble (Controlled Comparison)</option>
              </select>
              <div className="text-[10px] text-slate-500 font-medium leading-relaxed pt-0.5">
                {selectedModel === 'graphsage' ? (
                  <span>⭐ <strong>Primary Final Model:</strong> ESM-2 + GraphSAGE + XGBoost (Acc: 93.10%, ROC-AUC: 0.9753). Port 8000.</span>
                ) : (
                  <span>🔬 <strong>Controlled Comparison:</strong> ESM-2 + Standard GAT + XGBoost (Acc: 89.79%, ROC-AUC: 0.9582). Port 8001.</span>
                )}
              </div>
            </div>

            {/* Input Method Dropdown Selector */}
            <div className="mb-4 space-y-1">
              <label className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] ml-1 flex items-center gap-1">
                <Layers size={12} className="text-emerald-500" /> Custom Input Mode
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

            {/* Batch Upload */}
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

            {/* Mode Toggle */}
            <div className="border-t border-slate-100 pt-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {expertMode ? <FlaskConical size={14} className="text-violet-500" /> : <GraduationCap size={14} className="text-teal-500" />}
                  <span className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-500">
                    {expertMode ? 'Research Mode (Expert)' : 'Explorer Mode (General Audience)'}
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() => setExpertMode(v => !v)}
                  className={`relative w-12 h-6 rounded-full transition-colors duration-300 ${
                    expertMode ? 'bg-violet-500' : 'bg-teal-400'
                  }`}
                  title="Toggle between General Audience Explorer Mode and Expert Research Mode"
                >
                  <div className={`absolute top-1 w-4 h-4 bg-white rounded-full shadow transition-all duration-300 ${
                    expertMode ? 'left-7' : 'left-1'
                  }`} />
                </button>
              </div>
              <p className="text-[9px] text-slate-400 mt-1.5 leading-relaxed">
                {expertMode
                  ? 'Research Mode: Deep SHAP matrices, GraphSAGE graph signal, & raw metrics.'
                  : 'Explorer Mode: Simple plain-language explanations & key visual insights.'}
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

        {/* RIGHT: Multi-Page Analytical Workspace */}
        <main className="flex-1 bg-white/50 rounded-[3rem] border border-slate-100 p-8 relative overflow-hidden flex flex-col min-w-0 shadow-sm">
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
                key="result-workspace"
                initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }}
                className="h-full flex flex-col min-h-0"
              >
                {/* ── TOP NAV BAR: PAGE-WISE TABS & PDF EXPORT BUTTON ── */}
                <div className="flex flex-wrap items-center justify-between gap-4 bg-slate-900 text-white p-3.5 px-6 rounded-[2rem] border border-slate-800 shadow-xl mb-6">
                  
                  {/* Tabs Selector */}
                  <div className="flex items-center gap-1.5 bg-slate-950 p-1.5 rounded-2xl border border-slate-800">
                    <button
                      onClick={() => setActiveResultPage('probability')}
                      className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-black transition-all cursor-pointer ${
                        activeResultPage === 'probability'
                          ? 'bg-gradient-to-r from-emerald-500 to-teal-500 text-white shadow-lg shadow-emerald-500/20'
                          : 'text-slate-400 hover:text-white hover:bg-slate-900'
                      }`}
                    >
                      <Gauge size={15} />
                      <span>Page 1: Probability</span>
                    </button>

                    <button
                      onClick={() => setActiveResultPage('evidence')}
                      className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-black transition-all cursor-pointer ${
                        activeResultPage === 'evidence'
                          ? 'bg-gradient-to-r from-indigo-500 to-violet-500 text-white shadow-lg shadow-indigo-500/20'
                          : 'text-slate-400 hover:text-white hover:bg-slate-900'
                      }`}
                    >
                      <BarChart3 size={15} />
                      <span>Page 2: Evidence Weightage</span>
                    </button>

                    <button
                      onClick={() => setActiveResultPage('discovery')}
                      className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-black transition-all cursor-pointer ${
                        activeResultPage === 'discovery'
                          ? 'bg-gradient-to-r from-amber-500 to-rose-500 text-white shadow-lg shadow-amber-500/20'
                          : 'text-slate-400 hover:text-white hover:bg-slate-900'
                      }`}
                    >
                      <Sparkles size={15} />
                      <span>Page 3: Discovery Hub</span>
                    </button>
                  </div>

                  {/* Main Scientific PDF Download Button */}
                  <button
                    onClick={handleDownloadPDF}
                    disabled={exportingPdf}
                    className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl font-bold text-xs flex items-center gap-2 shadow-sm hover:shadow transition-all cursor-pointer active:scale-95 disabled:opacity-50"
                  >
                    {exportingPdf ? <Loader2 size={15} className="animate-spin" /> : <Download size={15} />}
                    <span>{exportingPdf ? 'Exporting...' : 'Download PDF'}</span>
                  </button>
                </div>

                {/* ── PAGE CONTENT AREA ── */}
                <div className="flex-1 overflow-y-auto no-scrollbar pr-1">
                  <AnimatePresence mode="wait">
                    
                    {/* PAGE 1: PROBABILITY METRICS & 3D STRUCTURE */}
                    {activeResultPage === 'probability' && (
                      <motion.div
                        key="page-1"
                        initial={{ opacity: 0, y: 15 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -15 }}
                        className="space-y-6"
                      >
                        <div className="flex justify-between items-center bg-white p-5 px-7 rounded-[2rem] border border-slate-100 shadow-sm">
                          <div>
                            <div className="flex items-center gap-2 mb-1">
                              <span className="text-[10px] font-black uppercase tracking-widest text-emerald-500">Page 1 of 3</span>
                              <span className={`px-2.5 py-0.5 rounded-full text-[9px] font-black uppercase tracking-wider ${
                                (result.active_model ?? selectedModel) === 'graphsage'
                                  ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                                  : 'bg-amber-50 text-amber-700 border border-amber-200'
                              }`}>
                                {(result.active_model ?? selectedModel) === 'graphsage' ? '⭐ Primary Final Model' : '🔬 Controlled Experimental Comparison'}
                              </span>
                            </div>
                            <h3 className="text-xl font-black text-slate-800 tracking-tight">
                              Interaction Probability & 3D Structural View
                            </h3>
                            <p className="text-xs text-slate-500 font-mono mt-0.5">
                              Model: <strong className="text-slate-700">{(result.active_model ?? selectedModel) === 'graphsage' ? 'ESM-2 + GraphSAGE + XGBoost' : 'ESM-2 + Standard GAT + XGBoost'}</strong> | Pair: <strong className="text-slate-800">{protein1}</strong> ↔ <strong className="text-slate-800">{protein2}</strong>
                            </p>
                          </div>
                          <button
                            onClick={() => handleExportCardFigure('page-1-container', 'Probability_and_Structure')}
                            className="px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl font-bold text-xs flex items-center gap-2 transition-all cursor-pointer"
                          >
                            <Camera size={14} className="text-emerald-600" /> Export Figure
                          </button>
                        </div>

                        {!expertMode && (
                          <div className="bg-emerald-50/80 border border-emerald-200/80 p-4 px-6 rounded-2xl flex items-start gap-3 text-slate-700 text-xs leading-relaxed shadow-sm">
                            <Sparkles size={18} className="text-emerald-500 shrink-0 mt-0.5" />
                            <div>
                              <span className="font-bold text-emerald-900 block mb-0.5">Summary:</span>
                              Predicted interaction probability between <span className="font-bold text-slate-900">{protein1}</span> and <span className="font-bold text-slate-900">{protein2}</span> is <strong className="text-emerald-700">{((result.interaction_probability ?? 0) * 100).toFixed(1)}%</strong> using the <strong>{(result.active_model ?? selectedModel) === 'graphsage' ? 'GraphSAGE Ensemble (Primary Final Model)' : 'Standard GAT Ensemble (Controlled Comparison)'}</strong>.
                            </div>
                          </div>
                        )}

                        <div id="page-1-container" className="grid grid-cols-1 md:grid-cols-3 gap-6">
                          
                          {/* Consensus Gauge Card */}
                          <div id="card-probability-gauge" className="bg-white p-6 rounded-[2.5rem] border border-slate-100 flex flex-col items-center justify-between relative shadow-sm">
                            <div className="w-full flex items-center justify-between">
                              <span className="px-3 py-1 bg-emerald-50 text-emerald-600 text-[9px] font-black rounded-lg uppercase tracking-widest">Prediction Output</span>
                              <button
                                onClick={() => handleExportCardFigure('card-probability-gauge', 'Probability_Gauge')}
                                className="text-slate-400 hover:text-emerald-600 transition-colors p-1"
                                title="Export Gauge Figure"
                              >
                                <Camera size={14} />
                              </button>
                            </div>

                            <div className="w-44 h-44 my-4 relative">
                              <ResponsiveContainer width="100%" height="100%">
                                <PieChart>
                                  <Pie data={[{v: result.interaction_probability*100}, {v: 100 - result.interaction_probability*100}]} innerRadius={60} outerRadius={76} startAngle={90} endAngle={-270} dataKey="v" paddingAngle={2}>
                                    <Cell fill={result.interaction_probability >= (result.threshold ?? (selectedModel === 'gat' ? 0.50 : 0.45)) ? "#10b981" : "#f43f5e"} />
                                    <Cell fill="#f1f5f9" />
                                  </Pie>
                                </PieChart>
                              </ResponsiveContainer>
                              <div className="absolute inset-0 flex flex-col items-center justify-center">
                                <span className="text-4xl font-black text-slate-800 tracking-tighter">{(result.interaction_probability*100).toFixed(1)}%</span>
                                <span className="text-[9px] font-black uppercase text-slate-400 tracking-wider">Probability</span>
                              </div>
                            </div>

                            {/* Prediction Label using exact backend threshold */}
                            {(() => {
                              const threshold = result.threshold ?? (selectedModel === 'gat' ? 0.50 : 0.45);
                              const isLikely = result.interaction_probability >= threshold;
                              return (
                                <div className={`w-full text-center py-2.5 px-4 rounded-2xl text-xs font-black uppercase tracking-wider shadow-sm flex items-center justify-center gap-2 ${
                                  isLikely ? 'bg-emerald-500 text-white shadow-emerald-200' : 'bg-slate-700 text-white shadow-slate-200'
                                }`}>
                                  {isLikely ? <CheckCircle2 size={16} /> : <XCircle size={16} />}
                                  <span>{isLikely ? 'Likely Interaction' : 'Unlikely Interaction'}</span>
                                  <span className="text-[10px] font-normal opacity-75 font-mono">(Thresh: {threshold.toFixed(2)})</span>
                                </div>
                              );
                            })()}

                            <div className="mt-4 w-full space-y-2">
                              {[
                                { 
                                  label: 'Classification', 
                                  value: result.interaction_probability >= (result.threshold ?? (selectedModel === 'gat' ? 0.50 : 0.45)) ? 'Likely Interaction' : 'Unlikely Interaction', 
                                  color: result.interaction_probability >= (result.threshold ?? (selectedModel === 'gat' ? 0.50 : 0.45)) ? 'text-emerald-700 bg-emerald-50 border-emerald-200' : 'text-slate-700 bg-slate-50 border-slate-200' 
                                },
                                { label: 'Model', value: (result.active_model ?? selectedModel) === 'graphsage' ? 'GraphSAGE Ensemble' : 'Standard GAT Ensemble', color: 'text-slate-700 bg-slate-50 border-slate-200' },
                                { label: 'Role', value: (result.active_model ?? selectedModel) === 'graphsage' ? 'Primary Final Model' : 'Controlled Comparison', color: (result.active_model ?? selectedModel) === 'graphsage' ? 'text-emerald-700 bg-emerald-50 border-emerald-200' : 'text-amber-700 bg-amber-50 border-amber-200' },
                                { label: 'Confidence Score', value: `${(result.confidence_score * 100).toFixed(1)}%`, color: 'text-indigo-600 bg-indigo-50 border-indigo-200' },
                                { label: 'ESM Sequence Signal', value: `${(result.esm_probability * 100).toFixed(1)}%`, color: 'text-teal-600 bg-teal-50 border-teal-200' },
                                { label: (result.active_model ?? selectedModel) === 'graphsage' ? 'GraphSAGE Graph Signal' : 'Standard GAT Graph Signal', value: `${(result.gat_probability * 100).toFixed(1)}%`, color: 'text-violet-600 bg-violet-50 border-violet-200' },
                              ].map((m, i) => (
                                <div key={i} className={`flex items-center justify-between px-3 py-1.5 rounded-xl border text-[10px] font-bold ${m.color}`}>
                                  <span>{m.label}</span>
                                  <span>{m.value}</span>
                                </div>
                              ))}
                            </div>
                          </div>

                          {/* 3D Structural Projection Card */}
                          <div id="card-3d-structure" className="md:col-span-2 bg-white p-6 md:p-8 rounded-[2.5rem] border border-slate-100 flex flex-col justify-between shadow-sm relative overflow-hidden w-full max-w-full text-slate-800 min-h-[500px]">
                            <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-100 w-full max-w-full">
                              <div className="flex items-center gap-2 min-w-0">
                                <span className="w-2.5 h-2.5 rounded-full bg-cyan-500 animate-pulse shrink-0" />
                                <span className="px-3 py-1 bg-cyan-50 text-cyan-600 border border-cyan-100 text-[10px] font-black rounded-lg uppercase tracking-widest truncate">
                                  3D Structural Projection Workbench
                                </span>
                              </div>
                              <button
                                onClick={() => handleExportCardFigure('card-3d-structure', '3D_Structural_Projection')}
                                className="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 rounded-lg text-xs font-bold flex items-center gap-1.5 transition-all cursor-pointer shrink-0"
                              >
                                <Camera size={13} className="text-cyan-600" /> Export Figure
                              </button>
                            </div>

                            <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-6 w-full max-w-full min-w-0 items-stretch my-2">
                              <div className="w-full max-w-full min-w-0 bg-slate-50/70 border border-slate-100 p-4 md:p-5 rounded-3xl flex flex-col justify-between">
                                <Protein3DView 
                                  pdbId={protein1} 
                                  fallbackPdbId="1tnr" 
                                  label={`Protein A: ${protein1}`} 
                                  selectedResidue={selectedResidueP1}
                                />
                              </div>
                              <div className="w-full max-w-full min-w-0 bg-slate-50/70 border border-slate-100 p-4 md:p-5 rounded-3xl flex flex-col justify-between">
                                <Protein3DView 
                                  pdbId={protein2} 
                                  fallbackPdbId="1a2y" 
                                  label={`Protein B: ${protein2}`} 
                                  selectedResidue={selectedResidueP2}
                                />
                              </div>
                            </div>
                          </div>

                          {/* ── DOWNSTREAM BIOLOGICAL INVESTIGATION DOCK ── */}
                          <div className="md:col-span-3 bg-gradient-to-r from-slate-900 via-slate-900 to-indigo-950 p-6 md:p-8 rounded-[2.5rem] border border-slate-800 text-white shadow-xl relative overflow-hidden">
                            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-white/10">
                              <div>
                                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-400 text-[10px] font-black uppercase tracking-widest border border-emerald-500/30 mb-2">
                                  <Sparkles size={12} /> Downstream Biological Workflows
                                </div>
                                <h4 className="text-lg md:text-xl font-black text-white tracking-tight">
                                  Next Research Actions for {protein1} ↔ {protein2}
                                </h4>
                                <p className="text-xs text-slate-400 mt-0.5">
                                  TransGraph-PPI is an integrated discovery platform. Select a downstream engine to continue your investigation:
                                </p>
                              </div>
                              <button
                                onClick={() => setActiveResultPage('discovery')}
                                className="px-4 py-2 bg-white/10 hover:bg-white/20 text-white text-xs font-bold rounded-xl border border-white/15 transition-all flex items-center gap-1.5 self-start md:self-auto cursor-pointer"
                              >
                                View Detailed Workbench <ChevronRight size={14} />
                              </button>
                            </div>

                            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mt-6">
                              {/* 1. 3D Structure */}
                              <Link
                                to={`/structure?protein=${protein1}`}
                                className="p-4 rounded-2xl bg-white/5 hover:bg-white/10 border border-white/10 hover:border-cyan-500/50 transition-all group flex flex-col justify-between"
                              >
                                <div className="flex items-start justify-between mb-3">
                                  <div className="p-2.5 rounded-xl bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
                                    <Boxes size={18} />
                                  </div>
                                  <span className="text-[10px] font-bold text-slate-400 group-hover:text-cyan-400 flex items-center gap-1 transition-colors">
                                    Open 3D <ArrowRight size={12} />
                                  </span>
                                </div>
                                <div>
                                  <p className="text-xs font-bold text-white group-hover:text-cyan-300 transition-colors">AlphaFold 3D Studio</p>
                                  <p className="text-[11px] text-slate-400 mt-1 leading-snug">
                                    Inspect full atomic CIF structures for {protein1} with Mol* viewer controls.
                                  </p>
                                </div>
                              </Link>

                              {/* 2. In-Silico Mutagenesis */}
                              <Link
                                to={`/mutation?p1=${protein1}&p2=${protein2}`}
                                className="p-4 rounded-2xl bg-white/5 hover:bg-white/10 border border-white/10 hover:border-teal-500/50 transition-all group flex flex-col justify-between"
                              >
                                <div className="flex items-start justify-between mb-3">
                                  <div className="p-2.5 rounded-xl bg-teal-500/20 text-teal-400 border border-teal-500/30">
                                    <Dna size={18} />
                                  </div>
                                  <span className="text-[10px] font-bold text-slate-400 group-hover:text-teal-400 flex items-center gap-1 transition-colors">
                                    Simulate <ArrowRight size={12} />
                                  </span>
                                </div>
                                <div>
                                  <p className="text-xs font-bold text-white group-hover:text-teal-300 transition-colors">In-Silico Mutagenesis</p>
                                  <p className="text-[11px] text-slate-400 mt-1 leading-snug">
                                    Scan point mutations on this complex to identify affinity destabilizing hotspots.
                                  </p>
                                </div>
                              </Link>

                              {/* 3. Drug Insights */}
                              <Link
                                to={`/drug-targets?q=${protein1}`}
                                className="p-4 rounded-2xl bg-white/5 hover:bg-white/10 border border-white/10 hover:border-indigo-500/50 transition-all group flex flex-col justify-between"
                              >
                                <div className="flex items-start justify-between mb-3">
                                  <div className="p-2.5 rounded-xl bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
                                    <Pill size={18} />
                                  </div>
                                  <span className="text-[10px] font-bold text-slate-400 group-hover:text-indigo-400 flex items-center gap-1 transition-colors">
                                    Screen <ArrowRight size={12} />
                                  </span>
                                </div>
                                <div>
                                  <p className="text-xs font-bold text-white group-hover:text-indigo-300 transition-colors">ChEMBL Drug Targets</p>
                                  <p className="text-[11px] text-slate-400 mt-1 leading-snug">
                                    Evaluate Therapeutic Target Priority Scores (TTPS) and approved drug leads for {protein1}.
                                  </p>
                                </div>
                              </Link>

                              {/* 4. Network Path */}
                              <Link
                                to={`/network?start=${protein1}&end=${protein2}`}
                                className="p-4 rounded-2xl bg-white/5 hover:bg-white/10 border border-white/10 hover:border-violet-500/50 transition-all group flex flex-col justify-between"
                              >
                                <div className="flex items-start justify-between mb-3">
                                  <div className="p-2.5 rounded-xl bg-violet-500/20 text-violet-400 border border-violet-500/30">
                                    <Share2 size={18} />
                                  </div>
                                  <span className="text-[10px] font-bold text-slate-400 group-hover:text-violet-400 flex items-center gap-1 transition-colors">
                                    Trace <ArrowRight size={12} />
                                  </span>
                                </div>
                                <div>
                                  <p className="text-xs font-bold text-white group-hover:text-violet-300 transition-colors">Interactome Path & Hubs</p>
                                  <p className="text-[11px] text-slate-400 mt-1 leading-snug">
                                    Trace shortest path and common interactors across the 12,000-node graph.
                                  </p>
                                </div>
                              </Link>

                              {/* 5. Compare WT vs Mutant */}
                              <Link
                                to={`/compare?p1=${protein1}&p2=${protein2}`}
                                className="p-4 rounded-2xl bg-white/5 hover:bg-white/10 border border-white/10 hover:border-amber-500/50 transition-all group flex flex-col justify-between"
                              >
                                <div className="flex items-start justify-between mb-3">
                                  <div className="p-2.5 rounded-xl bg-amber-500/20 text-amber-400 border border-amber-500/30">
                                    <GitCompare size={18} />
                                  </div>
                                  <span className="text-[10px] font-bold text-slate-400 group-hover:text-amber-400 flex items-center gap-1 transition-colors">
                                    Compare <ArrowRight size={12} />
                                  </span>
                                </div>
                                <div>
                                  <p className="text-xs font-bold text-white group-hover:text-amber-300 transition-colors">WT vs Mutant Comparison</p>
                                  <p className="text-[11px] text-slate-400 mt-1 leading-snug">
                                    Quantify exact probability delta between native and mutated sequence variants.
                                  </p>
                                </div>
                              </Link>

                              {/* 6. Bio-Copilot Assistant */}
                              <Link
                                to={`/assistant?q=${encodeURIComponent(`Explain the molecular mechanism and biological significance of the interaction between ${protein1} and ${protein2}`)}`}
                                className="p-4 rounded-2xl bg-white/5 hover:bg-white/10 border border-white/10 hover:border-rose-500/50 transition-all group flex flex-col justify-between"
                              >
                                <div className="flex items-start justify-between mb-3">
                                  <div className="p-2.5 rounded-xl bg-rose-500/20 text-rose-400 border border-rose-500/30">
                                    <Bot size={18} />
                                  </div>
                                  <span className="text-[10px] font-bold text-slate-400 group-hover:text-rose-400 flex items-center gap-1 transition-colors">
                                    Ask AI <ArrowRight size={12} />
                                  </span>
                                </div>
                                <div>
                                  <p className="text-xs font-bold text-white group-hover:text-rose-300 transition-colors">Bio-Copilot AI Consultation</p>
                                  <p className="text-[11px] text-slate-400 mt-1 leading-snug">
                                    Ask our domain LLM to synthesize disease linkages and cellular function for this pair.
                                  </p>
                                </div>
                              </Link>
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    )}

                    {/* PAGE 2: EVIDENCE WEIGHTAGE & EXPLAINER */}
                    {activeResultPage === 'evidence' && (
                      <motion.div
                        key="page-2"
                        initial={{ opacity: 0, y: 15 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -15 }}
                        className="space-y-6"
                      >
                        <div className="flex justify-between items-center bg-white p-5 px-7 rounded-[2rem] border border-slate-100 shadow-sm">
                          <div>
                            <span className="text-[10px] font-black uppercase tracking-widest text-indigo-500">Page 2 of 3</span>
                            <h3 className="text-xl font-black text-slate-800 tracking-tight">Evidence Weightage & Multi-Modal Rationale</h3>
                            <p className="text-xs text-slate-400 font-mono mt-0.5">Feature Importance & Decision Explainer</p>
                          </div>
                          <button
                            onClick={() => handleExportCardFigure('page-2-container', 'Evidence_Weightage')}
                            className="px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl font-bold text-xs flex items-center gap-2 transition-all cursor-pointer"
                          >
                            <Camera size={14} className="text-indigo-600" /> Export Figure
                          </button>
                        </div>

                        {!expertMode && (
                          <div className="bg-indigo-50/80 border border-indigo-200/80 p-4 px-6 rounded-2xl flex items-start gap-3 text-slate-700 text-xs leading-relaxed shadow-sm">
                            <Sparkles size={18} className="text-indigo-500 shrink-0 mt-0.5" />
                            <div>
                              <span className="font-bold text-indigo-900 block mb-0.5">Explorer Summary (How the AI makes its decision):</span>
                              The model evaluates three biological evidence sources: <strong>Amino Acid Sequences</strong>, <strong>3D Molecular Shapes</strong>, and <strong>Biological Graph Networks</strong>. The pie chart below shows how much weight each source contributed to this prediction.
                            </div>
                          </div>
                        )}

                        <div id="page-2-container" className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                          
                          {/* Evidence Weighting Card */}
                          <div id="card-evidence-weighting" className="bg-white p-8 rounded-[2.5rem] border border-slate-100 flex flex-col justify-between shadow-sm">
                            <div className="flex items-center justify-between mb-8">
                              <div className="flex items-center gap-3">
                                <div className="p-2.5 bg-indigo-50 rounded-xl text-indigo-500"><LayoutGrid size={18} /></div>
                                <div>
                                  <h4 className="text-xs font-black text-slate-800 uppercase tracking-widest">Expert Evidence Weighting</h4>
                                  <p className="text-[10px] text-slate-400">Ensemble Sub-network Contributions</p>
                                </div>
                              </div>
                              <button
                                onClick={() => handleExportCardFigure('card-evidence-weighting', 'Evidence_Weighting_Bars')}
                                className="text-slate-400 hover:text-indigo-600 p-1 transition-colors"
                              >
                                <Camera size={14} />
                              </button>
                            </div>

                            <div className="space-y-8 flex-1 flex flex-col justify-center">
                              {[
                                { label: 'Protein Language Model (ESM-2)', val: result.esm_probability*100, color: 'bg-emerald-500', desc: 'Sequence embedding compatibility score' },
                                { label: 'Network Topology (GraphSAGE)', val: result.gat_probability*100, color: 'bg-indigo-500', desc: 'Graph centrality & neighborhood interaction score' },
                                { label: 'Jury Consensus Agreement', val: result.confidence_score*100, color: 'bg-amber-500', desc: 'Model variance & prediction stability factor' }
                              ].map((sig, i) => (
                                <div key={i} className="space-y-2">
                                  <div className="flex justify-between items-center text-[10px] font-black">
                                    <span className="text-slate-500 uppercase tracking-[0.15em]">{sig.label}</span>
                                    <span className="text-slate-800 font-mono text-xs">{sig.val.toFixed(1)}%</span>
                                  </div>
                                  <div className="h-3 w-full bg-slate-100 rounded-full overflow-hidden shadow-inner">
                                    <motion.div 
                                      initial={{ width: 0 }} animate={{ width: `${sig.val}%` }}
                                      transition={{ duration: 1, delay: i*0.2 }}
                                      className={`h-full rounded-full ${sig.color} shadow-md`}
                                    />
                                  </div>
                                  <p className="text-[10px] text-slate-400 italic">{sig.desc}</p>
                                </div>
                              ))}
                            </div>
                          </div>

                          {/* Explainer Card */}
                          <div id="card-explainer-rationale" className="bg-slate-900 p-8 rounded-[2.5rem] flex flex-col justify-between shadow-2xl border border-slate-800 text-white relative overflow-hidden">
                            <div className="absolute top-0 right-0 p-8 opacity-5 text-white pointer-events-none"><ShieldCheck size={140} /></div>
                            
                            <div className="flex items-center justify-between border-b border-white/10 pb-4 mb-6">
                              <div className="flex items-center gap-3">
                                <div className="p-2.5 bg-emerald-500/20 rounded-xl text-emerald-400 border border-emerald-500/30">
                                  <Sparkles size={18} />
                                </div>
                                <div>
                                  <h4 className="text-xs font-black text-white uppercase tracking-widest">Why this prediction?</h4>
                                  <p className="text-[10px] text-slate-400 font-mono">Automated Multi-Modal Rationale</p>
                                </div>
                              </div>
                              <div className="flex items-center gap-2">
                                <button
                                  onClick={() => handleExportCardFigure('card-explainer-rationale', 'Prediction_Rationale')}
                                  className="text-slate-400 hover:text-white p-1 transition-colors"
                                >
                                  <Camera size={14} />
                                </button>
                                <span className={`px-3 py-1 rounded-full text-[9px] font-black uppercase tracking-wider ${result.interaction_probability >= (result.threshold ?? (selectedModel === 'gat' ? 0.50 : 0.45)) ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40' : 'bg-rose-500/20 text-rose-300 border border-rose-500/40'}`}>
                                  {result.interaction_probability >= (result.threshold ?? (selectedModel === 'gat' ? 0.50 : 0.45)) ? 'Likely Interaction' : 'Unlikely Interaction'}
                                </span>
                              </div>
                            </div>

                            <div className="space-y-4 text-slate-300 text-xs font-medium leading-relaxed">
                              <div className="p-3.5 bg-white/5 rounded-2xl border border-white/5 space-y-1">
                                <p className="text-white font-bold flex items-center gap-2">
                                  <Zap size={14} className="text-amber-400" /> Language Model Motif Rationale
                                </p>
                                <p className="text-slate-300 opacity-90 text-[11px]">
                                  ESM-2 Transformer calculated a motif alignment score of <strong className="text-emerald-400">{(result.esm_probability * 100).toFixed(1)}%</strong> based on hydrophobic contact surface compatibility between target sequences.
                                </p>
                              </div>

                              <div className="p-3.5 bg-white/5 rounded-2xl border border-white/5 space-y-1">
                                <p className="text-white font-bold flex items-center gap-2">
                                  <Database size={14} className="text-indigo-400" /> Graph Network Topology
                                </p>
                                <p className="text-slate-300 opacity-90 text-[11px]">
                                  The {(result.active_model ?? selectedModel) === 'graphsage' ? 'GraphSAGE' : 'Standard GAT'} graph model scored interaction-graph neighborhood proximity at <strong className="text-indigo-400">{(result.gat_probability * 100).toFixed(1)}%</strong>, capturing functional sub-graphs in STRING DB topology.
                                </p>
                              </div>
                            </div>

                            <div className="flex items-center gap-4 text-[10px] text-slate-400 border-t border-white/10 pt-4 mt-4 font-mono">
                              <span>Actionable Next Step:</span>
                              <Link 
                                to={`/mutation?p1=${protein1}&p2=${protein2}`}
                                className="text-emerald-400 hover:text-emerald-300 font-bold flex items-center gap-1 transition-colors"
                              >
                                Run In-Silico Mutation Scan to pinpoint binding hotspots <ArrowRight size={12} />
                              </Link>
                            </div>
                          </div>

                        </div>

                        {/* SHAP Explanation Section (PART 8) */}
                        <div className="bg-white p-7 rounded-[2.5rem] border border-slate-100 shadow-sm space-y-4">
                          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                            <div className="flex items-center gap-2.5">
                              <div className="p-2 bg-indigo-50 text-indigo-600 rounded-xl">
                                <BarChart3 size={18} />
                              </div>
                              <div>
                                <h4 className="text-sm font-black text-slate-800 tracking-tight">SHAP Meta-Feature Attribution</h4>
                                <p className="text-[10px] text-slate-400 font-medium">SHAP is used to interpret the contribution of the XGBoost meta-features to the final prediction.</p>
                              </div>
                            </div>
                            <span className="text-[10px] font-bold text-slate-500 bg-slate-50 px-2.5 py-1 rounded-lg border border-slate-100">
                              7 Meta-Features
                            </span>
                          </div>

                          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                            {[
                              { name: 'p_seq (ESM-2)', desc: 'Sequence prediction', idx: 0 },
                              { name: 'p_graph (' + ((result.active_model ?? selectedModel) === 'graphsage' ? 'GraphSAGE' : 'Standard GAT') + ')', desc: 'Calibrated graph prediction', idx: 1 },
                              { name: 'conf_seq', desc: '|p_seq - 0.5|', idx: 2 },
                              { name: 'conf_graph', desc: '|p_graph - 0.5|', idx: 3 },
                              { name: 'diff', desc: '|p_seq - p_graph|', idx: 4 },
                              { name: 'max_conf', desc: 'max(conf_seq, conf_graph)', idx: 5 },
                              { name: 'consensus', desc: 'p_seq × p_graph', idx: 6 },
                            ].map((f, i) => {
                              const val = (result.shap_explanations && result.shap_explanations[f.idx]) ?? 0.0;
                              const isPositive = val >= 0;
                              return (
                                <div key={i} className="p-3 bg-slate-50/80 rounded-2xl border border-slate-100 space-y-1">
                                  <div className="flex items-center justify-between text-[11px] font-bold">
                                    <span className="text-slate-700 truncate font-mono">{f.name}</span>
                                    <span className={`font-mono text-xs font-black ${isPositive ? 'text-emerald-600' : 'text-rose-600'}`}>
                                      {val > 0 ? `+${val.toFixed(4)}` : val.toFixed(4)}
                                    </span>
                                  </div>
                                  <p className="text-[10px] text-slate-400 truncate">{f.desc}</p>
                                  <div className="h-1.5 w-full bg-slate-200 rounded-full overflow-hidden">
                                    <div 
                                      className={`h-full rounded-full ${isPositive ? 'bg-emerald-500' : 'bg-rose-500'}`}
                                      style={{ width: `${Math.min(Math.abs(val) * 200, 100)}%` }}
                                    />
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                          <p className="text-[10px] text-slate-400 italic">
                            * Note: SHAP values interpret the attribution of XGBoost meta-features within the decision ensemble and do not claim to prove direct biological causation.
                          </p>
                        </div>

                        {/* MODEL COMPARISON TABLE (PART 15) */}
                        <div className="bg-white p-7 rounded-[2.5rem] border border-slate-100 shadow-sm space-y-4">
                          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-3">
                            <div>
                              <h4 className="text-sm font-black text-slate-800 tracking-tight">System Architecture Comparison: GraphSAGE vs Standard GAT</h4>
                              <p className="text-[10px] text-slate-400 font-medium">Evaluated under identical ESM-2 sequence embeddings and XGBoost stacking on the untouched 20,172-pair test set</p>
                            </div>
                            <span className="px-2.5 py-1 bg-indigo-50 text-indigo-700 rounded-lg text-[10px] font-bold self-start sm:self-auto border border-indigo-100">
                              Untouched Test Set (20,172 pairs)
                            </span>
                          </div>

                          <div className="overflow-x-auto rounded-2xl border border-slate-100">
                            <table className="w-full text-xs text-left">
                              <thead>
                                <tr className="bg-slate-50 text-slate-500 font-black uppercase text-[10px] tracking-wider border-b border-slate-100">
                                  <th className="py-3 px-5">Evaluation Metric</th>
                                  <th className="py-3 px-5 text-emerald-800 bg-emerald-50/70">
                                    GraphSAGE Ensemble<br />
                                    <span className="text-[9px] font-bold text-emerald-600 uppercase tracking-widest">Primary Final Model</span>
                                  </th>
                                  <th className="py-3 px-5 text-amber-800 bg-amber-50/70">
                                    Standard GAT Ensemble<br />
                                    <span className="text-[9px] font-bold text-amber-600 uppercase tracking-widest">Controlled Experimental Comparison</span>
                                  </th>
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-slate-100 font-mono text-slate-700">
                                <tr className="hover:bg-slate-50/50">
                                  <td className="py-3 px-5 font-sans font-bold text-slate-800">Accuracy</td>
                                  <td className="py-3 px-5 font-black text-emerald-700 bg-emerald-50/30">93.10%</td>
                                  <td className="py-3 px-5 text-slate-700 bg-amber-50/20">89.79%</td>
                                </tr>
                                <tr className="hover:bg-slate-50/50">
                                  <td className="py-3 px-5 font-sans font-bold text-slate-800">Precision</td>
                                  <td className="py-3 px-5 font-black text-emerald-700 bg-emerald-50/30">0.9448</td>
                                  <td className="py-3 px-5 text-slate-700 bg-amber-50/20">0.9205</td>
                                </tr>
                                <tr className="hover:bg-slate-50/50">
                                  <td className="py-3 px-5 font-sans font-bold text-slate-800">Recall</td>
                                  <td className="py-3 px-5 font-black text-emerald-700 bg-emerald-50/30">0.9155</td>
                                  <td className="py-3 px-5 text-slate-700 bg-amber-50/20">0.8710</td>
                                </tr>
                                <tr className="hover:bg-slate-50/50">
                                  <td className="py-3 px-5 font-sans font-bold text-slate-800">F1-Score</td>
                                  <td className="py-3 px-5 font-black text-emerald-700 bg-emerald-50/30">0.9300</td>
                                  <td className="py-3 px-5 text-slate-700 bg-amber-50/20">0.8951</td>
                                </tr>
                                <tr className="hover:bg-slate-50/50">
                                  <td className="py-3 px-5 font-sans font-bold text-slate-800">ROC-AUC</td>
                                  <td className="py-3 px-5 font-black text-emerald-700 bg-emerald-50/30">0.9753</td>
                                  <td className="py-3 px-5 text-slate-700 bg-amber-50/20">0.9582</td>
                                </tr>
                                <tr className="hover:bg-slate-50/50">
                                  <td className="py-3 px-5 font-sans font-bold text-slate-800">PR-AUC</td>
                                  <td className="py-3 px-5 font-black text-emerald-700 bg-emerald-50/30">0.9801</td>
                                  <td className="py-3 px-5 text-slate-700 bg-amber-50/20">0.9627</td>
                                </tr>
                              </tbody>
                            </table>
                          </div>
                        </div>

                        {result.attention_explanation ? (
                          <GATAttentionPanel attention={result.attention_explanation} />
                        ) : (
                          <div className="bg-slate-50 border border-slate-200 p-4 px-6 rounded-2xl text-xs text-slate-500 flex items-start gap-2">
                            <Info size={14} className="shrink-0 mt-0.5 text-violet-500" />
                            <span>GraphSAGE has no attention mechanism. Select the <strong>ESM-2 + Standard GAT + XGBoost</strong> model to see which graph neighbours the GAT attended to for this pair.</span>
                          </div>
                        )}
                      </motion.div>
                    )}

                    {/* PAGE 3: INTEGRATED BIOLOGICAL DISCOVERY HUB & DOWNSTREAM WORKBENCH */}
                    {activeResultPage === 'discovery' && (
                      <motion.div
                        key="page-3"
                        initial={{ opacity: 0, y: 15 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -15 }}
                        className="space-y-6"
                      >
                        <div className="flex justify-between items-center bg-white p-5 px-7 rounded-[2rem] border border-slate-100 shadow-sm">
                          <div>
                            <span className="text-[10px] font-black uppercase tracking-widest text-amber-500">Page 3 of 3</span>
                            <h3 className="text-xl font-black text-slate-800 tracking-tight">Downstream Biological Discovery Hub</h3>
                            <p className="text-xs text-slate-400 font-mono mt-0.5">Automated Multi-Engine Pipelines for {protein1} ↔ {protein2}</p>
                          </div>
                          <button
                            onClick={() => handleExportCardFigure('page-3-container', 'Downstream_Discovery_Hub')}
                            className="px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl font-bold text-xs flex items-center gap-2 transition-all cursor-pointer"
                          >
                            <Camera size={14} className="text-amber-600" /> Export Figure
                          </button>
                        </div>

                        {!expertMode && (
                          <div className="bg-amber-50/80 border border-amber-200/80 p-4 px-6 rounded-2xl flex items-start gap-3 text-slate-700 text-xs leading-relaxed shadow-sm">
                            <Sparkles size={18} className="text-amber-500 shrink-0 mt-0.5" />
                            <div>
                              <span className="font-bold text-amber-900 block mb-0.5">Explorer Summary (What happens after predicting?):</span>
                              Interaction prediction is only the first step. TransGraph-PPI provides an end-to-end biological workbench: inspect atomic 3D structures, mutate binding interfaces in silico, evaluate therapeutic druggability in ChEMBL, and trace topological signaling cascades.
                            </div>
                          </div>
                        )}

                        <div id="page-3-container" className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                          
                          {/* Card 1: 3D Structure Studio */}
                          <div className="bg-white p-6 rounded-[2.5rem] border border-slate-100 shadow-sm flex flex-col justify-between hover:shadow-md transition-all">
                            <div>
                              <div className="flex items-center justify-between mb-4">
                                <div className="p-3 bg-cyan-50 rounded-2xl text-cyan-600">
                                  <Boxes size={22} />
                                </div>
                                <span className="px-3 py-1 bg-cyan-50 text-cyan-700 border border-cyan-200 text-[10px] font-black uppercase rounded-lg">AlphaFold CIF</span>
                              </div>
                              <h4 className="text-base font-black text-slate-800 mb-1">AlphaFold 3D Molecular Studio</h4>
                              <p className="text-xs text-slate-500 mb-4 leading-relaxed">
                                Inspect full atomic 3D tertiary conformations in the interactive PDBe Mol* viewer with secondary structure highlights and residue coordinate inspection.
                              </p>
                              <div className="p-3 bg-slate-50 rounded-xl border border-slate-100 text-[11px] text-slate-600 font-mono space-y-1 mb-4">
                                <p><span className="text-slate-400">Target A:</span> <strong>{protein1}</strong></p>
                                <p><span className="text-slate-400">Target B:</span> <strong>{protein2}</strong></p>
                              </div>
                            </div>
                            <div className="flex gap-2">
                              <Link
                                to={`/structure?protein=${protein1}`}
                                className="flex-1 py-2.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded-xl text-center text-xs font-bold transition-all shadow-sm shadow-cyan-200 flex items-center justify-center gap-1.5"
                              >
                                View {protein1.slice(0, 8)} <ExternalLink size={12} />
                              </Link>
                              <Link
                                to={`/structure?protein=${protein2}`}
                                className="flex-1 py-2.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl text-center text-xs font-bold transition-all flex items-center justify-center gap-1.5"
                              >
                                View {protein2.slice(0, 8)} <ExternalLink size={12} />
                              </Link>
                            </div>
                          </div>

                          {/* Card 2: In-Silico Mutagenesis */}
                          <div className="bg-white p-6 rounded-[2.5rem] border border-slate-100 shadow-sm flex flex-col justify-between hover:shadow-md transition-all">
                            <div>
                              <div className="flex items-center justify-between mb-4">
                                <div className="p-3 bg-teal-50 rounded-2xl text-teal-600">
                                  <Dna size={22} />
                                </div>
                                <span className="px-3 py-1 bg-teal-50 text-teal-700 border border-teal-200 text-[10px] font-black uppercase rounded-lg">Affinity Shift</span>
                              </div>
                              <h4 className="text-base font-black text-slate-800 mb-1">In-Silico Mutation Scanner</h4>
                              <p className="text-xs text-slate-500 mb-4 leading-relaxed">
                                Introduce single or multiple point mutations along the amino acid sequence to determine whether binding affinity is preserved, enhanced, or disrupted.
                              </p>
                              <div className="p-3 bg-slate-50 rounded-xl border border-slate-100 text-[11px] text-slate-600 font-mono space-y-1 mb-4">
                                <p><span className="text-slate-400">Wild-Type Prob:</span> <strong>{((result.interaction_probability || 0) * 100).toFixed(1)}%</strong></p>
                                <p><span className="text-slate-400">Pair Mode:</span> <strong>{protein1} ↔ {protein2}</strong></p>
                              </div>
                            </div>
                            <Link
                              to={`/mutation?p1=${protein1}&p2=${protein2}`}
                              className="w-full py-2.5 bg-teal-600 hover:bg-teal-500 text-white rounded-xl text-center text-xs font-bold transition-all shadow-sm shadow-teal-200 flex items-center justify-center gap-1.5"
                            >
                              Launch Mutagenesis Scanner <ArrowRight size={14} />
                            </Link>
                          </div>

                          {/* Card 3: ChEMBL Drug Target Discovery */}
                          <div className="bg-white p-6 rounded-[2.5rem] border border-slate-100 shadow-sm flex flex-col justify-between hover:shadow-md transition-all">
                            <div>
                              <div className="flex items-center justify-between mb-4">
                                <div className="p-3 bg-indigo-50 rounded-2xl text-indigo-600">
                                  <Pill size={22} />
                                </div>
                                <span className="px-3 py-1 bg-indigo-50 text-indigo-700 border border-indigo-200 text-[10px] font-black uppercase rounded-lg">ChEMBL 34</span>
                              </div>
                              <h4 className="text-base font-black text-slate-800 mb-1">Therapeutic Target Profiler</h4>
                              <p className="text-xs text-slate-500 mb-4 leading-relaxed">
                                Cross-reference the EMBL-EBI ChEMBL database to retrieve FDA-approved drug indications, clinical phase candidates, and bioactivity assays for this target.
                              </p>
                              <div className="p-3 bg-slate-50 rounded-xl border border-slate-100 text-[11px] text-slate-600 font-mono space-y-1 mb-4">
                                <p><span className="text-slate-400">Priority Engine:</span> <strong>TTPS Multi-Factor</strong></p>
                                <p><span className="text-slate-400">Search Target:</span> <strong>{protein1}</strong></p>
                              </div>
                            </div>
                            <Link
                              to={`/drug-targets?q=${protein1}`}
                              className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-center text-xs font-bold transition-all shadow-sm shadow-indigo-200 flex items-center justify-center gap-1.5"
                            >
                              Explore Drug Candidates <ArrowRight size={14} />
                            </Link>
                          </div>

                          {/* Card 4: Interactome Topology & Shortest Path */}
                          <div className="bg-white p-6 rounded-[2.5rem] border border-slate-100 shadow-sm flex flex-col justify-between hover:shadow-md transition-all">
                            <div>
                              <div className="flex items-center justify-between mb-4">
                                <div className="p-3 bg-violet-50 rounded-2xl text-violet-600">
                                  <Share2 size={22} />
                                </div>
                                <span className="px-3 py-1 bg-violet-50 text-violet-700 border border-violet-200 text-[10px] font-black uppercase rounded-lg">STRING v12</span>
                              </div>
                              <h4 className="text-base font-black text-slate-800 mb-1">Interactome Network Topology</h4>
                              <p className="text-xs text-slate-500 mb-4 leading-relaxed">
                                Trace the shortest biological pathway, identify shared hub interactors, and evaluate degree and betweenness centralities on the global interactome graph.
                              </p>
                              <div className="p-3 bg-slate-50 rounded-xl border border-slate-100 text-[11px] text-slate-600 font-mono space-y-1 mb-4">
                                <p><span className="text-slate-400">Route:</span> <strong>{protein1} ➔ {protein2}</strong></p>
                                <p><span className="text-slate-400">Graph Size:</span> <strong>12,000+ Nodes</strong></p>
                              </div>
                            </div>
                            <div className="flex gap-2">
                              <Link
                                to={`/network?start=${protein1}&end=${protein2}`}
                                className="flex-1 py-2.5 bg-violet-600 hover:bg-violet-500 text-white rounded-xl text-center text-xs font-bold transition-all shadow-sm shadow-violet-200 flex items-center justify-center gap-1"
                              >
                                2D Subnetwork <ArrowRight size={12} />
                              </Link>
                              <Link
                                to={`/network-3d`}
                                className="flex-1 py-2.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl text-center text-xs font-bold transition-all flex items-center justify-center gap-1"
                              >
                                3D Globe <ArrowRight size={12} />
                              </Link>
                            </div>
                          </div>

                          {/* Card 5: Wildtype vs Mutant Comparison */}
                          <div className="bg-white p-6 rounded-[2.5rem] border border-slate-100 shadow-sm flex flex-col justify-between hover:shadow-md transition-all">
                            <div>
                              <div className="flex items-center justify-between mb-4">
                                <div className="p-3 bg-amber-50 rounded-2xl text-amber-600">
                                  <GitCompare size={22} />
                                </div>
                                <span className="px-3 py-1 bg-amber-50 text-amber-700 border border-amber-200 text-[10px] font-black uppercase rounded-lg">Side-by-Side</span>
                              </div>
                              <h4 className="text-base font-black text-slate-800 mb-1">WT vs Mutant Comparator</h4>
                              <p className="text-xs text-slate-500 mb-4 leading-relaxed">
                                Perform rigorous side-by-side benchmarking of the native wild-type protein complex versus mutated variants to observe exact probability delta shifts.
                              </p>
                              <div className="p-3 bg-slate-50 rounded-xl border border-slate-100 text-[11px] text-slate-600 font-mono space-y-1 mb-4">
                                <p><span className="text-slate-400">Baseline Prob:</span> <strong>{((result.interaction_probability || 0) * 100).toFixed(1)}%</strong></p>
                                <p><span className="text-slate-400">Delta Mode:</span> <strong>Point Perturbation</strong></p>
                              </div>
                            </div>
                            <Link
                              to={`/compare?p1=${protein1}&p2=${protein2}`}
                              className="w-full py-2.5 bg-amber-600 hover:bg-amber-500 text-white rounded-xl text-center text-xs font-bold transition-all shadow-sm shadow-amber-200 flex items-center justify-center gap-1.5"
                            >
                              Open Comparative Matrix <ArrowRight size={14} />
                            </Link>
                          </div>

                          {/* Card 6: Bio-Copilot AI Consultation */}
                          <div className="bg-white p-6 rounded-[2.5rem] border border-slate-100 shadow-sm flex flex-col justify-between hover:shadow-md transition-all">
                            <div>
                              <div className="flex items-center justify-between mb-4">
                                <div className="p-3 bg-rose-50 rounded-2xl text-rose-600">
                                  <Bot size={22} />
                                </div>
                                <span className="px-3 py-1 bg-rose-50 text-rose-700 border border-rose-200 text-[10px] font-black uppercase rounded-lg">Bio-LLM</span>
                              </div>
                              <h4 className="text-base font-black text-slate-800 mb-1">TransGraph Bio-Copilot</h4>
                              <p className="text-xs text-slate-500 mb-4 leading-relaxed">
                                Launch an AI consultation with our domain-specialized biological reasoning engine to synthesize disease etiology, clinical pathways, and functional context.
                              </p>
                              <div className="p-3 bg-slate-50 rounded-xl border border-slate-100 text-[11px] text-slate-600 font-mono space-y-1 mb-4">
                                <p><span className="text-slate-400">Context:</span> <strong>Predicted Interaction</strong></p>
                                <p><span className="text-slate-400">Query Pre-Fill:</span> <strong>Molecular Mechanism</strong></p>
                              </div>
                            </div>
                            <Link
                              to={`/assistant?q=${encodeURIComponent(`Explain the molecular mechanism and biological significance of the interaction between ${protein1} and ${protein2}`)}`}
                              className="w-full py-2.5 bg-rose-600 hover:bg-rose-500 text-white rounded-xl text-center text-xs font-bold transition-all shadow-sm shadow-rose-200 flex items-center justify-center gap-1.5"
                            >
                              Consult AI Copilot <ArrowRight size={14} />
                            </Link>
                          </div>

                        </div>
                      </motion.div>
                    )}

                    {/* End of Page Tabs */}

                  </AnimatePresence>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>

      {/* ── BATCH RESULTS TABLE ── */}
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
                  <th className="text-center px-4 py-3 font-black uppercase tracking-wider">GraphSAGE</th>
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
                      setActiveResultPage('probability');
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
        </motion.div>
      )}

      {/* ── HIDDEN PRINTABLE CONTAINER FOR SCIENTIFIC PDF EXPORT ── */}
      {result && (
        <div 
          id="scientific-report-pdf-content" 
          style={{ display: 'none', position: 'absolute', left: '-9999px', top: 0, width: '800px' }}
          className="bg-white text-slate-900 p-8 space-y-8 font-sans"
        >
          {/* Header */}
          <div className="border-b-2 border-emerald-600 pb-4 flex justify-between items-end">
            <div>
              <h1 className="text-2xl font-black text-slate-900 tracking-tight">TransGraph-PPI Scientific Report</h1>
              <p className="text-xs font-semibold text-emerald-600 uppercase tracking-widest">Deep Multimodal Protein Interaction & Localization Analysis</p>
            </div>
            <div className="text-right text-[10px] font-mono text-slate-500">
              <p>Timestamp: {new Date().toLocaleString()}</p>
              <p>Model: ESM2-GraphSAGE-XGBoost</p>
            </div>
          </div>

          {/* Pair Metadata Box */}
          <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 grid grid-cols-2 gap-4 text-xs font-mono">
            <div>
              <span className="text-slate-400 font-bold block uppercase text-[9px]">Target Protein A</span>
              <strong className="text-slate-800 text-sm">{protein1}</strong>
            </div>
            <div>
              <span className="text-slate-400 font-bold block uppercase text-[9px]">Target Protein B</span>
              <strong className="text-slate-800 text-sm">{protein2}</strong>
            </div>
          </div>

          {/* PAGE 1 PDF SECTION */}
          <div className="space-y-4">
            <h2 className="text-sm font-black uppercase tracking-wider text-emerald-700 border-b pb-1">1. Consensus Interaction Probability</h2>
            <div className="bg-emerald-50/50 p-6 rounded-2xl border border-emerald-200 flex items-center justify-between">
              <div>
                <span className="text-4xl font-black text-emerald-600">{(result.interaction_probability * 100).toFixed(1)}%</span>
                <span className="block text-xs font-bold text-slate-600 mt-1">Consensus Interaction Score</span>
              </div>
              <div className="text-right space-y-1 text-xs font-medium">
                <p>ESM-2 Language Signal: <strong>{(result.esm_probability * 100).toFixed(1)}%</strong></p>
                <p>GraphSAGE Graph Signal: <strong>{(result.gat_probability * 100).toFixed(1)}%</strong></p>
                <p>Ensemble Confidence: <strong>{(result.confidence_score * 100).toFixed(1)}%</strong></p>
              </div>
            </div>
          </div>

          {/* PAGE 2 PDF SECTION */}
          <div className="space-y-4 pt-4" style={{ pageBreakBefore: 'always' }}>
            <h2 className="text-sm font-black uppercase tracking-wider text-indigo-700 border-b pb-1">2. Evidence Weightage & Multi-Modal Explainer</h2>
            
            <div className="grid grid-cols-2 gap-4 text-xs">
              <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 space-y-2">
                <h3 className="font-bold text-slate-800">Sequence Embedding (ESM-2)</h3>
                <p className="text-slate-600 text-[11px] leading-relaxed">
                  Evaluates physical-chemical complementarity of hydrophobic surfaces across amino acid sequences.
                </p>
              </div>

              <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 space-y-2">
                <h3 className="font-bold text-slate-800">GraphSAGE Graph Model</h3>
                <p className="text-slate-600 text-[11px] leading-relaxed">
                  Scores the pair from neighborhood aggregation over the STRING-derived training interaction graph.
                </p>
              </div>
            </div>

            <div className="p-4 bg-slate-900 text-white rounded-xl space-y-2 text-xs">
              <h3 className="font-bold text-emerald-400 uppercase tracking-widest text-[10px]">Automated Rationale Summary</h3>
              <p className="text-slate-300 text-[11px] leading-relaxed">
                The prediction model concluded a {result.interaction_probability > 0.5 ? 'POSITIVE binding' : 'NON-INTERACTING'} classification with {(result.confidence_score * 100).toFixed(1)}% agreement between sequence representations and topological network signals.
              </p>
            </div>
          </div>

            <div className="pt-8 border-t border-slate-200 text-center text-[10px] text-slate-400 font-mono">
              End of Scientific Report — Generated by TransGraph-PPI Analytical Engine
            </div>
        </div>
      )}

    </div>
  );
};

export default Predict;
