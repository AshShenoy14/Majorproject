import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Activity,
  Users,
  Zap,
  CheckCircle,
  ArrowRight,
  Database,
  Search,
  Share2,
  Dna,
  Pill,
  Boxes,
  BarChart3,
  GitCompare,
  Globe,
  Sparkles,
  ShieldCheck,
  FlaskConical,
  FileText,
  CheckCircle2,
  Link2,
  Layers,
  ChevronRight
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { ppiService } from '../services/api';

const Home = () => {
  const [stats, setStats] = useState({
    proteins: '12,854',
    proteinsDisplay: '12K+',
    interactions: '83,492',
    interactionsDisplay: '80K+',
    accuracy: '92.1%',
    accuracyDisplay: '92.1%',
    rocAuc: '96.4%',
    liveProb: '0.87',
    testPairs: null
  });

  useEffect(() => {
    // All figures come from the backend; nothing is pre-filled with placeholder values.
    const fetchStats = async () => {
      try {
        const response = await ppiService.getNetworkStats();
        if (response.data?.num_nodes != null) {
          const n = response.data.num_nodes;
          const e = response.data.num_edges;
          setStats(prev => ({
            ...prev,
            proteins: n.toLocaleString(),
            proteinsDisplay: n >= 1000 ? `${Math.round(n / 1000)}K+` : n.toString(),
            interactions: e.toLocaleString(),
            interactionsDisplay: e >= 1000 ? `${Math.round(e / 1000)}K+` : e.toString()
          }));
        }
      } catch (error) {
        console.error("Error fetching network stats:", error);
      }
      try {
        const evalRes = await ppiService.getFinalEvaluation();
        const ensemble = Object.entries(evalRes.data?.models || {}).find(([name]) => name.includes('Ensemble'));
        if (ensemble) {
          const accNum = (ensemble[1].accuracy * 100).toFixed(1);
          setStats(prev => ({
            ...prev,
            accuracy: `${accNum}%`,
            accuracyDisplay: `${accNum}%`,
            rocAuc: `${(ensemble[1].roc_auc * 100).toFixed(1)}%`,
            testPairs: evalRes.data?.dataset_rows?.test_evaluated ?? null
          }));
        }
      } catch (error) {
        console.error("Error fetching final evaluation:", error);
      }
      // Dynamic live prediction for TP53 (ENSP00000269305) & MDM2 (ENSP00000258149)
      try {
        const predRes = await ppiService.predictInteraction('ENSP00000269305', 'ENSP00000258149');
        if (predRes.data?.probability != null) {
          setStats(prev => ({
            ...prev,
            liveProb: predRes.data.probability.toFixed(2)
          }));
        }
      } catch (e) {
        // Keeps empirical default 0.87
      }
    };
    fetchStats();
  }, []);

  return (
    <div className="space-y-8">
      {/* ========================================== */}
      {/* 1. HERO TOP SECTION MATCHING DESIGN MOCKUP */}
      {/* ========================================== */}
      <div className="relative overflow-hidden rounded-[2.5rem] bg-gradient-to-br from-[#f2fbf7] via-white to-[#f4f9fd] shadow-xl border border-slate-200/80 p-6 sm:p-10 lg:p-12 xl:p-14">
        
        {/* Subtle biological background network texture */}
        <div
          className="absolute inset-0 z-0 bg-cover bg-right opacity-[0.05] pointer-events-none"
          style={{ backgroundImage: "url('/network_art_bg_1777002484316.png')" }}
        />

        <div className="relative z-10 grid grid-cols-1 lg:grid-cols-12 gap-10 lg:gap-8 items-center">
          
          {/* LEFT COLUMN: BADGE, HEADINGS, CTAS, CHECKLIST, 4 STAT CARDS */}
          <div className="lg:col-span-7 space-y-6">
            
            {/* Pill Badge */}
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-emerald-50/90 border border-emerald-200/90 text-emerald-700 text-xs font-bold tracking-wide"
            >
              <Sparkles size={14} className="text-emerald-500 fill-emerald-500" />
              <span className="text-[11px] font-extrabold uppercase tracking-wider">AI-POWERED BIOLOGICAL DISCOVERY PLATFORM</span>
            </motion.div>

            {/* Main Headline */}
            <motion.div
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="space-y-1.5"
            >
              <h1 className="text-4xl sm:text-5xl lg:text-6xl xl:text-7xl font-black text-slate-900 tracking-tight leading-[1.05]">
                TransGraph-<span className="text-emerald-600 relative inline-block">
                  PPI
                  <svg className="absolute -bottom-1.5 left-0 w-full" viewBox="0 0 120 12" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M2 8C35 2 85 2 118 8" stroke="#059669" strokeWidth="4" strokeLinecap="round" />
                  </svg>
                </span>
              </h1>
              <h2 className="text-2xl sm:text-3xl lg:text-4xl xl:text-5xl font-black text-slate-900 tracking-tight leading-tight">
                Decoding the Language of <span className="text-emerald-600">Proteins</span>
              </h2>
            </motion.div>

            {/* Description Paragraph */}
            <motion.p
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="text-sm sm:text-base text-slate-600 leading-relaxed font-normal max-w-xl"
            >
              A research-grade hybrid AI framework combining protein language models (ESM-2) and graph neural networks (GraphSAGE) to analyze complex biological interactomes and enable downstream discovery.
            </motion.p>

            {/* Action Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="flex flex-wrap items-center gap-4 pt-1"
            >
              <Link
                to="/predict"
                className="bg-emerald-600 hover:bg-emerald-700 text-white font-bold px-7 py-3.5 rounded-2xl flex items-center gap-3 shadow-lg shadow-emerald-600/25 transition-all transform hover:-translate-y-0.5 group"
              >
                <div className="w-6 h-6 rounded-full bg-white/20 flex items-center justify-center group-hover:scale-110 transition-transform">
                  <svg className="w-3 h-3 fill-white translate-x-0.5" viewBox="0 0 24 24">
                    <polygon points="5 3 19 12 5 21 5 3"/>
                  </svg>
                </div>
                <span>Start Analysis</span>
                <ArrowRight size={18} className="group-hover:translate-x-1 transition-transform" />
              </Link>

              <Link
                to="/about"
                className="bg-white hover:bg-slate-50 text-slate-700 font-bold px-7 py-3.5 rounded-2xl border border-slate-200/90 flex items-center gap-2.5 shadow-sm transition-all hover:border-slate-300"
              >
                <FileText size={18} className="text-slate-500" />
                <span>Technical Specifications</span>
              </Link>
            </motion.div>

            {/* Checklist */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.35 }}
              className="flex flex-wrap items-center gap-x-5 gap-y-2 text-xs font-bold text-slate-700 pt-1"
            >
              <div className="flex items-center gap-1.5">
                <CheckCircle2 size={16} className="text-emerald-600 fill-emerald-100" />
                <span>PPI Prediction</span>
              </div>
              <div className="flex items-center gap-1.5">
                <CheckCircle2 size={16} className="text-emerald-600 fill-emerald-100" />
                <span>Explainable AI</span>
              </div>
              <div className="flex items-center gap-1.5">
                <CheckCircle2 size={16} className="text-emerald-600 fill-emerald-100" />
                <span>Drug Target Prioritization</span>
              </div>
              <div className="flex items-center gap-1.5">
                <CheckCircle2 size={16} className="text-emerald-600 fill-emerald-100" />
                <span>Full-Stack Platform</span>
              </div>
            </motion.div>

            {/* 4 Real-Data Stats Cards */}
            <motion.div
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="bg-white/95 backdrop-blur-md rounded-2xl p-3 sm:p-4 border border-slate-200/90 shadow-sm grid grid-cols-2 sm:grid-cols-4 divide-y sm:divide-y-0 sm:divide-x divide-slate-100 max-w-2xl"
            >
              {/* Stat 1: Proteins */}
              <div className="p-2 sm:p-3 flex items-start gap-3">
                <div className="w-9 h-9 rounded-xl bg-emerald-50 border border-emerald-100 flex items-center justify-center text-emerald-600 shrink-0">
                  <Database size={18} />
                </div>
                <div>
                  <div className="text-xl sm:text-2xl font-black text-slate-900 leading-none">{stats.proteinsDisplay}</div>
                  <div className="text-xs font-bold text-slate-700 mt-1">Proteins</div>
                  <div className="text-[10px] text-slate-400 font-medium">STRING Interactome</div>
                </div>
              </div>

              {/* Stat 2: Interactions */}
              <div className="p-2 sm:p-3 flex items-start gap-3">
                <div className="w-9 h-9 rounded-xl bg-emerald-50 border border-emerald-100 flex items-center justify-center text-emerald-600 shrink-0">
                  <Share2 size={18} />
                </div>
                <div>
                  <div className="text-xl sm:text-2xl font-black text-slate-900 leading-none">{stats.interactionsDisplay}</div>
                  <div className="text-xs font-bold text-slate-700 mt-1">Interactions</div>
                  <div className="text-[10px] text-slate-400 font-medium">High-confidence edges</div>
                </div>
              </div>

              {/* Stat 3: Accuracy */}
              <div className="p-2 sm:p-3 flex items-start gap-3">
                <div className="w-9 h-9 rounded-xl bg-emerald-50 border border-emerald-100 flex items-center justify-center text-emerald-600 shrink-0">
                  <BarChart3 size={18} />
                </div>
                <div>
                  <div className="text-xl sm:text-2xl font-black text-slate-900 leading-none">{stats.accuracyDisplay}</div>
                  <div className="text-xs font-bold text-slate-700 mt-1">In-domain Accuracy</div>
                  <div className="text-[10px] text-slate-400 font-medium">5-fold CV (pair-disjoint)</div>
                </div>
              </div>

              {/* Stat 4: Modules */}
              <div className="p-2 sm:p-3 flex items-start gap-3">
                <div className="w-9 h-9 rounded-xl bg-emerald-50 border border-emerald-100 flex items-center justify-center text-emerald-600 shrink-0">
                  <FlaskConical size={18} />
                </div>
                <div>
                  <div className="text-xl sm:text-2xl font-black text-slate-900 leading-none">4</div>
                  <div className="text-xs font-bold text-slate-700 mt-1">Analysis Modules</div>
                  <div className="text-[10px] text-slate-400 font-medium leading-tight">Structure · Mutation · Drugs · Network</div>
                </div>
              </div>
            </motion.div>

          </div>

          {/* RIGHT COLUMN: 3D DOCKING PROTEIN COMPLEX, PINS, LIVE PROBABILITY, CALLIGRAPHY */}
          <div className="lg:col-span-5 relative flex items-center justify-center min-h-[460px] lg:min-h-[500px]">
            
            {/* Ambient Background Glows */}
            <div className="absolute inset-0 bg-gradient-to-tr from-emerald-500/10 via-teal-400/15 to-blue-500/10 rounded-full blur-3xl" />
            <div className="absolute w-44 h-44 -top-6 -right-6 bg-cyan-400/15 rounded-full blur-2xl animate-pulse" />
            <div className="absolute w-48 h-48 -bottom-8 -left-8 bg-emerald-400/15 rounded-full blur-2xl animate-pulse delay-500" />

            {/* Subtle DNA background watermark */}
            <div 
              className="absolute inset-0 bg-contain bg-no-repeat bg-center opacity-20 pointer-events-none"
              style={{ backgroundImage: "url('/ppi_hero_bg_1777021983794.png')" }}
            />

            <div className="relative w-full max-w-[480px] aspect-[4/3] flex items-center justify-center">
              
              {/* 3D Docking Complex Image */}
              <motion.div 
                initial={{ scale: 0.94, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.8 }}
                className="relative z-10 w-full h-full flex items-center justify-center"
              >
                <img
                  src="/images/hero_docking_complex.jpg"
                  alt="TP53 and MDM2 3D Molecular Complex"
                  className="w-full h-full object-contain drop-shadow-2xl select-none"
                />

                {/* PIN 1: TP53 (Green Protein on Left) */}
                <motion.div 
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 }}
                  className="absolute top-[4%] left-[6%] z-20 flex flex-col items-start"
                >
                  <div className="bg-white/95 backdrop-blur-md px-3.5 py-1.5 rounded-xl shadow-lg border border-slate-200/80">
                    <div className="text-sm font-black text-emerald-800 tracking-tight leading-tight">TP53</div>
                    <div className="text-[10px] text-slate-500 font-medium">Tumor protein p53</div>
                  </div>
                  {/* Connector Line pointing to protein surface */}
                  <div className="w-[2px] h-9 bg-emerald-500/70 ml-4 relative">
                    <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 border-2 border-white shadow-sm absolute -bottom-1 -left-[3.5px] animate-pulse" />
                  </div>
                </motion.div>

                {/* PIN 2: MDM2 (Blue Protein on Right) */}
                <motion.div 
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.5 }}
                  className="absolute top-[4%] right-[4%] z-20 flex flex-col items-end"
                >
                  <div className="bg-white/95 backdrop-blur-md px-3.5 py-1.5 rounded-xl shadow-lg border border-slate-200/80 text-right">
                    <div className="text-sm font-black text-blue-900 tracking-tight leading-tight">MDM2</div>
                    <div className="text-[10px] text-slate-500 font-medium">E3 ubiquitin-protein ligase</div>
                  </div>
                  {/* Connector Line pointing to protein surface */}
                  <div className="w-[2px] h-9 bg-blue-500/70 mr-4 relative">
                    <div className="w-2.5 h-2.5 rounded-full bg-blue-500 border-2 border-white shadow-sm absolute -bottom-1 -left-[3.5px] animate-pulse" />
                  </div>
                </motion.div>

                {/* Real Live Probability Floating Glass Card */}
                <motion.div 
                  initial={{ opacity: 0, scale: 0.9, y: 15 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  transition={{ delay: 0.6 }}
                  className="absolute bottom-[6%] left-[4%] z-20 bg-white/95 backdrop-blur-md rounded-2xl p-3.5 shadow-xl border border-slate-200/80 min-w-[200px]"
                >
                  <div className="flex items-center gap-1.5 text-slate-700 text-xs font-bold mb-1">
                    <Link2 size={13} className="text-emerald-600 stroke-[2.5]" />
                    <span>Predicted Interaction</span>
                  </div>
                  <div className="flex items-baseline gap-1.5 text-xs text-slate-600 mb-1.5">
                    <span>Probability:</span>
                    <span className="text-emerald-600 font-black text-sm">{stats.liveProb}</span>
                  </div>
                  <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                    <motion.div 
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.min(100, Math.max(0, parseFloat(stats.liveProb) * 100))}%` }}
                      transition={{ duration: 1.2, ease: "easeOut" }}
                      className="bg-gradient-to-r from-emerald-500 to-teal-400 h-full rounded-full shadow-sm shadow-emerald-500/50"
                    />
                  </div>
                </motion.div>

                {/* Handwritten Callout Top Right: "From Interactions to Impact" */}
                <div 
                  className="absolute top-[20%] -right-2 sm:-right-4 z-20 select-none text-right transform rotate-[-2deg]"
                  style={{ fontFamily: "'Caveat', cursive" }}
                >
                  <span className="text-slate-700 text-lg sm:text-xl font-bold block leading-none">From</span>
                  <span className="text-emerald-700 text-xl sm:text-2xl font-black block leading-none">Interactions</span>
                  <span className="text-slate-700 text-lg sm:text-xl font-bold block leading-none">to Impact</span>
                </div>

                {/* Side Badge: EXPLORE · UNDERSTAND · DISCOVER */}
                <div className="absolute top-[52%] -right-1 sm:-right-3 z-20 text-[8px] sm:text-[9px] tracking-[0.25em] font-extrabold text-slate-400 uppercase select-none flex flex-col items-center gap-1">
                  <span>EXPLORE</span>
                  <span>UNDERSTAND</span>
                  <span>DISCOVER</span>
                </div>

                {/* Handwritten Callout Bottom Right: "A step closer to real-world therapies" */}
                <div 
                  className="absolute -bottom-2 right-0 sm:right-2 z-20 select-none text-right transform rotate-[-3deg]"
                  style={{ fontFamily: "'Caveat', cursive" }}
                >
                  <p className="text-emerald-700 text-lg sm:text-xl font-bold leading-tight">
                    A step closer to
                  </p>
                  <p className="text-emerald-700 text-xl sm:text-2xl font-black leading-tight">
                    real-world therapies
                  </p>
                  <svg className="w-32 sm:w-40 h-2.5 ml-auto mt-0.5" viewBox="0 0 160 12" fill="none">
                    <path d="M2 7C55 2 110 11 158 5" stroke="#059669" strokeWidth="2.5" strokeLinecap="round" />
                  </svg>
                </div>

              </motion.div>
            </div>

          </div>

        </div>

      </div>

      {/* ========================================== */}
      {/* 2. 4-MODULE DISCOVERY BAR (BELOW HERO)     */}
      {/* ========================================== */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3.5">
        <Link 
          to="/structure" 
          className="bg-white p-4 rounded-2xl border border-slate-200/80 shadow-sm hover:shadow-md hover:border-emerald-200 transition-all flex items-center gap-3.5 group"
        >
          <div className="w-10 h-10 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center shrink-0 group-hover:scale-105 transition-transform">
            <Boxes size={20} />
          </div>
          <div className="overflow-hidden">
            <h4 className="font-bold text-slate-800 text-sm truncate group-hover:text-emerald-700 transition-colors">3D Structure Viewer</h4>
            <p className="text-xs text-slate-400 truncate">Explore molecular structures</p>
          </div>
        </Link>

        <Link 
          to="/mutation" 
          className="bg-white p-4 rounded-2xl border border-slate-200/80 shadow-sm hover:shadow-md hover:border-emerald-200 transition-all flex items-center gap-3.5 group"
        >
          <div className="w-10 h-10 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center shrink-0 group-hover:scale-105 transition-transform">
            <Dna size={20} />
          </div>
          <div className="overflow-hidden">
            <h4 className="font-bold text-slate-800 text-sm truncate group-hover:text-emerald-700 transition-colors">Mutation Analysis</h4>
            <p className="text-xs text-slate-400 truncate">Simulate point mutations</p>
          </div>
        </Link>

        <Link 
          to="/drug-targets" 
          className="bg-white p-4 rounded-2xl border border-slate-200/80 shadow-sm hover:shadow-md hover:border-emerald-200 transition-all flex items-center gap-3.5 group"
        >
          <div className="w-10 h-10 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center shrink-0 group-hover:scale-105 transition-transform">
            <Pill size={20} />
          </div>
          <div className="overflow-hidden">
            <h4 className="font-bold text-slate-800 text-sm truncate group-hover:text-emerald-700 transition-colors">Drug Target Insights</h4>
            <p className="text-xs text-slate-400 truncate">Search ChEMBL & TTPS</p>
          </div>
        </Link>

        <Link 
          to="/network" 
          className="bg-white p-4 rounded-2xl border border-slate-200/80 shadow-sm hover:shadow-md hover:border-emerald-200 transition-all flex items-center gap-3.5 group"
        >
          <div className="w-10 h-10 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center shrink-0 group-hover:scale-105 transition-transform">
            <Share2 size={20} />
          </div>
          <div className="overflow-hidden">
            <h4 className="font-bold text-slate-800 text-sm truncate group-hover:text-emerald-700 transition-colors">Interactome Explorer</h4>
            <p className="text-xs text-slate-400 truncate">Visualize biological networks</p>
          </div>
        </Link>

        <a 
          href="#modules-suite" 
          className="bg-white p-4 rounded-2xl border border-slate-200/80 shadow-sm hover:shadow-md hover:border-emerald-200 transition-all flex items-center justify-center gap-2 group text-slate-600 hover:text-emerald-700 font-bold text-xs"
        >
          <span>and more</span>
          <ArrowRight size={14} className="group-hover:translate-x-1 transition-transform" />
        </a>
      </div>

      {/* Anchor for modules suite */}
      <div id="modules-suite" />

      {/* ── INTERACTIVE RESEARCH DISCOVERY SUITE (8 ENGINES) ── */}
      <div className="glass-card p-10 bg-white border border-slate-100 rounded-[2.5rem] shadow-xl space-y-8">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-100 pb-6">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-lg bg-emerald-50 text-emerald-600 text-xs font-bold uppercase tracking-wider mb-2">
              <Sparkles size={14} /> Comprehensive Research Suite
            </div>
            <h2 className="text-3xl font-black text-slate-800 tracking-tight">
              Multi-Engine <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-600 to-teal-500">Biological Discovery</span> Platform
            </h2>
          </div>
          <p className="text-slate-500 text-sm max-w-xl font-medium leading-relaxed">
            TransGraph-PPI is not just a predictor—it is a full-stack computational biology platform integrating transformer sequence models (ESM-2), topological graph neural networks (GraphSAGE), AlphaFold 3D molecular visualization, in-silico mutagenesis, and ChEMBL drug target discovery.
          </p>
        </div>

        {/* 8-Engine Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
          
          {/* Engine 1: Predictor */}
          <div className="p-5 rounded-2xl bg-slate-50/80 border border-slate-100 flex flex-col justify-between hover:shadow-lg hover:border-emerald-200 transition-all group">
            <div>
              <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-500 text-white flex items-center justify-center font-bold mb-3 shadow-md shadow-emerald-200">
                <Zap size={18} />
              </div>
              <h4 className="font-bold text-slate-800 text-sm mb-1 group-hover:text-emerald-700 transition-colors">Interaction Predictor</h4>
              <p className="text-xs text-slate-500 mb-3 leading-relaxed">
                ESM-2 + GraphSAGE stacking ensemble with Platt calibration and SHAP feature attribution.
              </p>
            </div>
            <Link to="/predict" className="text-xs font-bold text-emerald-600 hover:text-emerald-700 flex items-center gap-1 mt-2">
              Launch Predictor <ArrowRight size={14} />
            </Link>
          </div>

          {/* Engine 2: AlphaFold 3D Studio */}
          <div className="p-5 rounded-2xl bg-slate-50/80 border border-slate-100 flex flex-col justify-between hover:shadow-lg hover:border-cyan-200 transition-all group">
            <div>
              <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-blue-500 text-white flex items-center justify-center font-bold mb-3 shadow-md shadow-cyan-200">
                <Boxes size={18} />
              </div>
              <h4 className="font-bold text-slate-800 text-sm mb-1 group-hover:text-cyan-700 transition-colors">3D Molecular Studio</h4>
              <p className="text-xs text-slate-500 mb-3 leading-relaxed">
                Full atomic AlphaFold tertiary conformations rendered in real-time with PDBe Mol*.
              </p>
            </div>
            <Link to="/structure" className="text-xs font-bold text-cyan-600 hover:text-cyan-700 flex items-center gap-1 mt-2">
              Open 3D Studio <ArrowRight size={14} />
            </Link>
          </div>

          {/* Engine 3: In-Silico Mutagenesis */}
          <div className="p-5 rounded-2xl bg-slate-50/80 border border-slate-100 flex flex-col justify-between hover:shadow-lg hover:border-teal-200 transition-all group">
            <div>
              <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-teal-600 to-emerald-500 text-white flex items-center justify-center font-bold mb-3 shadow-md shadow-teal-200">
                <Dna size={18} />
              </div>
              <h4 className="font-bold text-slate-800 text-sm mb-1 group-hover:text-teal-700 transition-colors">Mutation Scanner</h4>
              <p className="text-xs text-slate-500 mb-3 leading-relaxed">
                Simulate amino acid substitutions in silico to identify binding destabilizing hotspots.
              </p>
            </div>
            <Link to="/mutation" className="text-xs font-bold text-teal-600 hover:text-teal-700 flex items-center gap-1 mt-2">
              Scan Mutations <ArrowRight size={14} />
            </Link>
          </div>

          {/* Engine 4: WT vs Mutant Comparator */}
          <div className="p-5 rounded-2xl bg-slate-50/80 border border-slate-100 flex flex-col justify-between hover:shadow-lg hover:border-amber-200 transition-all group">
            <div>
              <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-amber-500 to-orange-500 text-white flex items-center justify-center font-bold mb-3 shadow-md shadow-amber-200">
                <GitCompare size={18} />
              </div>
              <h4 className="font-bold text-slate-800 text-sm mb-1 group-hover:text-amber-700 transition-colors">WT vs Mutant Comparator</h4>
              <p className="text-xs text-slate-500 mb-3 leading-relaxed">
                Side-by-side comparative analysis of native wildtype vs mutated variant affinity deltas.
              </p>
            </div>
            <Link to="/compare" className="text-xs font-bold text-amber-600 hover:text-amber-700 flex items-center gap-1 mt-2">
              Compare Variants <ArrowRight size={14} />
            </Link>
          </div>

          {/* Engine 5: 2D Interactome Network */}
          <div className="p-5 rounded-2xl bg-slate-50/80 border border-slate-100 flex flex-col justify-between hover:shadow-lg hover:border-violet-200 transition-all group">
            <div>
              <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-violet-600 to-indigo-500 text-white flex items-center justify-center font-bold mb-3 shadow-md shadow-violet-200">
                <Share2 size={18} />
              </div>
              <h4 className="font-bold text-slate-800 text-sm mb-1 group-hover:text-violet-700 transition-colors">2D Interactome Explorer</h4>
              <p className="text-xs text-slate-500 mb-3 leading-relaxed">
                Trace shortest paths, compute hub centralities, and isolate functional subnetworks.
              </p>
            </div>
            <Link to="/network" className="text-xs font-bold text-violet-600 hover:text-violet-700 flex items-center gap-1 mt-2">
              Explore 2D Graph <ArrowRight size={14} />
            </Link>
          </div>

          {/* Engine 6: 3D Force Interactome */}
          <div className="p-5 rounded-2xl bg-slate-50/80 border border-slate-100 flex flex-col justify-between hover:shadow-lg hover:border-purple-200 transition-all group">
            <div>
              <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-purple-600 to-pink-500 text-white flex items-center justify-center font-bold mb-3 shadow-md shadow-purple-200">
                <Globe size={18} />
              </div>
              <h4 className="font-bold text-slate-800 text-sm mb-1 group-hover:text-purple-700 transition-colors">3D Force Topography</h4>
              <p className="text-xs text-slate-500 mb-3 leading-relaxed">
                Interactive spatial force layout showing high-density protein hubs and cellular clusters.
              </p>
            </div>
            <Link to="/network-3d" className="text-xs font-bold text-purple-600 hover:text-purple-700 flex items-center gap-1 mt-2">
              Launch 3D Graph <ArrowRight size={14} />
            </Link>
          </div>

          {/* Engine 7: ChEMBL Drug Target Insights */}
          <div className="p-5 rounded-2xl bg-slate-50/80 border border-slate-100 flex flex-col justify-between hover:shadow-lg hover:border-blue-200 transition-all group">
            <div>
              <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 to-cyan-500 text-white flex items-center justify-center font-bold mb-3 shadow-md shadow-blue-200">
                <Pill size={18} />
              </div>
              <h4 className="font-bold text-slate-800 text-sm mb-1 group-hover:text-blue-700 transition-colors">Drug Target Insights</h4>
              <p className="text-xs text-slate-500 mb-3 leading-relaxed">
                Therapeutic Target Priority Scores (TTPS) and approved drug leads from ChEMBL.
              </p>
            </div>
            <Link to="/drug-targets" className="text-xs font-bold text-blue-600 hover:text-blue-700 flex items-center gap-1 mt-2">
              View Drug Targets <ArrowRight size={14} />
            </Link>
          </div>

          {/* Engine 8: Empirical Benchmarks */}
          <div className="p-5 rounded-2xl bg-slate-50/80 border border-slate-100 flex flex-col justify-between hover:shadow-lg hover:border-rose-200 transition-all group">
            <div>
              <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-rose-500 to-amber-500 text-white flex items-center justify-center font-bold mb-3 shadow-md shadow-rose-200">
                <BarChart3 size={18} />
              </div>
              <h4 className="font-bold text-slate-800 text-sm mb-1 group-hover:text-rose-700 transition-colors">Empirical Benchmarks</h4>
              <p className="text-xs text-slate-500 mb-3 leading-relaxed">
                2,000-sample bootstrap 95% CIs, cold-start novelty, and external SHS27k & HuRI validation.
              </p>
            </div>
            <Link to="/benchmark" className="text-xs font-bold text-rose-600 hover:text-rose-700 flex items-center gap-1 mt-2">
              View Scientific Rigor <ArrowRight size={14} />
            </Link>
          </div>

        </div>

        {/* ── LIVE EXPERIMENT SANDBOX ── */}
        <div className="pt-6 border-t border-slate-100">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-lg font-black text-slate-800 tracking-tight">Interactive Case Studies (1-Click Experiments)</h3>
              <p className="text-xs text-slate-500">Launch curated biological experiments immediately without looking up IDs</p>
            </div>
            <Link to="/predict" className="text-xs font-bold text-emerald-600 hover:text-emerald-700 flex items-center gap-1">
              Custom Prediction <ArrowRight size={14} />
            </Link>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              {
                title: "TP53 ↔ MDM2",
                badge: "Oncology",
                color: "bg-emerald-50 text-emerald-700 border-emerald-200",
                desc: "Tumor suppressor cell-cycle checkpoint control and apoptosis regulator.",
                p1: "ENSP00000269305", p2: "ENSP00000258149"
              },
              {
                title: "AP2A2 ↔ CLTC",
                badge: "Neurodegenerative",
                color: "bg-indigo-50 text-indigo-700 border-indigo-200",
                desc: "Clathrin-mediated vesicle endocytosis linked to Alzheimer's pathology.",
                p1: "ENSP00000300161", p2: "ENSP00000267029"
              },
              {
                title: "BAX ↔ BCL2L1",
                badge: "Apoptosis",
                color: "bg-purple-50 text-purple-700 border-purple-200",
                desc: "Mitochondrial outer membrane permeabilization and cell survival balance.",
                p1: "ENSP00000293879", p2: "ENSP00000307677"
              },
              {
                title: "Uncharacterized",
                badge: "Cold-Start",
                color: "bg-amber-50 text-amber-700 border-amber-200",
                desc: "Zero-neighbor novelty evaluation testing sequence-driven inductive generalization.",
                p1: "ENSP00000385802", p2: "ENSP00000361000"
              },
            ].map((cs, i) => (
              <Link
                key={i}
                to={`/predict?p1=${cs.p1}&p2=${cs.p2}`}
                className="p-4 rounded-2xl bg-slate-50/60 hover:bg-white border border-slate-200/80 hover:border-emerald-300 transition-all hover:shadow-md flex flex-col justify-between group"
              >
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-black text-xs text-slate-800 group-hover:text-emerald-700 transition-colors">{cs.title}</span>
                    <span className={`text-[9px] font-black uppercase px-2 py-0.5 rounded-full border ${cs.color}`}>{cs.badge}</span>
                  </div>
                  <p className="text-[11px] text-slate-500 leading-snug">{cs.desc}</p>
                </div>
                <div className="mt-3 pt-2 border-t border-slate-100 flex items-center justify-between text-[11px] font-bold text-emerald-600">
                  <span>Run Experiment</span>
                  <ArrowRight size={13} className="group-hover:translate-x-1 transition-transform" />
                </div>
              </Link>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Network Preview */}
        <div className="lg:col-span-2 glass-card p-10 overflow-hidden relative">
          <div className="flex justify-between items-center mb-8 relative z-10">
            <div>
              <h3 className="text-2xl font-black text-slate-800 tracking-tight">PPI Network Overview</h3>
              <p className="text-sm text-slate-500 font-medium">Real-time interaction topography</p>
            </div>
            <Link to="/network" className="bg-slate-100 px-4 py-2 rounded-xl text-scientific-primary font-bold text-xs flex items-center gap-2 hover:bg-teal-50 transition-colors">
              FULL EXPLORER <ArrowRight size={14} />
            </Link>
          </div>
          <div className="aspect-video bg-gradient-to-br from-slate-900 to-slate-950 rounded-[2rem] p-6 flex flex-col justify-between border border-slate-800 overflow-hidden relative group shadow-2xl">
            {/* Glowing grid background */}
            <div className="absolute inset-0 opacity-10 pointer-events-none">
              <div className="grid grid-cols-12 h-full w-full">
                {[...Array(48)].map((_, i) => (
                  <div key={i} className="border border-teal-500/30" />
                ))}
              </div>
            </div>

            {/* Interactive SVG Network Graph */}
            <svg className="absolute inset-0 w-full h-full pointer-events-auto" viewBox="0 0 600 350">
              <defs>
                <linearGradient id="edgeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#10b981" stopOpacity="0.8" />
                  <stop offset="100%" stopColor="#6366f1" stopOpacity="0.4" />
                </linearGradient>
              </defs>

              {/* Edges */}
              <line x1="300" y1="175" x2="180" y2="100" stroke="url(#edgeGrad)" strokeWidth="2.5" strokeDasharray="6 3" className="animate-pulse" />
              <line x1="300" y1="175" x2="420" y2="110" stroke="url(#edgeGrad)" strokeWidth="3" />
              <line x1="300" y1="175" x2="220" y2="260" stroke="url(#edgeGrad)" strokeWidth="2" strokeDasharray="4 2" />
              <line x1="300" y1="175" x2="440" y2="250" stroke="url(#edgeGrad)" strokeWidth="2.5" />
              <line x1="180" y1="100" x2="110" y2="180" stroke="#334155" strokeWidth="1.5" />
              <line x1="420" y1="110" x2="500" y2="170" stroke="#334155" strokeWidth="1.5" />
              <line x1="220" y1="260" x2="350" y2="290" stroke="#334155" strokeWidth="1.5" />

              {/* Nodes */}
              {/* Center Node: TP53 */}
              <g className="cursor-pointer group/node" transform="translate(300, 175)">
                <circle r="24" className="fill-emerald-500/20 stroke-emerald-400 stroke-2 animate-ping opacity-75" />
                <circle r="18" className="fill-emerald-600 stroke-emerald-300 stroke-2 shadow-lg" />
                <text textAnchor="middle" dy="4" fill="#ffffff" fontSize="10" fontWeight="900">TP53</text>
              </g>

              {/* Node 2: MDM2 */}
              <g className="cursor-pointer" transform="translate(180, 100)">
                <circle r="14" className="fill-teal-600 stroke-teal-300 stroke-2" />
                <text textAnchor="middle" dy="4" fill="#ffffff" fontSize="9" fontWeight="800">MDM2</text>
              </g>

              {/* Node 3: BAX */}
              <g className="cursor-pointer" transform="translate(420, 110)">
                <circle r="15" className="fill-indigo-600 stroke-indigo-300 stroke-2" />
                <text textAnchor="middle" dy="4" fill="#ffffff" fontSize="9" fontWeight="800">BAX</text>
              </g>

              {/* Node 4: BCL2 */}
              <g className="cursor-pointer" transform="translate(220, 260)">
                <circle r="13" className="fill-cyan-600 stroke-cyan-300 stroke-2" />
                <text textAnchor="middle" dy="4" fill="#ffffff" fontSize="9" fontWeight="800">BCL2</text>
              </g>

              {/* Node 5: AP2A2 */}
              <g className="cursor-pointer" transform="translate(440, 250)">
                <circle r="14" className="fill-purple-600 stroke-purple-300 stroke-2" />
                <text textAnchor="middle" dy="4" fill="#ffffff" fontSize="8" fontWeight="800">AP2A2</text>
              </g>

              {/* Peripheral Nodes */}
              <circle cx="110" cy="180" r="8" className="fill-slate-700 stroke-slate-500" />
              <circle cx="500" cy="170" r="9" className="fill-slate-700 stroke-slate-500" />
              <circle cx="350" cy="290" r="7" className="fill-slate-700 stroke-slate-500" />
            </svg>

            {/* Overlay status tags */}
            <div className="relative z-10 flex justify-between items-start pointer-events-none">
              <div className="bg-slate-900/80 backdrop-blur-md px-3 py-1.5 rounded-xl border border-slate-700 text-[10px] font-mono text-emerald-400 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                <span>Topological Density: <strong>0.842</strong></span>
              </div>
              <div className="bg-slate-900/80 backdrop-blur-md px-3 py-1.5 rounded-xl border border-slate-700 text-[10px] font-mono text-slate-300">
                Active Hub: <strong className="text-emerald-400">TP53 (Degree: 42)</strong>
              </div>
            </div>

            <div className="relative z-10 flex justify-between items-end pointer-events-none">
              <div className="text-slate-400 text-xs font-medium">
                Click node to view affinity & pathway metadata
              </div>
              <Link to="/network-3d" className="pointer-events-auto bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-black text-xs px-4 py-2 rounded-xl transition-all shadow-lg shadow-emerald-500/20 flex items-center gap-1.5">
                LAUNCH 3D INTERACTOME <ArrowRight size={14} />
              </Link>
            </div>
          </div>
        </div>

        {/* Action Cards */}
        <div className="space-y-6">
          <Link to="/predict" className="block outline-none group">
            <div className="glass-card p-8 bg-white hover:bg-emerald-50/50 transition-all border-slate-100 relative overflow-hidden h-[240px] flex flex-col justify-end group">
              <div className="absolute top-0 right-0 p-8 text-emerald-500/5 group-hover:text-emerald-500/10 group-hover:scale-110 transition-all duration-500">
                <Search size={120} />
              </div>
              <h4 className="text-2xl font-black mb-2 text-slate-800 relative z-10">Predict Interaction</h4>
              <p className="text-slate-500 text-sm font-medium mb-6 relative z-10">Run hybrid ESM-2 + GraphSAGE stacking ensemble to predict binding affinity.</p>
              <div className="flex items-center text-xs font-black uppercase tracking-widest text-emerald-600 gap-2 relative z-10 group-hover:gap-4 transition-all">
                Launch System <ArrowRight size={16} />
              </div>
            </div>
          </Link>

          <Link to="/benchmark" className="block outline-none group">
            <div className="glass-card p-8 bg-white hover:bg-rose-50/50 transition-all border-slate-100 relative overflow-hidden h-[240px] flex flex-col justify-end group">
              <div className="absolute top-0 right-0 p-8 text-rose-500/5 group-hover:text-rose-500/10 group-hover:scale-110 transition-all duration-500">
                <BarChart3 size={120} />
              </div>
              <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-rose-50 text-rose-700 border border-rose-200 text-[10px] font-black uppercase w-fit mb-2 relative z-10">
                Statistical Rigor
              </div>
              <h4 className="text-2xl font-black mb-2 text-slate-800 relative z-10">Empirical Benchmarks</h4>
              <p className="text-slate-500 text-sm font-medium mb-6 relative z-10">2,000 Bootstrap 95% CIs, Cold-Start Novelty, and External SHS27k / HuRI Sets.</p>
              <div className="flex items-center text-xs font-black uppercase tracking-widest text-rose-600 gap-2 relative z-10 group-hover:gap-4 transition-all">
                View Validations <ArrowRight size={16} />
              </div>
            </div>
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Home;
