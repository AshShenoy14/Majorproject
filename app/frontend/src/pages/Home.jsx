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
  Share2
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { ppiService } from '../services/api';

const StatCard = ({ icon: Icon, label, value, subtext, color }) => (
  <motion.div 
    whileHover={{ y: -5 }}
    className="glass-card p-6 flex flex-col items-start"
  >
    <div className={`p-3 rounded-xl mb-4 ${color}`}>
      <Icon size={24} className="text-white" />
    </div>
    <h3 className="text-3xl font-bold text-slate-800">{value}</h3>
    <p className="text-sm font-semibold text-slate-500 uppercase tracking-wider mt-1">{label}</p>
    <p className="text-xs text-slate-400 mt-2">{subtext}</p>
  </motion.div>
);

const Home = () => {
  const [stats, setStats] = useState({
    proteins: '14,208',
    interactions: '42,519',
    accuracy: '94.2%',
    predictions: '1,204'
  });

  useEffect(() => {
    // Fetch real stats if available
    const fetchStats = async () => {
      try {
        const response = await ppiService.getNetworkStats();
        if (response.data) {
          setStats(prev => ({
            ...prev,
            proteins: response.data.num_nodes || prev.proteins,
            interactions: response.data.num_edges || prev.interactions
          }));
        }
      } catch (error) {
        console.error("Error fetching stats:", error);
      }
    };
    fetchStats();
  }, []);

  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-[3rem] bg-white min-h-[450px] flex items-center shadow-xl border border-slate-100">
        {/* Background Image with Overlay */}
        <div 
          className="absolute inset-0 z-0 bg-cover bg-center opacity-[0.06]"
          style={{ backgroundImage: "url('/ppi_hero_bg_1777021983794.png')" }}
        />
        <div className="absolute inset-0 z-0 bg-gradient-to-r from-white via-white/40 to-transparent" />
        
        <div className="relative z-10 p-16 max-w-3xl">
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-50 text-emerald-600 text-[10px] font-black uppercase tracking-[0.2em] mb-8"
          >
            <Zap size={14} className="text-emerald-500" /> AI-Powered Biological Discovery System
          </motion.div>
          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-6xl font-black mb-6 leading-[1.1] text-slate-800 tracking-tight"
          >
            Decoding the Language of <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-600 to-teal-400">Proteins</span>
          </motion.h1>
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="text-xl text-slate-500 mb-10 leading-relaxed font-medium"
          >
            <span className="font-cursive text-emerald-600 text-2xl">TransGraph PPI</span> is a state-of-the-art hybrid AI framework combining protein language models (ESM-2) 
            and graph neural networks (GAT) to analyze complex biological interactomes.
          </motion.p>
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="flex gap-5"
          >
            <Link to="/predict" className="btn-primary shadow-2xl shadow-emerald-200">
              Start Analysis <ArrowRight size={20} />
            </Link>
            <Link to="/about" className="px-8 py-3 bg-slate-50 text-slate-600 rounded-2xl font-bold transition-all duration-300 border border-slate-100 hover:bg-slate-100 hover:text-slate-800 flex items-center justify-center gap-2">
              Technical Specs
            </Link>
          </motion.div>
        </div>
        
        {/* Floating elements for visual interest */}
        <div className="absolute top-1/4 right-20 w-48 h-48 bg-emerald-100/30 blur-[80px] rounded-full animate-pulse" />
        <div className="absolute bottom-1/4 right-40 w-64 h-64 bg-teal-100/20 blur-[100px] rounded-full animate-pulse delay-700" />
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard 
          icon={Users} 
          label="Proteins Catalogued" 
          value={stats.proteins} 
          subtext="+124 added this week"
          color="bg-gradient-to-br from-teal-400 to-teal-600"
        />
        <StatCard 
          icon={Share2} 
          label="Interactions Analyzed" 
          value={stats.interactions} 
          subtext="Verified by STRING DB"
          color="bg-gradient-to-br from-blue-400 to-blue-600"
        />
        <StatCard 
          icon={CheckCircle} 
          label="Model Accuracy" 
          value={stats.accuracy} 
          subtext="Validated on Yeast subset"
          color="bg-gradient-to-br from-purple-400 to-purple-600"
        />
        <StatCard 
          icon={Activity} 
          label="Predictions Logged" 
          value={stats.predictions} 
          subtext="System-wide activity"
          color="bg-gradient-to-br from-orange-400 to-orange-600"
        />
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
               <Link to="/network" className="pointer-events-auto bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-black text-xs px-4 py-2 rounded-xl transition-all shadow-lg shadow-emerald-500/20 flex items-center gap-1.5">
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
              <p className="text-slate-500 text-sm font-medium mb-6 relative z-10">Run hybrid ESM+GAT model to predict binding probability.</p>
              <div className="flex items-center text-xs font-black uppercase tracking-widest text-emerald-600 gap-2 relative z-10 group-hover:gap-4 transition-all">
                Launch System <ArrowRight size={16} />
              </div>
            </div>
          </Link>

          <Link to="/drug-targets" className="block outline-none group">
            <div className="glass-card p-8 hover:bg-white transition-all h-[240px] flex flex-col justify-end relative overflow-hidden">
               <div className="absolute top-0 right-0 p-8 text-scientific-accent/5 group-hover:text-scientific-accent/10 transition-all duration-500">
                <Database size={120} />
              </div>
              <h4 className="text-2xl font-black mb-2 text-slate-800 relative z-10">Drug Insights</h4>
              <p className="text-slate-500 text-sm font-medium mb-6 relative z-10">Identify high-centrality proteins for therapeutic targeting.</p>
              <div className="flex items-center text-xs font-black uppercase tracking-widest text-scientific-accent gap-2 relative z-10">
                Open Database <ArrowRight size={16} />
              </div>
            </div>
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Home;
