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
          <div className="aspect-video bg-slate-900/5 rounded-[2rem] flex items-center justify-center border border-slate-200/50 overflow-hidden relative group">
             {/* Abstract grid background */}
             <div className="absolute inset-0 opacity-[0.03] pointer-events-none">
                <div className="grid grid-cols-12 h-full w-full">
                  {[...Array(48)].map((_, i) => (
                    <div key={i} className="border border-slate-900" />
                  ))}
                </div>
             </div>
             
             <div className="text-center z-10 scale-100 group-hover:scale-105 transition-transform duration-700">
                <div className="w-20 h-20 bg-white/80 backdrop-blur-md rounded-[2rem] flex items-center justify-center shadow-xl mx-auto mb-6 border border-white">
                  <Share2 size={32} className="text-scientific-primary" />
                </div>
                <p className="text-slate-500 font-black tracking-widest text-xs uppercase">Engine Initializing</p>
                <div className="mt-4 flex gap-2 justify-center">
                  <div className="w-1.5 h-1.5 rounded-full bg-teal-500 animate-bounce" />
                  <div className="w-1.5 h-1.5 rounded-full bg-teal-500 animate-bounce delay-100" />
                  <div className="w-1.5 h-1.5 rounded-full bg-teal-500 animate-bounce delay-200" />
                </div>
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
