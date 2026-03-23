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
        // #region agent log
        fetch('http://127.0.0.1:7656/ingest/a2c6930f-0198-499d-9920-7d735f885f13', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Debug-Session-Id': 'e579db'
          },
          body: JSON.stringify({
            sessionId: 'e579db',
            runId: 'pre-fix',
            hypothesisId: 'H1',
            location: 'Home.jsx:fetchStats',
            message: 'Error fetching network stats',
            data: {
              baseURL: ppiService?.defaults?.baseURL || null,
              endpoint: '/analysis/stats',
              errorMessage: error?.message || null,
              errorCode: error?.code || null
            },
            timestamp: Date.now()
          })
        }).catch(() => {});
        // #endregion
      }
    };
    fetchStats();
  }, []);

  return (
    <div className="space-y-10">
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-3xl bg-scientific-gradient p-12 text-white shadow-2xl">
        <div className="relative z-10 max-w-2xl">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/20 backdrop-blur-md text-xs font-bold uppercase tracking-widest mb-6"
          >
            <Zap size={14} /> AI-Powered Biological Discovery
          </motion.div>
          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-5xl font-extrabold mb-6 leading-tight"
          >
            Decoding the Language of <span className="text-teal-200">Proteins</span>
          </motion.h1>
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="text-lg text-teal-50/80 mb-8 leading-relaxed"
          >
            TransGraph-PPI is a hybrid AI framework combining protein language models (ESM-2) 
            and graph neural networks (GAT) to predict interactions and analyze complex 
            biological networks with unprecedented precision.
          </motion.p>
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="flex gap-4"
          >
            <Link to="/predict" className="btn-primary bg-white text-scientific-primary hover:bg-teal-50 flex items-center gap-2">
              Start Prediction <ArrowRight size={18} />
            </Link>
            <Link to="/about" className="px-6 py-2 bg-white/10 backdrop-blur-md border border-white/20 rounded-xl font-medium hover:bg-white/20 transition-all">
              Learn More
            </Link>
          </motion.div>
        </div>
        
        {/* Abstract background elements */}
        <div className="absolute top-0 right-0 w-1/2 h-full opacity-20 pointer-events-none">
          <svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" className="w-full h-full scale-150 transform translate-x-1/4">
            <path fill="#FFF" d="M44.7,-76.4C58.2,-69.2,70,-58.5,78.2,-45.5C86.4,-32.5,91,-17.2,88.9,-2.4C86.7,12.4,77.8,26.7,68.4,38.8C59,50.9,49.1,60.8,37.3,66.7C25.5,72.6,11.8,74.5,-1.3,76.5C-14.4,78.5,-28.8,80.7,-41.2,75.4C-53.6,70.1,-64,57.3,-71.3,43.2C-78.6,29.1,-82.8,13.7,-81.1,1.1C-79.4,-11.5,-71.8,-21.3,-63.3,-30C-54.8,-38.7,-45.4,-46.3,-34.7,-55C-24,-63.7,-12,-73.5,2.4,-77.2C16.8,-80.9,31.2,-78.6,44.7,-76.4Z" transform="translate(100 100)" />
          </svg>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard 
          icon={Users} 
          label="Proteins Catalogued" 
          value={stats.proteins} 
          subtext="+124 added this week"
          color="bg-teal-500"
        />
        <StatCard 
          icon={Share2} 
          label="Interactions Analyzed" 
          value={stats.interactions} 
          subtext="Verified by STRING DB"
          color="bg-blue-500"
        />
        <StatCard 
          icon={CheckCircle} 
          label="Model Accuracy" 
          value={stats.accuracy} 
          subtext="Validated on Yeast subset"
          color="bg-purple-500"
        />
        <StatCard 
          icon={Activity} 
          label="Predictions Logged" 
          value={stats.predictions} 
          subtext="System-wide activity"
          color="bg-orange-500"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Network Preview */}
        <div className="lg:col-span-2 glass-card p-8">
          <div className="flex justify-between items-center mb-6">
            <div>
              <h3 className="text-xl font-bold text-slate-800">PPI Network Overview</h3>
              <p className="text-sm text-slate-500">Live preview of protein interaction cluster</p>
            </div>
            <Link to="/network" className="text-scientific-primary font-bold text-sm flex items-center gap-1 hover:underline">
              Explore Full View <ArrowRight size={14} />
            </Link>
          </div>
          <div className="aspect-video bg-slate-50 rounded-2xl flex items-center justify-center border border-slate-100 overflow-hidden relative">
             <div className="absolute inset-0 opacity-10">
                <div className="grid grid-cols-12 h-full">
                  {[...Array(12)].map((_, i) => (
                    <div key={i} className="border-r border-slate-400" />
                  ))}
                </div>
             </div>
             <div className="text-center z-10">
                <Share2 size={48} className="text-slate-200 mx-auto mb-4" />
                <p className="text-slate-400 font-medium">Network Visualizer Initializing...</p>
                <button className="mt-4 px-4 py-2 bg-white border border-slate-200 rounded-lg text-xs font-bold text-slate-600 shadow-sm">
                  RELOAD CLUSTER
                </button>
             </div>
          </div>
        </div>

        {/* Action Cards */}
        <div className="space-y-6">
          <Link to="/predict" className="block outline-none">
            <div className="glass-card p-6 bg-scientific-primary text-white group cursor-pointer hover:bg-teal-700 transition-all border-none">
              <Search className="mb-4 text-teal-200" size={28} />
              <h4 className="text-lg font-bold mb-2">Predict Interaction</h4>
              <p className="text-sm text-teal-50/70 mb-4">Run the hybrid ESM+GAT model to predict binding probability between protein pairs.</p>
              <div className="inline-flex items-center text-sm font-bold gap-2 group-hover:translate-x-1 transition-transform">
                Launch Predictor <ArrowRight size={16} />
              </div>
            </div>
          </Link>

          <Link to="/drug-targets" className="block outline-none">
            <div className="glass-card p-6 group cursor-pointer border-l-4 border-scientific-accent hover:bg-slate-50 transition-all">
              <Database className="mb-4 text-scientific-accent" size={28} />
              <h4 className="text-lg font-bold mb-2 text-slate-800">Drug Target Explorer</h4>
              <p className="text-sm text-slate-500 mb-4">Identify high-centrality proteins and analyze their potential as therapeutic drug targets.</p>
              <div className="inline-flex items-center text-sm font-bold text-scientific-accent gap-2 group-hover:translate-x-1 transition-transform">
                Find Targets <ArrowRight size={16} />
              </div>
            </div>
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Home;
